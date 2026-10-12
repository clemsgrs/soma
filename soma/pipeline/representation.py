"""Task-free representation evaluation: CRoMa over frozen embeddings, no training.

``Pipeline.run`` dispatches here in one line when the config has a ``representation``
section; the run still gets the managed layout, the drift guards and a recorded status.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from rich.console import Console

from soma.config import RepresentationConfig
from soma.data import CachedSetSource, require_coverage
from soma.pipeline._legacy import LegacyFolds, LegacySamples
from soma.pipeline.aggregate import save_summary
from soma.pipeline.build import attach_source
from soma.pipeline.console import build_completed_run_panel
from soma.pipeline.layout import RunRecorder, open_run

if TYPE_CHECKING:
    from soma.pipeline.run import PipelineResult

__all__ = ["evaluate_representation", "run_representation"]


def run_representation(pipeline) -> "PipelineResult":
    """Run a ``representation`` config: open the run, attach the source, score CRoMa."""
    config = pipeline.config
    layout, _ = open_run(config, num_folds=pipeline.splits.num_folds)
    with RunRecorder(layout, config) as recorder:
        context = attach_source(
            config,
            source=pipeline.source,
            dataset=pipeline.dataset,
            splits=pipeline.splits,
            image_manifest=pipeline.image_manifest,
            run_dir=layout.run_dir,
        )
        log_path = layout.run_dir / "run.log"
        log_path.write_text("Starting task-free CRoMa representation evaluation.\n", encoding="utf-8")
        result = evaluate_representation(
            feature_store=context.source,
            dataset=context.dataset,
            splits=context.splits,
            representation=config.representation,
            run_dir=layout.run_dir,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("Completed task-free CRoMa representation evaluation.\n")
        Console().print(build_completed_run_panel(summary_metrics=result.summary))
        recorder.complete(result.summary)
        return recorder.with_identity(result)



def _representation_split_ids(
    splits: LegacyFolds, representation: RepresentationConfig
) -> list[str]:
    matches: list[tuple[int, tuple[str, ...]]] = []
    for fold_index, fold in enumerate(splits.folds):
        if representation.split == "train":
            ids = fold.train
        elif representation.split == "tune":
            ids = fold.tune
        else:
            ids = fold.tests.get(representation.split, ())
        if ids:
            matches.append((fold_index, ids))
    if len(matches) != 1:
        raise ValueError(
            f"representation.split={representation.split!r} must select a non-empty "
            f"cohort in exactly one fold; found it in {len(matches)} folds. "
            "Cross-validation and fold aggregation are not supported."
        )
    ids = list(matches[0][1])
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"representation.split={representation.split!r} contains duplicate membership."
        )
    return ids


def _required_representation_text(value: object, *, field_name: str, sample_id: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(
            f"Representation sample {sample_id!r} requires a non-empty {field_name}."
        )
    return str(value).strip()


def evaluate_representation(
    feature_store: CachedSetSource,
    dataset: LegacySamples,
    splits: LegacyFolds,
    representation: RepresentationConfig,
    run_dir: str | Path,
) -> "PipelineResult":
    """Evaluate selected frozen tile embeddings directly with CRoMa's public API."""
    from croma import CRoMa

    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    # A pinned/retried attempt reuses its run directory. Remove any prior successful
    # result before validation so a failed re-evaluation cannot leave rankable stale data.
    summary_path.unlink(missing_ok=True)

    selected_membership = _representation_split_ids(splits, representation)
    selected_set = set(selected_membership)
    selected_ids = [sample_id for sample_id in dataset.sample_ids if sample_id in selected_set]
    if len(selected_ids) != len(selected_membership) or set(selected_ids) != selected_set:
        raise ValueError(
            "Representation split membership does not match the dataset manifest IDs."
        )

    require_coverage(feature_store, selected_ids)
    manifest_rows: list[dict[str, str]] = []
    feature_rows: list[np.ndarray] = []
    feature_dim: int | None = None
    for sample_id in selected_ids:
        record = dataset.samples[sample_id]
        label = _required_representation_text(
            record.targets.get("label"), field_name="label", sample_id=sample_id
        )
        group_id = _required_representation_text(
            record.metadata.get("group_id"), field_name="group_id", sample_id=sample_id
        )
        confounder = _required_representation_text(
            record.metadata.get(representation.confounder_column),
            field_name=representation.confounder_column,
            sample_id=sample_id,
        )
        feature = feature_store.load(sample_id)
        if feature.ndim != 1:
            raise ValueError(
                "representation v1 requires one rank-1 feature per selected sample; "
                f"sample {sample_id!r} has rank {feature.ndim}."
            )
        if feature_dim is None:
            feature_dim = int(feature.shape[0])
        elif int(feature.shape[0]) != feature_dim:
            raise ValueError(
                "representation embeddings must have one common feature dimension; "
                f"sample {sample_id!r} has {feature.shape[0]}, expected {feature_dim}."
            )
        values = feature.detach().cpu().to(torch.float32).numpy()
        if not np.isfinite(values).all():
            raise ValueError(
                f"Representation embedding for sample {sample_id!r} contains non-finite values."
            )
        feature_rows.append(values)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(record.image_path),
                "label": label,
                "group_id": group_id,
                representation.confounder_column: confounder,
            }
        )

    features = np.stack(feature_rows, axis=0)
    manifest = pd.DataFrame(manifest_rows)
    if manifest["sample_id"].tolist() != selected_ids:
        raise AssertionError("CRoMa manifest IDs are not in selected dataset-manifest order.")
    if len(manifest) != features.shape[0] or features.shape[0] != len(selected_ids):
        raise AssertionError("CRoMa feature and manifest row counts differ.")

    result = CRoMa.compute(
        features,
        manifest,
        confounder_column=representation.confounder_column,
        evaluation_design=representation.evaluation_design,
        m=representation.m,
        alpha=representation.alpha,
    )
    metric_values = (float(result.value), float(result.f0), float(result.ltm_alpha))
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError(
            "CRoMa returned a non-finite median, F(0), or LTM10 value; check cohort "
            "support and embeddings."
        )

    summary = {
        f"{representation.split}/croma_median": metric_values[0],
        f"{representation.split}/croma_f0": metric_values[1],
        f"{representation.split}/croma_ltm10": metric_values[2],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    save_summary(summary, summary_path)

    sample_values = np.asarray(result.sample_values_aligned, dtype=float)
    if sample_values.ndim == 1 and len(sample_values) == len(selected_ids):
        pd.DataFrame(
            {"sample_id": selected_ids, "croma": sample_values}
        ).to_csv(run_dir / "croma_samples.csv", index=False)

    from soma.pipeline.run import PipelineResult

    return PipelineResult(fold_results=[], summary=summary, run_dir=run_dir)
