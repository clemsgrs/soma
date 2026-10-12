"""Pipeline — standalone step functions and Pipeline orchestrator.

Layer 1 (standalone):
    train_one_fold()  — train + evaluate one fold
    train()           — train all folds + summarize

Layer 2 (orchestrator):
    Pipeline          — wires everything together from a PipelineConfig
"""

from __future__ import annotations

import logging
import json
import functools
import inspect
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

from soma.data import require_coverage
from soma._logging import ensure_default_logging
from soma.aggregators.registry import aggregator_registry
from soma.artifact_mirror import ArtifactMirror, restore_run_from_mirror
from soma.atomic_io import atomic_write_json
from soma.cache import resolve_cache_root
from soma.class_scheme import resolve_classes
from soma.decoders.registry import build_decoder_for_grid
from soma.config import (
    AggregatorConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderMemberConfig,
    EvalConfig,
    HeatmapConfig,
    MasksConfig,
    NormalizationConfig,
    PipelineConfig,
    PreprocessingConfig,
    ProjectionConfig,
    RepresentationConfig,
    TaskConfig,
    TrainingConfig,
    config_yaml_dict,
    save_config,
    validate_feature_adaptor_compatibility,
)
from soma.data import (
    AnnotationManifest,
    CachedLabelMapSource,
    Cohort,
    FoldSplit,
    GridGeometry,
    ImageManifest,
    LabelMapEntry,
    LabelMapSource,
    PointEntry,
    PointSource,
    TileImageSource,
)
from soma.data.cohort import targets_frame
from soma.data.manifests import ImageEntry
from soma.data._legacy import (
    LegacyFolds,
    LegacyRecord,
    LegacySamples,
    records_for_pipeline,
)
from soma.dense.composite import CompositeDenseFeatureStore
from soma.dense.reader import CLASS_SCHEME_KEYS, resolve_class_scheme
from soma import evaluation as _verbs
from soma.evaluation import Evaluation
from soma.evaluation.metrics import metric_registry, resolve_metrics
from soma.evaluation.summary import summarize_values
from soma.extraction import FeatureExtractor, _release_parent_cuda_state
from soma.data import CachedSetSource
from soma.output_layout import (
    build_experiment_spec,
    create_run_metadata,
    has_successful_run,
    register_test_result,
    resolve_managed_output_paths,
    test_identity_digest,
    update_latest_pointer,
    update_run_index,
    write_experiment_metadata,
    write_run_metadata,
)
from soma.preprocessing.resolution import resolve_pipeline_preprocessing
from soma.tasks.classification import (
    BranchAwareClassificationHead,
    MulticlassClassificationHead,
)
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.feature_adaptor import (
    FeatureAdaptor,
    build_feature_adaptor,
    feature_adaptor_output_dim,
    write_feature_adapter_sidecar,
)
from soma.training.collate import bag_collate_fn, cox_window_collate, hierarchical_bag_collate_fn
from soma.training.model import DenseModel, EmbeddingModel, MILModel
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.sample_dataset import SampleDataset, sample_collate_fn
from soma.training.fold_planning import plan_dense_fold
from soma.training.seed import seed_everything
from soma.learn import (
    Checkpoint,
    EarlyStopping,
    FitResult,
    Learner,
    Progress,
    Trainer,
    learner_registry,
    resolve_leaf,
)
from soma.learn.grid import grid_loaders


logger = logging.getLogger(__name__)


def _log_cuda_memory(tag: str) -> None:
    """Log the torch CUDA allocator state for memory diagnostics.

    nvidia-smi's per-process "used" = CUDA context + cuDNN/cuBLAS kernel images +
    this ``reserved`` figure. Logging it right after the encoder is released
    exposes how much of the working set is fixed framework/context overhead (the
    gap between nvidia-smi "used" and ``reserved``) versus live tensors.
    """
    if not torch.cuda.is_available():
        return
    gib = 1024**3
    logger.info(
        "CUDA memory [%s]: allocated=%.2f GiB  reserved=%.2f GiB  max_reserved=%.2f GiB",
        tag,
        torch.cuda.memory_allocated() / gib,
        torch.cuda.memory_reserved() / gib,
        torch.cuda.max_memory_reserved() / gib,
    )
    # Reset the peak so a later read reflects the decoder-training phase only.
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Result of training + evaluation for a single fold."""

    fold: int
    # None for the pixel-classifier segmentation path — it has no torch Trainer / epoch
    # history (the classifier owns its own fit loop internally).
    train_result: FitResult | None
    tune_evaluation: Evaluation
    test_evaluations: dict[str, Evaluation]  # split_name → evaluation


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run across all folds."""

    fold_results: list[FoldResult]
    summary: dict[str, float]
    run_dir: Path
    #: Folds still without a ``metrics.json`` when this launch ended (``run.folds`` left
    #: them to another launch). Non-empty means no ``summary.json`` was written.
    pending_folds: tuple[int, ...] = ()
    #: The manifests this run trained and scored against, as ``experiment.json`` and
    #: ``run.yaml`` record them: the dataset and splits identity digests and the
    #: test-identity digest (issue #247). Empty for the layer-1 helpers, which have no
    #: managed run; :meth:`Pipeline.run` always fills them.
    dataset_checksum: str = ""
    splits_checksum: str = ""
    test_checksum: str = ""


@dataclass(frozen=True)
class _FeatureSourceContext:
    """Feature source plus the dataset/splits it trains against.

    On the live grid path ``feature_store`` is the pixels' image source and
    ``encoder_stage`` the frozen encoder that re-encodes them.
    """

    feature_store: object
    dataset: object
    splits: LegacyFolds
    encoder_stage: object = None


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
) -> PipelineResult:
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
    _save_summary(summary, summary_path)

    sample_values = np.asarray(result.sample_values_aligned, dtype=float)
    if sample_values.ndim == 1 and len(sample_values) == len(selected_ids):
        pd.DataFrame(
            {"sample_id": selected_ids, "croma": sample_values}
        ).to_csv(run_dir / "croma_samples.csv", index=False)

    return PipelineResult(fold_results=[], summary=summary, run_dir=run_dir)


def _format_fold_summary(
    fold: int,
    train_count: int,
    tune_count: int,
    tests_counts: dict[str, int],
    empty_sample_ids_by_split: dict[str, list[str]] | None = None,
) -> str:
    tests_str = " ".join(f"{name}={count}" for name, count in sorted(tests_counts.items()))
    base = f"Fold {fold}: train={train_count} tune={tune_count} {tests_str}"
    if not empty_sample_ids_by_split:
        return base

    empty_counts = {
        split: len(sample_ids) for split, sample_ids in empty_sample_ids_by_split.items()
    }
    if not any(empty_counts.values()):
        return base

    extras: list[str] = []
    train_empty = empty_counts.get("train", 0)
    if train_empty:
        extras.append(f"train empty dropped={train_empty}")

    eval_fallback_counts = {
        split: count
        for split, count in empty_counts.items()
        if split != "train" and count
    }
    if eval_fallback_counts:
        fallback_text = ", ".join(
            f"{split}={count}" for split, count in sorted(eval_fallback_counts.items())
        )
        extras.append(f"eval empty fallback={fallback_text}")

    if not extras:
        return base
    return f"{base} | {' | '.join(extras)}"


def _resolve_tune_is_test_split(fold_split: FoldSplit, fold_label: str) -> str:
    test_split_names = fold_split.test_split_names
    if len(test_split_names) != 1:
        raise ValueError(
            f"{fold_label} has {len(test_split_names)} test splits; "
            "training.tune_is_test=True requires exactly one test split"
        )
    return test_split_names[0]


def _patient_ids_for_records(records: list[LegacyRecord]) -> list[str]:
    seen: set[str] = set()
    patient_ids: list[str] = []
    for record in records:
        if record.patient_id is None:
            raise ValueError(
                f"Sample '{record.sample_id}' has no patient_id. "
                "All samples must have a patient_id for dataset_type='patient'."
            )
        if record.patient_id not in seen:
            seen.add(record.patient_id)
            patient_ids.append(record.patient_id)
    return patient_ids


def _write_dense_source_provenance(run_dir: Path, feature_store: object) -> None:
    provenance = getattr(feature_store, "provenance", None)
    if provenance is None:
        return
    path = run_dir / "dense_source.json"
    atomic_write_json(path, provenance.to_dict(), sort_keys=True)
    logger.info("Dense source provenance saved to %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _loader_kwargs(training: TrainingConfig) -> dict[str, object]:
    """Build DataLoader kwargs from TrainingConfig.

    ``persistent_workers`` requires ``num_workers > 0`` in PyTorch, so it is
    auto-downgraded to False when workers are disabled.
    """
    return {
        "batch_size": training.batch_size,
        "num_workers": training.num_workers,
        "pin_memory": training.pin_memory,
        "persistent_workers": training.persistent_workers and training.num_workers > 0,
    }


def _make_loaders(
    dataset_cls,
    collate_fn,
    train_items,
    tune_items,
    test_items_by_split: dict,
    training: TrainingConfig,
    feature_store: CachedSetSource,
    target_fn,
    *,
    train_batch_sampler=None,
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """Create train, tune, and per-split test DataLoaders with a common pattern."""
    loader_kwargs = _loader_kwargs(training)
    train_dataset = dataset_cls(train_items, feature_store, target_fn)
    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    else:
        sampler_loader_kwargs = dict(loader_kwargs)
        sampler_loader_kwargs.pop("batch_size")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            **sampler_loader_kwargs,
        )
    tune_loader = DataLoader(
        dataset_cls(tune_items, feature_store, target_fn),
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loaders = {
        split_name: DataLoader(
            dataset_cls(items, feature_store, target_fn),
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        for split_name, items in test_items_by_split.items()
    }
    return train_loader, tune_loader, test_loaders


def _segmentation_roi_batch_sampler(
    records: list[LegacyRecord],
    population,
    *,
    num_classes: int,
    training: TrainingConfig,
    fold: int,
):
    """Build the explicit cached-grid training-batch contract, when requested."""
    if training.roi_batch_sampling is None:
        return None

    from soma.learn.roi_sampler import SegmentationRoiBatchSampler

    selected_population = population.subset([record.sample_id for record in records])

    return SegmentationRoiBatchSampler(
        sample_ids=[record.sample_id for record in records],
        class_pixel_counts=selected_population.class_pixel_counts,
        batch_size=training.batch_size,
        gradient_accumulation=training.gradient_accumulation,
        draws_per_epoch=training.roi_draws_per_epoch,
        strategy=training.roi_batch_sampling,
        class_request_ratios=training.class_request_ratios,
        seed=training.seed + fold,
    )


def _event_balanced_train_loader(
    dataset,
    collate_fn,
    *,
    events: list[int],
    training: TrainingConfig,
    min_events_per_window: int,
    window_size: int | None = None,
) -> DataLoader:
    """Training loader whose batches each hold >= one event (for batched Cox).

    Replaces the ordinary shuffled train loader: the Cox partial likelihood is
    undefined on an event-free batch, so the risk set must contain events.
    ``batch_sampler`` cannot coexist with ``batch_size``/``shuffle``, so those are
    dropped here; the sampler owns batching and per-epoch reshuffling.

    ``window_size`` sets the risk-set size: it is ``training.batch_size`` in
    padded mode (each window is one padded batch) and ``cox_window`` in
    accumulation mode (each window is N un-padded bags, with ``batch_size`` pinned
    to 1). The matching ``collate_fn`` decides padded vs un-padded.
    """
    from soma.training.survival_sampler import EventBalancedBatchSampler

    sampler = EventBalancedBatchSampler(
        events,
        batch_size=window_size if window_size is not None else training.batch_size,
        min_events_per_window=min_events_per_window,
        seed=training.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=training.num_workers,
        pin_memory=training.pin_memory,
        persistent_workers=training.persistent_workers and training.num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Layer 1 — Standalone step functions
# ---------------------------------------------------------------------------


class _SupportFeatures:
    """A re-iterable stream of the Support split's feature tensors, one sample at a time.

    The feature adaptor fits its stages in pipeline order, so a fitted projection needs a
    *second* pass over rows the normalize stage has already transformed. A generator would
    arrive at that pass exhausted; this restarts the stream on every ``__iter__`` while
    still never holding more than one sample's tiles at once.
    """

    def __init__(self, feature_store: CachedSetSource, records) -> None:
        self._feature_store = feature_store
        self._sample_ids = [record.sample_id for record in records]

    def __iter__(self):
        return (self._feature_store.load(sample_id) for sample_id in self._sample_ids)


class _SupportGrids:
    """A re-iterable stream of the Support ROIs' dense grids, **channel-axis last**.

    The dense path's features are ``(d, h, w)`` grids, not rows; every spatial position is
    one feature vector, so the fit population is *all positions in the Support ROIs*.
    Moving the channel axis last hands :meth:`FeatureAdaptor.fit` exactly the row layout it
    expects (last axis = feature dim), which is why one fit implementation serves the MIL,
    embedding and dense paths alike. Re-iterable for the same reason as
    :class:`_SupportFeatures` — a fitted projection needs a second pass.
    """

    def __init__(self, feature_store, records) -> None:
        self._feature_store = feature_store
        self._sample_ids = [record.sample_id for record in records]

    def __iter__(self):
        return (
            self._feature_store.load(sample_id).movedim(0, -1)
            for sample_id in self._sample_ids
        )


def _fit_feature_adaptor(
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None,
    *,
    support_features: Iterable[torch.Tensor],
    num_support: int,
    feature_dim: int,
    encoder_identity: str,
    fold_dir: Path,
) -> FeatureAdaptor | None:
    """Build and fit the fold's feature adaptor, or return ``None`` when it has no stage.

    Shared by every path that carries an adaptor (the tile-encoder MIL path, issue #283;
    the slide-encoder embedding path, issue #285; the single-encoder dense path, issue
    #286), because the contract they share is the load-bearing part: the fit is over the
    **Support split alone** — "K means K" — and it happens *before* anything trainable is
    constructed, so the aggregator/decoder/head can be built against the adaptor's
    (possibly rewired) output width.

    ``support_features`` is the path's re-iterable stream of Support tensors whose **last**
    axis is the feature dim (:class:`_SupportFeatures` for rows, :class:`_SupportGrids` for
    dense grids). Streaming one sample at a time keeps a whole cohort off the heap; the
    stream must be re-iterable because a fitted projection needs a second pass over
    already-normalized rows. Held-out samples only ever pass through ``forward()``, which
    never touches the buffers, so the transform is leak-free by construction.
    """
    feature_adaptor = build_feature_adaptor(
        normalization,
        projection,
        num_features=feature_dim,
        encoder_identity=encoder_identity,
    )
    if feature_adaptor is None:
        return None
    feature_adaptor.fit(support_features)
    logger.info(
        "Fitted feature adaptor on %d Support sample(s): normalization=%s "
        "(%d channel(s) at the eps floor), projection=%s (%d → %d dims)",
        num_support,
        feature_adaptor.method,
        feature_adaptor.num_eps_floored,
        feature_adaptor.projection_method,
        feature_dim,
        feature_adaptor.output_dim,
    )
    write_feature_adapter_sidecar(
        feature_adaptor, fold_dir, n_support_samples=num_support
    )
    return feature_adaptor


def train_one_fold(
    feature_store: CachedSetSource,
    dataset: LegacySamples,
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    dataset_type: str = "slide",
    evaluation: EvalConfig | None = None,
    aggregator: AggregatorConfig | None = None,
    fold: int = 0,
    num_folds: int = 1,
    preprocessing: PreprocessingConfig | None = None,
    heatmaps: HeatmapConfig | None = None,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
) -> FoldResult:
    """Train and evaluate a single fold.

    Args:
        feature_store: Precomputed embeddings.
        dataset: LegacySamples with sample records and label_map.
        fold_split: Train/tune/test sample IDs for this fold.
        dataset_type: ``"slide"`` for WSI pipelines; ``"tile"`` for tile-image
            pipelines where each sample is a single encoded patch.
        aggregator: Aggregator configuration, or None for slide-level or
            tile-dataset features.
        task: Task head configuration.
        evaluation: Evaluation configuration (metrics, subgroups). Defaults to EvalConfig().
        training: Training loop configuration.
        fold_dir: Directory for checkpoint, metrics, predictions.
        fold: Fold index (for FoldResult metadata).
        heatmaps: When provided and enabled, attention scores are saved during
            the test evaluation pass (no separate inference pass needed).
        normalization: Feature-adaptor normalization. When it asks for a transform,
            a :class:`~soma.training.feature_adaptor.FeatureAdaptor` is fit on this
            fold's Support (train) features and inserted ahead of the aggregator.
            ``None``/``none`` means no adaptor at all.
        projection: Feature-adaptor label-free projection, applied after
            ``normalization``. When active the aggregator/head is built against
            ``projection.target_dim`` rather than the encoder's native dim, so the
            downstream trainable capacity is equal across a roster of encoders.
        encoder_identity: Name of the encoder behind the features; seeds the
            ``random`` projection so two encoders never share a matrix.

    Returns:
        FoldResult with the training result and the tune/test evaluations.
    """
    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(training.seed, fold=fold)

    feature_dim = feature_store.feature_dim
    all_test_ids = {sid for ids in fold_split.tests.values() for sid in ids}
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"
    tune_from_test_split_name: str | None = None
    if dataset_type == "patient":
        # Patient-level: features are indexed by patient_id; splits are slide-based.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records_by_split: dict[str, list[LegacyRecord]] = {
            split_name: [dataset.samples[sid] for sid in ids]
            for split_name, ids in fold_split.tests.items()
        }
        train_patient_ids = _patient_ids_for_records(train_records)
        tune_patient_ids = _patient_ids_for_records(tune_records)
        test_patient_ids_by_split = {
            split_name: _patient_ids_for_records(records)
            for split_name, records in test_records_by_split.items()
        }
        raw_train_patient_ids = list(train_patient_ids)
        raw_tune_patient_ids = list(tune_patient_ids)
        raw_test_patient_ids_by_split = {
            split_name: list(patient_ids)
            for split_name, patient_ids in test_patient_ids_by_split.items()
        }

        if feature_store.has_feature_manifest:
            manifest_statuses = feature_store.feature_statuses
            manifest_patient_ids = set(manifest_statuses)
            split_patient_ids = (
                set(train_patient_ids)
                | set(tune_patient_ids)
                | {patient_id for ids in test_patient_ids_by_split.values() for patient_id in ids}
            )
            missing_manifest_ids = sorted(split_patient_ids - manifest_patient_ids)
            if missing_manifest_ids:
                msg = (
                    f"Patient feature manifest is missing patient(s) required by fold {fold}: "
                    f"{missing_manifest_ids}"
                )
                raise ValueError(msg)

            expected_feature_ids = {
                patient_id for patient_id, status in manifest_statuses.items() if status == "success"
            }
            empty_patient_ids = {
                patient_id for patient_id, status in manifest_statuses.items() if status == "empty"
            }
            unexpected_missing_ids = sorted(expected_feature_ids - set(feature_store.sample_ids))
            if unexpected_missing_ids:
                msg = (
                    f"Feature store is missing expected patient embedding(s) for fold {fold}: "
                    f"{unexpected_missing_ids}"
                )
                raise ValueError(msg)

            train_patient_ids = [pid for pid in train_patient_ids if pid in expected_feature_ids]
            tune_patient_ids = [pid for pid in tune_patient_ids if pid in expected_feature_ids]
            test_patient_ids_by_split = {
                split_name: [pid for pid in patient_ids if pid in expected_feature_ids]
                for split_name, patient_ids in test_patient_ids_by_split.items()
            }
            empty_sample_ids_by_split = {
                "train": [pid for pid in raw_train_patient_ids if pid in empty_patient_ids],
                "tune": [pid for pid in raw_tune_patient_ids if pid in empty_patient_ids],
                **{
                    split_name: [pid for pid in patient_ids if pid in empty_patient_ids]
                    for split_name, patient_ids in raw_test_patient_ids_by_split.items()
                },
            }
        else:
            split_patient_ids = (
                set(train_patient_ids)
                | set(tune_patient_ids)
                | {patient_id for ids in test_patient_ids_by_split.values() for patient_id in ids}
            )
            missing_patient_ids = sorted(split_patient_ids - set(feature_store.sample_ids))
            if missing_patient_ids:
                msg = (
                    f"Patient feature store is missing patient embedding(s) required by fold {fold}: "
                    f"{missing_patient_ids}"
                )
                raise ValueError(msg)
            empty_sample_ids_by_split = None
    elif feature_store.has_feature_manifest:
        manifest_statuses = feature_store.feature_statuses
        manifest_sample_ids = set(manifest_statuses)
        split_sample_ids = set(fold_split.train) | set(fold_split.tune) | all_test_ids
        missing_manifest_ids = sorted(split_sample_ids - manifest_sample_ids)
        if missing_manifest_ids:
            msg = (
                f"Feature manifest is missing sample(s) required by fold {fold}: "
                f"{missing_manifest_ids}"
            )
            raise ValueError(msg)

        expected_feature_ids = {
            sample_id for sample_id, status in manifest_statuses.items() if status == "success"
        }
        empty_sample_ids = {
            sample_id for sample_id, status in manifest_statuses.items() if status == "empty"
        }
        unexpected_missing_ids = sorted(expected_feature_ids - set(feature_store.sample_ids))
        if unexpected_missing_ids:
            msg = (
                f"Feature store is missing expected sample(s) for fold {fold}: "
                f"{unexpected_missing_ids}"
            )
            raise ValueError(msg)

        empty_sample_ids_by_split = {
            "train": [sid for sid in fold_split.train if sid in empty_sample_ids],
            "tune": [sid for sid in fold_split.tune if sid in empty_sample_ids],
            **{
                split_name: [sid for sid in ids if sid in empty_sample_ids]
                for split_name, ids in fold_split.tests.items()
            },
        }

        # Build datasets from samples that are expected to have features.
        train_records = _records_for_sample_ids(dataset, fold_split.train, expected_feature_ids)
        tune_records = _records_for_sample_ids(dataset, fold_split.tune, expected_feature_ids)
        test_records_by_split = {
            split_name: _records_for_sample_ids(dataset, ids, expected_feature_ids)
            for split_name, ids in fold_split.tests.items()
        }
    else:
        # Legacy path: no manifest, so every listed sample must have a feature file.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records_by_split = {
            split_name: [dataset.samples[sid] for sid in ids]
            for split_name, ids in fold_split.tests.items()
        }
        empty_sample_ids_by_split = None

    if training.tune_is_test:
        tune_from_test_split_name = _resolve_tune_is_test_split(fold_split, _fp)
        logger.warning(
            "%s uses test split '%s' as tune because tune_is_test=True; "
            "checkpoint selection and test reporting use the same samples",
            _fp,
            tune_from_test_split_name,
        )
        if dataset_type == "patient":
            tune_patient_ids = list(test_patient_ids_by_split[tune_from_test_split_name])
            raw_tune_patient_ids = list(raw_test_patient_ids_by_split[tune_from_test_split_name])
            tune_records = list(test_records_by_split[tune_from_test_split_name])
        else:
            tune_records = list(test_records_by_split[tune_from_test_split_name])
        if empty_sample_ids_by_split is not None:
            empty_sample_ids_by_split["tune"] = list(
                empty_sample_ids_by_split.get(tune_from_test_split_name, [])
            )

    if evaluation.holdout_test:
        # Tune-only model-selection run: drop every declared test split so no test
        # loader is built, no test inference runs, and metrics/summary carry tune
        # only. Done after tune_is_test has resolved tune from the test split, so
        # tune selection and training are unaffected. The split stays in splits.csv;
        # it is simply not touched (not even feature-validated below).
        test_records_by_split = {}
        if dataset_type == "patient":
            test_patient_ids_by_split = {}
            raw_test_patient_ids_by_split = {}
        if empty_sample_ids_by_split is not None:
            empty_sample_ids_by_split = {
                name: ids
                for name, ids in empty_sample_ids_by_split.items()
                if name in ("train", "tune")
            }

    if dataset_type == "patient":
        if not train_patient_ids:
            msg = f"{_fp} has no training patients with available features"
            raise ValueError(msg)
        fallback_to_train = not tune_patient_ids and not (
            empty_sample_ids_by_split and empty_sample_ids_by_split.get("tune")
        )
        if fallback_to_train and not training.allow_missing_tune:
            msg = f"{_fp} has no tuning patients with available features"
            raise ValueError(msg)
        if fallback_to_train and training.allow_missing_tune:
            logger.warning(
                "%s has no tuning patients with available features; "
                "using train split as tune because allow_missing_tune=True",
                _fp,
            )
            tune_patient_ids = list(train_patient_ids)
            raw_tune_patient_ids = list(raw_train_patient_ids)
            tune_records = list(train_records)
        for split_name, patient_ids in test_patient_ids_by_split.items():
            split_empty_ids = (
                empty_sample_ids_by_split.get(split_name, []) if empty_sample_ids_by_split else []
            )
            if not patient_ids and not split_empty_ids:
                msg = f"{_fp} has no patients with available features in split '{split_name}'"
                raise ValueError(msg)
        train_count = len(train_patient_ids)
        tune_count = len(tune_patient_ids)
        tests_counts = {name: len(patient_ids) for name, patient_ids in test_patient_ids_by_split.items()}
    else:
        if not train_records:
            msg = f"{_fp} has no training samples with available features"
            raise ValueError(msg)
        fallback_to_train = not tune_records and not (
            empty_sample_ids_by_split and empty_sample_ids_by_split.get("tune")
        )
        if fallback_to_train and not training.allow_missing_tune:
            msg = f"{_fp} has no tuning samples with available features"
            raise ValueError(msg)
        if fallback_to_train and training.allow_missing_tune:
            logger.warning(
                "%s has no tuning samples with available features; "
                "using train split as tune because allow_missing_tune=True",
                _fp,
            )
            tune_records = list(train_records)
        for split_name, records in test_records_by_split.items():
            split_empty_ids = (
                empty_sample_ids_by_split.get(split_name, []) if empty_sample_ids_by_split else []
            )
            if not records and not split_empty_ids:
                msg = f"{_fp} has no samples with available features in split '{split_name}'"
                raise ValueError(msg)
        train_count = len(train_records)
        tune_count = len(tune_records)
        tests_counts = {name: len(recs) for name, recs in test_records_by_split.items()}

    summary = _format_fold_summary(
        fold=fold,
        train_count=train_count,
        tune_count=tune_count,
        tests_counts=tests_counts,
        empty_sample_ids_by_split=empty_sample_ids_by_split,
    )
    logger.info(summary)

    # Validate subgroup columns exist in dataset metadata
    if evaluation.subgroups.columns:
        sample = next(iter(dataset.samples.values()))
        missing = [c for c in evaluation.subgroups.columns if c not in sample.metadata]
        if missing:
            raise ValueError(
                f"Subgroup column(s) not found in dataset metadata: {missing}. "
                f"Available: {sorted(sample.metadata)}"
            )

    task_cls = task_registry.get(task.name)
    # Validate survival columns before auto_params reads them, so a missing/
    # malformed column raises a clear message instead of a cryptic KeyError.
    # Every head of the survival family is validated, built-in or registered.
    if task_cls.task_family == "survival":
        from soma.tasks.survival import resolve_survival_task, validate_survival_dataset

        task_cls, survival_loss = resolve_survival_task(task.name, task.params)
        validate_survival_dataset(
            dataset.records, dataset_type, loss=survival_loss, num_bins=task.params.get("num_bins")
        )
    task_params = {**task_cls.auto_params(dataset.records), **task.params, "metrics": evaluation.metrics}
    task_params.pop("loss", None)

    task_family = task_cls.task_family

    # Feature adaptor (issues #283, #284, #285). The tile-encoder MIL path and the
    # slide-encoder embedding path both fit one; the patient, tile-dataset and
    # hierarchical paths are separate slices, so a normalization or projection request
    # they cannot honor is refused rather than silently ignored — a silently-dropped
    # transform is a false result.
    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        dataset_type=dataset_type,
        is_hierarchical=feature_store.rank == 3,
    )

    # The task head owns its target contract. Build the head first, then derive
    # the per-sample target_fn and per-key collation from it.
    if dataset_type == "patient":
        # Patient-level path: pretrained patient encoder produced (D,) per patient
        if aggregator is not None:
            raise ValueError("aggregator must be None for dataset_type='patient'")
        head = task_cls(input_dim=feature_dim, **task_params)
        target_fn = head.extract_targets
        patient_record_map = dataset.patient_record_map
        _patient_collate = functools.partial(patient_collate_fn, target_dtypes=head.target_dtypes)

        _patient_loader_kwargs = _loader_kwargs(training)
        train_loader = DataLoader(
            PatientDataset(train_patient_ids, patient_record_map, feature_store, target_fn),
            shuffle=True,
            collate_fn=_patient_collate,
            **_patient_loader_kwargs,
        )
        tune_loader = DataLoader(
            PatientDataset(tune_patient_ids, patient_record_map, feature_store, target_fn),
            shuffle=False,
            collate_fn=_patient_collate,
            **_patient_loader_kwargs,
        )
        test_loaders: dict[str, DataLoader] = {
            split_name: DataLoader(
                PatientDataset(
                    test_patient_ids_by_split[split_name],
                    patient_record_map, feature_store, target_fn,
                ),
                shuffle=False,
                collate_fn=_patient_collate,
                **_patient_loader_kwargs,
            )
            for split_name, records in test_records_by_split.items()
        }
        if getattr(head, "needs_event_balanced_batches", False):
            train_loader = _event_balanced_train_loader(
                PatientDataset(train_patient_ids, patient_record_map, feature_store, target_fn),
                _patient_collate,
                events=[
                    int(patient_record_map[pid].targets["event"]) for pid in train_patient_ids
                ],
                training=training,
                min_events_per_window=head.min_events_per_window,
            )
        model: torch.nn.Module = EmbeddingModel(task_head=head)
    elif dataset_type == "tile" or feature_store.rank == 1:
        # Single-embedding path: one pre-computed vector per sample, no aggregation.
        # Fit the adaptor BEFORE the head is constructed, from the Support split's
        # embeddings and nothing else — "K means K", and here K rows is literally all
        # there is, which is why the PCA preflight is load-bearing on this path.
        if feature_store.rank == 1 and aggregator is not None:
            raise ValueError("aggregator must be None for slide-level features")
        feature_adaptor = _fit_feature_adaptor(
            normalization,
            projection,
            support_features=_SupportFeatures(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=feature_dim,
            encoder_identity=encoder_identity,
            fold_dir=fold_dir,
        )
        # The dim rewire (issue #284): with a projection active the head is built against
        # `target_dim`, not the encoder's native dim.
        head = task_cls(
            input_dim=feature_adaptor_output_dim(
                feature_adaptor, num_features=feature_dim
            ),
            **task_params,
        )
        target_fn = head.extract_targets
        _sample_collate = functools.partial(sample_collate_fn, target_dtypes=head.target_dtypes)
        train_loader, tune_loader, test_loaders = _make_loaders(
            SampleDataset, _sample_collate,
            train_records, tune_records, test_records_by_split,
            training, feature_store, target_fn,
        )
        if getattr(head, "needs_event_balanced_batches", False):
            train_loader = _event_balanced_train_loader(
                SampleDataset(train_records, feature_store, target_fn),
                _sample_collate,
                events=[int(r.targets["event"]) for r in train_records],
                training=training,
                min_events_per_window=head.min_events_per_window,
            )
        model: torch.nn.Module = EmbeddingModel(
            task_head=head, feature_adaptor=feature_adaptor
        )
    elif feature_store.rank == 3:
        if aggregator is None:
            raise ValueError("aggregator must be provided for hierarchical features")
        if aggregator.name != "hipt":
            raise ValueError("hierarchical features require the hipt aggregator")
        if preprocessing is None:
            raise ValueError("hierarchical features require resolved preprocessing")
        hipt_params = _resolve_hipt_params(preprocessing, aggregator)
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=feature_dim, **hipt_params)
        head = task_cls(input_dim=agg.output_dim, **task_params)
        target_fn = head.extract_targets
        _hier_collate = functools.partial(hierarchical_bag_collate_fn, target_dtypes=head.target_dtypes)
        train_loader, tune_loader, test_loaders = _make_loaders(
            HierarchicalBagDataset, _hier_collate,
            train_records, tune_records, test_records_by_split,
            training, feature_store, target_fn,
        )
        if getattr(head, "needs_event_balanced_batches", False):
            train_events = [int(r.targets["event"]) for r in train_records]
            if getattr(head, "accumulates_predictions", False):
                _cox_collate = functools.partial(
                    cox_window_collate,
                    target_dtypes=head.target_dtypes,
                )
                train_loader = _event_balanced_train_loader(
                    HierarchicalBagDataset(train_records, feature_store, target_fn),
                    _cox_collate,
                    events=train_events,
                    training=training,
                    min_events_per_window=head.min_events_per_window,
                    window_size=head.accumulation_window,
                )
            else:
                train_loader = _event_balanced_train_loader(
                    HierarchicalBagDataset(train_records, feature_store, target_fn),
                    _hier_collate,
                    events=train_events,
                    training=training,
                    min_events_per_window=head.min_events_per_window,
                )
        model = MILModel(aggregator=agg, task_head=head)
    else:
        # Tile-level MIL path
        if aggregator is None:
            raise ValueError("aggregator must be provided for tile-level features")
        # Fit the adaptor BEFORE anything trainable is constructed, from the Support
        # split's tiles and nothing else — "K means K".
        feature_adaptor = _fit_feature_adaptor(
            normalization,
            projection,
            support_features=_SupportFeatures(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=feature_dim,
            encoder_identity=encoder_identity,
            fold_dir=fold_dir,
        )
        # The dim rewire (issue #284): with a projection active the aggregator is built
        # against `target_dim`, not the encoder's native dim, so the trainable capacity
        # below is identical across a roster of encoders of differing width.
        adapted_dim = feature_adaptor_output_dim(
            feature_adaptor, num_features=feature_dim
        )
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=adapted_dim, **aggregator.params)
        head_cls = _bag_head_class(task_cls, task_name=task.name, aggregator_name=aggregator.name)
        head = head_cls(input_dim=agg.output_dim, **task_params)
        target_fn = head.extract_targets
        _bag_collate = functools.partial(bag_collate_fn, target_dtypes=head.target_dtypes)
        train_loader, tune_loader, test_loaders = _make_loaders(
            BagDataset, _bag_collate,
            train_records, tune_records, test_records_by_split,
            training, feature_store, target_fn,
        )
        # Cox MIL: replace the train loader with an event-balanced one. Tune/test
        # keep the ordinary bag loaders — eval is full-cohort and order-agnostic.
        if getattr(head, "needs_event_balanced_batches", False):
            train_events = [int(r.targets["event"]) for r in train_records]
            if getattr(head, "accumulates_predictions", False):
                # Accumulation mode: windows of cox_window un-padded bags.
                _cox_collate = functools.partial(cox_window_collate, target_dtypes=head.target_dtypes)
                train_loader = _event_balanced_train_loader(
                    BagDataset(train_records, feature_store, target_fn),
                    _cox_collate,
                    events=train_events,
                    training=training,
                    min_events_per_window=head.min_events_per_window,
                    window_size=head.accumulation_window,
                )
            else:
                # Padded mode: batch_size bags padded to the window max, masked.
                train_loader = _event_balanced_train_loader(
                    BagDataset(train_records, feature_store, target_fn),
                    _bag_collate,
                    events=train_events,
                    training=training,
                    min_events_per_window=head.min_events_per_window,
                )
        model = MILModel(
            aggregator=agg, task_head=head, feature_adaptor=feature_adaptor
        )

    # Samples without features are scored with the training split's baseline (class
    # prior / mean / constant risk), flagged as placeholders.
    baseline = _verbs.Baseline.fit(
        targets_frame(train_records),
        task_family=task_family,
        classes=head.classes,
        target_keys=head.target_keys,
    )

    # Train; the learner comes back in its selected state (Checkpoint restores it).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_result = _fit_fold(
        _build_learner(model, training, shape="mil"),
        train_loader,
        tune_loader,
        training=training,
        fold_dir=fold_dir,
        device=device,
        fold=fold,
        num_folds=num_folds,
    )
    learner = train_result.learner

    save_attention = (
        heatmaps is not None
        and heatmaps.enabled
        and aggregator is not None
        and not feature_store.rank == 1
        and not feature_store.rank == 3
    )

    empty_eval_sample_ids = empty_sample_ids_by_split or {}
    if dataset_type == "patient":
        # Patient ids score against the patient's first slide in cohort order, as
        # Cohort.collapse does.
        if fallback_to_train:
            tune_output_sample_ids = tuple(raw_train_patient_ids)
            tune_empty_sample_ids = empty_eval_sample_ids.get("train", [])
        else:
            tune_output_sample_ids = tuple(raw_tune_patient_ids)
            tune_empty_sample_ids = empty_eval_sample_ids.get("tune", [])
        tune_records_by_id = patient_record_map
        test_plan = {
            split_name: (tuple(raw_test_patient_ids_by_split[split_name]), patient_record_map)
            for split_name in test_loaders
        }
    else:
        if fallback_to_train:
            tune_output_sample_ids = fold_split.train
            tune_empty_sample_ids = empty_eval_sample_ids.get("train", [])
        elif tune_from_test_split_name is not None:
            tune_output_sample_ids = fold_split.tests[tune_from_test_split_name]
            tune_empty_sample_ids = empty_eval_sample_ids.get("tune", [])
        else:
            tune_output_sample_ids = fold_split.tune
            tune_empty_sample_ids = empty_eval_sample_ids.get("tune", [])
        tune_records_by_id = dataset.samples
        test_plan = {
            split_name: (fold_split.tests[split_name], dataset.samples) for split_name in test_loaders
        }

    # Score every split through the public verbs: predict → fill placeholders →
    # evaluate against the split's targets. Test splits also write their predictions,
    # subgroup breakdown and (with heatmaps on) attention maps.
    splits = {"tune": (tune_loader, tune_output_sample_ids, tune_empty_sample_ids, tune_records_by_id)}
    for split_name, (output_ids, records_by_id) in test_plan.items():
        splits[split_name] = (
            test_loaders[split_name],
            output_ids,
            empty_eval_sample_ids.get(split_name, []),
            records_by_id,
        )
    # The head resolved its metrics for its own outputs (a multi-output regression
    # reports mean_pearson, not the family's single-value defaults).
    metric_names = list(getattr(head, "metrics", None) or resolve_metrics(task_family, evaluation.metrics))
    evaluations: dict[str, Evaluation] = {}
    for split_name, (loader, output_ids, empty_ids, records_by_id) in splits.items():
        is_test = split_name != "tune"
        callbacks = []
        if is_test and save_attention:
            from soma.heatmaps import SaveAttention

            callbacks.append(SaveAttention(fold_dir / "attention" / split_name))
        predictions = _verbs.Predictions.concat(
            [_verbs.predict(learner, loader, callbacks), baseline.predict(empty_ids)]
        )
        targets = targets_frame([records_by_id[sid] for sid in output_ids])
        targets.index = pd.Index([str(sid) for sid in output_ids], name="sample_id", dtype=object)
        evaluations[split_name] = _verbs.evaluate(
            predictions,
            targets,
            metrics=metric_names,
            subgroups=evaluation.subgroups.columns if is_test else (),
        )
        if is_test:
            predictions.to_parquet(fold_dir / f"predictions_{split_name}.parquet")
            if evaluation.subgroups.columns:
                atomic_write_json(
                    fold_dir / f"subgroup_metrics_{split_name}.json",
                    evaluations[split_name].subgroups,
                )

    tune_evaluation = evaluations.pop("tune")
    _save_metrics(tune_evaluation, evaluations, fold_dir / "metrics.json")
    _save_training_history(train_result.history, fold_dir / "training_history.json")

    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_evaluation=tune_evaluation,
        test_evaluations=evaluations,
    )


def _dense_spacings_match(a: float | None, b: float | None, *, tol: float = 1e-9) -> bool:
    """True when two read-spacings agree: both flat (``None``) or equal within ``tol``."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _check_segmentation_grid_spacing(feature_store, records, preprocessing) -> None:
    """Fail when a cached grid was extracted at another spacing than the config requests.

    ROI masks are read at each grid's recorded spacing, so they register either way; this
    catches features extracted under another configuration, e.g. passed as
    ``Pipeline(feature_dir=...)``. Under ``native_if_coarser`` the spacing differs per
    source, so each record is checked against the extractor's own rule.
    """
    requested = preprocessing.requested_spacing_um if preprocessing is not None else None
    for record in records:
        expected_spacing_um = (
            None if requested is None else preprocessing.effective_spacing_um(record.spacing_at_level_0)
        )
        grid_spacing_um = feature_store.spacing_um(record.sample_id)
        if not _dense_spacings_match(grid_spacing_um, expected_spacing_um):
            raise ValueError(
                f"segmentation sample '{record.sample_id}' dense-grid spacing "
                f"({grid_spacing_um} µm/px) does not match the configured spacing "
                f"({expected_spacing_um} µm/px); re-extract the features for this "
                "configuration."
            )


def _segmentation_label_maps(head: SegmentationHead, records: "Sequence[LegacyRecord]"):
    """The label-map target source of a segmentation run's records.

    Annotation-sampled ROIs (``region`` set) read the mask crops ROI sampling stored
    through a :class:`~soma.data.CachedLabelMapSource`; pre-cropped tiles read their
    rasters through a :class:`~soma.data.LabelMapSource`. Both fold the class scheme in.
    """
    records = list(records)
    params = head.label_map_read_params
    if records and any(record.region is not None for record in records):
        crops: dict[str, Path] = {}
        for record in records:
            if record.region is None or record.label_mask_crop_path is None:
                raise ValueError(
                    f"segmentation sample '{record.sample_id}': a slide-manifest ROI needs its "
                    "stored mask crop (label_mask_crop_path), which ROI sampling writes."
                )
            crops[record.sample_id] = record.label_mask_crop_path
        return CachedLabelMapSource(
            crops, size=params["size"], label_remap=params["label_remap"], ignore_index=head.ignore_index
        )
    entries = []
    for record in records:
        if record.label_mask_path is None:
            raise ValueError(f"segmentation sample '{record.sample_id}' has no label_mask_path")
        entries.append(
            LabelMapEntry(
                sample_id=record.sample_id,
                label_mask_path=record.label_mask_path,
                reference_path=record.image_path,
                spacing_at_level_0=record.spacing_at_level_0,
            )
        )
    return LabelMapSource(entries, **params)


def _detection_points(task: TaskConfig, records: "Sequence[LegacyRecord]"):
    """``(num_classes, class names, PointSource)`` of a detection run's records."""
    num_classes, class_names, class_remap, dropped = resolve_classes(
        task.params, excluded_key="drop", subject="detection"
    )
    entries = []
    for record in records:
        if record.points_path is None:
            raise ValueError(f"detection sample '{record.sample_id}' has no points_path")
        entries.append(
            PointEntry(
                sample_id=record.sample_id,
                points_path=record.points_path,
                ignore_mask_path=record.ignore_mask_path,
            )
        )
    return num_classes, class_names, PointSource(entries, class_remap=class_remap, drop=dropped)


def _flat_tile_images(records: "Sequence[LegacyRecord]", *, size, spacing_um: float | None = None, spacing_policy: str = "strict", backend: str = "auto", tolerance: float = 0.05):
    """A :class:`~soma.data.TileImageSource` over pre-cropped tiles, or ``None`` for
    slide-manifest ROIs (their pixels are windows of a whole slide)."""
    if any(record.region is not None for record in records):
        return None
    return TileImageSource(
        [
            ImageEntry(
                sample_id=record.sample_id,
                image_path=record.image_path,
                spacing_at_level_0=record.spacing_at_level_0,
            )
            for record in records
        ],
        size=size,
        spacing_um=spacing_um,
        spacing_policy=spacing_policy,
        backend=backend,
        tolerance=tolerance,
    )


def _overlay_images(records: "Sequence[LegacyRecord]", *, size, preprocessing):
    """The pixels :class:`~soma.evaluation.SaveOverlays` blends over: the flat tiles, or
    each slide-manifest ROI's window of its whole slide at the run spacing."""
    spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    spacing_policy = preprocessing.spacing_policy if preprocessing is not None else "strict"
    backend = preprocessing.backend if preprocessing is not None else "auto"
    tolerance = float(preprocessing.tolerance) if preprocessing is not None else 0.05
    if spacing_um is None or not any(record.region is not None for record in records):
        return _flat_tile_images(
            records,
            size=size,
            spacing_um=spacing_um,
            spacing_policy=spacing_policy,
            backend=backend,
            tolerance=tolerance,
        )
    from soma.data.images import RegionImageSource

    return RegionImageSource(
        [
            ImageEntry(
                sample_id=record.sample_id,
                image_path=record.image_path,
                spacing_at_level_0=record.spacing_at_level_0,
            )
            for record in records
        ],
        regions={record.sample_id: record.region for record in records},
        size=size,
        spacing_um=spacing_um,
        spacing_policy=spacing_policy,
        backend=backend,
        tolerance=tolerance,
    )


def _grid_augment(augmentation: dict | None, head, geometry):
    """Build the configured ``{name, params}`` augmentation, injecting the run's
    ``ignore_index`` and ``target_size`` where the augmentation accepts them."""
    if augmentation is None:
        return None
    from soma.learn import augment_registry

    cls, params, _ = resolve_leaf(augmentation, augment_registry)
    try:
        accepted = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        accepted = {}
    injected = {
        "ignore_index": getattr(head, "ignore_index", None),
        "target_size": tuple(int(v) for v in geometry.target_size),
    }
    for key, value in injected.items():
        if key in accepted and key not in params and value is not None:
            params[key] = value
    return cls(**params)


def _uniform_spacing(source, sample_ids: Sequence[str]) -> float:
    """The one µm/px every sample of a detection run is read at."""
    spacings = {round(float(source.spacing(sid)), 9) for sid in sample_ids}
    if len(spacings) != 1:
        raise ValueError(
            "Detection requires one effective spacing across the run (its distances are in "
            f"µm); found {sorted(spacings)}."
        )
    return float(next(iter(spacings)))


def train_one_grid_fold(
    feature_store,
    dataset: LegacySamples,
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    decoder: DecoderConfig | None,
    dataset_type: str,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    masks: "MasksConfig | None" = None,
    fold: int = 0,
    num_folds: int = 1,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
    encoder_stage=None,
    augmentation: dict | None = None,
    roi_population_cache_root: str | Path | None = None,
    resolved_roi_population: "SegmentationRoiPopulation | None" = None,
    roi_source_fingerprint_cache: dict[str, dict[str, object]] | None = None,
    roi_resolved_backend_cache: dict[tuple[str, str, bool], str] | None = None,
) -> FoldResult:
    """Train and evaluate one grid fold (segmentation or detection) through the verbs.

    The target source, the head, the decoder and a :class:`DenseModel` are built; the
    fold's grid loaders feed ``_fit_fold``; ``soma.predict`` returns the head's noun on
    tune and every test split; detection sweeps its per-class threshold on tune (without
    a tune split it scores at the head's ``validation_threshold``); ``soma.evaluate``
    scores each split. ``feature_store`` is a :class:`~soma.data.GridSource` (cached
    grids) or, with ``encoder_stage`` (a frozen encoder), an
    :class:`~soma.data.ImageSource` re-encoded every step, where ``augmentation`` (a
    ``{name, params}`` leaf) augments the train split.

    Writes ``metrics.json``, ``training_history.json`` and the predictions:
    ``predictions_<split>.parquet`` (points, every peak above ``min_score``) or
    ``predictions_<split>/`` (label-map PNGs + ``index.parquet``).
    """
    if decoder is None:
        raise ValueError(f"dataset_type={dataset_type!r} requires a decoder configuration")
    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"
    live = encoder_stage is not None
    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        feature_mode="live" if live else "cached",
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
    )

    fold_plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=training,
        fold_label=_fp,
        logger=logger,
        holdout_test=evaluation.holdout_test,
    )
    train_records = fold_plan.train_records
    tune_records = fold_plan.tune_records
    test_records_by_split = fold_plan.test_records_by_split
    all_records = fold_plan.all_records
    all_ids = [record.sample_id for record in all_records]
    require_coverage(feature_store, all_ids, what="pixels" if live else "features")

    # The head's geometry is bound here: the encoder stage's frame (live) or the cached
    # grids' (the loaders check every sample shares it), anchored at level 0 by the source.
    reference = feature_store.geometry(train_records[0].sample_id)
    if live:
        geometry = GridGeometry.from_dense(
            encoder_stage.geometry,
            origin_level0=reference.origin_level0,
            level0_px_per_token_px=reference.level0_px_per_token_px,
        )
        feature_dim = int(encoder_stage.feature_dim)
    else:
        geometry = reference
        feature_dim = int(feature_store.feature_dim)
        if dataset_type == "segmentation":
            _check_segmentation_grid_spacing(feature_store, all_records, preprocessing)

    if dataset_type == "segmentation":
        head = _build_segmentation_head(
            task=task, evaluation=evaluation, preprocessing=preprocessing, masks=masks, geometry=geometry
        )
        targets = _segmentation_label_maps(head, all_records)
    else:
        num_classes, class_names, targets = _detection_points(task, all_records)
        params = {k: v for k, v in task.params.items() if k not in ("num_classes", "classes", "drop")}
        if "match_distance" not in params:
            raise ValueError(
                "dataset_type='detection' requires task.params.match_distance (the F1 "
                "matching distance δ, in µm)."
            )
        head = task_registry.get(task.name)(
            num_classes=num_classes,
            geometry=geometry,
            spacing_um=_uniform_spacing(feature_store, all_ids),
            metrics=evaluation.metrics,
            class_names=class_names,
            **params,
        )

    # Feature adaptor (issue #286), fit on the Support ROIs' grid positions before the
    # decoder is built. Cached only (refused above for live).
    feature_adaptor = None
    if not live:
        feature_adaptor = _fit_feature_adaptor(
            normalization,
            projection,
            support_features=_SupportGrids(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=feature_dim,
            encoder_identity=encoder_identity,
            fold_dir=fold_dir,
        )
    decoder_obj = build_decoder_for_grid(
        decoder.name,
        decoder.params,
        geometry=geometry,
        input_dim=feature_adaptor_output_dim(feature_adaptor, num_features=feature_dim),
        num_classes=head.num_classes,
    )
    model = DenseModel(decoder_obj, head, encoder=encoder_stage, adaptor=feature_adaptor)

    images = None
    if model.consumes_image:
        # An image-prior decoder emits logits at the mask resolution; the head's
        # resize-to-encoded + crop must then be the identity.
        if tuple(geometry.encoded_size) != tuple(geometry.target_size):
            raise ValueError(
                f"decoder '{decoder.name}' consumes the image and emits logits at the mask "
                f"size {tuple(geometry.target_size)}, but the encoder input is padded to "
                f"{tuple(geometry.encoded_size)}; image-prior decoders need an unpadded tile."
            )
        if not live:
            images = _flat_tile_images(all_records, size=geometry.target_size)
            if images is None:
                raise ValueError(
                    f"decoder '{decoder.name}' reads each tile's pixels; slide-manifest ROIs "
                    "carry no flat tile to read"
                )

    sampler = None
    if training.roi_batch_sampling is not None:
        if live or dataset_type != "segmentation":
            raise ValueError("Explicit ROI batch sampling is a cached segmentation training contract.")
        if resolved_roi_population is None:
            resolved_roi_population = _resolve_segmentation_roi_population(
                cache_root=(
                    Path(roi_population_cache_root)
                    if roi_population_cache_root is not None
                    else fold_dir / "segmentation_roi_population"
                ),
                records=all_records,
                head=head,
                label_maps=targets,
                source_fingerprint_cache=roi_source_fingerprint_cache,
                resolved_backend_cache=roi_resolved_backend_cache,
            )
        roi_population = resolved_roi_population.subset(all_ids)
        (fold_dir / "segmentation_roi_population.json").write_text(
            json.dumps(roi_population.provenance(), indent=2, sort_keys=True), encoding="utf-8"
        )
        sampler = _segmentation_roi_batch_sampler(
            train_records, roi_population, num_classes=head.num_classes, training=training, fold=fold
        )

    loader_kwargs = _loader_kwargs(training)
    fold_loaders = grid_loaders(
        feature_store,
        head=head,
        train=[record.sample_id for record in train_records],
        tune=[record.sample_id for record in tune_records],
        tests={name: [r.sample_id for r in records] for name, records in test_records_by_split.items()},
        batch_size=loader_kwargs.pop("batch_size"),
        num_workers=loader_kwargs.pop("num_workers"),
        targets=targets,
        augment=_grid_augment(augmentation, head, geometry) if live else None,
        # An image-prior decoder reads the raw pixels, so the stage preprocesses them.
        preprocess=encoder_stage.preprocessor() if live and not model.consumes_image else None,
        sampler=sampler,
        images=images,
        loader_kwargs=loader_kwargs,
    )
    logger.info(
        _format_fold_summary(
            fold=fold,
            train_count=len(train_records),
            tune_count=len(tune_records),
            tests_counts={name: len(recs) for name, recs in test_records_by_split.items()},
            empty_sample_ids_by_split=None,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_result = _fit_fold(
        _build_learner(model, training, shape="dense"),
        fold_loaders.train,
        fold_loaders.tune,
        training=training,
        fold_dir=fold_dir,
        device=device,
        fold=fold,
        num_folds=num_folds,
    )
    if sampler is not None:
        (fold_dir / "roi_batch_sampling.json").write_text(
            json.dumps(sampler.audit(), indent=2, sort_keys=True), encoding="utf-8"
        )
    learner = train_result.learner

    overlay_images = images or _overlay_images(all_records, size=geometry.target_size, preprocessing=preprocessing)

    def callbacks(split_name: str) -> list[object]:
        out: list[object] = []
        overlays = (
            evaluation.save_segmentation_overlays
            if dataset_type == "segmentation"
            else evaluation.save_detection_overlays
        )
        if overlays:
            out.append(_verbs.SaveOverlays(fold_dir / "overlays" / split_name, images=overlay_images))
        if dataset_type == "segmentation" and evaluation.save_segmentation_probabilities:
            out.append(_verbs.SaveProbabilities(fold_dir / "probabilities" / split_name))
        if dataset_type == "detection" and evaluation.save_detection_heatmaps:
            out.append(_verbs.SaveHeatmaps(fold_dir / "heatmaps" / split_name))
        return out

    metrics: list[object] = list(resolve_metrics(head.task_family, evaluation.metrics))
    if dataset_type == "detection":
        metrics = [{"name": name, "params": {"matching": head.matching}} for name in metrics]

    def predicted(split_name: str, loader):
        predictions = _verbs.predict(learner, loader, callbacks(split_name))
        if dataset_type == "segmentation":
            predictions.save(fold_dir / f"predictions_{split_name}")
        else:
            predictions.to_parquet(fold_dir / f"predictions_{split_name}.parquet")
        return predictions

    tune_ids = [record.sample_id for record in tune_records]
    tune_predictions = predicted("tune", fold_loaders.tune)
    threshold = None
    if dataset_type == "detection":
        # Without a tune split, train stands in and a sweep there would pick an
        # in-sample cut: score at the head's fixed validation threshold instead.
        if fold_plan.tune_from_train:
            threshold = head.validation_threshold
        else:
            threshold, _ = _verbs.sweep_threshold(
                tune_predictions, targets.subset(tune_ids), per_class=True, matching=head.matching
            )
        tune_predictions = tune_predictions.with_threshold(threshold)
    tune_evaluation = _verbs.evaluate(tune_predictions, targets.subset(tune_ids), metrics=metrics)
    test_evaluations: dict[str, Evaluation] = {}
    for split_name, loader in fold_loaders.tests.items():
        predictions = predicted(split_name, loader)
        if threshold is not None:
            predictions = predictions.with_threshold(threshold)
        split_ids = [record.sample_id for record in test_records_by_split[split_name]]
        test_evaluations[split_name] = _verbs.evaluate(
            predictions, targets.subset(split_ids), metrics=metrics
        )

    _save_metrics(tune_evaluation, test_evaluations, fold_dir / "metrics.json")
    _save_training_history(train_result.history, fold_dir / "training_history.json")
    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_evaluation=tune_evaluation,
        test_evaluations=test_evaluations,
    )


def _bag_head_class(task_cls: type, *, task_name: str, aggregator_name: str) -> type:
    """Pick the head class for a bag-level run.

    ``clam_mb`` emits one branch per class and needs a branch-aware head; every other
    task family (binary, ordinal, regression, survival, ...) has no per-class branch to
    route through. A registered head that already handles branch representations (the
    built-in ``branch_aware_classification`` or a user subclass of it) is kept as is, so
    its custom loss, metrics and constructor parameters survive; the ordinary multiclass
    head is swapped for the built-in branch-aware one.
    """
    if aggregator_name != "clam_mb":
        return task_cls
    task_family = str(task_cls.task_family)
    if task_family != "multiclass_classification":
        raise ValueError(
            f"clam_mb does not support task '{task_name}' (family "
            f"'{task_family}'); it only supports the multiclass_classification "
            "family. Use a different aggregator."
        )
    if getattr(task_cls, "supports_branch_representation", False):
        return task_cls
    if task_cls is MulticlassClassificationHead:
        return BranchAwareClassificationHead
    raise ValueError(
        f"clam_mb requires custom task head '{task_name}' to support branch "
        "representations. Subclass BranchAwareClassificationHead to preserve "
        "your custom loss and metrics."
    )


def _build_segmentation_head(
    *,
    task: TaskConfig,
    evaluation: EvalConfig,
    preprocessing: PreprocessingConfig | None,
    masks: "MasksConfig | None",
    geometry,
) -> SegmentationHead:
    """Build the fold-independent segmentation head bound to ``geometry``.

    The head class is the one registered under ``task.name`` (a user subclass of
    :class:`SegmentationHead` with a custom loss or metric reduction, or the built-in).
    Its label maps come from :func:`_segmentation_label_maps`.
    """
    head_cls = task_registry.get(task.name)
    num_classes, class_names, label_remap = resolve_class_scheme(
        task.params, annotation_rasters=masks is not None
    )
    seg_params = {
        key: value
        for key, value in task.params.items()
        if key not in CLASS_SCHEME_KEYS
    }
    mask_spacing_um = (
        preprocessing.requested_spacing_um if preprocessing is not None else None
    )
    return head_cls(
        num_classes=num_classes,
        geometry=geometry,
        metrics=evaluation.metrics,
        spacing_um=float(mask_spacing_um) if mask_spacing_um is not None else None,
        spacing_policy=(
            preprocessing.spacing_policy if preprocessing is not None else "strict"
        ),
        backend=preprocessing.mask_backend if preprocessing is not None else "auto",
        image_backend=preprocessing.backend if preprocessing is not None else "auto",
        tolerance=float(preprocessing.tolerance) if preprocessing is not None else 0.05,
        label_remap=label_remap,
        pixel_mapping=masks.pixel_mapping if masks is not None else None,
        class_names=class_names if len(class_names) == num_classes else None,
        **seg_params,
    )


def _segmentation_run_records(
    dataset: LegacySamples, splits: LegacyFolds
) -> list[LegacyRecord]:
    """Return every ROI named by the run's folds, once, in manifest order."""
    sample_ids = {
        sample_id
        for fold_split in splits.folds
        for sample_id in (
            *fold_split.train,
            *fold_split.tune,
            *(sample_id for ids in fold_split.tests.values() for sample_id in ids),
        )
    }
    return [record for sample_id, record in dataset.samples.items() if sample_id in sample_ids]


def _resolve_segmentation_roi_population(
    *,
    cache_root: str | Path,
    records: list[LegacyRecord],
    head: SegmentationHead,
    label_maps,
    source_fingerprint_cache: dict[str, dict[str, object]] | None,
    resolved_backend_cache: dict[tuple[str, str, bool], str] | None,
) -> "SegmentationRoiPopulation":
    """Resolve one population with the run's shared cache policy."""
    from soma.learn.roi_population import resolve_segmentation_roi_population

    return resolve_segmentation_roi_population(
        cache_root,
        records,
        lambda record: {"mask": label_maps.load(record.sample_id)},
        num_classes=head.num_classes,
        target_identity=head.target_identity,
        workers=max(1, min(8, os.cpu_count() or 1)),
        source_fingerprint_cache=source_fingerprint_cache,
        resolved_backend_cache=resolved_backend_cache,
    )


def train(
    feature_store: CachedSetSource,
    dataset: LegacySamples,
    splits: LegacyFolds,
    task: TaskConfig,
    training: TrainingConfig,
    run_dir: str | Path,
    aggregator: AggregatorConfig | None = None,
    decoder: DecoderConfig | None = None,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    masks: "MasksConfig | None" = None,
    heatmaps: HeatmapConfig | None = None,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
    dataset_type: str = "slide",
    encoder_stage=None,
    augmentation: dict | None = None,
    test_digest: str | None = None,
    overwrite_test: bool = False,
    run_id: str | None = None,
    artifact_mirror: ArtifactMirror | None = None,
    resume_completed_folds: bool = False,
    roi_population_cache_root: str | Path | None = None,
    folds: Sequence[int] | None = None,
) -> PipelineResult:
    """Train and evaluate all folds, then summarize.

    ``folds`` restricts this launch to those fold indices; other pending folds are left
    for another launch into the same ``run_dir``, and the summary waits for all of them.

    Args:
        feature_store: Precomputed embeddings.
        dataset: LegacySamples with sample records and label_map.
        splits: Cross-validation splits (1 or more folds).
        dataset_type: ``"slide"`` for WSI pipelines; ``"tile"`` for tile-image
            pipelines.
        aggregator: Aggregator configuration, or None for slide-level or
            tile-dataset features.
        task: Task head configuration.
        evaluation: Evaluation configuration (metrics, subgroups). Defaults to EvalConfig().
        training: Training loop configuration.
        run_dir: Root directory — each fold gets a fold_N/ subdirectory.
        heatmaps: When provided and enabled, attention scores are captured
            during the test evaluation pass and saved to fold_N/attention/.
        normalization: Feature-adaptor normalization, fit per fold on that fold's
            Support (train) split. Defaults to no adaptor.
        projection: Feature-adaptor label-free projection to a common width, fit per
            fold on that fold's Support split. Defaults to no projection.
        encoder_identity: Name of the encoder behind the features; seeds the
            ``random`` projection.
        encoder_stage: The frozen encoder stage of a live grid run (shared by every
            fold); ``feature_store`` is then the pixels' image source.
        augmentation: ``{name, params}`` augmentation of a live grid run's train split.

    Returns:
        PipelineResult with per-fold results and aggregated summary.
    """
    # Feature-adaptor path guard (issues #283-#286): a requested transform a path cannot
    # honor must fail, never be silently dropped. Composite (multi-encoder) dense streams
    # keep their per-member ``member_norm``; fitting one z-score over a channel-concatenated
    # grid would silently treat several encoders as one.
    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        dataset_type=dataset_type,
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
    )
    shape = extraction_shape(dataset_type)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if roi_population_cache_root is None:
        roi_population_cache_root = run_dir / "segmentation_roi_population"
    evaluation = evaluation or EvalConfig()
    include_tune_in_summary = evaluation.holdout_test

    single_fold = splits.num_folds == 1

    def _fold_dir(fold_idx: int) -> Path:
        # A single fold owns run_dir directly; multi-fold folds get fold_N/ subdirs.
        return run_dir if single_fold else run_dir / f"fold_{fold_idx}"

    if not single_fold:
        _reject_undeclared_fold_dirs(run_dir, splits.num_folds)
    if folds is not None:
        unknown = sorted(set(folds) - set(range(splits.num_folds)))
        if single_fold or unknown:
            raise ValueError(
                f"run.folds={list(folds)} does not fit the splits: they declare "
                f"{splits.num_folds} fold(s), indexed from 0."
            )
    # A fold with a metrics.json is complete; any without one is pending. A run with
    # pending folds is a resume in progress (issue #244).
    pending_folds = [
        i for i in range(splits.num_folds) if not (_fold_dir(i) / "metrics.json").exists()
    ]
    run_roi_population = None
    roi_source_fingerprint_cache: dict[str, dict[str, object]] = {}
    roi_resolved_backend_cache: dict[tuple[str, str, bool], str] = {}

    # Test-results clobber guard (issue #247). Experiment identity is test-invariant, so a
    # run dir may be re-scored against several test sets. Reserve the test-identity slot
    # under this run: a fresh identity records and proceeds (default single-test behavior
    # unchanged); an already-scored identity is refused unless overwrite_test — the test
    # splits are then held out so the prior result is left intact. Skipped when there is no
    # test scoring to guard (holdout_test) or the caller opted out (test_digest is None).
    if test_digest is not None and not evaluation.holdout_test:
        test_split_names = sorted(
            {name for fold in splits.folds for name in fold.test_split_names}
        )
        decision = register_test_result(
            run_dir,
            test_digest,
            split_names=test_split_names,
            run_id=run_id,
            overwrite=overwrite_test,
        )
        if decision.skipped and pending_folds:
            # Resume (issue #244): the digest was recorded by the original run, but folds
            # remain unscored. Scoring them is not a re-score — the fold-skip guard below
            # skips every already-scored fold, so no completed result is clobbered.
            logger.info(
                "Test identity %s already recorded, but %d fold(s) are unscored; resuming "
                "and scoring them (completed folds are skipped, not re-scored).",
                test_digest[:12],
                len(pending_folds),
            )
        elif decision.skipped:
            logger.warning(
                "Test identity %s was ALREADY scored for this run (recorded %s, splits %s); "
                "skipping test inference so the existing result is not clobbered. Set "
                "evaluation.overwrite_test=True to re-score.",
                test_digest[:12],
                (decision.prior or {}).get("recorded_at", "?"),
                (decision.prior or {}).get("split_names", []),
            )
            evaluation = replace(evaluation, holdout_test=True)

    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        fold_dir = _fold_dir(fold_idx)
        fold_was_complete = (fold_dir / "metrics.json").exists()
        # Resume fold-skip guard (issue #244): a fold that already wrote metrics.json is
        # complete, so on a relaunch into the same run dir we skip retraining it. The
        # summary is rebuilt from every fold's metrics.json on disk below, so a skipped
        # fold still counts. Gated to multi-fold: a single fold owns run_dir itself and
        # has no partial-run to preserve.
        if (not single_fold or resume_completed_folds) and fold_was_complete:
            logger.info(
                "Fold %d already complete (found %s); skipping (resume).",
                fold_idx,
                fold_dir / "metrics.json",
            )
            if artifact_mirror is not None:
                artifact_mirror.fold_completed(fold=fold_idx, fold_dir=fold_dir)
            continue
        if folds is not None and fold_idx not in folds:
            logger.info("Fold %d is not in run.folds; leaving it to another launch.", fold_idx)
            continue
        if shape == "grid":
            if (
                dataset_type == "segmentation"
                and training.roi_batch_sampling is not None
                and encoder_stage is None
                and run_roi_population is None
            ):
                run_records = _segmentation_run_records(dataset, splits)
                run_head = _build_segmentation_head(
                    task=task,
                    evaluation=evaluation,
                    preprocessing=preprocessing,
                    masks=masks,
                    geometry=feature_store.geometry(run_records[0].sample_id),
                )
                run_roi_population = _resolve_segmentation_roi_population(
                    cache_root=roi_population_cache_root,
                    records=run_records,
                    head=run_head,
                    label_maps=_segmentation_label_maps(run_head, run_records),
                    source_fingerprint_cache=roi_source_fingerprint_cache,
                    resolved_backend_cache=roi_resolved_backend_cache,
                )
            result = train_one_grid_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                task=task,
                decoder=decoder,
                dataset_type=dataset_type,
                evaluation=evaluation,
                training=training,
                fold_dir=fold_dir,
                preprocessing=preprocessing,
                masks=masks,
                fold=fold_idx,
                num_folds=splits.num_folds,
                normalization=normalization,
                projection=projection,
                encoder_identity=encoder_identity,
                encoder_stage=encoder_stage,
                augmentation=augmentation,
                roi_population_cache_root=roi_population_cache_root,
                resolved_roi_population=run_roi_population,
                roi_source_fingerprint_cache=roi_source_fingerprint_cache,
                roi_resolved_backend_cache=roi_resolved_backend_cache,
            )
        else:
            result = train_one_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                dataset_type=dataset_type,
                aggregator=aggregator,
                task=task,
                evaluation=evaluation,
                training=training,
                fold_dir=fold_dir,
                fold=fold_idx,
                num_folds=splits.num_folds,
                preprocessing=preprocessing,
                heatmaps=heatmaps,
                normalization=normalization,
                projection=projection,
                encoder_identity=encoder_identity,
            )
        if result.train_result is not None:
            # The FitResult keeps the fold's learner alive; move it off the accelerator so
            # completed folds do not hold GPU memory while the next one trains.
            result.train_result.learner.cpu()
        fold_results.append(result)
        if artifact_mirror is not None and not fold_was_complete:
            artifact_mirror.fold_completed(fold=fold_idx, fold_dir=fold_dir)

    still_pending = tuple(
        i for i in range(splits.num_folds) if not (_fold_dir(i) / "metrics.json").exists()
    )
    if still_pending:
        # Only reachable with run.folds. A summary over some of the folds would read as the
        # run's result, so the launch that finds every fold complete writes it instead.
        logger.info(
            "Fold(s) %s still pending; summary.json is left to the launch that completes them.",
            list(still_pending),
        )
        return PipelineResult(
            fold_results=fold_results, summary={}, run_dir=run_dir, pending_folds=still_pending
        )

    # Aggregate from every fold's metrics.json on disk, not just this session's
    # fold_results — so a resumed run (some folds skipped above) still summarizes all
    # folds. For a fresh run every fold is on disk, so this equals the in-memory path.
    summary = _aggregate_fold_metrics_from_disk(
        run_dir,
        single_fold=single_fold,
        include_tune=include_tune_in_summary,
        num_folds=splits.num_folds,
    )
    _save_summary(summary, run_dir / "summary.json")

    return PipelineResult(
        fold_results=fold_results,
        summary=summary,
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Layer 2 — Pipeline orchestrator
# ---------------------------------------------------------------------------


def _build_run_summary_panel(
    *,
    encoder: EncoderConfig | None,
    preprocessing: PreprocessingConfig,
    aggregator: AggregatorConfig | None,
    task: TaskConfig,
    feature_store: CachedSetSource,
    dataset_type: str = "slide",
    decoder: DecoderConfig | None = None,
) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", no_wrap=True)
    grid.add_column()

    # Encoder
    if encoder is not None:
        grid.add_row("encoder", encoder.name)
    else:
        grid.add_row("encoder", "[dim]pre-extracted[/dim]")

    # Spacing — only meaningful for slide/WSI pipelines
    if dataset_type != "tile":
        spacing = preprocessing.requested_spacing_um
        grid.add_row("spacing", f"{spacing} µm" if spacing is not None else "[dim]—[/dim]")

    # Aggregator (set shape) or decoder (grid shape) — the trainable component.
    if dataset_type in ("segmentation", "detection"):
        grid.add_row("decoder", decoder.name if decoder is not None else "[dim]—[/dim]")
    elif aggregator is not None:
        grid.add_row("aggregator", aggregator.name)
    else:
        grid.add_row("aggregator", "[dim]—[/dim]")

    # Task
    grid.add_row("task", task.name.replace("_", " "))

    # Feature level + dim
    if dataset_type == "patient":
        level = "patient"
    elif dataset_type == "tile":
        level = "tile (encoded)"
    elif dataset_type in ("segmentation", "detection"):
        # A grid source is not a set source (no rank); branch before rank is touched.
        level = "dense (segmentation)" if dataset_type == "segmentation" else "dense (detection)"
    elif feature_store.rank == 1:
        level = "slide"
    elif feature_store.rank == 3:
        level = "hierarchical"
    else:
        level = "tile"
    feature_dim = getattr(feature_store, "feature_dim", None)
    if feature_dim is None:
        grid.add_row("features", f"{level}  ·  live re-encoding")
    else:
        grid.add_row("features", f"{level}  ·  dim={feature_dim}")

    return Panel.fit(
        grid,
        title="[bold]run summary[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _build_completed_run_panel(*, summary_metrics: dict[str, float]) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", no_wrap=True)
    grid.add_column()

    def _split_metric(split_name: str, metric: str) -> float | None:
        for suffix in ("_mean", ""):
            key = f"{split_name}/{metric}{suffix}"
            if key in summary_metrics:
                return summary_metrics[key]
        return None

    split_names = sorted({key.split("/", 1)[0] for key in summary_metrics if "/" in key})
    for split_name in split_names:
        primary_metric_name: str | None = None
        primary_metric_mean: float | None = None
        for key, value in summary_metrics.items():
            prefix = f"{split_name}/"
            if not key.startswith(prefix):
                continue
            if key.endswith("_mean"):
                metric_name = key[len(prefix):-5]
            else:
                metric_name = key[len(prefix):]
            if metric_name.endswith("_std"):
                continue
            if metric_name in {
                "coverage",
                "num_samples",
                "num_real_samples",
                "num_placeholder_samples",
            }:
                continue
            primary_metric_name = metric_name
            primary_metric_mean = value
            break

        coverage = _split_metric(split_name, "coverage")
        num_samples = _split_metric(split_name, "num_samples")
        num_real = _split_metric(split_name, "num_real_samples")
        num_placeholder = _split_metric(split_name, "num_placeholder_samples")
        if (
            coverage is None
            or num_samples is None
            or num_real is None
            or num_placeholder is None
        ):
            continue
        metric_text = (
            f"{primary_metric_name}={primary_metric_mean:.4f}  ·  "
            if primary_metric_name is not None and primary_metric_mean is not None
            else ""
        )
        coverage_text = (
            f"{metric_text}{coverage * 100:.1f}%"
            f" ({int(round(num_real))}/{int(round(num_samples))})"
            f"  ·  placeholder={int(round(num_placeholder))}"
        )
        grid.add_row(f"{split_name} coverage", coverage_text)

    if not grid.rows:
        grid.add_row("coverage", "[dim]—[/dim]")

    return Panel.fit(
        grid,
        title="[bold]completed run[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _resolve_hipt_params(preprocessing: PreprocessingConfig, aggregator: AggregatorConfig) -> dict[str, object]:
    patch_size = preprocessing.requested_tile_size_px or preprocessing.read_tile_size_px
    region_size = preprocessing.requested_region_size_px or preprocessing.read_region_size_px
    tile_multiple = preprocessing.region_tile_multiple
    if patch_size is None or region_size is None:
        raise ValueError("hierarchical preprocessing must resolve patch and region sizes")
    patch_size = int(patch_size)
    region_size = int(region_size)
    if tile_multiple is None:
        if region_size % patch_size != 0:
            raise ValueError(
                "hierarchical preprocessing requires requested_region_size_px to be divisible by requested_tile_size_px"
            )
        tile_multiple = region_size // patch_size
    tile_multiple = int(tile_multiple)
    if region_size != patch_size * tile_multiple:
        raise ValueError(
            "hierarchical preprocessing requires requested_region_size_px to equal "
            "requested_tile_size_px × region_tile_multiple"
        )

    params = {
        key: value
        for key, value in aggregator.params.items()
        if key not in {"region_size", "patch_size", "tile_multiple"}
    }
    params.update(
        {
            "region_size": region_size,
            "patch_size": patch_size,
        }
    )
    return params


def _guard_resume_config_drift(run_dir: Path, config: PipelineConfig) -> None:
    """Refuse to resume into a run dir whose saved config differs from the current one.

    Only a resume (``resume``/``run_id`` set) that lands on an existing ``config.yaml``
    is checked — a fresh run, or a pinned run id naming a not-yet-created dir, has
    nothing to compare against. The comparison is over the persisted YAML form
    (:func:`config_yaml_dict`, which omits the ``resume``/``run_id`` directives), so an
    otherwise-identical relaunch passes while a changed seed, split file, encoder, or
    eval protocol is caught before it silently mixes incompatible folds (issue #244)."""
    if not (config.resume or config.run_id):
        return
    saved_path = run_dir / "config.yaml"
    if not saved_path.is_file():
        return
    saved = yaml.safe_load(saved_path.read_text(encoding="utf-8")) or {}
    # Round-trip the current config through YAML so both sides are the same primitive
    # shapes (tuples→lists, Paths→str) and compare cleanly.
    current = yaml.safe_load(yaml.safe_dump(config_yaml_dict(config)))
    # Shared-storage routing is operational: a resume may repoint a pending spool without
    # changing the training recipe that produced its checkpoint.
    for payload in (saved, current):
        run = payload.get("run")
        if isinstance(run, dict):
            run.pop("mirror_root", None)
    if saved == current:
        _guard_resume_manifest_drift(run_dir, config)
        return
    differing = sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )
    raise ValueError(
        f"Refusing to resume run '{run_dir.name}': its saved config.yaml differs from the "
        f"current config (differing top-level sections: {differing}). Resuming would mix "
        "folds trained under different recipes. Start a fresh run, or reconcile the config "
        "to match the existing run."
    )


def _guard_resume_manifest_drift(run_dir: Path, config: PipelineConfig) -> None:
    """Refuse to resume when the train/tune manifest *content* changed under the same paths.

    ``config.yaml`` only records the dataset/splits file paths. The run's ``run.yaml``
    carries the ``experiment_id`` minted from the train+tune slice content, so recompute
    that identity for the current files and compare: an edited label or reshuffled fold
    with unchanged paths is caught here rather than mixing folds trained on different data.
    """
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.is_file():
        return
    saved = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
    saved_experiment_id = saved.get("experiment_id")
    if not saved_experiment_id:
        return
    current_experiment_id = build_experiment_spec(config).experiment_id
    if saved_experiment_id == current_experiment_id:
        return
    raise ValueError(
        f"Refusing to resume run '{run_dir.name}': the train/tune content of "
        f"{config.dataset_csv} / {config.splits_csv} differs from what the run was trained "
        f"on (experiment identity {str(saved_experiment_id)[:12]} → "
        f"{current_experiment_id[:12]}). Resuming would mix folds trained on different "
        "data. Start a fresh run, or restore the original manifests."
    )


class _RunRecorder:
    """Context manager that tracks run metadata lifecycle (running → completed/failed).

    On enter: writes initial "running" metadata and appends to the per-run index.
    On success: call ``complete(summary_metrics)`` then exit normally.
    On exception: writes "failed" metadata and (if no prior successful run exists)
    advances the latest pointer before re-raising.
    """

    def __init__(self, layout, config: "PipelineConfig") -> None:
        self._layout = layout
        self._config = config
        self._metadata = None
        self._summary_metrics: dict[str, float] | None = None
        self._pending_folds: tuple[int, ...] = ()

    def __enter__(self) -> "_RunRecorder":
        layout = self._layout
        self._metadata = create_run_metadata(
            config=self._config,
            experiment=layout.experiment,
            run_dir=layout.run_dir,
            run_id=layout.run_id,
            status="running",
            started_at=datetime.now().astimezone().isoformat(),
        )
        write_run_metadata(layout.run_dir, self._metadata)
        self._write_indexes(self._metadata)
        return self

    def complete(self, summary_metrics: dict[str, float]) -> None:
        self._summary_metrics = summary_metrics

    def partial(self, pending_folds: tuple[int, ...]) -> None:
        """This launch trained its ``run.folds`` share; other folds are still pending."""
        self._pending_folds = pending_folds

    def with_identity(self, result: "PipelineResult") -> "PipelineResult":
        """The result with the manifest checksums the run's metadata files record."""
        experiment = self._layout.experiment
        return replace(
            result,
            dataset_checksum=experiment.dataset_checksum,
            splits_checksum=experiment.splits_checksum,
            test_checksum=self._metadata.test_checksum,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        layout = self._layout
        if exc_type is not None:
            metadata = self._metadata.with_updates(
                status="failed",
                finished_at=datetime.now().astimezone().isoformat(),
                error=str(exc_val),
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
            if not has_successful_run(layout.experiment_dir):
                update_latest_pointer(layout.experiment_dir, layout.run_dir)
        elif self._pending_folds:
            # Not "completed": the leaderboard ranks completed runs, and this one has no
            # summary yet. The latest pointer stays where it was for the same reason.
            metadata = self._metadata.with_updates(
                status="partial",
                finished_at=datetime.now().astimezone().isoformat(),
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
        else:
            metadata = self._metadata.with_updates(
                status="completed",
                finished_at=datetime.now().astimezone().isoformat(),
                summary_metrics=self._summary_metrics or {},
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
            update_latest_pointer(layout.experiment_dir, layout.run_dir)
        return False

    def _write_indexes(self, metadata) -> None:
        # Each run touches ONLY its own dir + the append-friendly per-run index. The
        # experiment-level projection is no longer written here: it was an unlocked
        # read-modify-rewrite of a shared CSV that silently lost rows when concurrent
        # sweep runs finished together (ADR 0003). The leaderboard now rebuilds that
        # projection on demand by scanning the self-describing run dirs.
        layout = self._layout
        update_run_index(layout.index_dir / "runs.csv", metadata)


def composite_member_extraction_spec(
    member: EncoderMemberConfig, preprocessing: PreprocessingConfig
) -> tuple[EncoderConfig, PreprocessingConfig]:
    """The ``(EncoderConfig, PreprocessingConfig)`` one composite member extracts with.

    Mirrors the single-encoder path exactly: the member's encoder config is run through
    :func:`resolve_preprocessing_config` (the same encoder-aware fill
    ``resolve_pipeline_preprocessing`` applies when ``encoder:`` is set), so a member's
    dense cache key is identical to the key the same encoder produces in a single-encoder
    run — composite reuse of already-extracted per-encoder caches depends on this.
    The per-member sliding window (CONCH native-448 vs H0-mini native-224, …) falls back
    to the run's shared window when the member leaves it unset; both land on the returned
    preprocessing config's ``dense_window_size`` / ``dense_window_overlap``.

    Note: the encoder-aware fill sets ``ref_tile_size_px`` / ``read_tile_size_px``, which
    are part of the dense cache key. Composite member caches extracted before this fill
    was applied to members key differently and are not reused.
    """
    from soma.encoders.validation import resolve_preprocessing_config

    member_encoder = EncoderConfig(
        name=member.name,
        precision=member.precision,
        batch_size=member.batch_size,
        adaptive_batching=member.adaptive_batching,
        output_variant=member.output_variant,
        allow_non_recommended_settings=member.allow_non_recommended_settings,
    )
    member_prep = replace(
        resolve_preprocessing_config(member_encoder, preprocessing),
        feature_kind=member.feature_kind,
        attention=member.attention,
        dense_window_size=(
            member.dense_window_size
            if member.dense_window_size is not None
            else preprocessing.dense_window_size
        ),
        dense_window_overlap=(
            member.dense_window_overlap
            if member.dense_window_overlap is not None
            else preprocessing.dense_window_overlap
        ),
    )
    return member_encoder, member_prep


def build_composite_dense_store(
    config: PipelineConfig, dataset: LegacySamples, *, output_root: Path
):
    """Extract every composite member into its own cache; return the concat view (§7).

    Public so that out-of-core consumers (a project protocol's rescore step) rebuild the
    exact store a composite run trained on, instead of re-deriving member geometry.
    Each member carries its own ``feature_kind`` / ``attention`` / ``member_norm``
    spec (cross-defaulted in :class:`PipelineConfig`), so the heterogeneous setup
    (several FMs, some attention, some patch-feature) needs no special-casing. Members
    share the run's spacing + supervision ``target_size`` (v1); their token grids may
    differ and are combined at load time by :class:`CompositeDenseFeatureStore` per the
    composite's ``concat_resolution`` / ``concat_grid_size``.

    Args:
        config: A config with ``composite`` set.
        dataset: The manifest the members are extracted over (the caller's, so a
            rescore reuses the manifest it already holds).
        output_root: Where per-member feature dirs (``member_<i>_<name>``) are written.
            Cache hits make these cheap views; the cache root itself comes from
            ``config.cache`` (default ``<run.output_root>/feature_cache``).
    """
    from soma.dense.composite import CompositeDenseFeatureStore
    from soma.dense_extraction import _DenseImageExtractor

    composite = config.composite
    if composite is None:
        raise ValueError("build_composite_dense_store requires config.composite.")
    preprocessing = resolve_pipeline_preprocessing(config)
    target_size = preprocessing.requested_tile_size_px
    if target_size is None:
        raise ValueError(
            "multi-encoder extraction requires "
            "preprocessing.requested_tile_size_px (the mask/tile supervision size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "multi-encoder extraction requires an explicit or auto-resolved "
            "preprocessing.requested_spacing_um."
        )
    cache_config = replace(
        config.cache,
        root_dir=resolve_cache_root(
            config.cache, feature_dir=output_root, output_root=config.output_root
        ),
    )
    member_stores = []
    try:
        for index, member in enumerate(composite.encoders):
            member_encoder, member_prep = composite_member_extraction_spec(
                member, preprocessing
            )
            extractor = _DenseImageExtractor(
                dataset,
                member_encoder,
                target_size=int(target_size),
                spacing_um=float(preprocessing.requested_spacing_um),
                backend=preprocessing.backend,
                tolerance=float(preprocessing.tolerance),
                window_size=member_prep.dense_window_size,
                overlap=float(member_prep.dense_window_overlap),
                execution=config.execution,
                cache=cache_config,
                preprocessing=member_prep,
            )
            member_dir = Path(output_root) / f"member_{index}_{member.name}"
            member_stores.append(extractor.run(feature_dir=member_dir))
    finally:
        _release_parent_cuda_state()
        _log_cuda_memory("after composite dense extraction release")
    return CompositeDenseFeatureStore(
        member_stores,
        concat_resolution=composite.concat_resolution or "target",
        concat_grid_size=composite.concat_grid_size,
        member_norms=[m.member_norm or "none" for m in composite.encoders],
    )


def live_encoder_for(config: PipelineConfig):
    """The frozen encoder stage a ``feature_mode='live'`` config re-encodes with.

    Resolves the config's preprocessing (tile size, spacing, dense window, feature kind,
    attention recipe) and the encoder's precision, then builds the stage through
    :func:`soma.extract.live_encoder`. Shared by :class:`Pipeline` and by whole-image
    inference over a finished run (:func:`soma.dense.predict.build_dense_models`).
    """
    from soma import extract
    from soma.encoders.validation import resolve_encoder_precision

    if config.encoder is None:
        raise ValueError("PipelineConfig.encoder is required for feature_mode='live'.")
    preprocessing = resolve_pipeline_preprocessing(config)
    if preprocessing.requested_tile_size_px is None:
        raise ValueError(
            "feature_mode='live' requires preprocessing.requested_tile_size_px (the "
            "supervision tile size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "feature_mode='live' requires a spacing: set preprocessing.requested_spacing_um "
            "or use an encoder that advertises a single supported_spacing_um."
        )
    return extract.live_encoder(
        config.encoder.name,
        target_size=int(preprocessing.requested_tile_size_px),
        spacing_um=float(preprocessing.requested_spacing_um),
        window=preprocessing.dense_window_size,
        overlap=float(preprocessing.dense_window_overlap),
        feature_kind=preprocessing.feature_kind or "patch_features",
        precision=resolve_encoder_precision(config.encoder, encoder_name=config.encoder.name),
        output_variant=config.encoder.output_variant,
        allow_non_recommended_settings=config.encoder.allow_non_recommended_settings,
        tolerance=float(preprocessing.tolerance),
        backend=preprocessing.backend,
        attention_blocks=tuple(preprocessing.attention.blocks),
        attention_include_registers=bool(preprocessing.attention.include_registers),
        num_gpus=config.execution.num_gpus,
    )


def extraction_shape(dataset_type: str) -> str:
    """The feature shape a ``dataset_type`` trains on: ``grid`` for dense tasks, else ``set``."""
    return "grid" if dataset_type in {"segmentation", "detection"} else "set"


def extraction_unit(config: PipelineConfig) -> str:
    """What one manifest row is to extraction: a whole slide or a pre-cropped image."""
    if config.dataset_type in {"slide", "patient"}:
        return "slide"
    if config.dataset_type in {"segmentation", "detection"} and config.preprocessing.masks is not None:
        return "slide"
    return "tile"


def resolve_task_class(config: PipelineConfig):
    """The registered head class a config trains (survival resolves its loss variant)."""
    if config.task is None:
        return None
    task_cls = task_registry.get(config.task.name)
    if task_cls.task_family == "survival":
        from soma.tasks.survival import resolve_survival_task

        task_cls, _ = resolve_survival_task(config.task.name, config.task.params)
    return task_cls


def load_pipeline_data(
    config: PipelineConfig,
) -> tuple[Cohort, ImageManifest, AnnotationManifest | None]:
    """Read the cohort and manifests a config names, validated against its task head.

    The head's ``resolve_target_schema(task.params)`` says which target keys the cohort
    must carry (``label``, ``time``/``event``, ``value`` or ``value_<k>`` ...);
    ``task.targets`` maps a key to a differently named CSV column.
    ``training.tune_is_test`` reuses each fold's tune split for test reporting via
    :meth:`Cohort.with_test_from_tune`.
    """
    task_cls = resolve_task_class(config)
    schema = task_cls.resolve_target_schema(config.task.params) if task_cls is not None else {}
    if config.representation is not None and config.task is None:
        schema = {"label": object}
    renames = dict(config.task.targets) if config.task is not None and config.task.targets else {}
    unknown = sorted(set(renames) - set(schema))
    if unknown:
        raise ValueError(
            f"task.targets names key(s) {unknown} the task head does not declare; "
            f"its target keys are {sorted(schema)}."
        )
    targets = {key: renames.get(key, key) for key in schema}
    cohort = Cohort.from_csv(
        config.dataset_csv,
        config.splits_csv,
        targets=targets,
        dtypes=schema,
        unit="patient_id" if config.dataset_type == "patient" else None,
        allow_missing_test=config.training.tune_is_test,
    )
    if config.training.tune_is_test:
        cohort = cohort.with_test_from_tune()
    image_manifest = ImageManifest.from_csv(config.dataset_csv)
    annotation_manifest = None
    if config.dataset_type in {"segmentation", "detection"}:
        masks = config.preprocessing.masks
        annotation_manifest = AnnotationManifest.from_csv(
            config.dataset_csv,
            pixel_mapping=masks.pixel_mapping if masks is not None else None,
        )
        if config.dataset_type == "segmentation" and not annotation_manifest.has_label_masks:
            raise ValueError("dataset_type='segmentation' needs a label_mask_path column.")
        if config.dataset_type == "detection" and not annotation_manifest.has_points:
            raise ValueError("dataset_type='detection' needs a points_path column.")
    return cohort, image_manifest, annotation_manifest


class Pipeline:
    """Orchestrates the full pipeline: extract → train all folds → summarize.

    Args:
        config: Complete pipeline specification.
        feature_dir: Path to pre-extracted features. If provided, skips extraction.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        feature_dir: str | Path | None = None,
    ) -> None:
        ensure_default_logging()
        self._config = config
        self._cohort, self._image_manifest, self._annotation_manifest = load_pipeline_data(config)
        # Legacy bridge (delete in slice 7, #581): pipeline.py still consumes fat records.
        self._dataset = records_for_pipeline(
            self._cohort,
            self._image_manifest,
            self._annotation_manifest,
            path=config.dataset_csv,
        )
        self._splits = LegacyFolds.from_cohort(self._cohort)
        self._feature_dir = Path(feature_dir) if feature_dir else None

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def cohort(self) -> Cohort:
        return self._cohort

    @property
    def image_manifest(self) -> ImageManifest:
        return self._image_manifest

    @property
    def annotation_manifest(self) -> AnnotationManifest | None:
        return self._annotation_manifest

    @property
    def dataset(self) -> LegacySamples:
        """Legacy fat records (delete in slice 7, #581)."""
        return self._dataset

    @property
    def splits(self) -> LegacyFolds:
        return self._splits

    def run(self) -> PipelineResult:
        """Run the full pipeline: save config → load features → train all folds → summarize."""
        restored_run_id = restore_run_from_mirror(
            self._config, num_folds=self._splits.num_folds
        )
        layout = resolve_managed_output_paths(self._config, run_id=restored_run_id)
        layout.output_root.mkdir(parents=True, exist_ok=True)
        layout.experiment_dir.mkdir(parents=True, exist_ok=True)
        (layout.experiment_dir / "runs").mkdir(parents=True, exist_ok=True)
        layout.run_dir.mkdir(parents=True, exist_ok=True)
        layout.index_dir.mkdir(parents=True, exist_ok=True)
        write_experiment_metadata(layout.experiment_dir, layout.experiment)
        # Resume drift guard (issue #244): refuse to resume into a run dir whose saved
        # config differs from the current one — mixing incompatible folds silently would
        # corrupt the aggregate. Runs before save_config overwrites the record.
        _guard_resume_config_drift(layout.run_dir, self._config)
        save_config(self._config, layout.run_dir / "config.yaml")
        artifact_mirror: ArtifactMirror | None = None
        if self._config.mirror_root is not None:
            relative_run_dir = layout.run_dir.relative_to(layout.output_root)
            artifact_mirror = ArtifactMirror(
                run_dir=layout.run_dir,
                mirror_run_dir=Path(self._config.mirror_root) / relative_run_dir,
            )
            artifact_mirror.retry_pending()

        with _RunRecorder(layout, self._config) as recorder:
            source_context = self._get_feature_source_context(run_dir=layout.run_dir)
            store = source_context.feature_store
            training_dataset = source_context.dataset
            training_splits = source_context.splits
            _write_dense_source_provenance(layout.run_dir, store)
            if self._config.representation is not None:
                log_path = layout.run_dir / "run.log"
                log_path.write_text(
                    "Starting task-free CRoMa representation evaluation.\n",
                    encoding="utf-8",
                )
                result = evaluate_representation(
                    feature_store=store,
                    dataset=training_dataset,
                    splits=training_splits,
                    representation=self._config.representation,
                    run_dir=layout.run_dir,
                )
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("Completed task-free CRoMa representation evaluation.\n")
                Console().print(_build_completed_run_panel(summary_metrics=result.summary))
                recorder.complete(result.summary)
                return recorder.with_identity(result)

            preprocessing = resolve_pipeline_preprocessing(self._config)
            Console().print(
                _build_run_summary_panel(
                    encoder=self._config.encoder,
                    preprocessing=preprocessing,
                    aggregator=self._config.aggregator,
                    task=self._config.task,
                    feature_store=store,
                    dataset_type=self._config.dataset_type,
                    decoder=self._config.decoder,
                )
            )

            roi_population_cache_root = (
                resolve_cache_root(
                    self._config.cache,
                    feature_dir=layout.run_dir / "features",
                    output_root=self._config.output_root,
                )
                / "segmentation_roi_population"
                if self._config.cache.enabled
                else layout.run_dir / "segmentation_roi_population"
            )
            result = train(
                feature_store=store,
                dataset=training_dataset,
                splits=training_splits,
                dataset_type=self._config.dataset_type,
                aggregator=self._config.aggregator,
                decoder=self._config.decoder,
                encoder_stage=source_context.encoder_stage,
                augmentation=self._config.augmentation,
                task=self._config.task,
                evaluation=self._config.evaluation,
                training=self._config.training,
                run_dir=layout.run_dir,
                preprocessing=preprocessing,
                masks=self._config.preprocessing.masks,
                heatmaps=self._config.heatmaps,
                normalization=self._config.normalization,
                projection=self._config.projection,
                encoder_identity=(
                    self._config.encoder.name if self._config.encoder is not None else ""
                ),
                test_digest=test_identity_digest(self._config),
                overwrite_test=self._config.evaluation.overwrite_test,
                run_id=layout.run_id,
                artifact_mirror=artifact_mirror,
                resume_completed_folds=bool(self._config.resume or self._config.run_id),
                roi_population_cache_root=roi_population_cache_root,
                folds=self._config.folds,
            )

            if result.pending_folds:
                # Heatmaps and the completed status describe a whole run.
                recorder.partial(result.pending_folds)
                return recorder.with_identity(result)

            if self._config.heatmaps.enabled:
                from soma.heatmaps import render_heatmaps
                logger.info("Rendering attention heatmaps...")
                render_heatmaps(
                    run_dir=layout.run_dir,
                    dataset=training_dataset,
                    tiling_dir=layout.run_dir / "tiling",
                    heatmap_config=self._config.heatmaps,
                    seg_downsample=self._config.preprocessing.seg_downsample,
                )

            Console().print(_build_completed_run_panel(summary_metrics=result.summary))
            recorder.complete(result.summary)

        return recorder.with_identity(result)

    def _get_feature_source_context(self, *, run_dir: Path) -> _FeatureSourceContext:
        is_dense = self._config.dataset_type in ("segmentation", "detection")
        annotation_sampled = (
            self._config.dataset_type == "segmentation"
            and self._config.preprocessing.masks is not None
        )
        if annotation_sampled and self._feature_dir is not None:
            raise ValueError(
                "feature_dir is not supported for annotation-sampled segmentation; "
                "FeatureExtractor owns the effective ROI dataset and cache identity."
            )
        if annotation_sampled and self._config.feature_mode == "live":
            raise NotImplementedError(
                "Live extraction over annotation-sampled segmentation ROIs is outside "
                "the persistent FeatureExtractor contract."
            )
        if annotation_sampled and self._config.composite is not None:
            raise NotImplementedError(
                "Multi-encoder extraction over annotation-sampled segmentation ROIs is "
                "not implemented."
            )
        if self._feature_dir is not None:
            from soma.dense import CachedGridSource

            source = (
                self._cache_backed_dense_source(
                    CachedGridSource(self._feature_dir),
                    kind="dense_cache",
                    dataset_csv=self._config.dataset_csv,
                )
                if is_dense
                else CachedSetSource(self._feature_dir)
            )
            return _FeatureSourceContext(
                feature_store=source,
                dataset=self._dataset,
                splits=self._splits,
            )

        # Live re-encoding and multi-encoder composition are intentionally outside the
        # single-encoder persistent extraction interface.
        if is_dense and self._config.feature_mode == "live":
            return self._live_source_context()
        if is_dense and self._config.composite is not None:
            return _FeatureSourceContext(
                feature_store=self._get_dense_source(run_dir=run_dir),
                dataset=self._dataset,
                splits=self._splits,
            )

        if self._config.encoder is None:
            raise ValueError("PipelineConfig.encoder is required when feature_dir is not provided.")
        cache_config = replace(
            self._config.cache,
            root_dir=resolve_cache_root(
                self._config.cache, feature_dir=run_dir, output_root=self._config.output_root
            ),
        )
        extraction = FeatureExtractor(
            self._dataset,
            self._config.encoder,
            resolve_pipeline_preprocessing(self._config),
            shape=extraction_shape(self._config.dataset_type),
            unit=extraction_unit(self._config),
            output_root=run_dir,
            execution=self._config.execution,
            cache=cache_config,
        ).extract()
        return _FeatureSourceContext(
            feature_store=extraction.source,
            dataset=extraction.dataset,
            splits=self._splits.project(extraction.dataset),
        )

    def _cache_backed_dense_source(
        self,
        store,
        *,
        kind: str,
        dataset_csv: str | Path,
        parent_dataset_csv: str | Path | None = None,
    ):
        from soma.dense import DenseSourceProvenance

        store.provenance = DenseSourceProvenance(
            kind=kind,
            feature_dir=getattr(store, "feature_dir", None),
            dataset_csv=dataset_csv,
            parent_dataset_csv=parent_dataset_csv,
        )
        return store

    def _live_source_context(self) -> _FeatureSourceContext:
        """The live grid path: the frozen encoder stage (built once, shared by every fold)
        and the image source of the pixels it re-encodes."""
        stage = live_encoder_for(self._config)
        preprocessing = resolve_pipeline_preprocessing(self._config)
        images = TileImageSource.from_manifest(
            self._image_manifest,
            size=int(preprocessing.requested_tile_size_px),
            spacing_um=float(preprocessing.requested_spacing_um),
            spacing_policy=preprocessing.spacing_policy,
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
        )
        return _FeatureSourceContext(
            feature_store=images, dataset=self._dataset, splits=self._splits, encoder_stage=stage
        )

    def _get_dense_source(self, *, run_dir: Path):
        """Multi-encoder composite: extract each member into its own cache, then present a
        load-time channel-concat view (design §7)."""
        if self._config.composite is None:
            raise RuntimeError("Dense source selection reached no composite branch.")
        return self._cache_backed_dense_source(
            build_composite_dense_store(
                self._config, self._dataset, output_root=run_dir / "features"
            ),
            kind="composite_dense_cache",
            dataset_csv=self._config.dataset_csv,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _records_for_sample_ids(
    dataset: LegacySamples,
    sample_ids: tuple[str, ...],
    allowed_sample_ids: set[str],
) -> list[LegacyRecord]:
    return [
        dataset.samples[sample_id]
        for sample_id in sample_ids
        if sample_id in allowed_sample_ids
    ]


def _save_metrics(
    tune_evaluation: Evaluation,
    test_evaluations: dict[str, Evaluation],
    path: Path,
) -> None:
    data: dict[str, dict[str, float]] = {"tune": tune_evaluation.metrics}
    for split_name, split_evaluation in test_evaluations.items():
        data[split_name] = split_evaluation.metrics
    atomic_write_json(path, data)


#: ``History`` columns that are not validation-summary keys.
_HISTORY_LOOP_COLUMNS = ("epoch", "train_loss", "lr", "elapsed_seconds")


def history_rows(history: pd.DataFrame) -> list[dict[str, object]]:
    """The ``History`` frame as JSON-ready rows: ``epoch`` an int, every other column a
    float, and ``None`` where an epoch was not evaluated."""
    rows: list[dict[str, object]] = []
    for record in history.to_dict("records"):
        row: dict[str, object] = {}
        for key, value in record.items():
            if key == "epoch":
                row[key] = int(value)
            elif value is None or (isinstance(value, float) and math.isnan(value)):
                row[key] = None
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def peak_per_metric(history: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Per-metric best value across all epochs, with the (1-based) epoch it occurred at.

    DIAGNOSTIC ONLY. This describes a model that never existed at any single epoch (each
    metric peaks at its own epoch), so it is positively biased and must never feed
    checkpoint selection, ``summary.json`` fold aggregates, or any results table. It is
    recorded in ``training_history.json`` purely to judge whether the configured
    ``monitor`` is a good selection proxy. Direction follows
    :meth:`soma.evaluation.metric_registry.higher_is_better`; ``loss`` is excluded.
    """
    peaks: dict[str, dict[str, float | int]] = {}
    for name in history.columns:
        if name in _HISTORY_LOOP_COLUMNS or name == "loss":
            continue
        for epoch, raw in zip(history["epoch"], history[name]):
            value = float(raw)
            if not np.isfinite(value):
                continue
            current = peaks.get(name)
            better = current is None or (
                value > current["value"]
                if metric_registry.higher_is_better(name)
                else value < current["value"]
            )
            if better:
                peaks[name] = {"epoch": int(epoch) + 1, "value": value}
    return peaks


def _save_training_history(history: pd.DataFrame, path: Path) -> None:
    data = {
        "epochs": history_rows(history),
        # Diagnostic only (see peak_per_metric): it never reaches summaries or results
        # tables.
        "peak_per_metric": peak_per_metric(history),
    }
    atomic_write_json(path, data)


def _optimizer_leaf(training: TrainingConfig) -> dict[str, object]:
    """The config's optimizer leaf with ``learning_rate`` / ``weight_decay`` as its params."""
    leaf = training.optimizer if isinstance(training.optimizer, dict) else {"name": training.optimizer}
    params = {"lr": training.learning_rate, "weight_decay": training.weight_decay}
    params.update(leaf.get("params") or {})
    return {**leaf, "params": params}


def _scheduler_leaves(training: TrainingConfig) -> tuple[object, ...]:
    schedulers = training.scheduler if isinstance(training.scheduler, list) else [training.scheduler]
    return tuple(leaf for leaf in schedulers if leaf is not None and leaf != "none")


def _build_learner(model: torch.nn.Module, training: TrainingConfig, *, shape: str) -> Learner:
    """Wrap ``model`` in the configured learner (``training.learner``) or the shape's."""
    cls, params, _ = resolve_leaf(training.learner or {"name": shape}, learner_registry)
    return cls(
        model,
        optimizer=_optimizer_leaf(training),
        schedulers=_scheduler_leaves(training),
        **params,
    )


def _fit_fold(
    learner: Learner,
    train_loader,
    tune_loader,
    *,
    training: TrainingConfig,
    fold_dir: Path,
    device: torch.device,
    fold: int,
    num_folds: int,
) -> FitResult:
    """Run the generic Trainer with the callbacks the training config asks for.

    ``Checkpoint`` writes ``best_model.pt`` into ``fold_dir`` and leaves the learner in
    its selected state; ``patience=None`` (required by ``checkpoint_selection='last'``)
    installs no ``EarlyStopping``.
    """
    callbacks: list[object] = [
        Checkpoint(
            monitor=training.monitor,
            mode=training.monitor_mode,
            selection=training.checkpoint_selection,
            dirpath=fold_dir,
        )
    ]
    if training.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=training.monitor,
                mode=training.monitor_mode,
                patience=training.patience,
                hold_until_moves=training.hold_patience_until_monitor_moves,
            )
        )
    callbacks.append(Progress(label=f"fold {fold + 1}/{num_folds}" if num_folds > 1 else None))
    trainer = Trainer(
        max_epochs=training.epochs,
        max_steps=training.max_steps,
        accumulate_grad_batches=training.gradient_accumulation,
        val_every_n_epochs=training.tune_every_n_epochs,
        device=device,
        callbacks=callbacks,
    )
    return trainer.fit(learner, train_loader, tune_loader)


def _summarize_fold_metric_dicts(
    per_fold: list[dict[str, dict[str, float]]], *, include_tune: bool
) -> dict[str, float]:
    """Core aggregation over per-fold ``{split_name: {metric: value}}`` dicts.

    For a single fold, emit values directly; for multiple folds, emit mean and std.
    Normally the summary covers the test splits only. ``include_tune`` (set for
    ``evaluation.holdout_test`` runs, which have no test splits) additionally emits
    the tune metrics under a ``tune/`` prefix so a tune-only selection sweep produces
    a non-empty, rankable summary.

    Shared by the in-memory and on-disk aggregators so a run assembled from mixed
    this-session + previously-completed folds (a resume, issue #244) summarizes
    identically to a single-shot run."""
    if not per_fold:
        return {}

    single_fold = len(per_fold) == 1
    summary: dict[str, float] = {}
    # Map each reported split name to its per-fold metric dicts. Tune leads (when
    # requested) so it heads the summary; test splits follow in sorted order.
    metrics_by_split: dict[str, list[dict[str, float]]] = {}
    if include_tune:
        metrics_by_split["tune"] = [f["tune"] for f in per_fold if "tune" in f]
    for split_name in sorted({s for f in per_fold for s in f if s != "tune"}):
        metrics_by_split[split_name] = [f[split_name] for f in per_fold if split_name in f]

    for split_name, split_metrics in metrics_by_split.items():
        if not split_metrics:
            continue
        metric_keys = list(split_metrics[0].keys())
        for key in metric_keys:
            values = [m[key] for m in split_metrics if key in m]
            if not values:
                continue
            if single_fold:
                summary[f"{split_name}/{key}"] = float(values[0])
            else:
                stats = summarize_values(values)
                summary[f"{split_name}/{key}_mean"] = stats.mean
                summary[f"{split_name}/{key}_std"] = stats.std
                if stats.n_nan:
                    # Folds whose split could not support the metric (nan) are
                    # excluded from mean/std; surface how many so the reader knows.
                    summary[f"{split_name}/{key}_nan_folds"] = stats.n_nan

    return summary


def _fold_result_to_metric_dict(fr: FoldResult) -> dict[str, dict[str, float]]:
    """Project a FoldResult onto the ``{split: metrics}`` shape ``metrics.json`` holds."""
    out = {"tune": dict(fr.tune_evaluation.metrics)}
    for split_name, split_evaluation in fr.test_evaluations.items():
        out[split_name] = dict(split_evaluation.metrics)
    return out


def _aggregate_fold_metrics(
    fold_results: list[FoldResult], *, include_tune: bool = False
) -> dict[str, float]:
    """Aggregate in-memory fold results (see :func:`_summarize_fold_metric_dicts`)."""
    return _summarize_fold_metric_dicts(
        [_fold_result_to_metric_dict(fr) for fr in fold_results], include_tune=include_tune
    )


def _aggregate_fold_metrics_from_disk(
    run_dir: Path, *, single_fold: bool, include_tune: bool, num_folds: int
) -> dict[str, float]:
    """Aggregate every fold's persisted ``metrics.json`` under ``run_dir``.

    Reading from disk rather than the in-memory ``fold_results`` is what makes a
    resumed run's ``summary.json`` correct (issue #244): folds skipped this session
    (already complete on disk) re-enter the summary, so a resumed 5-fold summary
    equals the single-shot one. On a fresh run every fold is on disk too, so the
    result is identical to the in-memory path."""
    if single_fold:
        metrics_paths = [run_dir / "metrics.json"]
    else:
        _reject_undeclared_fold_dirs(run_dir, num_folds)
        metrics_paths = [run_dir / f"fold_{i}" / "metrics.json" for i in range(num_folds)]
    per_fold = [
        json.loads(path.read_text()) for path in metrics_paths if path.is_file()
    ]
    return _summarize_fold_metric_dicts(per_fold, include_tune=include_tune)


def _reject_undeclared_fold_dirs(run_dir: Path, num_folds: int) -> None:
    """Refuse a run dir holding ``fold_*`` dirs beyond what the current splits declare.

    Such a dir comes from a previous launch with a different split file; folding its
    ``metrics.json`` into the summary would silently mix two protocols."""
    stray = sorted(
        path.name
        for path in run_dir.glob("fold_*")
        if path.is_dir()
        and path.name[len("fold_"):].isdigit()
        and int(path.name[len("fold_"):]) >= num_folds
    )
    if stray:
        raise ValueError(
            f"Run dir {run_dir} holds fold dir(s) {stray} that the current splits do not "
            f"declare ({num_folds} fold(s)). They were written under a different split "
            "file; start a fresh run or remove them before resuming."
        )


def _save_summary(summary: dict[str, float], path: Path) -> None:
    atomic_write_json(path, summary)
