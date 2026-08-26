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
import csv
import functools
import inspect
import math
from collections.abc import Iterable
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

from soma.aggregators.registry import aggregator_registry
from soma.decoders.registry import build_decoder_for_grid
from soma.config import (
    AggregatorConfig,
    DecoderConfig,
    EncoderConfig,
    EvalConfig,
    HeatmapConfig,
    MasksConfig,
    NormalizationConfig,
    PipelineConfig,
    PixelClassifierConfig,
    PreprocessingConfig,
    ProjectionConfig,
    RepresentationConfig,
    SamplingConfig,
    TaskConfig,
    TrainingConfig,
    config_yaml_dict,
    save_config,
    validate_feature_adaptor_compatibility,
)
from soma.dataset import (
    Dataset,
    DetectionManifest,
    FoldSplit,
    SampleRecord,
    SegmentationManifest,
    Splits,
    load_manifest,
)
from soma.dense.live import LiveSegmentationSource
from soma.evaluation.metrics import resolve_metrics
from soma.evaluation.metrics import compute_metrics
from soma.evaluation.dense_artifacts import DenseArtifactWriter
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.extraction import FeatureExtractor, _release_parent_cuda_state
from soma.features import FeatureStore
from soma.encoders.validation import resolve_preprocessing_config
from soma.output_layout import (
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
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator
from soma.tasks.classification import BranchAwareClassificationHead
from soma.tasks.registry import task_registry
from soma.tasks.detection import DetectionHead
from soma.tasks.segmentation import SegmentationHead
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.feature_adaptor import (
    FeatureAdaptor,
    build_feature_adaptor,
    feature_adaptor_output_dim,
    write_feature_adapter_sidecar,
)
from soma.training.collate import bag_collate_fn, cox_window_collate, hierarchical_bag_collate_fn
from soma.training.model import (
    EmbeddingModel,
    LiveSegmentationModel,
    MILModel,
    SegmentationModel,
)
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.sample_dataset import SampleDataset, SampleBatch, sample_collate_fn
from soma.training.detection_dataset import DetectionDataset, detection_collate_fn
from soma.training.fold_planning import plan_dense_fold
from soma.training.segmentation_dataset import (
    LiveSegmentationDataset,
    SegmentationDataset,
    segmentation_collate_fn,
)
from soma.training.seed import seed_everything
from soma.training.trainer import (
    Trainer,
    TrainResult,
    accumulate_dense_stats,
    epoch_log_to_dict,
    peak_per_metric,
)
from soma.reporting import generate_report_from_result
from soma.reporting.subgroups import subgroup_data_for_predictions, subgroup_report_for_predictions


logger = logging.getLogger(__name__)


def _log_cuda_memory(tag: str) -> None:
    """Print the torch CUDA allocator state for memory diagnostics.

    nvidia-smi's per-process "used" = CUDA context + cuDNN/cuBLAS kernel images +
    this ``reserved`` figure. Printing it right after the encoder is released
    exposes how much of the working set is fixed framework/context overhead (the
    gap between nvidia-smi "used" and ``reserved``) versus live tensors. ``print``,
    not ``logger`` — so it always shows without opting into a log level, matching
    the dense-mode announce in dense_extraction.
    """
    if not torch.cuda.is_available():
        return
    gib = 1024**3
    print(
        f"CUDA memory [{tag}]: "
        f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB  "
        f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB  "
        f"max_reserved={torch.cuda.max_memory_reserved() / gib:.2f} GiB"
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
    train_result: TrainResult | None
    tune_report: EvaluationReport
    test_reports: dict[str, EvaluationReport]  # split_name → report


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run across all folds."""

    fold_results: list[FoldResult]
    summary: dict[str, float]
    run_dir: Path


@dataclass(frozen=True)
class _FeatureSourceContext:
    """Feature source plus the dataset/splits it trains against."""

    feature_store: object
    dataset: object
    splits: Splits


@dataclass(frozen=True)
class _DeterministicBaseline:
    """Deterministic fallback prediction derived from the training split."""

    task_family: str
    probabilities: list[float] | None = None
    predicted_label: int | None = None
    predicted_value: float | None = None
    raw_score: float | None = None
    risk_score: float | None = None


def _representation_split_ids(
    splits: Splits, representation: RepresentationConfig
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
    feature_store: FeatureStore,
    dataset: Dataset,
    splits: Splits,
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

    feature_store.validate_coverage(selected_ids)
    manifest_rows: list[dict[str, str]] = []
    feature_rows: list[np.ndarray] = []
    feature_dim: int | None = None
    for sample_id in selected_ids:
        record = dataset.samples[sample_id]
        label = _required_representation_text(
            record.label, field_name="label", sample_id=sample_id
        )
        group_id = _required_representation_text(
            record.group_id, field_name="group_id", sample_id=sample_id
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
    if float(result.undefined_frac) != 0.0:
        raise ValueError(
            "CRoMa left selected samples undefined; the cohort lacks the required "
            "same/other-confounder neighbour support for m=5."
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


def _patient_ids_for_records(records: list[SampleRecord]) -> list[str]:
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


def _patient_placeholder_records(records: list[SampleRecord]) -> dict[str, SampleRecord]:
    placeholder_records: dict[str, SampleRecord] = {}
    for record in records:
        if record.patient_id is None or record.patient_id in placeholder_records:
            continue
        placeholder_records[record.patient_id] = record
    return placeholder_records


def _write_dense_source_provenance(run_dir: Path, feature_store: object) -> None:
    provenance = getattr(feature_store, "provenance", None)
    if provenance is None:
        return
    path = run_dir / "dense_source.json"
    path.write_text(json.dumps(provenance.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
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
    feature_store: FeatureStore,
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
    records: list[SampleRecord],
    target_fn,
    *,
    num_classes: int,
    training: TrainingConfig,
    fold: int,
):
    """Build the explicit cached-grid training-batch contract, when requested."""
    if training.roi_batch_sampling is None:
        return None
    if num_classes != 4:
        raise ValueError(
            "Explicit ROI batch sampling currently requires exactly four segmentation "
            f"classes, got {num_classes}."
        )

    from soma.training.segmentation_roi_sampler import SegmentationRoiBatchSampler

    class_pixel_counts = []
    for record in records:
        mask = target_fn(record)["mask"]
        class_pixel_counts.append(
            [int((mask == class_index).sum().item()) for class_index in range(4)]
        )

    draws_per_epoch = training.roi_draws_per_epoch
    if draws_per_epoch is None:
        draws_per_epoch = (len(records) // training.batch_size) * training.batch_size
        if draws_per_epoch == 0:
            raise ValueError(
                "Explicit ROI batch sampling needs at least one whole training batch; "
                f"got {len(records)} ROIs and batch_size={training.batch_size}."
            )
    return SegmentationRoiBatchSampler(
        sample_ids=[record.sample_id for record in records],
        class_pixel_counts=class_pixel_counts,
        batch_size=training.batch_size,
        draws_per_epoch=draws_per_epoch,
        strategy=training.roi_batch_sampling,
        seed=training.seed + fold,
    )


def _make_live_loaders(
    source: "LiveSegmentationSource",
    collate_fn,
    train_records: list[SampleRecord],
    tune_records: list[SampleRecord],
    test_records_by_split: dict[str, list[SampleRecord]],
    training: TrainingConfig,
    *,
    num_classes: int,
    ignore_index: int,
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """Live-path loaders: augmentation on the **train** split only, deterministic eval.

    The augment-on-train asymmetry is why this cannot ride the uniform ``_make_loaders``
    (which builds every split the same way). Tune/test datasets are built with
    augmentation off so evaluation re-encodes deterministically.
    """
    from soma.dense.augment import build_segmentation_augmentation

    loader_kwargs = _loader_kwargs(training)
    train_augment = build_segmentation_augmentation(source.augmentation, ignore_index=ignore_index)

    def _make(records: list[SampleRecord], augment) -> LiveSegmentationDataset:
        return LiveSegmentationDataset(
            records,
            geometry=source.geometry,
            preprocessor=source.preprocessor,
            spacing_um=source.spacing_um,
            backend=source.backend,
            tolerance=source.tolerance,
            num_classes=num_classes,
            ignore_index=ignore_index,
            augment=augment,
        )

    train_loader = DataLoader(
        _make(train_records, train_augment), shuffle=True, collate_fn=collate_fn, **loader_kwargs
    )
    tune_loader = DataLoader(
        _make(tune_records, None), shuffle=False, collate_fn=collate_fn, **loader_kwargs
    )
    test_loaders = {
        split_name: DataLoader(
            _make(records, None), shuffle=False, collate_fn=collate_fn, **loader_kwargs
        )
        for split_name, records in test_records_by_split.items()
    }
    return train_loader, tune_loader, test_loaders


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

    def __init__(self, feature_store: FeatureStore, records) -> None:
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


def _validate_dense_feature_adaptor(
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None,
    *,
    feature_store,
) -> None:
    """Resolve runtime dense-source facts, then apply the shared compatibility rule."""
    from soma.dense.composite import CompositeDenseFeatureStore

    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        feature_mode=(
            "live" if isinstance(feature_store, LiveSegmentationSource) else "cached"
        ),
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
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
    feature_store: FeatureStore,
    dataset: Dataset,
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
        dataset: Dataset with sample records and label_map.
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
        FoldResult with training result + tune/test evaluation reports.
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
        test_records_by_split: dict[str, list[SampleRecord]] = {
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
        raw_train_placeholder_records = _patient_placeholder_records(train_records)
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
            unexpected_missing_ids = sorted(expected_feature_ids - set(feature_store.available_samples))
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
            missing_patient_ids = sorted(split_patient_ids - set(feature_store.available_samples))
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
        unexpected_missing_ids = sorted(expected_feature_ids - set(feature_store.available_samples))
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
    # ``task.params.loss`` (nll | cox) selects the discrete-NLL or continuous-Cox
    # head; it is a routing key, not a head constructor argument, so it is
    # validated here and stripped from the params before instantiation.
    if task.name == "survival":
        from soma.tasks.survival import resolve_survival_head, validate_survival_dataset

        survival_loss = task.params.get("loss", "nll")
        validate_survival_dataset(dataset, dataset_type, loss=survival_loss)
        task_cls = resolve_survival_head(survival_loss)
    task_params = {**task_cls.auto_params(dataset), **task.params, "metrics": evaluation.metrics}
    task_params.pop("loss", None)

    task_family = task_cls.task_family
    resolved_metric_names = resolve_metrics(task_family, evaluation.metrics)

    # Feature adaptor (issues #283, #284, #285). The tile-encoder MIL path and the
    # slide-encoder embedding path both fit one; the patient, tile-dataset and
    # hierarchical paths are separate slices, so a normalization or projection request
    # they cannot honor is refused rather than silently ignored — a silently-dropped
    # transform is a false result.
    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        dataset_type=dataset_type,
        is_hierarchical=feature_store.is_hierarchical,
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
                    int(patient_record_map[pid].metadata["event"]) for pid in train_patient_ids
                ],
                training=training,
                min_events_per_window=head.min_events_per_window,
            )
        model: torch.nn.Module = EmbeddingModel(task_head=head)
    elif dataset_type == "tile" or feature_store.is_slide_level:
        # Single-embedding path: one pre-computed vector per sample, no aggregation.
        # Fit the adaptor BEFORE the head is constructed, from the Support split's
        # embeddings and nothing else — "K means K", and here K rows is literally all
        # there is, which is why the PCA preflight is load-bearing on this path.
        if feature_store.is_slide_level and aggregator is not None:
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
                events=[int(r.metadata["event"]) for r in train_records],
                training=training,
                min_events_per_window=head.min_events_per_window,
            )
        model: torch.nn.Module = EmbeddingModel(
            task_head=head, feature_adaptor=feature_adaptor
        )
    elif feature_store.is_hierarchical:
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
            train_events = [int(r.metadata["event"]) for r in train_records]
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
        if aggregator.name == "clam_mb" and task.name == "binary_classification":
            raise ValueError(
                "clam_mb does not support binary_classification; "
                "use multiclass_classification or a different aggregator."
            )
        elif aggregator.name == "clam_mb" and task.name == "multiclass_classification":
            head = BranchAwareClassificationHead(input_dim=agg.output_dim, **task_params)
        else:
            head = task_cls(input_dim=agg.output_dim, **task_params)
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
            train_events = [int(r.metadata["event"]) for r in train_records]
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

    deterministic_baseline = _build_deterministic_baseline(
        train_records,
        target_fn=target_fn,
        task_family=task_family,
        num_classes=int(getattr(head, "num_classes", 0)),
    )

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        tune_loader=tune_loader,
        config=training,
        fold_dir=fold_dir,
        device=device,
        fold=fold,
        num_folds=num_folds,
    )
    train_result = trainer.fit()

    # Load selected checkpoint and evaluate
    checkpoint = torch.load(
        train_result.checkpoint_path, weights_only=True, map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    save_attention = (
        heatmaps is not None
        and heatmaps.enabled
        and aggregator is not None
        and not feature_store.is_slide_level
        and not feature_store.is_hierarchical
    )

    empty_eval_sample_ids = empty_sample_ids_by_split or {}
    if dataset_type == "patient":
        if fallback_to_train:
            tune_output_sample_ids = tuple(raw_train_patient_ids)
            tune_placeholder_records = raw_train_placeholder_records
            tune_empty_sample_ids = empty_eval_sample_ids.get("train", [])
        else:
            tune_output_sample_ids = tuple(raw_tune_patient_ids)
            tune_placeholder_records = _patient_placeholder_records(tune_records)
            tune_empty_sample_ids = empty_eval_sample_ids.get("tune", [])
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
        tune_placeholder_records = None
    tune_report = _evaluate_split_with_placeholders(
        model,
        tune_loader,
        "tune",
        device,
        output_sample_ids=tune_output_sample_ids,
        empty_sample_ids=tune_empty_sample_ids,
        dataset=dataset,
        target_fn=target_fn,
        baseline=deterministic_baseline,
        metric_names=resolved_metric_names,
        task_family=task_family,
        placeholder_records_by_id=tune_placeholder_records,
    )

    test_reports: dict[str, EvaluationReport] = {}
    for split_name, test_loader in test_loaders.items():
        attention_dir: Path | None = None
        if save_attention:
            attention_dir = fold_dir / "attention" / split_name
            attention_dir.mkdir(parents=True, exist_ok=True)
        test_reports[split_name] = _evaluate_split_with_placeholders(
            model,
            test_loader,
            split_name,
            device,
            output_sample_ids=(
                tuple(raw_test_patient_ids_by_split[split_name])
                if dataset_type == "patient"
                else fold_split.tests[split_name]
            ),
            empty_sample_ids=empty_eval_sample_ids.get(split_name, []),
            dataset=dataset,
            target_fn=target_fn,
            baseline=deterministic_baseline,
            metric_names=resolved_metric_names,
            task_family=task_family,
            attention_dir=attention_dir,
            aggregator_name=aggregator.name if aggregator is not None else None,
            placeholder_records_by_id=(
                _patient_placeholder_records(test_records_by_split[split_name])
                if dataset_type == "patient"
                else None
            ),
        )

    # Save metrics, predictions, and training history
    _save_metrics(tune_report, test_reports, fold_dir / "metrics.json")
    _save_training_history(train_result.history, fold_dir / "training_history.json")

    task_family = task.name
    resolved_metrics = resolve_metrics(task_family, evaluation.metrics)
    for split_name, test_report in test_reports.items():
        predictions_path = fold_dir / f"predictions_{split_name}.csv"
        subgroup_data = subgroup_data_for_predictions(
            dataset,
            test_report.predictions,
            evaluation.subgroups.columns,
        )
        _save_predictions(test_report, predictions_path, subgroup_data=subgroup_data)

        # Save subgroup metrics when subgroup columns are configured
        if evaluation.subgroups.columns:
            predictions_df = _build_predictions_df(test_report, subgroup_data)
            sg_out = subgroup_report_for_predictions(
                task_family=task_family,
                metrics=resolved_metrics,
                predictions_df=predictions_df,
                subgroup_columns=evaluation.subgroups.columns,
            )
            (fold_dir / f"subgroup_metrics_{split_name}.json").write_text(json.dumps(sg_out, indent=2))

    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_report=tune_report,
        test_reports=test_reports,
    )


def _dense_spacings_match(a: float | None, b: float | None, *, tol: float = 1e-9) -> bool:
    """True when two read-spacings agree: both flat (``None``) or equal within ``tol``."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _segmentation_label_remap(masks: "MasksConfig | None", num_classes: int, ignore_index: int):
    """Derive the raw-pixel → class-index LUT for slide-manifest masks (None otherwise).

    Slide-manifest annotation rasters carry the dataset's own pixel vocabulary, so soma
    remaps them to contiguous class indices (+ ignore) from ``masks.pixel_mapping``. The
    pre-cropped-tile / flat-mask path has no ``masks:`` block and stays remap-free. Fails
    loud if the mapping's class count disagrees with ``task.num_classes`` (see
    :func:`soma.dense.reader.build_label_remap` for how ``background`` — when present —
    selects the ignore-label mode).
    """
    if masks is None:
        return None
    from soma.dense.reader import build_label_remap

    lut, _ = build_label_remap(
        masks.pixel_mapping, num_classes=num_classes, ignore_index=ignore_index
    )
    return lut


def train_one_segmentation_fold(
    feature_store: "DenseFeatureSource | LiveSegmentationSource",
    dataset: SegmentationManifest,
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    decoder: DecoderConfig | None,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    masks: "MasksConfig | None" = None,
    fold: int = 0,
    num_folds: int = 1,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
) -> FoldResult:
    """Train and evaluate a single dense-segmentation fold.

    Separate from :func:`train_one_fold` because the scalar path's manifest-status
    filtering, deterministic baseline, placeholder predictions, ``SamplePrediction``
    and CSV machinery are all scalar-shaped and do not apply to dense rasters, and
    because the model is ``decoder + SegmentationHead`` (not aggregator/embedding +
    head). The split→records selection and ``tune_is_test``/``allow_missing_tune``
    semantics are reused.

    Two data planes share this body (design §13.B-3), distinguished by
    ``feature_store``: a :class:`~soma.dense.DenseFeatureSource` drives the **cached**
    path (read pre-extracted grids + head-loaded masks with explicit provenance), a
    :class:`~soma.dense.live.LiveSegmentationSource` drives the **live** path
    (re-encode augmented image+mask tiles through the frozen encoder each step). Only
    five things differ — geometry source, feature_dim, coverage check, dataset, and
    model — and they are handled with inline branches here; everything else (records,
    num_classes, decoder/head build, trainer, eval, summary) is shared.

    Args:
        normalization: Feature-adaptor normalization (issue #286). When active, a
            :class:`~soma.training.feature_adaptor.FeatureAdaptor` is fit **channel-axis**
            over all positions in this fold's Support ROIs and inserted ahead of the
            decoder. **Cached only** — a request on the live path is refused, because the
            augmented live stream must never receive an unfit or mis-fit transform.
        projection: Feature-adaptor label-free projection, applied after
            ``normalization``. The frozen ``d -> target_dim`` map composes ahead of the
            decoder's own learnable 1x1 projection conv, so the decoder is simply built
            against ``target_dim`` and its body is unchanged.
        encoder_identity: Name of the encoder behind the grids; seeds the ``random``
            projection so two encoders never share a matrix.
    """
    if decoder is None:
        raise ValueError("dataset_type='segmentation' requires a decoder configuration")

    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

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

    is_live = isinstance(feature_store, LiveSegmentationSource)
    _validate_dense_feature_adaptor(normalization, projection, feature_store=feature_store)

    # num_classes is the single source — from task.params (no dataset auto-inject,
    # since segmentation has no scalar labels). Fed to BOTH decoder and head.
    seg_params = dict(task.params)
    num_classes = seg_params.pop("num_classes", None)
    if num_classes is None:
        raise ValueError(
            "dataset_type='segmentation' requires task.params.num_classes "
            "(the number of segmentation classes)."
        )
    num_classes = int(num_classes)

    # Geometry + feature_dim source (the first inline live/cached fork). Live computes
    # a single geometry from patch_size + target_size (no per-sample sidecar), so the
    # cohort is uniform by construction; cached reads the sidecar and asserts uniformity.
    if is_live:
        for record in all_records:
            if record.image_path is None or record.label_mask_path is None:
                raise ValueError(
                    f"live segmentation sample '{record.sample_id}' needs both image_path "
                    "and label_mask_path (the live path re-encodes from the raw tiles)."
                )
        geometry = feature_store.geometry
        ref_feature_dim = feature_store.feature_dim
    else:
        feature_store.validate_coverage([r.sample_id for r in all_records])
        # The head crop + decoder are built from one reference sample, which is only
        # correct if the run is uniform (fixed tile/grid size — the v1 assumption);
        # assert it loudly rather than silently misregister logits.
        ref_id = train_records[0].sample_id
        geometry = feature_store.geometry(ref_id)
        ref_feature_dim = feature_store.feature_dim
        for record in all_records:
            sid = record.sample_id
            if feature_store.geometry(sid) != geometry or int(feature_store.metadata(sid)["feature_dim"]) != ref_feature_dim:
                raise ValueError(
                    f"dense grid '{sid}' has geometry/feature_dim differing from reference "
                    f"'{ref_id}'; dataset_type='segmentation' v1 requires a uniform tile/grid "
                    "size across the cohort."
                )
    # Masks are read spacing-aware (hs2p) at the same µm/px the dense grids were
    # extracted at, so the target registers against the features. None ⇒ flat read.
    mask_spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    # Cached path: the grids were extracted at a fixed spacing recorded in the sidecar.
    # Reading masks at a different spacing would silently shift/scale the supervision
    # against the features — fail loud. (Live reads image+mask at one spacing each step,
    # so it registers by construction and needs no cross-check.)
    if not is_live:
        grid_spacing_um = feature_store.spacing_um(ref_id)
        if not _dense_spacings_match(grid_spacing_um, mask_spacing_um):
            raise ValueError(
                f"segmentation mask read-spacing ({mask_spacing_um} µm/px) does not match the "
                f"spacing the cached dense grids were extracted at ({grid_spacing_um} µm/px); "
                "the mask would misregister against the features. Re-extract the grids at the "
                "mask spacing, or set preprocessing.requested_spacing_um to match the grids."
            )
    head = SegmentationHead(
        num_classes=num_classes,
        geometry=geometry,
        metrics=evaluation.metrics,
        spacing_um=float(mask_spacing_um) if mask_spacing_um is not None else None,
        backend=preprocessing.backend if preprocessing is not None else "auto",
        tolerance=float(preprocessing.tolerance) if preprocessing is not None else 0.05,
        label_remap=_segmentation_label_remap(
            masks, num_classes, int(seg_params.get("ignore_index", 255))
        ),
        **seg_params,
    )
    target_fn = head.extract_targets

    # Feature adaptor (issue #286), fit BEFORE the decoder is constructed from the Support
    # ROIs' grid positions and nothing else — "K means K". Cached only (asserted above).
    feature_adaptor = (
        None
        if is_live
        else _fit_feature_adaptor(
            normalization,
            projection,
            support_features=_SupportGrids(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=ref_feature_dim,
            encoder_identity=encoder_identity,
            fold_dir=fold_dir,
        )
    )
    # The dim rewire: with a projection active the frozen `d -> target_dim` map composes
    # *ahead of* the decoder's own learnable 1x1 projection conv, so the decoder is simply
    # built against `target_dim`. Its body is untouched — and since that 1x1 is the
    # decoder's only d-dependent module, the whole decoder becomes encoder-dim-independent.
    decoder_input_dim = feature_adaptor_output_dim(
        feature_adaptor, num_features=ref_feature_dim
    )

    decoder_obj = build_decoder_for_grid(
        decoder.name,
        decoder.params,
        geometry=geometry,
        input_dim=decoder_input_dim,
        num_classes=num_classes,
    )
    if decoder_obj.num_classes != head.num_classes:
        raise ValueError(
            f"decoder num_classes ({decoder_obj.num_classes}) != head num_classes "
            f"({head.num_classes}) — a mismatch would misregister the logits."
        )

    seg_collate = functools.partial(segmentation_collate_fn, target_dtypes=head.target_dtypes)
    roi_batch_sampler = None
    if training.roi_batch_sampling is not None:
        if is_live:
            raise ValueError(
                "Explicit ROI batch sampling is a cached-grid training contract; "
                "feature_mode='live' is unsupported."
            )
        roi_batch_sampler = _segmentation_roi_batch_sampler(
            train_records,
            target_fn,
            num_classes=num_classes,
            training=training,
            fold=fold,
        )
    # Model + loaders (the remaining live/cached fork). Live wraps the shared frozen
    # encoder so each step re-encodes the augmented tiles; cached consumes pre-extracted
    # grids. The trainer, eval, and checkpoint reload paths below are identical.
    if is_live:
        model = LiveSegmentationModel(
            kit=feature_store.kit,
            decoder=decoder_obj,
            task_head=head,
        )
        train_loader, tune_loader, test_loaders = _make_live_loaders(
            feature_store, seg_collate,
            train_records, tune_records, test_records_by_split,
            training,
            num_classes=num_classes,
            ignore_index=head.ignore_index,
        )
    else:
        model = SegmentationModel(
            decoder=decoder_obj, task_head=head, feature_adaptor=feature_adaptor
        )
        train_loader, tune_loader, test_loaders = _make_loaders(
            SegmentationDataset, seg_collate,
            train_records, tune_records, test_records_by_split,
            training, feature_store, target_fn,
            train_batch_sampler=roi_batch_sampler,
        )

    summary = _format_fold_summary(
        fold=fold,
        train_count=len(train_records),
        tune_count=len(tune_records),
        tests_counts={name: len(recs) for name, recs in test_records_by_split.items()},
        empty_sample_ids_by_split=None,
    )
    logger.info(summary)

    # Live: the trainer must move inputs to the encoder's device (the encoder is not a
    # registered submodule, so model.to() won't relocate it). Cached: standard device.
    device = (
        torch.device(feature_store.device)
        if is_live
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        tune_loader=tune_loader,
        config=training,
        fold_dir=fold_dir,
        device=device,
        fold=fold,
        num_folds=num_folds,
    )
    train_result = trainer.fit()
    if roi_batch_sampler is not None:
        (fold_dir / "roi_batch_sampling.json").write_text(
            json.dumps(roi_batch_sampler.audit(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint = torch.load(train_result.checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tune_report = _evaluate_segmentation(
        model, tune_loader, "tune", device, dataset=dataset, output_dir=fold_dir,
        save_segmentation_overlays=evaluation.save_segmentation_overlays,
        save_segmentation_probabilities=evaluation.save_segmentation_probabilities,
    )
    test_reports = {
        split_name: _evaluate_segmentation(
            model, loader, split_name, device, dataset=dataset, output_dir=fold_dir,
            save_segmentation_overlays=evaluation.save_segmentation_overlays,
            save_segmentation_probabilities=evaluation.save_segmentation_probabilities,
        )
        for split_name, loader in test_loaders.items()
    }

    _save_metrics(tune_report, test_reports, fold_dir / "metrics.json")
    _save_training_history(train_result.history, fold_dir / "training_history.json")

    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_report=tune_report,
        test_reports=test_reports,
    )


def _resolve_detection_px(value_um: float, spacing_um: float | None, name: str) -> float:
    """Resolve a detection distance (δ / σ / NMS), given in **µm**, to target-frame px.

    Detection distances are always configured in µm — physically meaningful and
    spacing-invariant (the same value means the same tolerance regardless of which
    encoder / spacing the run uses, and there is no "px at which level?" ambiguity).
    The grid's µm/px spacing converts them to the target frame the heatmap and matching
    live in. Detection extraction always records a spacing, so this is always resolvable.
    """
    if spacing_um is None:
        raise ValueError(
            f"{name} is in µm but the dense grids carry no spacing; detection requires "
            "preprocessing.requested_spacing_um (recorded in the grid sidecar)."
        )
    if float(value_um) <= 0.0:
        raise ValueError(f"{name} must be > 0 µm, got {value_um}.")
    return float(value_um) / float(spacing_um)


def _resolve_detection_sample_spacings(feature_store, records):
    """Read per-sample extraction provenance and require one effective grid scale."""
    spacing_by_id = {
        str(record.sample_id): feature_store.spacing(str(record.sample_id))
        for record in records
    }
    effective_groups: dict[float, list[str]] = {}
    for sample_id, spacing in spacing_by_id.items():
        effective_groups.setdefault(float(spacing.effective_spacing_um), []).append(
            sample_id
        )
    if len(effective_groups) != 1:
        details = {
            value: sorted(sample_ids)
            for value, sample_ids in sorted(effective_groups.items())
        }
        raise ValueError(
            "Detection requires one effective_spacing_um across the run; "
            f"found {details}."
        )
    return spacing_by_id, next(iter(effective_groups))


def train_one_detection_fold(
    feature_store: "DenseFeatureSource",
    dataset: DetectionManifest,
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    decoder: DecoderConfig | None,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    fold: int = 0,
    num_folds: int = 1,
    checkpoint_path: str | Path | None = None,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
) -> FoldResult:
    """Train and evaluate a single dense-detection fold (heatmap regression, design §6-§7).

    The detection sibling of :func:`train_one_segmentation_fold`: the model is the same
    ``decoder + head`` (here a :class:`DetectionHead` regressing a per-class peak
    heatmap), built on cached dense grids. After training, the per-class score threshold
    is swept on the **tune** split and frozen (design §7) before the tune/test splits are
    scored with class-aware F1@δ. v1 is cached-only.

    When ``checkpoint_path`` is given, the train loop is skipped and the model is loaded
    from it instead — the eval-only path that regenerates a finished run's artifacts
    (overlays/heatmaps/metrics) under the current code without retraining. The threshold
    sweep and scoring are deterministic, so the regenerated metrics reproduce the original
    run's; ``FoldResult.train_result`` is ``None`` (no epoch history, mirroring the
    pixel-classifier path) and no ``training_history.json`` is written.

    ``normalization``/``projection``/``encoder_identity`` carry the feature adaptor (issue
    #286) on exactly the terms :func:`train_one_segmentation_fold` documents — this path is
    cached-only, so the ``live`` question does not arise. Under ``checkpoint_path`` the
    adaptor is rebuilt (unfitted) and the checkpoint's fitted buffers load into it, so an
    eval-only rerun re-applies the transform the run trained under.
    """
    if decoder is None:
        raise ValueError("dataset_type='detection' requires a decoder configuration")

    _validate_dense_feature_adaptor(normalization, projection, feature_store=feature_store)
    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

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

    # num_classes + detection knobs from task.params (no scalar-label auto-inject).
    det_params = dict(task.params)
    num_classes = det_params.pop("num_classes", None)
    if num_classes is None:
        raise ValueError(
            "dataset_type='detection' requires task.params.num_classes (the number of "
            "object classes)."
        )
    num_classes = int(num_classes)

    feature_store.validate_coverage([r.sample_id for r in all_records])
    ref_id = train_records[0].sample_id
    geometry = feature_store.geometry(ref_id)
    ref_feature_dim = feature_store.feature_dim
    for record in all_records:
        sid = record.sample_id
        if feature_store.geometry(sid) != geometry or int(feature_store.metadata(sid)["feature_dim"]) != ref_feature_dim:
            raise ValueError(
                f"dense grid '{sid}' has geometry/feature_dim differing from reference "
                f"'{ref_id}'; dataset_type='detection' v1 requires a uniform tile/grid size."
            )

    sample_spacings, effective_spacing_um = _resolve_detection_sample_spacings(
        feature_store, all_records
    )

    # Matching distance / σ / NMS radius are configured in µm and resolved to
    # target-frame pixels via the grid spacing (µm is spacing-invariant; px would
    # silently change the physical tolerance across encoders / spacings). δ is
    # required; σ defaults to δ/3 and NMS to δ.
    match_distance_um = det_params.pop("match_distance", None)
    if match_distance_um is None:
        raise ValueError(
            "dataset_type='detection' requires task.params.match_distance (the F1 "
            "matching distance δ, in µm)."
        )
    delta_px = _resolve_detection_px(
        float(match_distance_um), effective_spacing_um, "match_distance"
    )
    sigma_um = det_params.pop("sigma", None)
    sigma_px = (
        _resolve_detection_px(float(sigma_um), effective_spacing_um, "sigma")
        if sigma_um is not None
        else delta_px / 3.0
    )
    nms_um = det_params.pop("nms_distance", None)
    nms_px = (
        _resolve_detection_px(float(nms_um), effective_spacing_um, "nms_distance")
        if nms_um is not None
        else delta_px
    )

    head = DetectionHead(
        num_classes=num_classes,
        geometry=geometry,
        delta_px=delta_px,
        sigma_px=sigma_px,
        nms_distance_px=nms_px,
        sample_spacings=sample_spacings,
        metrics=evaluation.metrics,
        **det_params,
    )
    target_fn = head.extract_targets

    # Feature adaptor (issue #286). Training fits it on the Support ROIs' grid positions;
    # an eval-only rerun builds it *unfitted* and lets the checkpoint's buffers below load
    # into it, so the transform re-applied is exactly the one the run trained under (and
    # no QC sidecar is rewritten for a run that did not fit anything).
    if checkpoint_path is not None:
        feature_adaptor = build_feature_adaptor(
            normalization,
            projection,
            num_features=ref_feature_dim,
            encoder_identity=encoder_identity,
        )
    else:
        feature_adaptor = _fit_feature_adaptor(
            normalization,
            projection,
            support_features=_SupportGrids(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=ref_feature_dim,
            encoder_identity=encoder_identity,
            fold_dir=fold_dir,
        )

    # The dim rewire: the frozen projection composes ahead of the decoder's learnable 1x1
    # projection conv, so the decoder is built against `target_dim` and its body is unchanged.
    decoder_obj = build_decoder_for_grid(
        decoder.name,
        decoder.params,
        geometry=geometry,
        input_dim=feature_adaptor_output_dim(feature_adaptor, num_features=ref_feature_dim),
        num_classes=num_classes,
    )
    if decoder_obj.num_classes != head.num_classes:
        raise ValueError(
            f"decoder num_classes ({decoder_obj.num_classes}) != head num_classes "
            f"({head.num_classes}) — a mismatch would misregister the heatmap channels."
        )

    det_collate = functools.partial(detection_collate_fn, target_dtypes=head.target_dtypes)
    model = SegmentationModel(
        decoder=decoder_obj, task_head=head, feature_adaptor=feature_adaptor
    )
    train_loader, tune_loader, test_loaders = _make_loaders(
        DetectionDataset, det_collate,
        train_records, tune_records, test_records_by_split,
        training, feature_store, target_fn,
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
    if checkpoint_path is not None:
        # Eval-only: load a previously trained checkpoint and skip the train loop, so a
        # finished run's artifacts can be regenerated under the current code. No epoch
        # history exists → train_result stays None (the pixel-classifier path's precedent).
        train_result = None
        state = torch.load(checkpoint_path, weights_only=True, map_location=device)
    else:
        trainer = Trainer(
            model=model, train_loader=train_loader, tune_loader=tune_loader,
            config=training, fold_dir=fold_dir, device=device, fold=fold, num_folds=num_folds,
        )
        train_result = trainer.fit()
        state = torch.load(train_result.checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # Freeze the per-class score threshold on the tune split (design §7), then score.
    thresholds = _sweep_detection_thresholds(model, tune_loader, device, head)
    head.score_threshold = thresholds
    (fold_dir / "detection_thresholds.json").write_text(
        json.dumps({"score_threshold_per_class": thresholds}, indent=2), encoding="utf-8"
    )

    tune_report = _evaluate_detection(
        model, tune_loader, "tune", device, head=head, dataset=dataset, output_dir=fold_dir,
        save_detection_overlays=evaluation.save_detection_overlays,
        save_detection_heatmaps=evaluation.save_detection_heatmaps,
    )
    test_reports = {
        split_name: _evaluate_detection(
            model, loader, split_name, device, head=head, dataset=dataset, output_dir=fold_dir,
            save_detection_overlays=evaluation.save_detection_overlays,
            save_detection_heatmaps=evaluation.save_detection_heatmaps,
        )
        for split_name, loader in test_loaders.items()
    }

    _save_metrics(tune_report, test_reports, fold_dir / "metrics.json")
    if train_result is not None:
        _save_training_history(train_result.history, fold_dir / "training_history.json")

    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_report=tune_report,
        test_reports=test_reports,
    )


@torch.inference_mode()
def _sweep_detection_thresholds(
    model: torch.nn.Module,
    tune_loader: DataLoader,
    device: torch.device,
    head: DetectionHead,
) -> list[float]:
    """Collect tune-split predictions (all candidate peaks) and sweep per-class thresholds."""
    from soma.detection.matching import sweep_score_thresholds

    model.eval()
    saved = head.score_threshold
    head.score_threshold = 0.0  # keep every local maximum so the sweep sees all candidates
    pred_xy, pred_cls, pred_score, gt_xy, gt_cls = [], [], [], [], []
    try:
        for batch in tune_loader:
            out = model(batch.features.to(device))
            gt_points = batch.targets["gt_points"]
            for b in range(out.logits.shape[0]):
                xy, cls, score = head._predict_points(out.logits[b])
                gxy, gcls = head._strip_padding(gt_points[b])
                pred_xy.append(xy); pred_cls.append(cls); pred_score.append(score)
                gt_xy.append(gxy); gt_cls.append(gcls)
    finally:
        head.score_threshold = saved
    return sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls,
        num_classes=head.num_classes, delta=head.delta_px, method=head.matching,
    )


@torch.inference_mode()
def _evaluate_detection(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    *,
    head: DetectionHead,
    dataset: DetectionManifest,
    output_dir: Path | None = None,
    save_detection_overlays: bool = True,
    save_detection_heatmaps: bool = False,
) -> EvaluationReport:
    """Consolidated detection eval: decode + match **once** per image, fan out to all.

    Each image is decoded (NMS peaks at the frozen per-class ``head.score_threshold``)
    and class-aware F1@δ matched exactly once; that single result feeds three consumers
    so they cannot drift: the per-class ``(C, 3)`` TP/FP/FN counts → the headline metric
    (``finalize_eval_metrics``); the decoded points → the level-0 ``predictions_<split>.csv``
    (stitch-ready, design §4, byte-identical to the pre-consolidation output); and the
    assignment + heatmap + GT points → the :class:`DetectionArtifactWriter` (plain pred/GT
    overlays + per-image manifest + split-level metrics CSVs, plus opt-in raw heatmap
    overlays + npz when ``save_detection_heatmaps``). ``head.dense_stats`` /
    ``head.postprocess`` are left untouched (the Trainer's tune monitor still uses them).
    """
    from soma.detection.encode import transform_points_to_level0
    from soma.detection.matching import match_assignment
    from soma.evaluation.detection_artifacts import DetectionArtifactWriter

    model.eval()
    top, left, _, _ = head._crop_box
    writer = (
        DetectionArtifactWriter(
            head=head,
            split=split_name,
            output_dir=output_dir,
            dataset=dataset,
            save_detection_overlays=save_detection_overlays,
            save_detection_heatmaps=save_detection_heatmaps,
        )
        if output_dir is not None
        else None
    )
    stat_rows: list[torch.Tensor] = []
    pred_rows: list[str] = ["sample_id,x,y,class,score"]
    for batch in loader:
        out = model(batch.features.to(device))
        gt_points = batch.targets["gt_points"]
        for b, sid in enumerate(batch.sample_ids):
            heatmap = out.logits[b]
            pred_xy, pred_cls, pred_score = head._predict_points(heatmap)
            gt_xy, gt_cls = head._strip_padding(gt_points[b])
            # The one match the headline counts, the per-point CSV, and the overlays
            # all derive from (mirrors match_points' reduction, so counts are identical).
            assignment = match_assignment(
                pred_xy, pred_cls, pred_score, gt_xy, gt_cls,
                num_classes=head.num_classes, delta=head.delta_px, method=head.matching,
            )
            counts = np.zeros((head.num_classes, 3), dtype=np.int64)
            for c, m in enumerate(assignment):
                counts[c] = m.counts
            stat_rows.append(torch.from_numpy(counts).to(torch.long).unsqueeze(0))
            if pred_xy.shape[0]:
                spacing = head.spacing_for_sample(sid)
                xy_l0 = transform_points_to_level0(
                    pred_xy,
                    source_spacing_um=spacing.source_spacing_um,
                    effective_spacing_um=spacing.effective_spacing_um,
                    crop_top=top,
                    crop_left=left,
                )
                for (x, y), c, s in zip(xy_l0, pred_cls, pred_score):
                    pred_rows.append(f"{sid},{x:.3f},{y:.3f},{int(c)},{float(s):.4f}")
            if writer is not None:
                writer.add_image(
                    sample_id=sid, heatmap=heatmap,
                    pred_xy=pred_xy, pred_class=pred_cls, pred_score=pred_score,
                    assignment=assignment, gt_xy=gt_xy, gt_class=gt_cls,
                )

    metrics = head.finalize_eval_metrics(torch.cat(stat_rows, dim=0)) if stat_rows else {}
    if output_dir is not None:
        (Path(output_dir) / f"predictions_{split_name}.csv").write_text(
            "\n".join(pred_rows) + "\n", encoding="utf-8"
        )
    if writer is not None:
        writer.finalize()
    return EvaluationReport(split=split_name, metrics=metrics, predictions=[])


def train_one_pixel_classifier_fold(
    feature_store: "DenseFeatureSource",
    dataset: SegmentationManifest,
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    pixel_classifier: PixelClassifierConfig,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    masks: "MasksConfig | None" = None,
    fold: int = 0,
    num_folds: int = 1,
) -> FoldResult:
    """Train + evaluate one **decoder-free** segmentation fold (design §9).

    The pixel-classifier counterpart of :func:`train_one_segmentation_fold`: no torch
    ``Trainer``, no decoder, no checkpoints. It samples class-stratified pixels from the
    train split into ``(X, y)`` matrices (the tune split supplies ``X_val`` for early
    stopping), fits the swappable :class:`~soma.pixel_classifiers.base.PixelClassifier`
    once, then predicts **every** pixel of every tune/test tile and reuses the shared
    dense metrics + artifact writer verbatim. Cached features only (no live re-encode /
    augmentation — that would require re-encoding augmented images).

    The split→records selection, ``num_classes`` source, geometry/uniformity assertion,
    and mask-spacing cross-check are identical to the decoder fold; only the model and
    its (Trainer-free) fit/eval loop differ.
    """
    import numpy as np

    from soma.pixel_classifiers import pixel_classifier_registry
    from soma.pixel_classifiers.segmentation import (
        build_training_matrix,
        evaluate_pixel_classifier,
        inverse_frequency_sample_weight,
    )

    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

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
    feature_store.validate_coverage([r.sample_id for r in all_records])

    seg_params = dict(task.params)
    num_classes = seg_params.pop("num_classes", None)
    if num_classes is None:
        raise ValueError(
            "dataset_type='segmentation' requires task.params.num_classes "
            "(the number of segmentation classes)."
        )
    num_classes = int(num_classes)

    # Geometry + feature_dim from one reference sample; assert cohort uniformity.
    ref_id = train_records[0].sample_id
    geometry = feature_store.geometry(ref_id)
    ref_feature_dim = feature_store.feature_dim
    for record in all_records:
        sid = record.sample_id
        if feature_store.geometry(sid) != geometry or int(feature_store.metadata(sid)["feature_dim"]) != ref_feature_dim:
            raise ValueError(
                f"dense grid '{sid}' has geometry/feature_dim differing from reference "
                f"'{ref_id}'; dataset_type='segmentation' v1 requires a uniform tile/grid "
                "size across the cohort."
            )
    # Masks read at the same µm/px the grids were extracted at (else misregistration).
    mask_spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    grid_spacing_um = feature_store.spacing_um(ref_id)
    if not _dense_spacings_match(grid_spacing_um, mask_spacing_um):
        raise ValueError(
            f"segmentation mask read-spacing ({mask_spacing_um} µm/px) does not match the "
            f"spacing the cached dense grids were extracted at ({grid_spacing_um} µm/px); "
            "the mask would misregister against the features. Re-extract the grids at the "
            "mask spacing, or set preprocessing.requested_spacing_um to match the grids."
        )

    head = SegmentationHead(
        num_classes=num_classes,
        geometry=geometry,
        metrics=evaluation.metrics,
        spacing_um=float(mask_spacing_um) if mask_spacing_um is not None else None,
        backend=preprocessing.backend if preprocessing is not None else "auto",
        tolerance=float(preprocessing.tolerance) if preprocessing is not None else 0.05,
        label_remap=_segmentation_label_remap(
            masks, num_classes, int(seg_params.get("ignore_index", 255))
        ),
        **seg_params,
    )

    # Build the classifier (num_classes injected like a decoder). ``class_balanced_weights``
    # is a fold-level knob (inverse-frequency per-pixel sample weights), not a classifier
    # hyperparameter, so pop it before constructing.
    clf_params = dict(pixel_classifier.params)
    class_balanced_weights = bool(clf_params.pop("class_balanced_weights", False))
    clf = pixel_classifier_registry.get(pixel_classifier.name)(
        num_classes=num_classes, **clf_params
    )

    summary = _format_fold_summary(
        fold=fold,
        train_count=len(train_records),
        tune_count=len(tune_records),
        tests_counts={name: len(recs) for name, recs in test_records_by_split.items()},
        empty_sample_ids_by_split=None,
    )
    logger.info(summary)

    # Sampled pixel matrices: train fits, tune supplies early-stopping validation. The
    # tune budget is a fraction of train (it only guides stopping, not the fit).
    rng = np.random.default_rng(training.seed + fold)
    X_train, y_train = build_training_matrix(
        train_records, feature_store, head, max_pixels=training.max_train_pixels, rng=rng
    )
    val_budget = max(num_classes, training.max_train_pixels // 5)
    X_val, y_val = build_training_matrix(
        tune_records, feature_store, head, max_pixels=val_budget, rng=rng
    )
    sample_weight = (
        inverse_frequency_sample_weight(y_train, num_classes) if class_balanced_weights else None
    )
    logger.info(
        "%s: fitting '%s' on %d sampled train pixels (%d val), K=%d, num_classes=%d",
        _fp, pixel_classifier.name, len(y_train), len(y_val), ref_feature_dim, num_classes,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight)
    clf.save(fold_dir / "pixel_classifier")

    tune_report = evaluate_pixel_classifier(
        clf, tune_records, feature_store, head, "tune",
        dataset=dataset, output_dir=fold_dir,
        save_segmentation_overlays=evaluation.save_segmentation_overlays,
        save_segmentation_probabilities=evaluation.save_segmentation_probabilities,
    )
    test_reports = {
        split_name: evaluate_pixel_classifier(
            clf, records, feature_store, head, split_name,
            dataset=dataset, output_dir=fold_dir,
            save_segmentation_overlays=evaluation.save_segmentation_overlays,
            save_segmentation_probabilities=evaluation.save_segmentation_probabilities,
        )
        for split_name, records in test_records_by_split.items()
    }
    _save_metrics(tune_report, test_reports, fold_dir / "metrics.json")

    return FoldResult(
        fold=fold,
        train_result=None,
        tune_report=tune_report,
        test_reports=test_reports,
    )


def _probe_gene_metric_key(index: int, gene: str) -> str:
    """Stable, order-preserving per-gene Pearson metric key (``pearson_gene00_<sym>``)."""
    return f"pearson_gene{index:02d}_{gene}"


def _write_probe_predictions(
    path: Path, sample_ids: list[str], predictions: "np.ndarray", genes: list[str]
) -> None:
    """Write per-spot predicted gene vectors as ``sample_id`` + one column per gene."""
    frame = pd.DataFrame(predictions, columns=list(genes))
    frame.insert(0, "sample_id", sample_ids)
    frame.to_csv(path, index=False)


def train_one_probe_fold(
    feature_store: FeatureStore,
    dataset: "SpatialExpressionManifest",
    fold_split: FoldSplit,
    task: TaskConfig,
    training: TrainingConfig,
    fold_dir: str | Path,
    *,
    evaluation: EvalConfig | None = None,
    fold: int = 0,
    num_folds: int = 1,
) -> FoldResult:
    """Train + evaluate one **closed-form probe** fold (HEST spatial_expression, design §6).

    The ``regression`` family's non-gradient sibling: no torch ``Trainer``, no head, no
    tune split. It pulls the fold's cached per-spot embeddings, fits StandardScaler → PCA →
    multi-output Ridge on the train spots (see :func:`soma.training.probe.score_probe_fold`),
    predicts each test split's spots, and scores every gene by Pearson correlation pooled
    over that split's spots. The per-fold ``mean_pearson`` (mean over genes) plus the
    per-gene detail land in ``metrics.json``; the shared summary writer then averages over
    folds into the headline. Cached features only — embeddings are extracted once and
    reused across folds by the shared feature cache.
    """
    from soma.training.probe import DEFAULT_PCA_COMPONENTS, score_probe_fold

    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

    train_ids = [str(sid) for sid in fold_split.train]
    if not train_ids:
        raise ValueError(f"{_fp}: the closed-form probe needs a non-empty train split.")
    # HEST folds carry no tune split; the closed-form probe needs none (fixed Ridge alpha,
    # no early stopping) so tune is ignored entirely.
    test_ids_by_split = {
        name: [str(sid) for sid in ids] for name, ids in fold_split.tests.items()
    }
    all_ids = train_ids + [sid for ids in test_ids_by_split.values() for sid in ids]
    feature_store.validate_coverage(all_ids)

    genes = list(dataset.genes)
    n_genes = len(genes)
    pca_components = int(task.params.get("pca_components", DEFAULT_PCA_COMPONENTS))

    def _stack_features(ids: list[str]) -> np.ndarray:
        return np.stack(
            [feature_store.load(sid).reshape(-1).to(torch.float64).numpy() for sid in ids]
        )

    def _stack_targets(ids: list[str]) -> np.ndarray:
        return np.stack(
            [np.asarray(dataset.samples[sid].target, dtype=np.float64) for sid in ids]
        )

    x_train = _stack_features(train_ids)
    y_train = _stack_targets(train_ids)
    feature_dim = x_train.shape[1]
    max_components = min(len(train_ids), feature_dim)
    if pca_components > max_components:
        raise ValueError(
            f"{_fp}: PCA n_components={pca_components} exceeds the fold's rank "
            f"min(n_train={len(train_ids)}, feature_dim={feature_dim})={max_components}. "
            "Lower task.params.pca_components or provide more train spots."
        )

    logger.info(
        "%s: closed-form Ridge+PCA probe — %d train spots, %d genes, PCA=%d, feature_dim=%d",
        _fp, len(train_ids), n_genes, pca_components, feature_dim,
    )

    test_reports: dict[str, EvaluationReport] = {}
    for split_name, ids in test_ids_by_split.items():
        if not ids:
            test_reports[split_name] = EvaluationReport(
                split=split_name, metrics={}, predictions=[]
            )
            continue
        x_test = _stack_features(ids)
        y_test = _stack_targets(ids)
        score = score_probe_fold(
            x_train, y_train, x_test, y_test,
            pca_components=pca_components, seed=training.seed,
        )
        # Headline (mean over genes) + per-gene Pearson detail, all flat floats so the
        # shared summary writer averages each over folds.
        metrics = {"mean_pearson": float(score.mean_pearson)}
        for index, gene in enumerate(genes):
            metrics[_probe_gene_metric_key(index, gene)] = float(score.per_gene_pearson[index])
        test_reports[split_name] = EvaluationReport(
            split=split_name, metrics=metrics, predictions=[]
        )
        _write_probe_predictions(
            fold_dir / f"predictions_{split_name}.csv", ids, score.predictions, genes
        )
        logger.info(
            "%s [%s]: mean Pearson = %.4f over %d genes (%d spots)",
            _fp, split_name, score.mean_pearson, n_genes, len(ids),
        )

    # The probe has no tune split; report an empty tune (ignored by the summary writer).
    tune_report = EvaluationReport(split="tune", metrics={}, predictions=[])
    _save_metrics(tune_report, test_reports, fold_dir / "metrics.json")

    return FoldResult(
        fold=fold,
        train_result=None,
        tune_report=tune_report,
        test_reports=test_reports,
    )


def train(
    feature_store: FeatureStore,
    dataset: Dataset,
    splits: Splits,
    task: TaskConfig,
    training: TrainingConfig,
    run_dir: str | Path,
    aggregator: AggregatorConfig | None = None,
    decoder: DecoderConfig | None = None,
    pixel_classifier: PixelClassifierConfig | None = None,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    masks: "MasksConfig | None" = None,
    heatmaps: HeatmapConfig | None = None,
    normalization: NormalizationConfig | None = None,
    projection: ProjectionConfig | None = None,
    encoder_identity: str = "",
    dataset_type: str = "slide",
    test_digest: str | None = None,
    overwrite_test: bool = False,
    run_id: str | None = None,
) -> PipelineResult:
    """Train and evaluate all folds, then summarize.

    Args:
        feature_store: Precomputed embeddings.
        dataset: Dataset with sample records and label_map.
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

    Returns:
        PipelineResult with per-fold results and aggregated summary.
    """
    # Feature-adaptor path guard (issues #283-#286). The dense fold dispatchers below never
    # reach train_one_fold's own guard, so the streams *this* entrypoint owns are checked
    # here — a requested transform a path cannot honor must fail, never be silently dropped.
    #
    # Covered now: the single-encoder dense streams (`segmentation` / `detection` with a
    # decoder over one encoder's cached grids), alongside the tile-encoder MIL and
    # slide-encoder embedding paths in train_one_fold. Still refused:
    #
    #   * `spatial_expression` — one spot is a single embedding scored by the closed-form
    #     Ridge+PCA probe, not a dense grid, and the probe carries no adaptor.
    #   * the decoder-free `pixel_classifier` segmentation path — no torch model, so no
    #     buffers and no checkpoint to carry them.
    #   * composite (multi-encoder) dense streams — the top-level blocks are scoped to
    #     single-encoder streams; composites keep their per-member ``member_norm``, and
    #     fitting one z-score over a channel-concatenated grid would silently treat several
    #     encoders as one.
    from soma.dense.composite import CompositeDenseFeatureStore

    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        dataset_type=dataset_type,
        has_pixel_classifier=pixel_classifier is not None,
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluation or EvalConfig()

    single_fold = splits.num_folds == 1

    def _fold_dir(fold_idx: int) -> Path:
        # A single fold owns run_dir directly; multi-fold folds get fold_N/ subdirs.
        return run_dir if single_fold else run_dir / f"fold_{fold_idx}"

    # A fold with a metrics.json is complete; any without one is pending. A run with
    # pending folds is a resume in progress (issue #244).
    pending_folds = [
        i for i in range(splits.num_folds) if not (_fold_dir(i) / "metrics.json").exists()
    ]

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

    if dataset_type == "patient" or dataset.has_patient_ids:
        splits.validate_no_patient_leakage(dataset)

    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        fold_dir = _fold_dir(fold_idx)
        # Resume fold-skip guard (issue #244): a fold that already wrote metrics.json is
        # complete, so on a relaunch into the same run dir we skip retraining it. The
        # summary is rebuilt from every fold's metrics.json on disk below, so a skipped
        # fold still counts. Gated to multi-fold: a single fold owns run_dir itself and
        # has no partial-run to preserve.
        if not single_fold and (fold_dir / "metrics.json").exists():
            logger.info(
                "Fold %d already complete (found %s); skipping (resume).",
                fold_idx,
                fold_dir / "metrics.json",
            )
            continue
        if training.method == "ridge_pca_probe":
            # Closed-form Ridge+PCA probe (HEST spatial_expression): a sibling per-fold
            # trainer under the regression family, selected by the method flag rather than
            # forking the entrypoint. Reuses this fold loop, resumable CV, the feature
            # cache, and the summary writer.
            result = train_one_probe_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                task=task,
                training=training,
                evaluation=evaluation,
                fold_dir=fold_dir,
                fold=fold_idx,
                num_folds=splits.num_folds,
            )
        elif dataset_type == "segmentation" and pixel_classifier is not None:
            # Decoder-free path: the cross-defaulted feature_kind=cls_attention grids
            # feed a per-pixel classifier (no Trainer). Mutually exclusive with decoder,
            # enforced in PipelineConfig.
            result = train_one_pixel_classifier_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                task=task,
                pixel_classifier=pixel_classifier,
                evaluation=evaluation,
                training=training,
                fold_dir=fold_dir,
                preprocessing=preprocessing,
                masks=masks,
                fold=fold_idx,
                num_folds=splits.num_folds,
            )
        elif dataset_type == "segmentation":
            result = train_one_segmentation_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                task=task,
                decoder=decoder,
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
            )
        elif dataset_type == "detection":
            result = train_one_detection_fold(
                feature_store=feature_store,
                dataset=dataset,
                fold_split=fold_split,
                task=task,
                decoder=decoder,
                evaluation=evaluation,
                training=training,
                fold_dir=fold_dir,
                preprocessing=preprocessing,
                fold=fold_idx,
                num_folds=splits.num_folds,
                normalization=normalization,
                projection=projection,
                encoder_identity=encoder_identity,
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
        fold_results.append(result)

    # Aggregate from every fold's metrics.json on disk, not just this session's
    # fold_results — so a resumed run (some folds skipped above) still summarizes all
    # folds. For a fresh run every fold is on disk, so this equals the in-memory path.
    summary = _aggregate_fold_metrics_from_disk(
        run_dir, single_fold=single_fold, include_tune=evaluation.holdout_test
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
    feature_store: FeatureStore,
    dataset_type: str = "slide",
    decoder: DecoderConfig | None = None,
    pixel_classifier: PixelClassifierConfig | None = None,
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

    # Aggregator (MIL path), or decoder / pixel-classifier (segmentation) — the
    # trainable component.
    if dataset_type == "segmentation":
        if pixel_classifier is not None:
            grid.add_row("pixel_classifier", pixel_classifier.name)
            grid.add_row("feature_kind", str(preprocessing.feature_kind))
        else:
            grid.add_row("decoder", decoder.name if decoder is not None else "[dim]—[/dim]")
    elif dataset_type == "detection":
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
        # DenseFeatureStore is not a FeatureStore (no is_slide_level/is_hierarchical);
        # branch before those attrs are touched.
        level = "dense (segmentation)" if dataset_type == "segmentation" else "dense (detection)"
    elif feature_store.is_slide_level:
        level = "slide"
    elif feature_store.is_hierarchical:
        level = "hierarchical"
    else:
        level = "tile"
    grid.add_row("features", f"{level}  ·  dim={feature_store.feature_dim}")

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
    if saved == current:
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
        self._config = config
        # The load-time validator keyed on dataset_type selects the right manifest loader
        # (segmentation -> label_mask_path, detection -> points_path, else -> label); each loader
        # fail-fast validates its supervision column and exposes the samples/sample_ids
        # surface Splits needs.
        self._dataset = load_manifest(config.dataset_csv, config.dataset_type)
        self._splits = Splits(
            config.splits_csv,
            self._dataset,
            tune_is_test=config.training.tune_is_test,
        )
        self._feature_dir = Path(feature_dir) if feature_dir else None

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def splits(self) -> Splits:
        return self._splits

    def run(self) -> PipelineResult:
        """Run the full pipeline: save config → load features → train all folds → summarize."""
        layout = resolve_managed_output_paths(self._config)
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
                return result

            preprocessing = self._resolve_preprocessing()
            Console().print(
                _build_run_summary_panel(
                    encoder=self._config.encoder,
                    preprocessing=preprocessing,
                    aggregator=self._config.aggregator,
                    task=self._config.task,
                    feature_store=store,
                    dataset_type=self._config.dataset_type,
                    decoder=self._config.decoder,
                    pixel_classifier=self._config.pixel_classifier,
                )
            )

            result = train(
                feature_store=store,
                dataset=training_dataset,
                splits=training_splits,
                dataset_type=self._config.dataset_type,
                aggregator=self._config.aggregator,
                decoder=self._config.decoder,
                pixel_classifier=self._config.pixel_classifier,
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
            )

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

            try:
                report_path = generate_report_from_result(
                    result, self._config, dataset=training_dataset
                )
                logger.info("Report saved to %s", report_path)
            except Exception:
                logger.warning("Report generation failed", exc_info=True)

            Console().print(_build_completed_run_panel(summary_metrics=result.summary))
            recorder.complete(result.summary)

        return result

    def _get_feature_source_context(self, *, run_dir: Path) -> _FeatureSourceContext:
        if self._config.dataset_type in ("segmentation", "detection"):
            return self._get_dense_feature_source_context(run_dir=run_dir)
        return _FeatureSourceContext(
            feature_store=self._get_feature_store(run_dir=run_dir),
            dataset=self._dataset,
            splits=self._splits,
        )

    def _get_feature_store(self, *, run_dir: Path):
        if self._config.dataset_type in ("segmentation", "detection"):
            return self._get_dense_feature_source_context(run_dir=run_dir).feature_store
        if self._feature_dir is not None:
            return FeatureStore(self._feature_dir)

        if self._config.dataset_type in ("tile", "spatial_expression"):
            # spatial_expression (HEST) reuses the tile path: each spot is one PNG tile
            # encoded to a 1-D feature vector, cached once and reused across folds.
            if self._config.encoder is None:
                raise ValueError(
                    f"PipelineConfig.encoder is required for "
                    f"dataset_type={self._config.dataset_type!r} when feature_dir is not "
                    "provided."
                )
            from soma.tile_extraction import TileFeatureExtractor

            cache_config = self._config.cache
            if cache_config.root_dir is None:
                cache_config = replace(
                    cache_config,
                    root_dir=Path(self._config.output_root) / "feature_cache",
                )
            extractor = TileFeatureExtractor(
                self._dataset,
                self._config.encoder,
                execution=self._config.execution,
                cache=cache_config,
            )
            try:
                return extractor.run(feature_dir=run_dir / "features")
            finally:
                _release_parent_cuda_state()

        # Slide pipeline path
        if self._config.encoder is None:
            raise ValueError(
                "PipelineConfig.encoder is required when feature_dir is not provided."
            )
        preprocessing = self._resolve_preprocessing()

        cache_config = self._config.cache
        if cache_config.root_dir is None:
            cache_config = replace(
                cache_config,
                root_dir=Path(self._config.output_root) / "feature_cache",
            )
        extractor = FeatureExtractor(
            self._dataset,
            self._config.encoder,
            preprocessing,
            output_root=run_dir,
            execution=self._config.execution,
            cache=cache_config,
        )
        try:
            return extractor.run(feature_dir="features")
        finally:
            _release_parent_cuda_state()

    def _get_dense_feature_source_context(self, *, run_dir: Path) -> _FeatureSourceContext:
        if self._config.preprocessing.masks is not None:
            return self._build_slide_manifest_dense_context(run_dir=run_dir)
        return _FeatureSourceContext(
            feature_store=self._get_dense_source(run_dir=run_dir),
            dataset=self._dataset,
            splits=self._splits,
        )

    def _cache_backed_dense_source(
        self,
        store,
        *,
        kind: str,
        dataset_csv: str | Path,
        splits_csv: str | Path,
        parent_dataset_csv: str | Path | None = None,
        parent_splits_csv: str | Path | None = None,
    ):
        from soma.dense import CacheBackedDenseSource, DenseSourceProvenance

        return CacheBackedDenseSource(
            store,
            provenance=DenseSourceProvenance(
                kind=kind,
                feature_dir=getattr(store, "feature_dir", None),
                dataset_csv=dataset_csv,
                splits_csv=splits_csv,
                parent_dataset_csv=parent_dataset_csv,
                parent_splits_csv=parent_splits_csv,
            ),
        )

    def _get_dense_source(self, *, run_dir: Path):
        from soma.dense import DenseFeatureStore

        # Live re-encode path: no cached grids — hold the frozen encoder + geometry and
        # re-encode (augmented) tiles each step. Built before the fold loop so the
        # backbone loads once and is shared across folds.
        if self._config.feature_mode == "live":
            return self._build_live_segmentation_source()

        if self._feature_dir is not None:
            return self._cache_backed_dense_source(
                DenseFeatureStore(self._feature_dir),
                kind="dense_cache",
                dataset_csv=self._config.dataset_csv,
                splits_csv=self._config.splits_csv,
            )

        # Multi-encoder composite: extract each member into its own cache, then present a
        # load-time channel-concat view (design §7).
        if self._config.composite is not None:
            return self._cache_backed_dense_source(
                self._build_composite_dense_store(run_dir=run_dir),
                kind="composite_dense_cache",
                dataset_csv=self._config.dataset_csv,
                splits_csv=self._config.splits_csv,
            )

        dtype = self._config.dataset_type
        if self._config.encoder is None:
            raise ValueError(
                f"PipelineConfig.encoder is required for dataset_type={dtype!r} "
                "when feature_dir is not provided."
            )
        # Resolve preprocessing so requested_spacing_um defaults to the encoder's
        # supported spacing (without overriding an explicit value); the dense read is
        # spacing-aware (hs2p). requested_tile_size_px stays the supervision size.
        preprocessing = self._resolve_preprocessing()
        # The dense supervision size (tile/mask size) drives the pad-to-patch
        # geometry. Reuse the existing tile-size knob — a segmentation tile is an
        # image of this size — rather than duplicating it on the decoder config.
        target_size = preprocessing.requested_tile_size_px
        if target_size is None:
            raise ValueError(
                f"dataset_type={dtype!r} extraction requires "
                "preprocessing.requested_tile_size_px (the supervision tile size) "
                "when feature_dir is not provided."
            )
        if preprocessing.requested_spacing_um is None:
            raise ValueError(
                f"dataset_type={dtype!r} extraction requires a spacing — set "
                "preprocessing.requested_spacing_um or use an encoder that advertises "
                "a single supported_spacing_um."
            )
        from soma.dense_extraction import DenseTileFeatureExtractor

        cache_config = self._config.cache
        if cache_config.root_dir is None:
            cache_config = replace(
                cache_config,
                root_dir=Path(self._config.output_root) / "feature_cache",
            )
        extractor = DenseTileFeatureExtractor(
            self._dataset,
            self._config.encoder,
            target_size=int(target_size),
            spacing_um=float(preprocessing.requested_spacing_um),
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
            window_size=preprocessing.dense_window_size,
            overlap=float(preprocessing.dense_window_overlap),
            execution=self._config.execution,
            cache=cache_config,
            preprocessing=preprocessing,
        )
        try:
            return self._cache_backed_dense_source(
                extractor.run(feature_dir=run_dir / "features"),
                kind="dense_cache",
                dataset_csv=self._config.dataset_csv,
                splits_csv=self._config.splits_csv,
            )
        finally:
            _release_parent_cuda_state()
            _log_cuda_memory("after dense extraction release")

    def _build_slide_manifest_dense_context(self, *, run_dir: Path):
        """Sample ROIs from slides+masks, extract dense grids, return derived context.

        The derived ROI manifest + ROI splits (each ROI inherits its parent slide's split)
        are returned explicitly so the pipeline's configured slide-level dataset/splits stay
        intact. The grids are extracted via slide2vec + cached by soma; the sampling spec is
        folded into the dense cache key (distinct ``min_coverage``/spacing/strategy ⇒ distinct
        cache). Cached path only — the live/augmentation path over the same ROIs is A4.
        """
        from soma.dataset import SegmentationManifest, Splits
        from soma.dense_slide_extraction import (
            SlideManifestDenseExtractor,
            build_roi_manifest,
            sample_slide_rois,
        )

        dtype = self._config.dataset_type
        if dtype != "segmentation":
            raise ValueError(f"slide-manifest masks: config requires dataset_type='segmentation', got {dtype!r}.")
        if self._config.feature_mode == "live":
            raise NotImplementedError(
                "Live (augmentation) extraction over slide-manifest ROIs is not implemented "
                "yet (A4). Use the default cached feature_mode for slide-manifest segmentation."
            )
        if self._config.composite is not None:
            raise NotImplementedError(
                "Multi-encoder composite extraction over slide-manifest ROIs is not implemented "
                "yet. Use a single encoder for slide-manifest segmentation."
            )
        if self._feature_dir is not None:
            raise ValueError(
                "feature_dir is not supported with a slide-manifest masks: config — the ROIs "
                "are sampled and extracted from the slides, not read from pre-extracted grids."
            )
        if self._config.encoder is None:
            raise ValueError(
                "PipelineConfig.encoder is required for slide-manifest segmentation "
                "(feature_dir is not supported here)."
            )

        preprocessing = self._resolve_preprocessing()
        sampling = self._config.preprocessing.sampling or SamplingConfig()
        masks = self._config.preprocessing.masks

        cache_config = self._config.cache
        if cache_config.root_dir is None:
            cache_config = replace(cache_config, root_dir=Path(self._config.output_root) / "feature_cache")

        # Cross-run ROI sampling reuse (#365): resolve the roi_sampling cache before
        # sampling, run hs2p only over the missing slide set, publish the fresh per-slide
        # coords (zero-ROI outcomes included, so they hit next launch), and merge. Splits
        # are structurally excluded from the cache — the ROI manifest/splits are re-derived
        # below every launch from coords + the run's splits CSV, so a splits edit never
        # re-samples and never reads stale splits. Cache disabled ⇒ sample everything,
        # no cache I/O, exactly as before.
        if cache_config.enabled:
            from soma.cache import (
                resolve_cache_root,
                resolve_roi_sampling_cache,
                write_roi_sampling_coords,
            )

            cache_resolution = resolve_roi_sampling_cache(
                cache_root=resolve_cache_root(cache_config, feature_dir=run_dir / "features"),
                dataset=self._dataset,
                preprocessing=preprocessing,
            )
            miss_ids = cache_resolution.miss_sample_ids
            logger.info(
                "Sampling segmentation ROIs: %d slide(s) from cache, %d slide(s) to sample.",
                len(cache_resolution.coords_by_id),
                len(miss_ids),
            )
            fresh: dict[str, list[tuple[int, int]]] = {}
            if miss_ids:
                fresh = sample_slide_rois(
                    self._dataset,
                    masks=masks,
                    sampling=sampling,
                    preprocessing=preprocessing,
                    sample_ids=miss_ids,
                )
                write_roi_sampling_coords(
                    cache_resolution=cache_resolution, coords_by_sample_id=fresh
                )
            merged = {**cache_resolution.coords_by_id, **fresh}
        else:
            logger.info("Sampling segmentation ROIs from %d slides...", len(self._dataset.sample_ids))
            merged = sample_slide_rois(
                self._dataset, masks=masks, sampling=sampling, preprocessing=preprocessing
            )
        # Manifest row order follows the slide manifest regardless of the hit/miss mix,
        # so a warm relaunch derives a byte-identical ROI manifest.
        coords_by_slide = {sid: merged[sid] for sid in self._dataset.sample_ids}
        roi_dir = run_dir / "segmentation_rois"
        roi_manifest_csv, roi_splits_csv = build_roi_manifest(
            self._dataset, self._config.splits_csv, coords_by_slide, out_dir=roi_dir
        )
        roi_dataset = SegmentationManifest(roi_manifest_csv)
        roi_splits = Splits(
            roi_splits_csv,
            roi_dataset,
            tune_is_test=self._config.training.tune_is_test,
        )

        extractor = SlideManifestDenseExtractor(
            roi_dataset,
            self._config.encoder,
            masks=masks,
            sampling=sampling,
            preprocessing=preprocessing,
            execution=self._config.execution,
            cache=cache_config,
        )
        try:
            store = extractor.run(feature_dir=run_dir / "features")
        finally:
            _release_parent_cuda_state()
        return _FeatureSourceContext(
            feature_store=self._cache_backed_dense_source(
                store,
                kind="slide_manifest_dense_cache",
                dataset_csv=roi_manifest_csv,
                splits_csv=roi_splits_csv,
                parent_dataset_csv=self._config.dataset_csv,
                parent_splits_csv=self._config.splits_csv,
            ),
            dataset=roi_dataset,
            splits=roi_splits,
        )

    def _build_composite_dense_store(self, *, run_dir: Path):
        """Extract every member encoder into its own cache; return a concat view (§7).

        Each member carries its own ``feature_kind`` / ``attention`` / ``member_norm``
        spec (cross-defaulted in :class:`PipelineConfig`), so the heterogeneous setup
        (several FMs, some attention, some patch-feature) needs no special-casing. Members
        share the run's spacing + supervision ``target_size`` (v1); their token grids may
        differ and are combined at load time by :class:`CompositeDenseFeatureStore` per the
        composite's ``concat_resolution`` / ``concat_grid_size``.
        """
        from soma.dense.composite import CompositeDenseFeatureStore
        from soma.dense_extraction import DenseTileFeatureExtractor

        composite = self._config.composite
        preprocessing = self._resolve_preprocessing()
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
        cache_config = self._config.cache
        if cache_config.root_dir is None:
            cache_config = replace(
                cache_config, root_dir=Path(self._config.output_root) / "feature_cache"
            )
        member_stores = []
        try:
            for index, member in enumerate(composite.encoders):
                member_encoder = EncoderConfig(
                    name=member.name,
                    precision=member.precision,
                    batch_size=member.batch_size,
                    adaptive_batching=member.adaptive_batching,
                    output_variant=member.output_variant,
                    allow_non_recommended_settings=member.allow_non_recommended_settings,
                )
                # Per-member sliding window (CONCH native-448 vs H0-mini native-224, …);
                # fall back to the run's shared window when the member leaves it unset.
                member_window = (
                    member.dense_window_size
                    if member.dense_window_size is not None
                    else preprocessing.dense_window_size
                )
                member_overlap = (
                    member.dense_window_overlap
                    if member.dense_window_overlap is not None
                    else preprocessing.dense_window_overlap
                )
                member_prep = replace(
                    preprocessing,
                    feature_kind=member.feature_kind,
                    attention=member.attention,
                    dense_window_size=member_window,
                    dense_window_overlap=member_overlap,
                )
                extractor = DenseTileFeatureExtractor(
                    self._dataset,
                    member_encoder,
                    target_size=int(target_size),
                    spacing_um=float(preprocessing.requested_spacing_um),
                    backend=preprocessing.backend,
                    tolerance=float(preprocessing.tolerance),
                    window_size=member_window,
                    overlap=float(member_overlap),
                    execution=self._config.execution,
                    cache=cache_config,
                    preprocessing=member_prep,
                )
                member_dir = run_dir / "features" / f"member_{index}_{member.name}"
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

    @staticmethod
    def _resolve_composite_spacing(composite: "CompositeConfig") -> float:
        """Members' shared supported µm/px, for the requested_spacing_um auto-default.

        v1 reads every member at one spacing, so this succeeds only when all members
        advertise the *same* single supported spacing; a member with multiple supported
        spacings, or members that disagree, must be pinned via
        ``preprocessing.requested_spacing_um``.
        """
        from slide2vec.encoders.registry import resolve_preprocessing_requirements

        per_member: dict[str, float] = {}
        for member in composite.encoders:
            spacing = resolve_preprocessing_requirements(member.name)["spacing_um"]
            if isinstance(spacing, (list, tuple)):
                if len(spacing) != 1:
                    raise ValueError(
                        f"composite member '{member.name}' supports multiple spacings "
                        f"{list(spacing)}; set preprocessing.requested_spacing_um explicitly."
                    )
                spacing = spacing[0]
            per_member[member.name] = float(spacing)
        unique = sorted(set(per_member.values()))
        if len(unique) != 1:
            raise ValueError(
                "composite members do not share a single supported spacing "
                f"({per_member}); set preprocessing.requested_spacing_um explicitly "
                "(v1 reads every member at the same µm/px)."
            )
        return unique[0]

    def _build_live_segmentation_source(self) -> "LiveSegmentationSource":
        """Prepare one public DenseEncodeKit and share it across every fold.

        Soma owns the augmented-pixel handoff. slide2vec owns the dense normalization,
        resolved geometry, padding, precision/device transfer, frozen encoding,
        windowing/attention mode, and output dtype behind the public kit boundary.
        """
        from slide2vec import DenseImageOptions, ExecutionOptions, Model

        from soma.dense.live import LiveSegmentationSource
        from soma.encoders.validation import resolve_encoder_precision

        if self._config.encoder is None:
            raise ValueError(
                "PipelineConfig.encoder is required for feature_mode='live' segmentation."
            )
        preprocessing = self._resolve_preprocessing()
        target_size = preprocessing.requested_tile_size_px
        if target_size is None:
            raise ValueError(
                "feature_mode='live' segmentation requires preprocessing.requested_tile_size_px "
                "(the mask/tile supervision size)."
            )
        if preprocessing.requested_spacing_um is None:
            raise ValueError(
                "feature_mode='live' segmentation requires a spacing — set "
                "preprocessing.requested_spacing_um or use an encoder that advertises a "
                "single supported_spacing_um."
            )

        model = Model.from_preset(
            self._config.encoder.name,
            output_variant=self._config.encoder.output_variant,
            allow_non_recommended_settings=self._config.encoder.allow_non_recommended_settings,
        )
        precision = resolve_encoder_precision(
            self._config.encoder, encoder_name=self._config.encoder.name
        )
        window_size = preprocessing.dense_window_size
        overlap = float(preprocessing.dense_window_overlap)
        from soma.dense.sliding import describe_dense_mode

        # print, not logger: always visible regardless of logging config (same as the
        # cached extractor's announcement) so the resolved mode is never silent.
        print(f"Live segmentation dense mode: {describe_dense_mode(window_size, overlap)}")
        feature_kind = preprocessing.feature_kind or "patch_features"
        dense = DenseImageOptions(
            target_size=int(target_size),
            spacing_um=float(preprocessing.requested_spacing_um),
            tolerance=float(preprocessing.tolerance),
            backend=preprocessing.backend,
            pad_mode="reflect",
            image_pad_value=None,
            window_size=window_size,
            overlap=overlap,
            feature_kind=feature_kind,
            attention_blocks=(
                tuple(preprocessing.attention.blocks)
                if feature_kind == "cls_attention"
                else (-1,)
            ),
            attention_include_registers=(
                bool(preprocessing.attention.include_registers)
                if feature_kind == "cls_attention"
                else False
            ),
        )
        kit = model.prepare_dense_encoder(
            dense=dense,
            execution=ExecutionOptions(
                num_gpus=self._config.execution.num_gpus,
                precision=precision,
                output_dtype="fp32",
            ),
        )
        # Model.feature_dim is the pooled/backbone width and is not authoritative for
        # alternate dense outputs such as CLS attention. Probe only the public tensor
        # boundary so the decoder is built for the kit's actual resolved grid channels.
        probe_pixels = torch.zeros(
            (3, *kit.geometry.target_size), dtype=torch.uint8, device="cpu"
        )
        probe_batch = torch.stack([kit.preprocessor()(probe_pixels)])
        probe_grid = kit.encode(probe_batch)
        if probe_grid.ndim != 4 or tuple(int(v) for v in probe_grid.shape[-2:]) != tuple(
            int(v) for v in kit.geometry.grid_shape
        ):
            raise ValueError(
                "DenseEncodeKit returned an invalid probe grid: expected "
                f"(B, d, {kit.geometry.grid_shape[0]}, {kit.geometry.grid_shape[1]}), "
                f"got {tuple(probe_grid.shape)}."
            )
        return LiveSegmentationSource(
            kit=kit,
            device=model.device,
            feature_dim=int(probe_grid.shape[1]),
            augmentation=self._config.augmentation,
            spacing_um=float(preprocessing.requested_spacing_um),
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
        )

    def _resolve_preprocessing(self) -> "PreprocessingConfig":
        """Resolve preprocessing config, injecting HIPT-specific overrides if needed."""
        if self._config.dataset_type == "tile":
            # Tile-dataset pipelines have no WSI tiling step; preprocessing is irrelevant.
            return PreprocessingConfig()
        preprocessing = derive_preprocessing_for_aggregator(
            self._config.preprocessing,
            self._config.aggregator,
        )
        if self._config.encoder is not None:
            preprocessing = resolve_preprocessing_config(
                self._config.encoder,
                preprocessing,
            )
        elif self._config.composite is not None and preprocessing.requested_spacing_um is None:
            # Composite runs have no single encoder to feed into resolve_preprocessing_config,
            # but v1 still reads every member at one shared spacing. Resolve that default
            # here so extraction, run summary, and dense-fold spacing guards agree.
            preprocessing = replace(
                preprocessing,
                requested_spacing_um=self._resolve_composite_spacing(self._config.composite),
            )
        return preprocessing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    *,
    attention_dir: Path | None = None,
    aggregator_name: str | None = None,
) -> EvaluationReport:
    """Evaluate model on a split and build a report.

    Args:
        attention_dir: When provided, per-tile attention scores are saved as
            ``<sample_id>.npz`` files inside this directory during the forward
            pass — no separate inference pass is needed for heatmap generation.
        aggregator_name: Name of the aggregator (e.g. ``"abmil"``), used to
            decide whether to apply softmax before saving attention scores.
            Required when ``attention_dir`` is set.
    """
    all_logits = []
    all_targets: dict[str, list[torch.Tensor]] = {}
    all_sample_ids: list[str] = []

    for batch in loader:
        features = batch.features.to(device)
        if hasattr(batch, "mask"):
            out = model(features, mask=batch.mask.to(device))
        else:
            out = model(features)
        all_logits.append(out.logits.cpu())
        for key, value in batch.targets.items():
            all_targets.setdefault(key, []).append(value.cpu())
        all_sample_ids.extend(batch.sample_ids)

        if attention_dir is not None and out.tile_attention is not None:
            from soma.heatmaps import normalize_attention
            for i, sid in enumerate(batch.sample_ids):
                attn_i = out.tile_attention[i : i + 1]
                if hasattr(batch, "mask"):
                    attn_i = attn_i[..., batch.mask[i]]
                normalized = normalize_attention(attn_i, aggregator_name or "")
                np.savez_compressed(attention_dir / f"{sid}.npz", attention=normalized)

    logits = torch.cat(all_logits, dim=0)
    targets = {key: torch.cat(values, dim=0) for key, values in all_targets.items()}

    task_head = model.task_head
    metrics = task_head.compute_metrics(logits, targets)
    processed = task_head.postprocess(logits)

    predictions = []
    if "risk_scores" in processed:
        # Survival: true_label holds time, plus event and predicted risk.
        time = targets["time"].numpy()
        event = targets["event"].numpy()
        for i, sid in enumerate(all_sample_ids):
            predictions.append(
                SamplePrediction(
                    sample_id=sid,
                    true_label=float(time[i]),
                    event=float(event[i]),
                    risk_score=float(processed["risk_scores"][i]),
                )
            )
    elif "probabilities" in processed:
        # Classification
        y_true = targets["label"].numpy()
        for i, sid in enumerate(all_sample_ids):
            predictions.append(
                SamplePrediction(
                    sample_id=sid,
                    true_label=int(y_true[i]),
                    predicted_label=int(processed["predicted_labels"][i]),
                    probabilities=processed["probabilities"][i].tolist(),
                )
            )
    elif "raw_scores" in processed:
        # Ordinal classification: integer prediction + raw continuous score
        y_true = targets["label"].numpy()
        for i, sid in enumerate(all_sample_ids):
            predictions.append(
                SamplePrediction(
                    sample_id=sid,
                    true_label=int(y_true[i]),
                    predicted_label=int(processed["predicted_labels"][i]),
                    raw_score=float(processed["raw_scores"][i]),
                )
            )
    else:
        # Regression
        y_true = targets["value"].numpy()
        for i, sid in enumerate(all_sample_ids):
            predictions.append(
                SamplePrediction(
                    sample_id=sid,
                    true_label=float(y_true[i]),
                    predicted_value=float(processed["predictions"][i]),
                )
            )

    return EvaluationReport(
        split=split_name,
        metrics=metrics,
        predictions=predictions,
    )


@torch.inference_mode()
def _evaluate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    *,
    dataset: SegmentationManifest | None = None,
    output_dir: Path | None = None,
    save_segmentation_overlays: bool = True,
    save_segmentation_probabilities: bool = False,
) -> EvaluationReport:
    """Streaming dense evaluation: accumulate compact per-image confusion counts.

    Shares ``accumulate_dense_stats`` with ``Trainer._tune_streaming_metrics`` — never
    concatenates the dense ``(N, C, H, W)`` logits (which would OOM). The per-image
    ``dense_stats`` rows are concatenated along the image axis and reduced once via
    ``finalize_eval_metrics`` (the same reduce+filter path as ``compute_metrics``), so
    the report metric matches the training monitor exactly.

    When ``output_dir`` is given, a :class:`DenseArtifactWriter` streams per-tile
    prediction rasters, color overlays (fail-soft via ``dataset``'s source images),
    and a ``predictions_<split>.csv`` to disk — written per batch *before* the logits
    are discarded, so memory stays bounded. The returned ``EvaluationReport`` still
    carries ``predictions=[]``: the dense artifacts live on disk, not in the report.
    """
    head = model.task_head
    writer = (
        DenseArtifactWriter(
            head=head,
            split=split_name,
            output_dir=output_dir,
            dataset=dataset,
            save_segmentation_overlays=save_segmentation_overlays,
            save_segmentation_probabilities=save_segmentation_probabilities,
            # Slide-manifest ROIs (record.region set) carry a whole-slide image_path; the
            # writer reads each ROI window at the run geometry rather than opening the slide.
            spacing_um=getattr(head, "_spacing_um", None),
            backend=getattr(head, "_backend", "auto"),
            tolerance=getattr(head, "_tolerance", 0.05),
        )
        if output_dir is not None
        else None
    )
    stat_rows, _, _ = accumulate_dense_stats(model, loader, device, on_batch_output=writer)
    metrics = head.finalize_eval_metrics(torch.cat(stat_rows, dim=0)) if stat_rows else {}
    if writer is not None:
        writer.finalize()
    return EvaluationReport(split=split_name, metrics=metrics, predictions=[])


def _records_for_sample_ids(
    dataset: Dataset,
    sample_ids: tuple[str, ...],
    allowed_sample_ids: set[str],
) -> list[SampleRecord]:
    return [
        dataset.samples[sample_id]
        for sample_id in sample_ids
        if sample_id in allowed_sample_ids
    ]


def _build_deterministic_baseline(
    train_records: list[SampleRecord],
    *,
    target_fn,
    task_family: str,
    num_classes: int,
) -> _DeterministicBaseline:
    if not train_records:
        raise ValueError("Cannot build a deterministic baseline without training records")

    if task_family in {"binary_classification", "multiclass_classification"}:
        labels = np.asarray(
            [int(target_fn(record)["label"]) for record in train_records], dtype=int
        )
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        probabilities = (counts / counts.sum()).tolist()
        return _DeterministicBaseline(
            task_family=task_family,
            probabilities=probabilities,
            predicted_label=int(np.argmax(probabilities)),
        )

    if task_family == "ordinal_classification":
        raw_score = float(np.mean([int(target_fn(record)["label"]) for record in train_records]))
        predicted_label = int(np.clip(np.rint(raw_score), 0, max(num_classes - 1, 0)))
        return _DeterministicBaseline(
            task_family=task_family,
            predicted_label=predicted_label,
            raw_score=raw_score,
        )

    if task_family == "survival":
        # Empty-bag placeholders share a constant risk; they tie among themselves
        # and contribute ~0.5 concordance against real samples.
        return _DeterministicBaseline(task_family=task_family, risk_score=0.0)

    predicted_value = float(
        np.mean([float(target_fn(record)["value"]) for record in train_records])
    )
    return _DeterministicBaseline(
        task_family=task_family,
        predicted_value=predicted_value,
    )


def _make_placeholder_prediction(
    record: SampleRecord,
    *,
    target_fn,
    baseline: _DeterministicBaseline,
    sample_id: str | None = None,
) -> SamplePrediction:
    targets = target_fn(record)
    if baseline.task_family == "survival":
        return SamplePrediction(
            sample_id=sample_id or record.sample_id,
            true_label=float(targets["time"]),
            event=float(targets["event"]),
            risk_score=baseline.risk_score,
            is_placeholder=True,
            missing_reason="no_tiles",
        )

    true_value: int | float
    if baseline.task_family == "regression":
        true_value = float(targets["value"])
    else:
        true_value = int(targets["label"])

    return SamplePrediction(
        sample_id=sample_id or record.sample_id,
        true_label=true_value,
        predicted_label=baseline.predicted_label,
        probabilities=list(baseline.probabilities) if baseline.probabilities is not None else None,
        predicted_value=baseline.predicted_value,
        raw_score=baseline.raw_score,
        is_placeholder=True,
        missing_reason="no_tiles",
    )


def _compute_metrics_from_predictions(
    predictions: list[SamplePrediction],
    *,
    task_family: str,
    metric_names: list[str],
) -> dict[str, float]:
    if not predictions:
        return {
            "coverage": 0.0,
            "num_samples": 0,
            "num_real_samples": 0,
            "num_placeholder_samples": 0,
        }

    if task_family == "survival":
        from soma.evaluation.metrics import compute_survival_metrics

        event = np.asarray([float(pred.event) for pred in predictions], dtype=float)
        time = np.asarray([float(pred.true_label) for pred in predictions], dtype=float)
        risk = np.asarray([float(pred.risk_score) for pred in predictions], dtype=float)
        metrics = compute_survival_metrics(metric_names, event, time, risk)
    elif task_family in {"binary_classification", "multiclass_classification"}:
        y_true = np.asarray([int(pred.true_label) for pred in predictions], dtype=int)
        y_pred = np.asarray([int(pred.predicted_label) for pred in predictions], dtype=int)
        y_prob = np.asarray([pred.probabilities for pred in predictions], dtype=float)
        metrics = compute_metrics(task_family, metric_names, y_true, y_pred, y_prob=y_prob)
    elif task_family == "ordinal_classification":
        y_true = np.asarray([int(pred.true_label) for pred in predictions], dtype=int)
        y_pred = np.asarray([int(pred.predicted_label) for pred in predictions], dtype=int)
        metrics = compute_metrics(task_family, metric_names, y_true, y_pred)
    else:
        y_true = np.asarray([float(pred.true_label) for pred in predictions], dtype=float)
        y_pred = np.asarray([float(pred.predicted_value) for pred in predictions], dtype=float)
        metrics = compute_metrics(task_family, metric_names, y_true, y_pred)

    num_placeholder = sum(1 for pred in predictions if pred.is_placeholder)
    num_samples = len(predictions)
    num_real = num_samples - num_placeholder
    metrics["coverage"] = float(num_real / num_samples)
    metrics["num_samples"] = num_samples
    metrics["num_real_samples"] = num_real
    metrics["num_placeholder_samples"] = num_placeholder
    return metrics


def _evaluate_split_with_placeholders(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    *,
    output_sample_ids: tuple[str, ...],
    empty_sample_ids: list[str],
    dataset: Dataset,
    target_fn,
    baseline: _DeterministicBaseline,
    metric_names: list[str],
    task_family: str,
    attention_dir: Path | None = None,
    aggregator_name: str | None = None,
    placeholder_records_by_id: dict[str, SampleRecord] | None = None,
) -> EvaluationReport:
    real_report = (
        _evaluate(
            model,
            loader,
            split_name,
            device,
            attention_dir=attention_dir,
            aggregator_name=aggregator_name,
        )
        if len(loader.dataset) > 0
        else EvaluationReport(split=split_name, metrics={}, predictions=[])
    )
    placeholder_predictions = [
        _make_placeholder_prediction(
            (
                placeholder_records_by_id[sample_id]
                if placeholder_records_by_id is not None
                else dataset.samples[sample_id]
            ),
            target_fn=target_fn,
            baseline=baseline,
            sample_id=sample_id,
        )
        for sample_id in empty_sample_ids
    ]
    predictions_by_id = {
        pred.sample_id: pred
        for pred in [*real_report.predictions, *placeholder_predictions]
    }
    full_predictions = [predictions_by_id[sample_id] for sample_id in output_sample_ids]
    metrics = _compute_metrics_from_predictions(
        full_predictions,
        task_family=task_family,
        metric_names=metric_names,
    )
    return EvaluationReport(split=split_name, metrics=metrics, predictions=full_predictions)


def _save_metrics(
    tune_report: EvaluationReport,
    test_reports: dict[str, EvaluationReport],
    path: Path,
) -> None:
    data: dict[str, dict[str, float]] = {"tune": tune_report.metrics}
    for split_name, report in test_reports.items():
        data[split_name] = report.metrics
    path.write_text(json.dumps(data, indent=2))


def _save_training_history(history: list, path: Path) -> None:
    data = {
        "epochs": [epoch_log_to_dict(log) for log in history],
        # Diagnostic only (see soma.training.trainer.peak_per_metric); the reporting
        # layer drops this so it never reaches summaries or results tables.
        "peak_per_metric": peak_per_metric(history),
    }
    path.write_text(json.dumps(data, indent=2))


def _build_subgroup_data(
    dataset: Dataset,
    report: EvaluationReport,
    subgroup_columns: list[str],
) -> dict[str, dict[str, object]]:
    """Build a mapping of sample/patient ID to subgroup values for predictions."""
    return subgroup_data_for_predictions(dataset, report.predictions, subgroup_columns)


def _build_predictions_df(
    report: EvaluationReport,
    subgroup_data: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build an enriched predictions DataFrame in memory (no disk I/O)."""
    rows = []
    for pred in report.predictions:
        row: dict = {"sample_id": pred.sample_id, "true_label": pred.true_label}
        if pred.predicted_label is not None:
            row["predicted_label"] = pred.predicted_label
        if pred.probabilities is not None:
            for i, p in enumerate(pred.probabilities):
                row[f"prob_{i}"] = p
        if pred.raw_score is not None:
            row["raw_score"] = pred.raw_score
        if pred.predicted_value is not None:
            row["predicted_value"] = pred.predicted_value
        if pred.is_placeholder:
            row["is_placeholder"] = True
        if pred.missing_reason is not None:
            row["missing_reason"] = pred.missing_reason
        if subgroup_data and pred.sample_id in subgroup_data:
            row.update(subgroup_data[pred.sample_id])
        rows.append(row)
    return pd.DataFrame(rows)


def _save_predictions(
    report: EvaluationReport,
    path: Path,
    *,
    subgroup_data: dict[str, dict[str, object]] | None = None,
) -> None:
    with open(path, "w", newline="") as f:
        if not report.predictions:
            return

        first = report.predictions[0]
        extra_cols = sorted(next(iter(subgroup_data.values()))) if subgroup_data else []
        placeholder_cols: list[str] = []
        if any(pred.is_placeholder for pred in report.predictions):
            placeholder_cols.append("is_placeholder")
        if any(pred.missing_reason is not None for pred in report.predictions):
            placeholder_cols.append("missing_reason")

        def _write_row(writer: csv.DictWriter, row: dict, pred: SamplePrediction) -> None:
            if "is_placeholder" in placeholder_cols:
                row["is_placeholder"] = pred.is_placeholder
            if "missing_reason" in placeholder_cols:
                row["missing_reason"] = pred.missing_reason or ""
            if subgroup_data and pred.sample_id in subgroup_data:
                row.update(subgroup_data[pred.sample_id])
            writer.writerow(row)

        if first.risk_score is not None:
            # Survival: time (true_label), event indicator, predicted risk
            fieldnames = ["sample_id", "true_label", "event", "risk_score"] + placeholder_cols + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "event": pred.event,
                    "risk_score": f"{pred.risk_score:.6f}",
                }
                _write_row(writer, row, pred)
        elif first.probabilities is not None:
            # Classification: include class probabilities
            num_classes = len(first.probabilities)
            fieldnames = ["sample_id", "true_label", "predicted_label"] + [
                f"prob_{i}" for i in range(num_classes)
            ] + placeholder_cols + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row: dict = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "predicted_label": pred.predicted_label,
                }
                for i, p in enumerate(pred.probabilities):  # type: ignore[arg-type]
                    row[f"prob_{i}"] = f"{p:.6f}"
                _write_row(writer, row, pred)
        elif first.raw_score is not None:
            # Ordinal classification: integer prediction + raw continuous score
            fieldnames = ["sample_id", "true_label", "predicted_label", "raw_score"] + placeholder_cols + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "predicted_label": pred.predicted_label,
                    "raw_score": f"{pred.raw_score:.6f}",
                }
                _write_row(writer, row, pred)
        else:
            # Regression: predicted continuous value
            fieldnames = ["sample_id", "true_label", "predicted_value"] + placeholder_cols + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "predicted_value": f"{pred.predicted_value:.6f}",
                }
                _write_row(writer, row, pred)


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
                summary[f"{split_name}/{key}_mean"] = float(np.mean(values))
                summary[f"{split_name}/{key}_std"] = float(np.std(values))

    return summary


def _fold_result_to_metric_dict(fr: FoldResult) -> dict[str, dict[str, float]]:
    """Project a FoldResult onto the ``{split: metrics}`` shape ``metrics.json`` holds."""
    out = {"tune": dict(fr.tune_report.metrics)}
    for split_name, report in fr.test_reports.items():
        out[split_name] = dict(report.metrics)
    return out


def _aggregate_fold_metrics(
    fold_results: list[FoldResult], *, include_tune: bool = False
) -> dict[str, float]:
    """Aggregate in-memory fold results (see :func:`_summarize_fold_metric_dicts`)."""
    return _summarize_fold_metric_dicts(
        [_fold_result_to_metric_dict(fr) for fr in fold_results], include_tune=include_tune
    )


def _aggregate_fold_metrics_from_disk(
    run_dir: Path, *, single_fold: bool, include_tune: bool
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
        metrics_paths = sorted(run_dir.glob("fold_*/metrics.json"))
    per_fold = [
        json.loads(path.read_text()) for path in metrics_paths if path.is_file()
    ]
    return _summarize_fold_metric_dicts(per_fold, include_tune=include_tune)


def _save_summary(summary: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2))
