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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

from soma.aggregators.registry import aggregator_registry
from soma.decoders.registry import decoder_registry
from soma.config import (
    AggregatorConfig,
    DecoderConfig,
    EncoderConfig,
    EvalConfig,
    HeatmapConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)
from soma.dataset import Dataset, FoldSplit, SampleRecord, SegmentationManifest, Splits
from soma.dense.live import LiveSegmentationSource
from soma.evaluation.metrics import resolve_metrics
from soma.evaluation.metrics import compute_metrics
from soma.evaluation.dense_artifacts import DenseArtifactWriter
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.extraction import FeatureExtractor, _release_parent_cuda_state
from soma.features import FeatureStore
from soma.encoders.validation import resolve_preprocessing_config
from soma.output_layout import (
    count_run_directories,
    create_run_metadata,
    has_successful_run,
    resolve_managed_output_paths,
    update_experiment_index,
    update_latest_pointer,
    update_run_index,
    write_experiment_metadata,
    write_run_metadata,
)
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator
from soma.tasks.classification import BranchAwareClassificationHead
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.collate import bag_collate_fn, cox_window_collate, hierarchical_bag_collate_fn
from soma.training.model import (
    EmbeddingModel,
    LiveSegmentationModel,
    MILModel,
    SegmentationModel,
)
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.sample_dataset import SampleDataset, SampleBatch, sample_collate_fn
from soma.training.segmentation_dataset import (
    LiveSegmentationDataset,
    SegmentationDataset,
    segmentation_collate_fn,
)
from soma.training.seed import seed_everything
from soma.training.trainer import Trainer, TrainResult, accumulate_dense_stats, epoch_log_to_dict
from soma.reporting import generate_report_from_result
from soma.reporting.subgroups import subgroup_data_for_predictions, subgroup_report_for_predictions


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Result of training + evaluation for a single fold."""

    fold: int
    train_result: TrainResult
    tune_report: EvaluationReport
    test_reports: dict[str, EvaluationReport]  # split_name → report


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run across all folds."""

    fold_results: list[FoldResult]
    summary: dict[str, float]
    run_dir: Path


@dataclass(frozen=True)
class _DeterministicBaseline:
    """Deterministic fallback prediction derived from the training split."""

    task_family: str
    probabilities: list[float] | None = None
    predicted_label: int | None = None
    predicted_value: float | None = None
    raw_score: float | None = None
    risk_score: float | None = None


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
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    """Create train, tune, and per-split test DataLoaders with a common pattern."""
    loader_kwargs = _loader_kwargs(training)
    train_loader = DataLoader(
        dataset_cls(train_items, feature_store, target_fn),
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
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
            dense_transform=source.dense_transform,
            spacing_um=source.spacing_um,
            backend=source.backend,
            tolerance=source.tolerance,
            num_classes=num_classes,
            ignore_index=ignore_index,
            augment=augment,
            pad_mode=source.pad_mode,
            image_pad_value=source.image_pad_value,
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
        # Single-embedding path: one pre-computed vector per sample, no aggregation
        if feature_store.is_slide_level and aggregator is not None:
            raise ValueError("aggregator must be None for slide-level features")
        head = task_cls(input_dim=feature_dim, **task_params)
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
        model: torch.nn.Module = EmbeddingModel(task_head=head)
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
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=feature_dim, **aggregator.params)
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
        model = MILModel(aggregator=agg, task_head=head)

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

    # Load best checkpoint and evaluate
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


def train_one_segmentation_fold(
    feature_store: "DenseFeatureStore | LiveSegmentationSource",
    dataset: SegmentationManifest,
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
) -> FoldResult:
    """Train and evaluate a single dense-segmentation fold.

    Separate from :func:`train_one_fold` because the scalar path's manifest-status
    filtering, deterministic baseline, placeholder predictions, ``SamplePrediction``
    and CSV machinery are all scalar-shaped and do not apply to dense rasters, and
    because the model is ``decoder + SegmentationHead`` (not aggregator/embedding +
    head). The split→records selection and ``tune_is_test``/``allow_missing_tune``
    semantics are reused.

    Two data planes share this body (design §13.B-3), distinguished by
    ``feature_store``: a :class:`~soma.dense.DenseFeatureStore` drives the **cached**
    path (read pre-extracted grids + head-loaded masks), a
    :class:`~soma.dense.live.LiveSegmentationSource` drives the **live** path
    (re-encode augmented image+mask tiles through the frozen encoder each step). Only
    five things differ — geometry source, feature_dim, coverage check, dataset, and
    model — and they are handled with inline branches here; everything else (records,
    num_classes, decoder/head build, trainer, eval, summary) is shared.
    """
    if decoder is None:
        raise ValueError("dataset_type='segmentation' requires a decoder configuration")

    evaluation = evaluation or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(training.seed, fold=fold)
    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

    # Split -> records (reuse the legacy, no-manifest selection; dense grids are
    # required for every listed sample, validated below).
    train_records = [dataset.samples[sid] for sid in fold_split.train]
    tune_records = [dataset.samples[sid] for sid in fold_split.tune]
    test_records_by_split = {
        split_name: [dataset.samples[sid] for sid in ids]
        for split_name, ids in fold_split.tests.items()
    }
    if training.tune_is_test:
        tune_from_test_split_name = _resolve_tune_is_test_split(fold_split, _fp)
        tune_records = list(test_records_by_split[tune_from_test_split_name])

    if not train_records:
        raise ValueError(f"{_fp} has no training samples")
    if not tune_records:
        if not training.allow_missing_tune:
            raise ValueError(f"{_fp} has no tuning samples")
        logger.warning("%s has no tune samples; using train as tune (allow_missing_tune)", _fp)
        tune_records = list(train_records)
    for split_name, records in test_records_by_split.items():
        if not records:
            raise ValueError(f"{_fp} has no samples in split '{split_name}'")

    all_records = [*train_records, *tune_records, *(r for recs in test_records_by_split.values() for r in recs)]

    is_live = isinstance(feature_store, LiveSegmentationSource)

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
            if record.image_path is None or record.mask_path is None:
                raise ValueError(
                    f"live segmentation sample '{record.sample_id}' needs both image_path "
                    "and mask_path (the live path re-encodes from the raw tiles)."
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
        grid_spacing_um = feature_store.metadata(ref_id).get("spacing_um")
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
        **seg_params,
    )
    target_fn = head.extract_targets

    # Decoder: inject the auto-computed upsample depth only if the decoder accepts it
    # (LinearDecoder does not) and the user did not pin it. num_upsample_blocks brings
    # the token grid up to ~encoded resolution (the size the head interpolates to);
    # encoded/grid == patch_size per axis.
    decoder_cls = decoder_registry.get(decoder.name)
    decoder_params = dict(decoder.params)
    ctor_params = inspect.signature(decoder_cls.__init__).parameters
    if "num_upsample_blocks" in ctor_params and "num_upsample_blocks" not in decoder_params:
        ratio_h = geometry.encoded_size[0] / geometry.grid_shape[0]
        ratio_w = geometry.encoded_size[1] / geometry.grid_shape[1]
        decoder_params["num_upsample_blocks"] = max(0, math.ceil(math.log2(max(ratio_h, ratio_w))))
    decoder_obj = decoder_cls(
        input_dim=ref_feature_dim,
        num_classes=num_classes,
        **decoder_params,
    )
    if decoder_obj.num_classes != head.num_classes:
        raise ValueError(
            f"decoder num_classes ({decoder_obj.num_classes}) != head num_classes "
            f"({head.num_classes}) — a mismatch would misregister the logits."
        )

    seg_collate = functools.partial(segmentation_collate_fn, target_dtypes=head.target_dtypes)
    # Model + loaders (the remaining live/cached fork). Live wraps the shared frozen
    # encoder so each step re-encodes the augmented tiles; cached consumes pre-extracted
    # grids. The trainer, eval, and checkpoint reload paths below are identical.
    if is_live:
        model = LiveSegmentationModel(
            encoder=feature_store.encoder,
            decoder=decoder_obj,
            task_head=head,
            device=feature_store.device,
            precision=feature_store.precision,
            geometry=geometry,
            window_size=feature_store.window_size,
            overlap=feature_store.overlap,
        )
        train_loader, tune_loader, test_loaders = _make_live_loaders(
            feature_store, seg_collate,
            train_records, tune_records, test_records_by_split,
            training,
            num_classes=num_classes,
            ignore_index=head.ignore_index,
        )
    else:
        model = SegmentationModel(decoder=decoder_obj, task_head=head)
        train_loader, tune_loader, test_loaders = _make_loaders(
            SegmentationDataset, seg_collate,
            train_records, tune_records, test_records_by_split,
            training, feature_store, target_fn,
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

    checkpoint = torch.load(train_result.checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tune_report = _evaluate_segmentation(
        model, tune_loader, "tune", device, dataset=dataset, output_dir=fold_dir,
        save_probabilities=evaluation.save_probabilities,
    )
    test_reports = {
        split_name: _evaluate_segmentation(
            model, loader, split_name, device, dataset=dataset, output_dir=fold_dir,
            save_probabilities=evaluation.save_probabilities,
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


def train(
    feature_store: FeatureStore,
    dataset: Dataset,
    splits: Splits,
    task: TaskConfig,
    training: TrainingConfig,
    run_dir: str | Path,
    aggregator: AggregatorConfig | None = None,
    decoder: DecoderConfig | None = None,
    evaluation: EvalConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    heatmaps: HeatmapConfig | None = None,
    dataset_type: str = "slide",
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

    Returns:
        PipelineResult with per-fold results and aggregated summary.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if dataset_type == "patient" or dataset.has_patient_ids:
        splits.validate_no_patient_leakage(dataset)

    single_fold = splits.num_folds == 1
    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        fold_dir = run_dir if single_fold else run_dir / f"fold_{fold_idx}"
        if dataset_type == "segmentation":
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
                fold=fold_idx,
                num_folds=splits.num_folds,
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
            )
        fold_results.append(result)

    summary = _aggregate_fold_metrics(fold_results)
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

    # Aggregator (MIL path) or decoder (segmentation path) — the trainable component.
    if dataset_type == "segmentation":
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
    elif dataset_type == "segmentation":
        # DenseFeatureStore is not a FeatureStore (no is_slide_level/is_hierarchical);
        # branch before those attrs are touched.
        level = "dense (segmentation)"
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


class _RunRecorder:
    """Context manager that tracks run metadata lifecycle (running → completed/failed).

    On enter: writes initial "running" metadata and updates the run and experiment indexes.
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
        layout = self._layout
        update_run_index(layout.index_dir / "runs.csv", metadata)
        update_experiment_index(
            layout.index_dir / "experiments.csv",
            layout.experiment,
            num_runs=count_run_directories(layout.experiment_dir),
            latest_run_id=metadata.run_id,
            latest_status=metadata.status,
        )


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
        # Segmentation uses a mask-based manifest (image_path/mask_path, label
        # optional); it exposes the same samples/sample_ids surface Splits needs.
        if config.dataset_type == "segmentation":
            self._dataset = SegmentationManifest(config.dataset_csv)
        else:
            self._dataset = Dataset(config.dataset_csv)
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
        save_config(self._config, layout.run_dir / "config.yaml")

        with _RunRecorder(layout, self._config) as recorder:
            store = self._get_feature_store(run_dir=layout.run_dir)
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
                )
            )

            result = train(
                feature_store=store,
                dataset=self._dataset,
                splits=self._splits,
                dataset_type=self._config.dataset_type,
                aggregator=self._config.aggregator,
                decoder=self._config.decoder,
                task=self._config.task,
                evaluation=self._config.evaluation,
                training=self._config.training,
                run_dir=layout.run_dir,
                preprocessing=preprocessing,
                heatmaps=self._config.heatmaps,
            )

            if self._config.heatmaps.enabled:
                from soma.heatmaps import render_heatmaps
                logger.info("Rendering attention heatmaps...")
                render_heatmaps(
                    run_dir=layout.run_dir,
                    dataset=self._dataset,
                    tiling_dir=layout.run_dir / "tiling",
                    heatmap_config=self._config.heatmaps,
                    seg_downsample=self._config.preprocessing.seg_downsample,
                )

            try:
                report_path = generate_report_from_result(result, self._config, dataset=self._dataset)
                logger.info("Report saved to %s", report_path)
            except Exception:
                logger.warning("Report generation failed", exc_info=True)

            Console().print(_build_completed_run_panel(summary_metrics=result.summary))
            recorder.complete(result.summary)

        return result

    def _get_feature_store(self, *, run_dir: Path):
        if self._config.dataset_type == "segmentation":
            return self._get_dense_feature_store(run_dir=run_dir)
        if self._feature_dir is not None:
            return FeatureStore(self._feature_dir)

        if self._config.dataset_type == "tile":
            if self._config.encoder is None:
                raise ValueError(
                    "PipelineConfig.encoder is required for dataset_type='tile' "
                    "when feature_dir is not provided."
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
            execution=self._config.execution,
            cache=cache_config,
        )
        try:
            return extractor.run(feature_dir=run_dir / "features")
        finally:
            _release_parent_cuda_state()

    def _get_dense_feature_store(self, *, run_dir: Path):
        from soma.dense import DenseFeatureStore

        # Live re-encode path: no cached grids — hold the frozen encoder + geometry and
        # re-encode (augmented) tiles each step. Built before the fold loop so the
        # backbone loads once and is shared across folds.
        if self._config.feature_mode == "live":
            return self._build_live_segmentation_source()

        if self._feature_dir is not None:
            return DenseFeatureStore(self._feature_dir)

        if self._config.encoder is None:
            raise ValueError(
                "PipelineConfig.encoder is required for dataset_type='segmentation' "
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
                "dataset_type='segmentation' extraction requires "
                "preprocessing.requested_tile_size_px (the mask/tile supervision size) "
                "when feature_dir is not provided."
            )
        if preprocessing.requested_spacing_um is None:
            raise ValueError(
                "dataset_type='segmentation' extraction requires a spacing — set "
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
            return extractor.run(feature_dir=run_dir / "features")
        finally:
            _release_parent_cuda_state()

    def _build_live_segmentation_source(self) -> "LiveSegmentationSource":
        """Load the frozen encoder once and bundle it with geometry/transform for live.

        Mirrors :meth:`DenseTileFeatureExtractor.run`'s encoder construction (same
        ``load_model`` + ``dynamic_img_size`` + dense transform + resolved precision) so
        a live-no-aug run reproduces the cached features exactly. ``feature_dim`` comes
        from a probe forward — the same ``grid.shape[1]`` source the cached extractor
        uses — which also fails fast if the encoder lacks a patch grid.
        """
        import torch as _torch
        from slide2vec.inference import load_model
        from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx

        from soma.dense.geometry import compute_dense_geometry
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

        loaded = load_model(
            name=self._config.encoder.name,
            output_variant=self._config.encoder.output_variant,
            allow_non_recommended_settings=self._config.encoder.allow_non_recommended_settings,
            dynamic_img_size=True,
        )
        encoder = loaded.model
        device = loaded.device
        precision = resolve_encoder_precision(
            self._config.encoder, encoder_name=self._config.encoder.name
        )
        geometry = compute_dense_geometry(
            target_size=int(target_size), patch_size=encoder.patch_size
        )
        window_size = preprocessing.dense_window_size
        overlap = float(preprocessing.dense_window_overlap)
        from soma.dense.sliding import describe_dense_mode, resolve_window_geometry

        # print, not logger: always visible regardless of logging config (same as the
        # cached extractor's announcement) so the resolved mode is never silent.
        print(f"Live segmentation dense mode: {describe_dense_mode(window_size, overlap)}")

        # Probe feature_dim (d) on a single forward (same source of truth as the cached
        # extractor's grids.shape[1]; also a fail-fast dense-capability check). When
        # sliding, probe ONE resolved window rather than the full padded tile: the whole
        # point of a smaller window is to avoid the full-size forward (which can OOM at
        # large scale-ups), and d is spatial-size-independent, so a window probe yields
        # the same channel count.

        probe_h, probe_w = (
            geometry.encoded_size
            if window_size is None
            else resolve_window_geometry(geometry, window_size=window_size, overlap=overlap)[0]
        )
        dummy = _torch.zeros(1, 3, probe_h, probe_w, device=device)
        with _torch.no_grad(), slide_encode_autocast_ctx(device, precision):
            probe = encoder.encode_tiles_dense(dummy)
        if probe.ndim != 4:
            raise ValueError(
                f"encode_tiles_dense returned a {probe.ndim}-D tensor for encoder "
                f"'{self._config.encoder.name}'; expected (B, d, grid_h, grid_w)."
            )
        feature_dim = int(probe.shape[1])

        # Reflect padding (no out-of-distribution constant border), matching the cached
        # extractor's default; image_pad_value is unused for reflect.
        return LiveSegmentationSource(
            encoder=encoder,
            device=device,
            precision=precision,
            geometry=geometry,
            feature_dim=feature_dim,
            dense_transform=encoder.get_dense_transform(),
            augmentation=self._config.augmentation,
            spacing_um=float(preprocessing.requested_spacing_um),
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
            pad_mode="reflect",
            image_pad_value=None,
            window_size=window_size,
            overlap=overlap,
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
    save_probabilities: bool = False,
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
            save_probabilities=save_probabilities,
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
    data = [epoch_log_to_dict(log) for log in history]
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


def _aggregate_fold_metrics(fold_results: list[FoldResult]) -> dict[str, float]:
    """Compute per-split metrics summary. For a single fold, emit values directly.
    For multiple folds, emit mean and std."""
    if not fold_results:
        return {}

    single_fold = len(fold_results) == 1
    summary: dict[str, float] = {}
    test_split_names = sorted({s for fr in fold_results for s in fr.test_reports})

    for split_name in test_split_names:
        split_reports = [fr.test_reports[split_name] for fr in fold_results if split_name in fr.test_reports]
        if not split_reports:
            continue
        metric_keys = list(split_reports[0].metrics.keys())
        for key in metric_keys:
            values = [report.metrics[key] for report in split_reports if key in report.metrics]
            if not values:
                continue
            if single_fold:
                summary[f"{split_name}/{key}"] = float(values[0])
            else:
                summary[f"{split_name}/{key}_mean"] = float(np.mean(values))
                summary[f"{split_name}/{key}_std"] = float(np.std(values))

    return summary


def _save_summary(summary: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2))
