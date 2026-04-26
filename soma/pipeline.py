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
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import functools

import pandas as pd

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

from soma.aggregators.registry import aggregator_registry
from soma.config import (
    AggregatorConfig,
    EncoderConfig,
    EvalConfig,
    HeatmapConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)
from soma.dataset import Dataset, FoldSplit, SampleRecord, Splits
from soma.evaluation.metrics import compute_subgroup_metrics, compute_subgroup_stats, resolve_metrics
from soma.evaluation.metrics import compute_metrics
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
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.collate import bag_collate_fn, hierarchical_bag_collate_fn
from soma.training.model import EmbeddingModel, MILModel
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.sample_dataset import SampleDataset, SampleBatch, sample_collate_fn
from soma.training.seed import seed_everything
from soma.training.trainer import Trainer, TrainResult, epoch_log_to_dict
from soma.reporting import generate_report_from_result


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

    seed_everything(training.seed)

    label_map = dataset.label_map
    feature_dim = feature_store.feature_dim
    all_test_ids = {sid for ids in fold_split.tests.values() for sid in ids}
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

    _fp = f"Fold {fold}" if num_folds > 1 else "Run"

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
    task_params = {**task_cls.auto_params(dataset), **task.params, "metrics": evaluation.metrics}

    # Derive label encoding from the task head class
    label_dtype = task_cls.label_dtype
    if label_dtype == torch.float:
        def label_fn(record):
            return float(record.label)
    else:
        def label_fn(record):
            return label_map[record.label]
    task_family = task_cls.task_family
    resolved_metric_names = resolve_metrics(task_family, evaluation.metrics)
    deterministic_baseline = _build_deterministic_baseline(
        train_records,
        label_fn=label_fn,
        task_family=task_family,
        num_classes=len(label_map),
    )

    if dataset_type == "patient":
        # Patient-level path: pretrained patient encoder produced (D,) per patient
        if aggregator is not None:
            raise ValueError("aggregator must be None for dataset_type='patient'")
        patient_label_map = dataset.patient_label_map
        if label_dtype == torch.float:
            def patient_label_fn(pid, raw):
                return float(raw)
        else:
            def patient_label_fn(pid, raw):
                return label_map[raw]
        _patient_collate = functools.partial(patient_collate_fn, label_dtype=label_dtype)

        train_loader = DataLoader(
            PatientDataset(train_patient_ids, patient_label_map, feature_store, label_map, label_fn=patient_label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_patient_collate,
        )
        tune_loader = DataLoader(
            PatientDataset(tune_patient_ids, patient_label_map, feature_store, label_map, label_fn=patient_label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_patient_collate,
        )
        test_loaders: dict[str, DataLoader] = {
            split_name: DataLoader(
                PatientDataset(
                    test_patient_ids_by_split[split_name],
                    patient_label_map, feature_store, label_map, label_fn=patient_label_fn,
                ),
                batch_size=training.batch_size,
                shuffle=False,
                collate_fn=_patient_collate,
            )
            for split_name, records in test_records_by_split.items()
        }
        head = task_cls(input_dim=feature_dim, **task_params)
        model: torch.nn.Module = EmbeddingModel(task_head=head)
    elif dataset_type == "tile":
        # Tile-dataset path: each sample is a single encoded tile → task head directly
        _sample_collate = functools.partial(sample_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            SampleDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_sample_collate,
        )
        tune_loader = DataLoader(
            SampleDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_sample_collate,
        )
        test_loaders = {
            split_name: DataLoader(
                SampleDataset(records, feature_store, label_map, label_fn=label_fn),
                batch_size=training.batch_size,
                shuffle=False,
                collate_fn=_sample_collate,
            )
            for split_name, records in test_records_by_split.items()
        }
        head = task_cls(input_dim=feature_dim, **task_params)
        model = EmbeddingModel(task_head=head)
    elif feature_store.is_slide_level:
        # Slide-level path: skip aggregator, pass (B, D) directly to task head
        if aggregator is not None:
            msg = "aggregator must be None for slide-level features"
            raise ValueError(msg)
        _sample_collate = functools.partial(sample_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            SampleDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_sample_collate,
        )
        tune_loader = DataLoader(
            SampleDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_sample_collate,
        )
        test_loaders = {
            split_name: DataLoader(
                SampleDataset(records, feature_store, label_map, label_fn=label_fn),
                batch_size=training.batch_size,
                shuffle=False,
                collate_fn=_sample_collate,
            )
            for split_name, records in test_records_by_split.items()
        }
        head = task_cls(input_dim=feature_dim, **task_params)
        model: torch.nn.Module = EmbeddingModel(task_head=head)
    elif feature_store.is_hierarchical:
        if aggregator is None:
            raise ValueError("aggregator must be provided for hierarchical features")
        if aggregator.name != "hipt":
            raise ValueError("hierarchical features require the hipt aggregator")
        if preprocessing is None:
            raise ValueError("hierarchical features require resolved preprocessing")
        hipt_params = _resolve_hipt_params(preprocessing, aggregator)
        _hier_collate = functools.partial(hierarchical_bag_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            HierarchicalBagDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_hier_collate,
        )
        tune_loader = DataLoader(
            HierarchicalBagDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_hier_collate,
        )
        test_loaders = {
            split_name: DataLoader(
                HierarchicalBagDataset(records, feature_store, label_map, label_fn=label_fn),
                batch_size=training.batch_size,
                shuffle=False,
                collate_fn=_hier_collate,
            )
            for split_name, records in test_records_by_split.items()
        }
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=feature_dim, **hipt_params)
        head = task_cls(input_dim=agg.output_dim, **task_params)
        model = MILModel(aggregator=agg, task_head=head)
    else:
        # Tile-level MIL path
        _bag_collate = functools.partial(bag_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            BagDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_bag_collate,
        )
        tune_loader = DataLoader(
            BagDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_bag_collate,
        )
        test_loaders = {
            split_name: DataLoader(
                BagDataset(records, feature_store, label_map, label_fn=label_fn),
                batch_size=training.batch_size,
                shuffle=False,
                collate_fn=_bag_collate,
            )
            for split_name, records in test_records_by_split.items()
        }
        if aggregator is None:
            msg = "aggregator must be provided for tile-level features"
            raise ValueError(msg)
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
        model = MILModel(aggregator=agg, task_head=head)

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        tune_loader=tune_loader,
        config=training,
        fold_dir=fold_dir,
        device=device,
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
        tune_output_sample_ids = (
            tuple(raw_train_patient_ids) if fallback_to_train else tuple(raw_tune_patient_ids)
        )
        tune_placeholder_records = (
            raw_train_placeholder_records if fallback_to_train else _patient_placeholder_records(tune_records)
        )
        tune_empty_sample_ids = (
            empty_eval_sample_ids.get("train", [])
            if fallback_to_train
            else empty_eval_sample_ids.get("tune", [])
        )
    else:
        tune_output_sample_ids = fold_split.train if fallback_to_train else fold_split.tune
        tune_placeholder_records = None
        tune_empty_sample_ids = (
            empty_eval_sample_ids.get("train", [])
            if fallback_to_train
            else empty_eval_sample_ids.get("tune", [])
        )
    tune_report = _evaluate_split_with_placeholders(
        model,
        tune_loader,
        "tune",
        label_map,
        device,
        output_sample_ids=tune_output_sample_ids,
        empty_sample_ids=tune_empty_sample_ids,
        dataset=dataset,
        label_fn=label_fn,
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
            label_map,
            device,
            output_sample_ids=(
                tuple(raw_test_patient_ids_by_split[split_name])
                if dataset_type == "patient"
                else fold_split.tests[split_name]
            ),
            empty_sample_ids=empty_eval_sample_ids.get(split_name, []),
            dataset=dataset,
            label_fn=label_fn,
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
        subgroup_data = _build_subgroup_data(dataset, test_report, evaluation.subgroups.columns)
        _save_predictions(test_report, predictions_path, subgroup_data=subgroup_data)

        # Save subgroup metrics when subgroup columns are configured
        if evaluation.subgroups.columns:
            predictions_df = _build_predictions_df(test_report, subgroup_data)
            sg_metrics = compute_subgroup_metrics(task_family, resolved_metrics, predictions_df, evaluation.subgroups.columns)
            sg_stats = compute_subgroup_stats(task_family, resolved_metrics, predictions_df, evaluation.subgroups.columns)
            sg_out: dict = {"metrics": sg_metrics, "stats": sg_stats}
            (fold_dir / f"subgroup_metrics_{split_name}.json").write_text(json.dumps(sg_out, indent=2))

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

    if dataset_type == "patient":
        splits.validate_no_patient_leakage(dataset)

    single_fold = splits.num_folds == 1
    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        fold_dir = run_dir if single_fold else run_dir / f"fold_{fold_idx}"
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

    # Aggregator
    if aggregator is not None:
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
        self._dataset = Dataset(config.dataset_csv)
        self._splits = Splits(config.splits_csv, self._dataset)
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

        # Save config snapshot
        save_config(self._config, layout.run_dir / "config.yaml")
        started_at = datetime.now().astimezone().isoformat()
        run_metadata = create_run_metadata(
            config=self._config,
            experiment=layout.experiment,
            run_dir=layout.run_dir,
            run_id=layout.run_id,
            status="running",
            started_at=started_at,
        )
        write_run_metadata(layout.run_dir, run_metadata)
        update_run_index(layout.index_dir / "runs.csv", run_metadata)
        update_experiment_index(
            layout.index_dir / "experiments.csv",
            layout.experiment,
            num_runs=count_run_directories(layout.experiment_dir),
            latest_run_id=run_metadata.run_id,
            latest_status=run_metadata.status,
        )

        try:
            # Load feature store
            store = self._get_feature_store(run_dir=layout.run_dir)

            # Print a compact summary of the run configuration before training
            preprocessing = self._resolve_preprocessing()
            Console().print(
                _build_run_summary_panel(
                    encoder=self._config.encoder,
                    preprocessing=preprocessing,
                    aggregator=self._config.aggregator,
                    task=self._config.task,
                    feature_store=store,
                    dataset_type=self._config.dataset_type,
                )
            )

            result = train(
                feature_store=store,
                dataset=self._dataset,
                splits=self._splits,
                dataset_type=self._config.dataset_type,
                aggregator=self._config.aggregator,
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
                report_path = generate_report_from_result(result, self._config)
                logger.info("Report saved to %s", report_path)
            except Exception:
                logger.warning("Report generation failed", exc_info=True)

        except Exception as exc:
            failed_metadata = run_metadata.with_updates(
                status="failed",
                finished_at=datetime.now().astimezone().isoformat(),
                error=str(exc),
            )
            write_run_metadata(layout.run_dir, failed_metadata)
            update_run_index(layout.index_dir / "runs.csv", failed_metadata)
            update_experiment_index(
                layout.index_dir / "experiments.csv",
                layout.experiment,
                num_runs=count_run_directories(layout.experiment_dir),
                latest_run_id=failed_metadata.run_id,
                latest_status=failed_metadata.status,
            )
            if not has_successful_run(layout.experiment_dir):
                update_latest_pointer(layout.experiment_dir, layout.run_dir)
            raise

        completed_metadata = run_metadata.with_updates(
            status="completed",
            finished_at=datetime.now().astimezone().isoformat(),
            summary_metrics=result.summary,
        )
        Console().print(_build_completed_run_panel(summary_metrics=result.summary))
        write_run_metadata(layout.run_dir, completed_metadata)
        update_run_index(layout.index_dir / "runs.csv", completed_metadata)
        update_experiment_index(
            layout.index_dir / "experiments.csv",
            layout.experiment,
            num_runs=count_run_directories(layout.experiment_dir),
            latest_run_id=completed_metadata.run_id,
            latest_status=completed_metadata.status,
        )
        update_latest_pointer(layout.experiment_dir, layout.run_dir)

        return result

    def _get_feature_store(self, *, run_dir: Path) -> FeatureStore:
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
    label_map: dict[str | int, int],
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
    all_labels = []
    all_sample_ids: list[str] = []

    for batch in loader:
        features = batch.features.to(device)
        if hasattr(batch, "mask"):
            out = model(features, mask=batch.mask.to(device))
        else:
            out = model(features)
        all_logits.append(out.logits.cpu())
        all_labels.append(batch.labels)
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
    labels = torch.cat(all_labels, dim=0)

    task_head = model.task_head
    metrics = task_head.compute_metrics(logits, labels)
    processed = task_head.postprocess(logits)
    y_true = labels.numpy()

    predictions = []
    for i, sid in enumerate(all_sample_ids):
        if "probabilities" in processed:
            # Classification
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
    label_fn,
    task_family: str,
    num_classes: int,
) -> _DeterministicBaseline:
    if not train_records:
        raise ValueError("Cannot build a deterministic baseline without training records")

    if task_family in {"binary_classification", "multiclass_classification"}:
        labels = np.asarray([int(label_fn(record)) for record in train_records], dtype=int)
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        probabilities = (counts / counts.sum()).tolist()
        return _DeterministicBaseline(
            task_family=task_family,
            probabilities=probabilities,
            predicted_label=int(np.argmax(probabilities)),
        )

    if task_family == "ordinal_classification":
        raw_score = float(np.mean([int(label_fn(record)) for record in train_records]))
        predicted_label = int(np.clip(np.rint(raw_score), 0, max(num_classes - 1, 0)))
        return _DeterministicBaseline(
            task_family=task_family,
            predicted_label=predicted_label,
            raw_score=raw_score,
        )

    predicted_value = float(np.mean([float(label_fn(record)) for record in train_records]))
    return _DeterministicBaseline(
        task_family=task_family,
        predicted_value=predicted_value,
    )


def _make_placeholder_prediction(
    record: SampleRecord,
    *,
    label_fn,
    baseline: _DeterministicBaseline,
    sample_id: str | None = None,
) -> SamplePrediction:
    true_label = label_fn(record)
    true_value: int | float
    if baseline.task_family == "regression":
        true_value = float(true_label)
    else:
        true_value = int(true_label)

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

    if task_family in {"binary_classification", "multiclass_classification"}:
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
    label_map: dict[str | int, int],
    device: torch.device,
    *,
    output_sample_ids: tuple[str, ...],
    empty_sample_ids: list[str],
    dataset: Dataset,
    label_fn,
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
            label_map,
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
            label_fn=label_fn,
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
    """Build a mapping of sample_id → subgroup column values for enriching predictions."""
    if not subgroup_columns:
        return {}
    subgroup_data: dict[str, dict[str, object]] = {}
    patient_groups = dataset.patient_groups if dataset.has_patient_ids else {}
    for pred in report.predictions:
        if pred.sample_id in dataset.samples:
            subgroup_data[pred.sample_id] = {
                col: dataset.samples[pred.sample_id].metadata.get(col)
                for col in subgroup_columns
            }
            continue

        patient_records = patient_groups.get(pred.sample_id)
        if not patient_records:
            subgroup_data[pred.sample_id] = {}
            continue

        values: dict[str, object] = {}
        for col in subgroup_columns:
            observed_values = {record.metadata.get(col) for record in patient_records}
            if len(observed_values) > 1:
                raise ValueError(
                    f"Patient '{pred.sample_id}' has inconsistent subgroup metadata for column '{col}'."
                )
            values[col] = patient_records[0].metadata.get(col)
        subgroup_data[pred.sample_id] = values

    return subgroup_data


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

        if first.probabilities is not None:
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
