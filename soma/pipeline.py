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
from dataclasses import asdict, dataclass, replace
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
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.extraction import FeatureExtractor
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
from soma.training.model import MILModel
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.patient_model import PatientModel
from soma.training.seed import seed_everything
from soma.training.slide_dataset import SlideDataset, slide_collate_fn
from soma.training.slide_model import SlideModel
from soma.training.tile_dataset import TileDataset, tile_collate_fn
from soma.training.tile_model import TileClassifier
from soma.training.trainer import Trainer, TrainResult, _epoch_log_to_dict
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
    test_report: EvaluationReport


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run across all folds."""

    fold_results: list[FoldResult]
    summary: dict[str, float]
    run_dir: Path


def _format_fold_summary(
    fold: int,
    train_count: int,
    tune_count: int,
    test_count: int,
    dropped_by_split: dict[str, list[str]] | None = None,
) -> str:
    base = f"Fold {fold}: train={train_count} tune={tune_count} test={test_count}"
    if not dropped_by_split:
        return base

    dropped_counts = {split: len(sample_ids) for split, sample_ids in dropped_by_split.items()}
    if not any(dropped_counts.values()):
        return base

    dropped_text = ", ".join(f"{split}={count}" for split, count in dropped_counts.items())
    return f"{base} | dropped empty: {dropped_text}"


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
    eval: EvalConfig | None = None,
    aggregator: AggregatorConfig | None = None,
    fold: int = 0,
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
        eval: Evaluation configuration (metrics, subgroups). Defaults to EvalConfig().
        training: Training loop configuration.
        fold_dir: Directory for checkpoint, metrics, predictions.
        fold: Fold index (for FoldResult metadata).
        heatmaps: When provided and enabled, attention scores are saved during
            the test evaluation pass (no separate inference pass needed).

    Returns:
        FoldResult with training result + tune/test evaluation reports.
    """
    eval = eval or EvalConfig()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(training.seed)

    label_map = dataset.label_map
    feature_dim = feature_store.feature_dim
    if dataset_type == "patient":
        # Patient-level: features are indexed by patient_id; splits are slide-based.
        # Skip the sample-level manifest check and build records directly from splits.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records = [dataset.samples[sid] for sid in fold_split.test]
        dropped_by_split = None
    elif feature_store.has_feature_manifest:
        manifest_statuses = feature_store.feature_statuses
        manifest_sample_ids = set(manifest_statuses)
        split_sample_ids = set(fold_split.train) | set(fold_split.tune) | set(fold_split.test)
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

        dropped_by_split = {
            "train": [sid for sid in fold_split.train if sid in empty_sample_ids],
            "tune": [sid for sid in fold_split.tune if sid in empty_sample_ids],
            "test": [sid for sid in fold_split.test if sid in empty_sample_ids],
        }

        # Build datasets from samples that are expected to have features.
        train_records = _records_for_sample_ids(dataset, fold_split.train, expected_feature_ids)
        tune_records = _records_for_sample_ids(dataset, fold_split.tune, expected_feature_ids)
        test_records = _records_for_sample_ids(dataset, fold_split.test, expected_feature_ids)
    else:
        # Legacy path: no manifest, so every listed sample must have a feature file.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records = [dataset.samples[sid] for sid in fold_split.test]
        dropped_by_split = None

    if not train_records:
        msg = f"Fold {fold} has no training samples with available features"
        raise ValueError(msg)
    if not tune_records:
        msg = f"Fold {fold} has no tuning samples with available features"
        raise ValueError(msg)
    if not test_records:
        msg = f"Fold {fold} has no test samples with available features"
        raise ValueError(msg)

    summary = _format_fold_summary(
        fold=fold,
        train_count=len(train_records),
        tune_count=len(tune_records),
        test_count=len(test_records),
        dropped_by_split=dropped_by_split,
    )
    logger.info(summary)

    # Validate subgroup columns exist in dataset metadata
    if eval.subgroups.columns:
        sample = next(iter(dataset.samples.values()))
        missing = [c for c in eval.subgroups.columns if c not in sample.metadata]
        if missing:
            raise ValueError(
                f"Subgroup column(s) not found in dataset metadata: {missing}. "
                f"Available: {sorted(sample.metadata)}"
            )

    task_cls = task_registry.get(task.name)
    task_params = {**task_cls.auto_params(dataset), **task.params, "metrics": eval.metrics}

    # Derive label encoding from the task head class
    label_dtype = task_cls.label_dtype
    if label_dtype == torch.float:
        label_fn = lambda record: float(record.label)  # noqa: E731
    else:
        label_fn = lambda record: label_map[record.label]  # noqa: E731

    if dataset_type == "patient":
        # Patient-level path: pretrained patient encoder produced (D,) per patient
        if aggregator is not None:
            raise ValueError("aggregator must be None for dataset_type='patient'")
        patient_label_map = dataset.patient_label_map
        if label_dtype == torch.float:
            patient_label_fn = lambda pid, raw: float(raw)  # noqa: E731
        else:
            patient_label_fn = lambda pid, raw: label_map[raw]  # noqa: E731
        _patient_collate = functools.partial(patient_collate_fn, label_dtype=label_dtype)

        def _patient_ids_for_records(records: list[SampleRecord]) -> list[str]:
            seen: set[str] = set()
            ids: list[str] = []
            for r in records:
                if r.patient_id is None:
                    raise ValueError(
                        f"Sample '{r.sample_id}' has no patient_id. "
                        "All samples must have a patient_id for dataset_type='patient'."
                    )
                if r.patient_id not in seen:
                    seen.add(r.patient_id)
                    ids.append(r.patient_id)
            return ids

        train_patient_ids = _patient_ids_for_records(train_records)
        tune_patient_ids = _patient_ids_for_records(tune_records)
        test_patient_ids = _patient_ids_for_records(test_records)
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
        test_loader = DataLoader(
            PatientDataset(test_patient_ids, patient_label_map, feature_store, label_map, label_fn=patient_label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_patient_collate,
        )
        head = task_cls(input_dim=feature_dim, **task_params)
        model: torch.nn.Module = PatientModel(task_head=head)
    elif dataset_type == "tile":
        # Tile-dataset path: each sample is a single encoded tile → task head directly
        _tile_collate = functools.partial(tile_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            TileDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_tile_collate,
        )
        tune_loader = DataLoader(
            TileDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_tile_collate,
        )
        test_loader = DataLoader(
            TileDataset(test_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_tile_collate,
        )
        head = task_cls(input_dim=feature_dim, **task_params)
        model = TileClassifier(task_head=head)
    elif feature_store.is_slide_level:
        # Slide-level path: skip aggregator, pass (B, D) directly to task head
        if aggregator is not None:
            msg = "aggregator must be None for slide-level features"
            raise ValueError(msg)
        _slide_collate = functools.partial(slide_collate_fn, label_dtype=label_dtype)
        train_loader = DataLoader(
            SlideDataset(train_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=_slide_collate,
        )
        tune_loader = DataLoader(
            SlideDataset(tune_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_slide_collate,
        )
        test_loader = DataLoader(
            SlideDataset(test_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_slide_collate,
        )
        head = task_cls(input_dim=feature_dim, **task_params)
        model: torch.nn.Module = SlideModel(task_head=head)
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
        test_loader = DataLoader(
            HierarchicalBagDataset(test_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_hier_collate,
        )
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
        test_loader = DataLoader(
            BagDataset(test_records, feature_store, label_map, label_fn=label_fn),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=_bag_collate,
        )
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
    attention_dir = fold_dir / "attention" if save_attention else None
    if attention_dir is not None:
        attention_dir.mkdir(exist_ok=True)

    tune_report = _evaluate(model, tune_loader, "tune", label_map, device)
    test_report = _evaluate(
        model, test_loader, "test", label_map, device,
        attention_dir=attention_dir,
        aggregator_name=aggregator.name if aggregator is not None else None,
    )

    # Save metrics, predictions, and training history
    _save_metrics(tune_report, test_report, fold_dir / "metrics.json")
    subgroup_data = _build_subgroup_data(dataset, test_report, eval.subgroups.columns)
    _save_predictions(test_report, fold_dir / "predictions.csv", subgroup_data=subgroup_data)
    _save_training_history(train_result.history, fold_dir / "training_history.json")

    # Save subgroup metrics when subgroup columns are configured
    if eval.subgroups.columns:
        predictions_df = pd.read_csv(fold_dir / "predictions.csv")
        task_family = task.name
        resolved_metrics = resolve_metrics(task_family, eval.metrics)
        sg_metrics = compute_subgroup_metrics(task_family, resolved_metrics, predictions_df, eval.subgroups.columns)
        sg_stats = compute_subgroup_stats(task_family, resolved_metrics, predictions_df, eval.subgroups.columns)
        sg_out: dict = {"metrics": sg_metrics, "stats": sg_stats}
        (fold_dir / "subgroup_metrics.json").write_text(json.dumps(sg_out, indent=2))

    return FoldResult(
        fold=fold,
        train_result=train_result,
        tune_report=tune_report,
        test_report=test_report,
    )


def train(
    feature_store: FeatureStore,
    dataset: Dataset,
    splits: Splits,
    task: TaskConfig,
    training: TrainingConfig,
    run_dir: str | Path,
    aggregator: AggregatorConfig | None = None,
    eval: EvalConfig | None = None,
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
        eval: Evaluation configuration (metrics, subgroups). Defaults to EvalConfig().
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

    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        result = train_one_fold(
            feature_store=feature_store,
            dataset=dataset,
            fold_split=fold_split,
            dataset_type=dataset_type,
            aggregator=aggregator,
            task=task,
            eval=eval,
            training=training,
            fold_dir=run_dir / f"fold_{fold_idx}",
            fold=fold_idx,
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
        spacing: float | None = encoder.spacing_um if encoder is not None else None
        if spacing is None:
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
                eval=self._config.eval,
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
                    feature_store=store,
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
                cache=cache_config,
            )
            return extractor.run(feature_dir=run_dir / "features")

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
            cache=cache_config,
        )
        return extractor.run(feature_dir=run_dir / "features")

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
            from soma.heatmaps import _normalize_attention
            for i, sid in enumerate(batch.sample_ids):
                attn_i = out.tile_attention[i : i + 1]
                normalized = _normalize_attention(attn_i, aggregator_name or "")
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


def _save_metrics(
    tune_report: EvaluationReport,
    test_report: EvaluationReport,
    path: Path,
) -> None:
    data = {
        "tune": tune_report.metrics,
        "test": test_report.metrics,
    }
    path.write_text(json.dumps(data, indent=2))


def _save_training_history(history: list, path: Path) -> None:
    data = [_epoch_log_to_dict(log) for log in history]
    path.write_text(json.dumps(data, indent=2))


def _build_subgroup_data(
    dataset: Dataset,
    report: EvaluationReport,
    subgroup_columns: list[str],
) -> dict[str, dict[str, object]]:
    """Build a mapping of sample_id → subgroup column values for enriching predictions."""
    if not subgroup_columns:
        return {}
    return {
        pred.sample_id: {
            col: dataset.samples[pred.sample_id].metadata.get(col)
            for col in subgroup_columns
            if pred.sample_id in dataset.samples
        }
        for pred in report.predictions
    }


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

        if first.probabilities is not None:
            # Classification: include class probabilities
            num_classes = len(first.probabilities)
            fieldnames = ["sample_id", "true_label", "predicted_label"] + [
                f"prob_{i}" for i in range(num_classes)
            ] + extra_cols
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
                if subgroup_data and pred.sample_id in subgroup_data:
                    row.update(subgroup_data[pred.sample_id])
                writer.writerow(row)
        elif first.raw_score is not None:
            # Ordinal classification: integer prediction + raw continuous score
            fieldnames = ["sample_id", "true_label", "predicted_label", "raw_score"] + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "predicted_label": pred.predicted_label,
                    "raw_score": f"{pred.raw_score:.6f}",
                }
                if subgroup_data and pred.sample_id in subgroup_data:
                    row.update(subgroup_data[pred.sample_id])
                writer.writerow(row)
        else:
            # Regression: predicted continuous value
            fieldnames = ["sample_id", "true_label", "predicted_value"] + extra_cols
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pred in report.predictions:
                row = {
                    "sample_id": pred.sample_id,
                    "true_label": pred.true_label,
                    "predicted_value": f"{pred.predicted_value:.6f}",
                }
                if subgroup_data and pred.sample_id in subgroup_data:
                    row.update(subgroup_data[pred.sample_id])
                writer.writerow(row)


def _aggregate_fold_metrics(fold_results: list[FoldResult]) -> dict[str, float]:
    """Compute mean and std of each metric across folds."""
    if not fold_results:
        return {}

    # Collect all metric keys from test reports
    metric_keys = list(fold_results[0].test_report.metrics.keys())
    summary: dict[str, float] = {}

    for key in metric_keys:
        values = [fr.test_report.metrics[key] for fr in fold_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    return summary


def _save_summary(summary: dict[str, float], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2))
