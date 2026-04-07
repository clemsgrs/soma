"""Pipeline — standalone step functions and Pipeline orchestrator.

Layer 1 (standalone):
    train_one_fold()  — train + evaluate one fold
    train()           — train all folds + summarize

Layer 2 (orchestrator):
    Pipeline          — wires everything together from a PipelineConfig
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from soma.aggregators.registry import aggregator_registry
from soma.config import (
    AggregatorConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)
from soma.dataset import Dataset, FoldSplit, Splits
from soma.evaluation.metrics import compute_classification_metrics
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.features import FeatureStore
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator
from soma.tasks.registry import task_registry
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.collate import bag_collate_fn, hierarchical_bag_collate_fn
from soma.training.model import MILModel
from soma.training.seed import seed_everything
from soma.training.slide_dataset import SlideDataset, slide_collate_fn
from soma.training.slide_model import SlideModel
from soma.training.trainer import Trainer, TrainResult


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
    output_dir: Path


# ---------------------------------------------------------------------------
# Layer 1 — Standalone step functions
# ---------------------------------------------------------------------------


def train_one_fold(
    feature_store: FeatureStore,
    dataset: Dataset,
    fold_split: FoldSplit,
    aggregator: AggregatorConfig | None,
    task: TaskConfig,
    training: TrainingConfig,
    output_dir: str | Path,
    *,
    fold: int = 0,
    preprocessing: PreprocessingConfig | None = None,
) -> FoldResult:
    """Train and evaluate a single fold.

    Args:
        feature_store: Precomputed embeddings (tile-level or slide-level).
        dataset: Dataset with sample records and label_map.
        fold_split: Train/tune/test sample IDs for this fold.
        aggregator: Aggregator configuration, or None for slide-level features.
        task: Task head configuration.
        training: Training loop configuration.
        output_dir: Directory for checkpoint, metrics, predictions.
        fold: Fold index (for FoldResult metadata).

    Returns:
        FoldResult with training result + tune/test evaluation reports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(training.seed)

    label_map = dataset.label_map
    feature_dim = feature_store.feature_dim

    # Build datasets
    train_records = [dataset.samples[sid] for sid in fold_split.train]
    tune_records = [dataset.samples[sid] for sid in fold_split.tune]
    test_records = [dataset.samples[sid] for sid in fold_split.test]

    task_cls = task_registry.get(task.name)
    task_params = {"num_classes": dataset.num_classes, **task.params}

    if feature_store.is_slide_level:
        # Slide-level path: skip aggregator, pass (B, D) directly to task head
        if aggregator is not None:
            msg = "aggregator must be None for slide-level features"
            raise ValueError(msg)
        train_loader = DataLoader(
            SlideDataset(train_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=slide_collate_fn,
        )
        tune_loader = DataLoader(
            SlideDataset(tune_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=slide_collate_fn,
        )
        test_loader = DataLoader(
            SlideDataset(test_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=slide_collate_fn,
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
        train_loader = DataLoader(
            HierarchicalBagDataset(train_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=hierarchical_bag_collate_fn,
        )
        tune_loader = DataLoader(
            HierarchicalBagDataset(tune_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=hierarchical_bag_collate_fn,
        )
        test_loader = DataLoader(
            HierarchicalBagDataset(test_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=hierarchical_bag_collate_fn,
        )
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=feature_dim, **hipt_params)
        head = task_cls(input_dim=agg.output_dim, **task_params)
        model = MILModel(aggregator=agg, task_head=head)
    else:
        # Tile-level MIL path
        train_loader = DataLoader(
            BagDataset(train_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=True,
            collate_fn=bag_collate_fn,
        )
        tune_loader = DataLoader(
            BagDataset(tune_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=bag_collate_fn,
        )
        test_loader = DataLoader(
            BagDataset(test_records, feature_store, label_map),
            batch_size=training.batch_size,
            shuffle=False,
            collate_fn=bag_collate_fn,
        )
        if aggregator is None:
            msg = "aggregator must be provided for tile-level features"
            raise ValueError(msg)
        aggregator_cls = aggregator_registry.get(aggregator.name)
        agg = aggregator_cls(input_dim=feature_dim, **aggregator.params)
        head = task_cls(input_dim=agg.output_dim, **task_params)
        model = MILModel(aggregator=agg, task_head=head)

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=tune_loader,
        config=training,
        output_dir=output_dir,
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

    tune_report = _evaluate(model, tune_loader, "tune", label_map, device)
    test_report = _evaluate(model, test_loader, "test", label_map, device)

    # Save metrics and predictions
    _save_metrics(tune_report, test_report, output_dir / "metrics.json")
    _save_predictions(test_report, output_dir / "predictions.csv")

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
    aggregator: AggregatorConfig | None,
    task: TaskConfig,
    training: TrainingConfig,
    output_dir: str | Path,
    preprocessing: PreprocessingConfig | None = None,
) -> PipelineResult:
    """Train and evaluate all folds, then summarize.

    Args:
        feature_store: Precomputed embeddings (tile-level or slide-level).
        dataset: Dataset with sample records and label_map.
        splits: Cross-validation splits (1 or more folds).
        aggregator: Aggregator configuration, or None for slide-level features.
        task: Task head configuration.
        training: Training loop configuration.
        output_dir: Root directory — each fold gets a fold_N/ subdirectory.

    Returns:
        PipelineResult with per-fold results and aggregated summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_idx, fold_split in enumerate(splits.folds):
        result = train_one_fold(
            feature_store=feature_store,
            dataset=dataset,
            fold_split=fold_split,
            aggregator=aggregator,
            task=task,
            training=training,
            output_dir=output_dir / f"fold_{fold_idx}",
            fold=fold_idx,
            preprocessing=preprocessing,
        )
        fold_results.append(result)

    summary = _aggregate_fold_metrics(fold_results)
    _save_summary(summary, output_dir / "summary.json")

    return PipelineResult(
        fold_results=fold_results,
        summary=summary,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Layer 2 — Pipeline orchestrator
# ---------------------------------------------------------------------------


def _resolve_hipt_params(preprocessing: PreprocessingConfig, aggregator: AggregatorConfig) -> dict[str, object]:
    patch_size = preprocessing.target_tile_size_px or preprocessing.effective_tile_size_px
    region_size = preprocessing.target_region_size_px or preprocessing.effective_region_size_px
    tile_multiple = preprocessing.region_tile_multiple
    if patch_size is None or region_size is None:
        raise ValueError("hierarchical preprocessing must resolve patch and region sizes")
    patch_size = int(patch_size)
    region_size = int(region_size)
    if tile_multiple is None:
        if region_size % patch_size != 0:
            raise ValueError(
                "hierarchical preprocessing requires target_region_size_px to be divisible by target_tile_size_px"
            )
        tile_multiple = region_size // patch_size
    tile_multiple = int(tile_multiple)
    if region_size != patch_size * tile_multiple:
        raise ValueError(
            "hierarchical preprocessing requires target_region_size_px to equal "
            "target_tile_size_px × region_tile_multiple"
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
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        save_config(self._config, output_dir / "config.yaml")

        # Load feature store
        store = self._get_feature_store()

        return train(
            feature_store=store,
            dataset=self._dataset,
            splits=self._splits,
            aggregator=self._config.aggregator,
            task=self._config.task,
            training=self._config.training,
            output_dir=output_dir,
            preprocessing=self._resolve_preprocessing(),
        )

    def _get_feature_store(self) -> FeatureStore:
        if self._feature_dir is not None:
            store = FeatureStore(self._feature_dir)
        else:
            from soma.extraction import FeatureExtractor

            preprocessing = self._resolve_preprocessing()

            cache_config = self._config.cache
            if cache_config.root_dir is None:
                cache_config = replace(
                    cache_config,
                    root_dir=Path(self._config.output_dir).parent / "feature_cache",
                )
            extractor = FeatureExtractor(
                self._dataset,
                self._config.encoder,
                preprocessing,
                cache=cache_config,
            )
            store = extractor.run(Path(self._config.output_dir) / "features")
        store.validate_coverage(self._dataset.sample_ids)
        return store

    def _resolve_preprocessing(self) -> "PreprocessingConfig":
        """Resolve preprocessing config, injecting HIPT-specific overrides if needed."""
        return derive_preprocessing_for_aggregator(
            self._config.preprocessing,
            self._config.aggregator,
        )


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
) -> EvaluationReport:
    """Evaluate model on a split and build a report."""
    # Invert label_map for decoding
    idx_to_label = {v: k for k, v in label_map.items()}

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

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = labels.numpy()

    metrics = compute_classification_metrics(y_true, probs, preds)

    predictions = []
    for i, sid in enumerate(all_sample_ids):
        predictions.append(
            SamplePrediction(
                sample_id=sid,
                true_label=int(y_true[i]),
                predicted_label=int(preds[i]),
                probabilities=probs[i].tolist(),
            )
        )

    return EvaluationReport(
        split=split_name,
        metrics=metrics,
        predictions=predictions,
    )


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


def _save_predictions(report: EvaluationReport, path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        if not report.predictions:
            return
        num_classes = len(report.predictions[0].probabilities)
        fieldnames = ["sample_id", "true_label", "predicted_label"] + [
            f"prob_{i}" for i in range(num_classes)
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in report.predictions:
            row = {
                "sample_id": pred.sample_id,
                "true_label": pred.true_label,
                "predicted_label": pred.predicted_label,
            }
            for i, p in enumerate(pred.probabilities):
                row[f"prob_{i}"] = f"{p:.6f}"
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
