"""Tests for soma.pipeline — train_one_fold, train, and Pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import pytest

from soma.config import (
    AggregatorConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, FoldSplit, Splits
from soma.features import FeatureStore
from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_SAMPLES = 8


def _setup_synthetic_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset.csv, splits.csv, and feature .pt files.

    Returns (dataset_csv, splits_csv, feature_dir).
    """
    # Dataset CSV
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [f"/slides/s{i}.svs" for i in range(NUM_SAMPLES)],
            "label": ["tumor" if i % 2 == 0 else "normal" for i in range(NUM_SAMPLES)],
        }
    ).to_csv(dataset_csv, index=False)

    # Splits CSV — single fold: 6 train, 1 tune, 1 test
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * NUM_SAMPLES,
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "split": ["train"] * 6 + ["tune", "test"],
        }
    ).to_csv(splits_csv, index=False)

    # Feature files
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(42)
    for i in range(NUM_SAMPLES):
        n_tiles = 5 + i
        torch.save(torch.randn(n_tiles, D), feature_dir / f"s{i}.pt")

    return dataset_csv, splits_csv, feature_dir


def _setup_multifold_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset with 2 folds."""
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [f"/slides/s{i}.svs" for i in range(NUM_SAMPLES)],
            "label": ["tumor" if i % 2 == 0 else "normal" for i in range(NUM_SAMPLES)],
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    rows = []
    for fold in [0, 1]:
        for i in range(NUM_SAMPLES):
            if fold == 0:
                split = "train" if i < 6 else ("tune" if i == 6 else "test")
            else:
                split = "train" if i >= 2 else ("tune" if i == 0 else "test")
            rows.append({"fold": fold, "sample_id": f"s{i}", "split": split})
    pd.DataFrame(rows).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(42)
    for i in range(NUM_SAMPLES):
        torch.save(torch.randn(5 + i, D), feature_dir / f"s{i}.pt")

    return dataset_csv, splits_csv, feature_dir


# ---------------------------------------------------------------------------
# train_one_fold (Layer 1 — standalone)
# ---------------------------------------------------------------------------


class TestTrainOneFold:
    def test_returns_fold_result(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, learning_rate=1e-3, patience=10, batch_size=2),
            output_dir=tmp_path / "fold_0",
        )

        assert isinstance(result, FoldResult)
        assert result.fold == 0
        assert result.train_result is not None
        assert result.tune_report.split == "tune"
        assert result.test_report.split == "test"

    def test_saves_checkpoint(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        assert (fold_dir / "best_model.pt").exists()

    def test_saves_metrics_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        metrics_path = fold_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "test" in metrics
        assert "accuracy" in metrics["test"]

    def test_saves_predictions_csv(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        preds_path = fold_dir / "predictions.csv"
        assert preds_path.exists()
        preds_df = pd.read_csv(preds_path)
        assert "sample_id" in preds_df.columns
        assert "true_label" in preds_df.columns
        assert "predicted_label" in preds_df.columns

    def test_num_classes_auto_inferred(self, tmp_path: Path):
        """num_classes should be auto-inferred from dataset labels."""
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),  # no num_classes specified
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=tmp_path / "fold_0",
        )

        # Should have 2-class probabilities (tumor/normal)
        assert len(result.test_report.predictions[0].probabilities) == 2


# ---------------------------------------------------------------------------
# train (Layer 1 — all folds)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_single_fold_returns_pipeline_result(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].fold == 0

    def test_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        result = train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        assert len(result.fold_results) == 2
        assert result.fold_results[0].fold == 0
        assert result.fold_results[1].fold == 1
        assert "accuracy_mean" in result.summary
        assert "accuracy_std" in result.summary

    def test_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        summary = json.loads((output_dir / "summary.json").read_text())
        assert "accuracy_mean" in summary

    def test_fold_subdirectories(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        assert (output_dir / "fold_0" / "best_model.pt").exists()
        assert (output_dir / "fold_1" / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Pipeline (Layer 2 — orchestrator)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_run_single_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)
        result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.output_dir == output_dir

    def test_run_saves_config_yaml(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        Pipeline(config, feature_dir=feature_dir).run()

        assert (output_dir / "config.yaml").exists()

    def test_run_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        Pipeline(config, feature_dir=feature_dir).run()

        summary_path = output_dir / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "accuracy_mean" in summary

    def test_run_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        result = Pipeline(config, feature_dir=feature_dir).run()

        assert len(result.fold_results) == 2
        assert (output_dir / "fold_0" / "best_model.pt").exists()
        assert (output_dir / "fold_1" / "best_model.pt").exists()

        summary = json.loads((output_dir / "summary.json").read_text())
        assert "accuracy_mean" in summary
        assert "accuracy_std" in summary

    def test_dataset_and_splits_properties(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "output",
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)

        assert pipeline.dataset.num_classes == 2
        assert pipeline.splits.num_folds == 1
