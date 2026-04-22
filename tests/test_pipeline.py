"""Tests for soma.pipeline — train_one_fold, train, and Pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch
import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest
import yaml
from rich.console import Console

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    PreprocessingConfig as _PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, FoldSplit, Splits
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.features import FeatureStore
from soma.output_layout import build_experiment_spec
from soma.pipeline import (
    FoldResult,
    Pipeline,
    PipelineResult,
    _aggregate_fold_metrics,
    _build_subgroup_data,
    _build_completed_run_panel,
    _evaluate,
    train,
    train_one_fold,
)
from soma.training.collate import BagBatch
from soma.training.trainer import TrainResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_SAMPLES = 8
FIXED_RUN_ID = "2026-04-09_16-22-10__local"
_TISSUE_METHOD_SENTINEL = object()


def PreprocessingConfig(*args, tissue_method=_TISSUE_METHOD_SENTINEL, **kwargs):
    if tissue_method is _TISSUE_METHOD_SENTINEL:
        kwargs.setdefault("tissue_method", "hsv")
    else:
        kwargs["tissue_method"] = tissue_method
    return _PreprocessingConfig(*args, **kwargs)


def _expected_run_dir(config: PipelineConfig) -> Path:
    experiment = build_experiment_spec(config)
    return Path(config.output_root) / "experiments" / experiment.experiment_dirname / "runs" / FIXED_RUN_ID


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


def _setup_train_test_only_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset.csv, splits.csv, and feature files with no tune split."""
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [f"/slides/s{i}.svs" for i in range(NUM_SAMPLES)],
            "label": ["tumor" if i % 2 == 0 else "normal" for i in range(NUM_SAMPLES)],
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * NUM_SAMPLES,
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "split": ["train"] * 6 + ["test"] * 2,
        }
    ).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(42)
    for i in range(NUM_SAMPLES):
        n_tiles = 5 + i
        torch.save(torch.randn(n_tiles, D), feature_dir / f"s{i}.pt")

    return dataset_csv, splits_csv, feature_dir


def _write_feature_manifest(feature_dir: Path, statuses: dict[str, str]) -> None:
    rows = []
    for sample_id, status in statuses.items():
        rows.append(
            {
                "sample_id": sample_id,
                "feature_status": status,
                "feature_path": str((feature_dir / f"{sample_id}.pt").resolve()) if status == "success" else "",
                "num_tiles": 1 if status == "success" else 0,
                "feature_rank": 2,
                "feature_dim": D,
            }
        )
    pd.DataFrame(rows).to_csv(feature_dir / "process_list.csv", index=False)


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


def _setup_hierarchical_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a small dataset with hierarchical (3-D) feature tensors."""
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"/slides/s{i}.svs" for i in range(4)],
            "label": ["tumor", "normal", "tumor", "normal"],
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * 4,
            "sample_id": ["s0", "s1", "s2", "s3"],
            "split": ["train", "train", "tune", "test"],
        }
    ).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "hier_features"
    feature_dir.mkdir()
    torch.manual_seed(21)
    torch.save(torch.randn(2, 4, D), feature_dir / "s0.pt")
    torch.save(torch.randn(3, 4, D), feature_dir / "s1.pt")
    torch.save(torch.randn(1, 4, D), feature_dir / "s2.pt")
    torch.save(torch.randn(2, 4, D), feature_dir / "s3.pt")

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
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, learning_rate=1e-3, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_0",
        )

        assert isinstance(result, FoldResult)
        assert result.fold == 0
        assert result.train_result is not None
        assert result.tune_report.split == "tune"
        assert result.test_reports["test"].split == "test"

    def test_ignores_samples_without_features(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        (feature_dir / "s5.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "empty",
                "s6": "success",
                "s7": "success",
            },
        )
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_missing_feature"

        caplog.set_level(logging.INFO, logger="soma.pipeline")
        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
        )

        assert isinstance(result, FoldResult)
        assert (fold_dir / "best_model.pt").exists()
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            message == "Fold 0: train=5 tune=1 test=1 | train empty dropped=1"
            for message in messages
        )
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_errors_when_expected_feature_is_missing(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        (feature_dir / "s5.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "success",
                "s6": "success",
                "s7": "success",
            },
        )
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        with pytest.raises(ValueError, match="s5"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                aggregator=AggregatorConfig(name="mean_pool"),
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                fold_dir=tmp_path / "fold_missing_feature",
            )

    def test_requires_tune_split_by_default_when_missing(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_train_test_only_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        with pytest.raises(ValueError, match="no tuning samples"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                aggregator=AggregatorConfig(name="mean_pool"),
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                fold_dir=tmp_path / "fold_missing_tune",
            )

    def test_allow_missing_tune_uses_train_split_as_tune_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        dataset_csv, splits_csv, feature_dir = _setup_train_test_only_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        caplog.set_level(logging.WARNING, logger="soma.pipeline")

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(
                epochs=2,
                patience=10,
                batch_size=2,
                allow_missing_tune=True,
            ),
            fold_dir=tmp_path / "fold_allow_missing_tune",
        )

        assert [pred.sample_id for pred in result.tune_report.predictions] == [
            f"s{i}" for i in range(6)
        ]
        assert result.tune_report.split == "tune"
        assert result.tune_report.metrics["num_samples"] == 6
        assert any(
            record.levelno == logging.WARNING
            and record.getMessage()
            == "Run has no tuning samples with available features; using train split as tune because allow_missing_tune=True"
            for record in caplog.records
        )

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
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
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
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
        )

        metrics_path = fold_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "test" in metrics
        assert "auroc" in metrics["test"]

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
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
        )

        preds_path = fold_dir / "predictions_test.csv"
        assert preds_path.exists()
        preds_df = pd.read_csv(preds_path)
        assert "sample_id" in preds_df.columns
        assert "true_label" in preds_df.columns
        assert "predicted_label" in preds_df.columns

    def test_empty_tune_and_test_samples_use_deterministic_baselines(
        self, tmp_path: Path
    ):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        for sample_id in ("s6", "s7"):
            (feature_dir / f"{sample_id}.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "success",
                "s6": "empty",
                "s7": "empty",
            },
        )
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_empty_eval"

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
        )

        tune_predictions = result.tune_report.predictions
        assert [pred.sample_id for pred in tune_predictions] == ["s6"]
        assert tune_predictions[0].probabilities == [0.5, 0.5]
        assert tune_predictions[0].predicted_label == 0
        assert tune_predictions[0].is_placeholder is True
        assert tune_predictions[0].missing_reason == "no_tiles"
        assert result.tune_report.metrics["coverage"] == 0.0

        test_predictions = result.test_reports["test"].predictions
        assert [pred.sample_id for pred in test_predictions] == ["s7"]
        assert test_predictions[0].probabilities == [0.5, 0.5]
        assert test_predictions[0].predicted_label == 0
        assert test_predictions[0].is_placeholder is True
        assert test_predictions[0].missing_reason == "no_tiles"
        assert result.test_reports["test"].metrics["coverage"] == 0.0

        metrics = json.loads((fold_dir / "metrics.json").read_text())
        assert metrics["tune"]["coverage"] == 0.0
        assert metrics["test"]["coverage"] == 0.0
        assert metrics["tune"]["num_samples"] == 1
        assert metrics["tune"]["num_real_samples"] == 0
        assert metrics["tune"]["num_placeholder_samples"] == 1
        assert metrics["test"]["num_samples"] == 1
        assert metrics["test"]["num_real_samples"] == 0
        assert metrics["test"]["num_placeholder_samples"] == 1

        preds_df = pd.read_csv(fold_dir / "predictions_test.csv")
        assert preds_df["sample_id"].tolist() == ["s7"]
        assert preds_df["is_placeholder"].tolist() == [True]
        assert preds_df["missing_reason"].tolist() == ["no_tiles"]

    def test_saves_attention_npz_when_heatmaps_enabled(self, tmp_path: Path):
        """With heatmaps.enabled, attention .npz files should be written during the test pass."""
        from soma.config import HeatmapConfig

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=1, patience=10, batch_size=1),
            fold_dir=fold_dir,
            heatmaps=HeatmapConfig(enabled=True),
        )

        attention_dir = fold_dir / "attention" / "test"
        assert attention_dir.is_dir(), "attention/test/ dir should be created"
        npz_files = list(attention_dir.glob("*.npz"))
        assert len(npz_files) > 0, "at least one attention .npz should be saved"
        # Verify shape: (N,) for single-branch ABMIL
        data = np.load(npz_files[0])
        assert "attention" in data
        assert data["attention"].ndim == 1

    def test_evaluate_trims_attention_to_valid_tiles(self, tmp_path: Path):
        """Padded batch positions should not leak into saved attention arrays."""

        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        batch = BagBatch(
            features=torch.zeros(2, 4, D),
            mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
            labels=torch.tensor([0, 1], dtype=torch.long),
            sample_ids=("s0", "s1"),
        )

        class _DummyTaskHead:
            def compute_metrics(self, logits, labels):
                return {"accuracy": 1.0}

            def postprocess(self, logits):
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                predicted_labels = logits.argmax(dim=-1).cpu().numpy()
                return {"probabilities": probabilities, "predicted_labels": predicted_labels}

        class _DummyModel:
            task_head = _DummyTaskHead()

            def __call__(self, features, mask=None):
                return SimpleNamespace(
                    logits=torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
                    tile_attention=torch.tensor(
                        [
                            [0.1, 0.2, 0.3, 0.4],
                            [0.4, 0.3, 0.2, 0.1],
                        ],
                        dtype=torch.float32,
                    ),
                )

        _evaluate(
            _DummyModel(),
            [batch],
            "test",
            {"normal": 0, "tumor": 1},
            torch.device("cpu"),
            attention_dir=attention_dir,
            aggregator_name="abmil",
        )

        attn_s0 = np.load(attention_dir / "s0.npz")["attention"]
        attn_s1 = np.load(attention_dir / "s1.npz")["attention"]
        assert attn_s0.shape == (3,)
        assert attn_s1.shape == (4,)

    def test_no_attention_dir_when_heatmaps_disabled(self, tmp_path: Path):
        """Without heatmaps enabled, no attention/ directory should be written."""
        from soma.config import HeatmapConfig

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=1, patience=10, batch_size=1),
            fold_dir=fold_dir,
            heatmaps=HeatmapConfig(enabled=False),
        )

        assert not (fold_dir / "attention").exists()

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
            task=TaskConfig(name="binary_classification"),  # no num_classes specified
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_0",
        )

        # Should have 2-class probabilities (tumor/normal)
        assert len(result.test_reports["test"].predictions[0].probabilities) == 2


    def test_slide_level_features_no_aggregator(self, tmp_path: Path):
        """train_one_fold with 1-D features and aggregator=None uses SlideModel."""
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)

        # Slide-level: one (D,) tensor per sample
        slide_dir = tmp_path / "slide_feats"
        slide_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_dir / f"s{i}.pt")
        store = FeatureStore(slide_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=None,
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_slide",
        )

        assert isinstance(result, FoldResult)
        assert result.test_reports["test"] is not None
        assert "auroc" in result.test_reports["test"].metrics

    def test_slide_level_features_can_omit_aggregator(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)

        slide_dir = tmp_path / "slide_feats"
        slide_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_dir / f"s{i}.pt")
        store = FeatureStore(slide_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_slide_omitted",
        )

        assert isinstance(result, FoldResult)
        assert "auroc" in result.test_reports["test"].metrics

    def test_slide_level_features_with_aggregator_raises(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        slide_dir = tmp_path / "slide_feats"
        slide_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_dir / f"s{i}.pt")
        store = FeatureStore(slide_dir)

        with pytest.raises(ValueError, match="aggregator must be None"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                aggregator=AggregatorConfig(name="mean_pool"),
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                fold_dir=tmp_path / "fold_slide",
            )

    def test_hierarchical_features_use_hipt(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_hierarchical_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(
                name="hipt",
                params={
                    "embed_dim_region": 12,
                    "embed_dim_slide": 12,
                    "num_heads": 2,
                    "dropout": 0.0,
                },
            ),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_hier",
            preprocessing=PreprocessingConfig(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                requested_region_size_px=448,
                region_tile_multiple=2,
            ),
        )

        assert isinstance(result, FoldResult)
        assert store.is_hierarchical is True
        assert (tmp_path / "fold_hier" / "best_model.pt").exists()

    def test_hierarchical_features_reject_non_hipt_aggregator(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_hierarchical_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        with pytest.raises(ValueError, match="hierarchical features require the hipt aggregator"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                aggregator=AggregatorConfig(name="mean_pool"),
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                fold_dir=tmp_path / "fold_hier_error",
                preprocessing=PreprocessingConfig(
                    requested_tile_size_px=224,
                    requested_spacing_um=0.5,
                    requested_region_size_px=448,
                    region_tile_multiple=2,
                ),
            )


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
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            run_dir=tmp_path / "output",
        )

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].fold == 0

    def test_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        run_dir = tmp_path / "output"

        result = train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            run_dir=run_dir,
        )

        assert len(result.fold_results) == 2
        assert result.fold_results[0].fold == 0
        assert result.fold_results[1].fold == 1
        assert "test/auroc_mean" in result.summary
        assert "test/auroc_std" in result.summary

    def test_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        run_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            run_dir=run_dir,
        )

        summary = json.loads((run_dir / "summary.json").read_text())
        assert "test/auroc" in summary

    def test_saves_summary_json_with_coverage_aggregates(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        for sample_id in ("s6", "s7"):
            (feature_dir / f"{sample_id}.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "success",
                "s6": "empty",
                "s7": "empty",
            },
        )
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        run_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            run_dir=run_dir,
        )

        summary = json.loads((run_dir / "summary.json").read_text())
        assert "test/coverage" in summary
        assert "test/num_samples" in summary
        assert "test/num_real_samples" in summary
        assert "test/num_placeholder_samples" in summary

    def test_aggregate_fold_metrics_handles_inconsistent_test_split_names(self):
        train_result = TrainResult(
            best_epoch=0,
            best_tune_loss=0.1,
            best_tune_metrics={},
            history=[],
            checkpoint_path=Path("best_model.pt"),
        )
        fold_results = [
            FoldResult(
                fold=0,
                train_result=train_result,
                tune_report=EvaluationReport(split="tune", metrics={}, predictions=[]),
                test_reports={
                    "test": EvaluationReport(
                        split="test",
                        metrics={"auroc": 0.8},
                        predictions=[],
                    )
                },
            ),
            FoldResult(
                fold=1,
                train_result=train_result,
                tune_report=EvaluationReport(split="tune", metrics={}, predictions=[]),
                test_reports={
                    "test_external": EvaluationReport(
                        split="test_external",
                        metrics={"auroc": 0.7},
                        predictions=[],
                    )
                },
            ),
        ]

        summary = _aggregate_fold_metrics(fold_results)

        assert summary["test/auroc_mean"] == pytest.approx(0.8)
        assert summary["test_external/auroc_mean"] == pytest.approx(0.7)

    def test_completed_run_panel_includes_coverage_summary(self):
        panel = _build_completed_run_panel(
            summary_metrics={
                "test/auroc_mean": 0.75,
                "test/auroc_std": 0.02,
                "test/coverage_mean": 0.9,
                "test/num_samples_mean": 10.0,
                "test/num_real_samples_mean": 9.0,
                "test/num_placeholder_samples_mean": 1.0,
            }
        )
        console = Console(record=True, width=120)
        console.print(panel)
        rendered = console.export_text()

        assert "coverage" in rendered.lower()
        assert "auroc=0.7500" in rendered
        assert "90.0% (9/10)" in rendered
        assert "placeholder=1" in rendered

    def test_fold_subdirectories(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        run_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            run_dir=run_dir,
        )

        assert (run_dir / "fold_0" / "best_model.pt").exists()
        assert (run_dir / "fold_1" / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Pipeline (Layer 2 — orchestrator)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_run_single_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.run_dir == _expected_run_dir(config)

    def test_run_ignores_samples_without_features(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        (feature_dir / "s5.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "empty",
                "s6": "success",
                "s7": "success",
            },
        )
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert (_expected_run_dir(config) / "best_model.pt").exists()

    def test_run_saves_config_yaml(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, feature_dir=feature_dir).run()

        assert (_expected_run_dir(config) / "config.yaml").exists()

    def test_run_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, feature_dir=feature_dir).run()

        summary_path = _expected_run_dir(config) / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "test/auroc" in summary

    def test_run_saves_summary_json_with_coverage_aggregates(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        for sample_id in ("s6", "s7"):
            (feature_dir / f"{sample_id}.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "success",
                "s6": "empty",
                "s7": "empty",
            },
        )
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, feature_dir=feature_dir).run()

        summary = json.loads((_expected_run_dir(config) / "summary.json").read_text())
        assert "test/coverage" in summary
        assert "test/num_samples" in summary
        assert "test/num_real_samples" in summary
        assert "test/num_placeholder_samples" in summary

    def test_run_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        output_root = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert len(result.fold_results) == 2
        assert (_expected_run_dir(config) / "fold_0" / "best_model.pt").exists()
        assert (_expected_run_dir(config) / "fold_1" / "best_model.pt").exists()

        summary = json.loads((_expected_run_dir(config) / "summary.json").read_text())
        assert "test/auroc_mean" in summary
        assert "test/auroc_std" in summary

    def test_run_slide_level_features(self, tmp_path: Path):
        """Pipeline with aggregator=None should work with slide-level (1-D) features."""
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)

        # Create slide-level features (1-D per sample)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        torch.manual_seed(7)
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        output_root = tmp_path / "output_slide"
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=None,
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=slide_feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].test_reports["test"] is not None
        assert "auroc" in result.fold_results[0].test_reports["test"].metrics

    def test_run_slide_level_features_can_omit_aggregator(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)

        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        torch.manual_seed(7)
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        output_root = tmp_path / "output_slide_omitted"
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            aggregator=None,
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=slide_feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert "auroc" in result.fold_results[0].test_reports["test"].metrics

    def test_run_hierarchical_features_with_hipt(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_hierarchical_data(tmp_path)
        output_root = tmp_path / "output_hier"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            encoder=EncoderConfig(name="uni2"),
            preprocessing=PreprocessingConfig(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                requested_region_size_px=448,
                region_tile_multiple=2,
            ),
            aggregator=AggregatorConfig(
                name="hipt",
                params={
                    "embed_dim_region": 12,
                    "embed_dim_slide": 12,
                    "num_heads": 2,
                    "dropout": 0.0,
                },
            ),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert (_expected_run_dir(config) / "best_model.pt").exists()
        assert "auroc" in result.fold_results[0].test_reports["test"].metrics

    def test_resolve_preprocessing_populates_hipt_geometry(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "output_resolve",
            dataset_type="slide",
            encoder=EncoderConfig(name="uni2"),
            preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
            aggregator=AggregatorConfig(name="hipt", params={"tile_multiple": 6}),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)

        resolved = pipeline._resolve_preprocessing()

        assert resolved.requested_tile_size_px == 224
        assert resolved.read_tile_size_px == 224
        assert resolved.region_tile_multiple == 6
        assert resolved.requested_region_size_px == 1344
        assert resolved.read_region_size_px == 1344

    def test_run_auto_extracts_slide_features_without_feature_dir(self, tmp_path: Path):
        pytest.importorskip("soma.extraction")
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output_prism"

        def _fake_run(self, feature_dir, **kwargs):
            out = Path(feature_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.randn(D), out / f"s{i}.pt")
            return FeatureStore(out)

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="slide",
            encoder=EncoderConfig(name="prism"),
            aggregator=None,
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )

        with patch("soma.extraction.FeatureExtractor.run", new=_fake_run):
            with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
                result = Pipeline(config).run()

        assert isinstance(result, PipelineResult)
        assert (_expected_run_dir(config) / "features" / "s0.pt").exists()

    def test_pipeline_rejects_aggregator_for_slide_features(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "output_slide_error",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with pytest.raises(ValueError, match="aggregator must be None"):
            with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(config, feature_dir=slide_feature_dir).run()

        run_dir = _expected_run_dir(config)
        latest = run_dir.parents[1] / "latest"
        run_yaml = run_dir / "run.yaml"
        payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))

        assert payload["status"] == "failed"
        assert latest.is_symlink()
        assert latest.resolve() == run_dir.resolve()

    def test_pipeline_reuses_shared_cache_for_sibling_runs(self, tmp_path: Path):
        pytest.importorskip("slide2vec")
        from types import SimpleNamespace

        from hs2p import SlideSpec
        from slide2vec.encoders.base import SlideEncoder, TileEncoder
        from slide2vec.encoders.registry import encoder_registry
        from soma.slide2vec_adapter import LoadedTiling
        from soma.cache import record_feature_dim, record_sample_identity_signatures
        from soma.extraction import FeatureExtractor

        test_tile = "_test_pipeline_cache_tile"
        test_slide = "_test_pipeline_cache_slide"

        class _PipelineTileEncoder(TileEncoder):
            def __init__(self, **kwargs):
                self._device = torch.device("cpu")
                self._output_variant = kwargs.get("output_variant") or "default"

            def get_transform(self):
                return lambda x: x

            def encode_tiles(self, batch):
                return torch.ones(batch.shape[0], D)

            @property
            def encode_dim(self):
                return D

            @property
            def device(self):
                return self._device

            def to(self, device):
                self._device = torch.device(device)
                return self

        class _PipelineSlideEncoder(SlideEncoder):
            def __init__(self, **kwargs):
                self._device = torch.device("cpu")
                self._output_variant = kwargs.get("output_variant") or "default"

            def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0: int | None = None):
                return tile_features.mean(dim=0)

            @property
            def encode_dim(self):
                return D

            @property
            def device(self):
                return self._device

            def to(self, device):
                self._device = torch.device(device)
                return self

        if test_tile not in encoder_registry:
            encoder_registry.register(
                test_tile,
                _PipelineTileEncoder,
                metadata={
                    "level": "tile",
                    "input_size": 256,
                    "output_variants": {"default": {"encode_dim": D}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                },
            )
        if test_slide not in encoder_registry:
            encoder_registry.register(
                test_slide,
                _PipelineSlideEncoder,
                metadata={
                    "level": "slide",
                    "tile_encoder": test_tile,
                    "tile_encoder_output_variant": "default",
                    "output_variants": {"default": {"encode_dim": D}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                },
            )

        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        shared_cache = tmp_path / "shared-cache"

        loaded_tilings = [
            LoadedTiling(
                slide=SlideSpec(
                    sample_id=f"s{i}",
                    image_path=Path(f"/slides/s{i}.svs"),
                    mask_path=None,
                    spacing_at_level_0=None,
                ),
                tiling_result=SimpleNamespace(
                    x=np.array([0, 256], dtype=np.int64),
                    y=np.array([0, 256], dtype=np.int64),
                    tissue_fractions=np.array([1.0, 1.0], dtype=np.float32),
                    requested_tile_size_px=256,
                    requested_spacing_um=0.5,
                    read_tile_size_px=256,
                    read_spacing_um=0.5,
                    tile_size_lv0=256,
                    read_level=0,
                    use_padding=True,
                    is_within_tolerance=True,
                    sample_id=f"s{i}",
                    coordinates_npz_path=Path(f"/tmp/s{i}.npz"),
                    coordinates_meta_path=Path(f"/tmp/s{i}.meta.json"),
                ),
            )
            for i in range(NUM_SAMPLES)
        ]

        slide_config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "slide_run",
            dataset_type="slide",
            preprocessing=PreprocessingConfig(),
            cache=CacheConfig(root_dir=shared_cache),
            encoder=EncoderConfig(name=test_slide),
            aggregator=None,
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        tile_config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "tile_run",
            dataset_type="slide",
            preprocessing=PreprocessingConfig(),
            cache=CacheConfig(root_dir=shared_cache),
            encoder=EncoderConfig(name=test_tile),
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )

        def _fake_preprocess(self_, tiling_dir, *, skip_existing=True, backend="auto"):
            tiling_dir = Path(tiling_dir)
            tiling_dir.mkdir(parents=True, exist_ok=True)
            (tiling_dir / "process_list.csv").write_text("sample_id,tiling_status\n", encoding="utf-8")

        def _fake_populate_tile_cache(
            self_,
            *,
            cache_resolution,
            loaded_tilings,
            prepared_tilings,
            tiling_dir,
            preprocessing,
            encoder_name,
            output_variant,
            num_gpus,
        ):
            cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.ones(2, D), cache_resolution.feature_path_for_id(f"s{i}"))
            record_feature_dim(cache_resolution, D)
            record_sample_identity_signatures(cache_resolution, [f"s{i}" for i in range(NUM_SAMPLES)])

        def _fake_populate_slide_cache(
            self_,
            *,
            slide_cache,
            tile_cache,
            loaded_tilings,
            model_name,
            output_variant,
            num_gpus,
        ):
            slide_cache.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.ones(D), slide_cache.feature_path_for_id(f"s{i}"))
            record_feature_dim(slide_cache, D)
            record_sample_identity_signatures(slide_cache, [f"s{i}" for i in range(NUM_SAMPLES)])

        with patch("soma.extraction.torch.cuda.is_available", return_value=False), patch(
            "soma.extraction.torch.cuda.device_count", return_value=1
        ), patch.object(
            FeatureExtractor,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction._validate_runtime"), patch.object(
            FeatureExtractor,
            "_populate_tile_cache",
            autospec=True,
            side_effect=_fake_populate_tile_cache,
        ) as populate_tile_cache, patch.object(
            FeatureExtractor,
            "_populate_slide_cache",
            autospec=True,
            side_effect=_fake_populate_slide_cache,
        ) as populate_slide_cache:
            with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(slide_config).run()
            assert populate_tile_cache.called
            assert populate_slide_cache.called

        with patch("soma.extraction.torch.cuda.is_available", return_value=False), patch(
            "soma.extraction.torch.cuda.device_count", return_value=1
        ), patch.object(
            FeatureExtractor,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction._validate_runtime"), patch.object(
            FeatureExtractor,
            "_populate_tile_cache",
            side_effect=AssertionError("tile extraction should be reused"),
        ):
            with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(tile_config).run()

    def test_dataset_and_splits_properties(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "output",
            dataset_type="slide",
            task=TaskConfig(name="binary_classification"),
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)

        assert pipeline.dataset.num_classes == 2
        assert pipeline.splits.num_folds == 1

    def test_pipeline_writes_experiment_metadata_and_indexes(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "output",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )

        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        experiment_dir = result.run_dir.parents[1]
        experiment_yaml = experiment_dir / "experiment.yaml"
        run_yaml = result.run_dir / "run.yaml"
        latest = experiment_dir / "latest"
        runs_index = Path(config.output_root) / "indexes" / "runs.csv"

        assert experiment_yaml.exists()
        assert run_yaml.exists()
        assert latest.is_symlink()
        assert latest.resolve() == result.run_dir.resolve()

        with runs_index.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert rows == [
            {
                "run_id": FIXED_RUN_ID,
                "experiment_id": rows[0]["experiment_id"],
                "status": "completed",
                "started_at": rows[0]["started_at"],
                "finished_at": rows[0]["finished_at"],
                "seed": "0",
                "wandb_id": "",
                "git_sha": rows[0]["git_sha"],
                "run_dir": str(result.run_dir.resolve()),
                "error": "",
            }
        ]


# ---------------------------------------------------------------------------
# Patient pipeline
# ---------------------------------------------------------------------------

D_PATIENT = 768
NUM_PATIENTS = 4  # 2 slides each → 8 slides total


def _setup_patient_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset.csv (with patient_id), splits.csv, and patient feature .pt files.

    4 patients × 2 slides each = 8 slides.
    Splits are patient-clean: all slides for a patient go to the same split.
    """
    sample_ids = [f"s{i}" for i in range(8)]
    patient_ids = [f"p{i // 2}" for i in range(8)]  # p0 p0 p1 p1 p2 p2 p3 p3
    labels = ["tumor" if (i // 2) % 2 == 0 else "normal" for i in range(8)]

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": [f"/slides/{s}.svs" for s in sample_ids],
            "label": labels,
            "patient_id": patient_ids,
        }
    ).to_csv(dataset_csv, index=False)

    # Splits: p0+p1 in train, p2 in tune, p3 in test
    splits_rows = []
    for sid, pid in zip(sample_ids, patient_ids):
        if pid in ("p0", "p1"):
            split = "train"
        elif pid == "p2":
            split = "tune"
        else:
            split = "test"
        splits_rows.append({"fold": 0, "sample_id": sid, "split": split})
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(splits_rows).to_csv(splits_csv, index=False)

    # One patient feature per patient_id (1-D tensors)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(0)
    for pid in [f"p{i}" for i in range(NUM_PATIENTS)]:
        torch.save(torch.randn(D_PATIENT), feature_dir / f"{pid}.pt")

    return dataset_csv, splits_csv, feature_dir


def _write_patient_feature_manifest(feature_dir: Path, statuses: dict[str, str]) -> None:
    rows = []
    for patient_id, status in statuses.items():
        rows.append(
            {
                "sample_id": patient_id,
                "feature_status": status,
                "feature_path": str((feature_dir / f"{patient_id}.pt").resolve())
                if status == "success"
                else "",
            }
        )
    pd.DataFrame(rows).to_csv(feature_dir / "process_list.csv", index=False)


class TestPatientPipeline:
    def test_train_one_fold_returns_fold_result_and_saves_checkpoint(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_patient_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            dataset_type="patient",
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=fold_dir,
        )

        assert isinstance(result, FoldResult)
        assert result.fold == 0
        assert result.tune_report.split == "tune"
        assert result.test_reports["test"].split == "test"
        assert (fold_dir / "best_model.pt").exists()

    def test_train_one_fold_predictions_keyed_by_patient_id(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_patient_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            dataset_type="patient",
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_0",
        )

        test_ids = {pred.sample_id for pred in result.test_reports["test"].predictions}
        assert test_ids == {"p3"}

    def test_train_one_fold_patient_features_fail_fast_when_missing(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_patient_data(tmp_path)
        (feature_dir / "p2.pt").unlink()
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        with pytest.raises(ValueError, match="patient"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                dataset_type="patient",
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                fold_dir=tmp_path / "fold_0",
            )

    def test_train_one_fold_patient_empty_tune_uses_placeholder_fallback(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_patient_data(tmp_path)
        (feature_dir / "p2.pt").unlink()
        _write_patient_feature_manifest(
            feature_dir,
            {
                "p0": "success",
                "p1": "success",
                "p2": "empty",
                "p3": "success",
            },
        )
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            dataset_type="patient",
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold_0",
        )

        assert [pred.sample_id for pred in result.tune_report.predictions] == ["p2"]
        assert result.tune_report.predictions[0].is_placeholder is True
        assert result.tune_report.metrics["coverage"] == pytest.approx(0.0)

    def test_build_subgroup_data_uses_patient_metadata_for_patient_predictions(self, tmp_path: Path):
        dataset_csv, _, _ = _setup_patient_data(tmp_path)
        dataset_df = pd.read_csv(dataset_csv)
        dataset_df["site"] = [
            "site_a" if patient_id in {"p0", "p2"} else "site_b"
            for patient_id in dataset_df["patient_id"]
        ]
        dataset_df.to_csv(dataset_csv, index=False)
        dataset = Dataset(dataset_csv)
        report = EvaluationReport(
            split="test",
            metrics={},
            predictions=[
                SamplePrediction(sample_id="p2", true_label=1, predicted_label=1),
                SamplePrediction(sample_id="p3", true_label=0, predicted_label=0),
            ],
        )

        subgroup_data = _build_subgroup_data(dataset, report, ["site"])

        assert subgroup_data == {
            "p2": {"site": "site_a"},
            "p3": {"site": "site_b"},
        }

    def test_train_detects_patient_leakage(self, tmp_path: Path):
        dataset_csv, _, feature_dir = _setup_patient_data(tmp_path)
        dataset = Dataset(dataset_csv)

        # s0 belongs to p0 but is put in test; the other p0 slide (s1) stays in train
        bad_splits = []
        for sid in dataset.sample_ids:
            pid = dataset.samples[sid].patient_id
            if sid == "s0":
                split = "test"
            elif pid in ("p0", "p1"):
                split = "train"
            elif pid == "p2":
                split = "tune"
            else:
                split = "test"
            bad_splits.append({"fold": 0, "sample_id": sid, "split": split})

        splits_csv = tmp_path / "bad_splits.csv"
        pd.DataFrame(bad_splits).to_csv(splits_csv, index=False)

        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        with pytest.raises(ValueError, match="p0"):
            train(
                feature_store=store,
                dataset=dataset,
                splits=splits,
                dataset_type="patient",
                task=TaskConfig(name="binary_classification"),
                training=TrainingConfig(epochs=1, patience=5),
                run_dir=tmp_path / "run",
            )
