"""Tests for soma.reporting — training history persistence, data loading, and HTML generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from soma.config import AggregatorConfig, PipelineConfig, TaskConfig, TrainingConfig
from soma.evaluation.metrics import DEFAULT_METRICS, resolve_metrics
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.reporting import generate_report, load_run_data
from soma.reporting.data import FoldData, RunData, run_data_from_result
from soma.reporting.html import render_report
from soma.training.trainer import EpochLog, TrainResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_training_history(n_epochs: int = 5) -> list[dict]:
    return [
        {
            "epoch": i,
            "train_loss": 0.8 - i * 0.05,
            "tune_loss": 0.75 - i * 0.04,
            "tune_metrics": {"auroc": 0.5 + i * 0.05, "balanced_accuracy": 0.5 + i * 0.03},
            "lr": 1e-4,
        }
        for i in range(n_epochs)
    ]


def _make_binary_predictions(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": [i % 2 for i in range(n)],
        "predicted_label": [i % 2 for i in range(n)],
        "prob_0": [0.3 + (i % 2) * 0.4 for i in range(n)],
        "prob_1": [0.7 - (i % 2) * 0.4 for i in range(n)],
    })


def _make_regression_predictions(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": [float(i) for i in range(n)],
        "predicted_value": [float(i) + 0.1 * (i % 3 - 1) for i in range(n)],
    })


def _make_run_dir(
    tmp_path: Path,
    *,
    task_name: str = "binary_classification",
    metrics: list[str] | None = None,
    n_folds: int = 1,
    include_history: bool = True,
    predictions_fn=None,
) -> Path:
    """Create a synthetic run directory with all expected artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    config = {
        "dataset_csv": "/data/dataset.csv",
        "splits_csv": "/data/splits.csv",
        "output_root": "/output",
        "task": {"name": task_name, "params": {}, "metrics": metrics or []},
        "encoder": None,
        "aggregator": {"name": "abmil", "params": {}},
        "training": {
            "seed": 42,
            "epochs": 5,
            "learning_rate": 1e-4,
            "optimizer": "adam",
            "scheduler": "cosine",
            "patience": 10,
            "batch_size": 1,
        },
        "tags": [],
    }
    (run_dir / "config.yaml").write_text(_to_yaml(config))

    run_metadata = {
        "run_id": "2026-04-11_12-00-00__local",
        "status": "completed",
        "started_at": "2026-04-11T12:00:00+00:00",
        "finished_at": "2026-04-11T12:05:00+00:00",
        "seed": 42,
        "git_sha": "abcdef1234567890",
    }
    (run_dir / "run.yaml").write_text(_to_yaml(run_metadata))

    # summary with mean/std for each metric
    resolved = resolve_metrics(task_name, metrics or [])
    summary = {}
    for m in resolved:
        summary[f"{m}_mean"] = 0.75
        summary[f"{m}_std"] = 0.02
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    preds_fn = predictions_fn or (
        _make_regression_predictions if task_name == "regression" else _make_binary_predictions
    )

    for fold_idx in range(n_folds):
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir()

        if include_history:
            (fold_dir / "training_history.json").write_text(
                json.dumps(_make_training_history())
            )

        metrics_data = {
            "tune": {m: 0.80 for m in resolved},
            "test": {m: 0.75 for m in resolved},
        }
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))
        preds_fn().to_csv(fold_dir / "predictions.csv", index=False)

    return run_dir


def _to_yaml(data: dict) -> str:
    return yaml.dump(data, default_flow_style=False)


# ---------------------------------------------------------------------------
# Training history persistence
# ---------------------------------------------------------------------------


def test_save_training_history(tmp_path: Path) -> None:
    """_save_training_history writes a valid JSON array of epoch dicts."""
    from soma.training.trainer import EpochLog
    from soma.pipeline import _save_training_history

    history = [
        EpochLog(epoch=0, train_loss=0.8, tune_loss=0.75,
                 tune_metrics={"auroc": 0.55}, lr=1e-4),
        EpochLog(epoch=1, train_loss=0.6, tune_loss=0.60,
                 tune_metrics={"auroc": 0.70}, lr=9e-5),
    ]
    path = tmp_path / "training_history.json"
    _save_training_history(history, path)

    data = json.loads(path.read_text())
    assert len(data) == 2
    assert data[0] == {
        "epoch": 0,
        "train_loss": 0.8,
        "tune_loss": 0.75,
        "tune_metrics": {"auroc": 0.55},
        "lr": 1e-4,
    }
    assert data[1]["epoch"] == 1
    assert data[1]["tune_metrics"] == {"auroc": 0.70}


# ---------------------------------------------------------------------------
# Data loading from disk
# ---------------------------------------------------------------------------


def test_load_run_data_binary(tmp_path: Path) -> None:
    """load_run_data correctly parses a binary classification run directory."""
    run_dir = _make_run_dir(tmp_path, task_name="binary_classification")
    run_data = load_run_data(run_dir)

    assert run_data.task_family == "binary_classification"
    assert len(run_data.folds) == 1
    assert run_data.folds[0].fold == 0
    assert len(run_data.folds[0].training_history) == 5
    assert run_data.folds[0].training_history[0]["epoch"] == 0
    assert "auroc" in run_data.folds[0].tune_metrics
    assert "prob_1" in run_data.folds[0].predictions.columns


def test_load_run_data_multi_fold(tmp_path: Path) -> None:
    """load_run_data discovers and sorts fold directories correctly."""
    run_dir = _make_run_dir(tmp_path, n_folds=3)
    run_data = load_run_data(run_dir)

    assert len(run_data.folds) == 3
    assert [fd.fold for fd in run_data.folds] == [0, 1, 2]


def test_load_run_data_missing_history(tmp_path: Path) -> None:
    """load_run_data handles a run directory without training_history.json gracefully."""
    run_dir = _make_run_dir(tmp_path, include_history=False)
    run_data = load_run_data(run_dir)

    assert run_data.folds[0].training_history == []


def test_load_run_data_resolves_default_metrics(tmp_path: Path) -> None:
    """When task.metrics is empty, load_run_data uses default metrics for the family."""
    run_dir = _make_run_dir(tmp_path, task_name="binary_classification", metrics=[])
    run_data = load_run_data(run_dir)

    from soma.evaluation.metrics import DEFAULT_METRICS
    assert set(run_data.metrics) == set(DEFAULT_METRICS["binary_classification"])


def test_load_run_data_uses_user_metrics(tmp_path: Path) -> None:
    """When task.metrics is set, only those metrics are in run_data.metrics."""
    run_dir = _make_run_dir(
        tmp_path, task_name="binary_classification", metrics=["auroc", "f1"]
    )
    run_data = load_run_data(run_dir)
    assert run_data.metrics == ["auroc", "f1"]


# ---------------------------------------------------------------------------
# RunData from in-memory PipelineResult
# ---------------------------------------------------------------------------


def _make_mock_pipeline_result(tmp_path: Path) -> tuple:
    """Build a minimal mock PipelineResult and PipelineConfig."""
    history = [EpochLog(epoch=0, train_loss=0.7, tune_loss=0.65,
                        tune_metrics={"auroc": 0.6}, lr=1e-4)]
    train_result = TrainResult(
        best_epoch=0,
        best_tune_loss=0.65,
        best_tune_metrics={"auroc": 0.6},
        history=history,
        checkpoint_path=tmp_path / "best_model.pt",
    )

    predictions = [
        SamplePrediction(
            sample_id="s0", true_label=0, predicted_label=0, probabilities=[0.8, 0.2]
        ),
        SamplePrediction(
            sample_id="s1", true_label=1, predicted_label=1, probabilities=[0.3, 0.7]
        ),
    ]
    tune_report = EvaluationReport(split="tune", metrics={"auroc": 0.65}, predictions=[])
    test_report = EvaluationReport(split="test", metrics={"auroc": 0.60}, predictions=predictions)

    fold_result = MagicMock()
    fold_result.fold = 0
    fold_result.train_result = train_result
    fold_result.tune_report = tune_report
    fold_result.test_report = test_report

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "run.yaml").write_text(
        yaml.dump({"run_id": "test-run", "status": "running", "seed": 0})
    )

    result = MagicMock()
    result.fold_results = [fold_result]
    result.summary = {"auroc_mean": 0.60, "auroc_std": 0.0}
    result.run_dir = run_dir

    config = PipelineConfig(
        dataset_csv="/data/dataset.csv",
        splits_csv="/data/splits.csv",
        output_root="/output",
        task=TaskConfig(name="binary_classification", metrics=["auroc"]),
        training=TrainingConfig(seed=42),
        aggregator=AggregatorConfig(name="abmil"),
    )

    return result, config


def test_run_data_from_result(tmp_path: Path) -> None:
    """run_data_from_result converts in-memory objects to RunData correctly."""
    result, config = _make_mock_pipeline_result(tmp_path)
    run_data = run_data_from_result(result, config)

    assert run_data.task_family == "binary_classification"
    assert run_data.metrics == ["auroc"]
    assert len(run_data.folds) == 1

    fd = run_data.folds[0]
    assert len(fd.training_history) == 1
    assert fd.training_history[0]["epoch"] == 0
    assert fd.tune_metrics == {"auroc": 0.65}
    assert fd.test_metrics == {"auroc": 0.60}
    assert "prob_1" in fd.predictions.columns
    assert len(fd.predictions) == 2


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def test_generate_report_binary_creates_file(tmp_path: Path) -> None:
    """generate_report writes an HTML file for a binary classification run."""
    run_dir = _make_run_dir(tmp_path, task_name="binary_classification")
    report_path = generate_report(run_dir)

    assert report_path.exists()
    assert report_path.suffix == ".html"
    html = report_path.read_text()
    assert "<!DOCTYPE html>" in html
    assert "Experiment Report" in html


def test_generate_report_contains_key_sections(tmp_path: Path) -> None:
    """The generated HTML contains all expected section headers."""
    run_dir = _make_run_dir(tmp_path, task_name="binary_classification")
    html = generate_report(run_dir).read_text()

    assert "Configuration" in html
    assert "Test Results" in html
    assert "Training Curves" in html
    assert "Prediction Analysis" in html


def test_generate_report_regression(tmp_path: Path) -> None:
    """generate_report works for regression tasks and omits classification-specific sections."""
    run_dir = _make_run_dir(
        tmp_path, task_name="regression", predictions_fn=_make_regression_predictions
    )
    html = generate_report(run_dir).read_text()

    assert "Prediction Analysis" in html
    # ROC curve only appears for classification
    assert "ROC curve" not in html


def test_generate_report_without_history(tmp_path: Path) -> None:
    """Report generates successfully even when training_history.json is absent."""
    run_dir = _make_run_dir(tmp_path, include_history=False)
    html = generate_report(run_dir).read_text()

    assert "No training history available" in html


def test_generate_report_metric_curves_match_requested(tmp_path: Path) -> None:
    """Metric curves are shown only for user-requested metrics, not all metrics."""
    run_dir = _make_run_dir(
        tmp_path,
        task_name="binary_classification",
        metrics=["auroc"],
    )
    html = generate_report(run_dir).read_text()

    # auroc metric curve title should appear
    assert "Tune auroc per epoch" in html
    # balanced_accuracy was not requested, should not appear as a curve
    assert "Tune balanced_accuracy per epoch" not in html


def test_generate_report_custom_output_path(tmp_path: Path) -> None:
    """generate_report respects a custom output_path argument."""
    run_dir = _make_run_dir(tmp_path)
    custom = tmp_path / "custom_report.html"
    result_path = generate_report(run_dir, output_path=custom)

    assert result_path == custom
    assert custom.exists()


def test_generate_report_multi_fold(tmp_path: Path) -> None:
    """Multi-fold report generates without errors and shows summary stats."""
    run_dir = _make_run_dir(tmp_path, n_folds=3)
    html = generate_report(run_dir).read_text()

    assert "Fold 0" in html
    assert "Fold 1" in html
    assert "Fold 2" in html
    assert "Mean" in html or "±" in html
