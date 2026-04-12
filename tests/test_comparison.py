"""Tests for cross-run comparison: diff_configs, load_comparison_data, compare_runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from soma.evaluation.metrics import resolve_metrics
from soma.reporting import compare_runs, load_comparison_data
from soma.reporting.data import ComparisonData, diff_configs


# ---------------------------------------------------------------------------
# Helpers (mirrors test_reporting.py fixtures, kept local for independence)
# ---------------------------------------------------------------------------


def _make_training_history(n_epochs: int = 4) -> list[dict]:
    return [
        {
            "epoch": i,
            "train_loss": 0.8 - i * 0.05,
            "tune_loss": 0.75 - i * 0.04,
            "tune_metrics": {"auroc": 0.5 + i * 0.05},
            "lr": 1e-4,
        }
        for i in range(n_epochs)
    ]


def _make_binary_predictions(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": [i % 2 for i in range(n)],
        "predicted_label": [i % 2 for i in range(n)],
        "prob_0": [0.3 + (i % 2) * 0.4 for i in range(n)],
        "prob_1": [0.7 - (i % 2) * 0.4 for i in range(n)],
    })


def _make_run_dir(
    tmp_path: Path,
    *,
    aggregator: str = "abmil",
    learning_rate: float = 1e-4,
    run_id: str = "2026-04-11_12-00-00__local",
    n_folds: int = 1,
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    metrics = ["auroc"]
    config = {
        "dataset_csv": "/data/dataset.csv",
        "splits_csv": "/data/splits.csv",
        "output_root": "/output",
        "task": {"name": "binary_classification", "params": {}},
        "eval": {"metrics": metrics, "subgroups": {"columns": []}},
        "encoder": None,
        "aggregator": {"name": aggregator, "params": {}},
        "training": {
            "seed": 42,
            "epochs": 4,
            "learning_rate": learning_rate,
            "optimizer": "adam",
            "scheduler": "cosine",
            "patience": 10,
            "batch_size": 1,
        },
        "tags": [],
    }
    (run_dir / "config.yaml").write_text(yaml.dump(config))

    run_metadata = {"run_id": run_id, "status": "completed", "seed": 42}
    (run_dir / "run.yaml").write_text(yaml.dump(run_metadata))

    resolved = resolve_metrics("binary_classification", metrics)
    summary = {f"test/{m}_mean": 0.75 + 0.05 * (aggregator == "clam_sb") for m in resolved}
    summary.update({f"test/{m}_std": 0.02 for m in resolved})
    (run_dir / "summary.json").write_text(json.dumps(summary))

    for fold_idx in range(n_folds):
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir()
        (fold_dir / "training_history.json").write_text(json.dumps(_make_training_history()))
        metrics_data = {"tune": {m: 0.78 for m in resolved}, "test": {m: 0.75 for m in resolved}}
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_data))
        _make_binary_predictions().to_csv(fold_dir / "predictions_test.csv", index=False)

    return run_dir


# ---------------------------------------------------------------------------
# diff_configs
# ---------------------------------------------------------------------------


def test_diff_configs_detects_varying_fields() -> None:
    """Fields that differ across runs appear in diffs, not in shared."""
    configs = [
        {"aggregator": {"name": "abmil"}, "training": {"lr": 1e-4}},
        {"aggregator": {"name": "clam_sb"}, "training": {"lr": 1e-4}},
    ]
    shared, diffs = diff_configs(configs)

    assert "aggregator.name" not in shared
    assert diffs[0]["aggregator.name"] == "abmil"
    assert diffs[1]["aggregator.name"] == "clam_sb"
    assert shared.get("training.lr") == 1e-4


def test_diff_configs_all_identical() -> None:
    """When all configs are identical, shared contains everything and diffs are empty."""
    cfg = {"aggregator": {"name": "abmil"}, "training": {"lr": 1e-4}}
    shared, diffs = diff_configs([cfg, cfg.copy()])

    assert shared["aggregator.name"] == "abmil"
    assert shared["training.lr"] == 1e-4
    assert all(len(d) == 0 for d in diffs)


def test_diff_configs_nested_diff() -> None:
    """A difference nested inside training is correctly detected."""
    configs = [
        {"training": {"lr": 1e-4, "epochs": 50}},
        {"training": {"lr": 1e-3, "epochs": 50}},
    ]
    shared, diffs = diff_configs(configs)

    assert "training.lr" not in shared
    assert shared.get("training.epochs") == 50
    assert diffs[0]["training.lr"] == 1e-4
    assert diffs[1]["training.lr"] == 1e-3


def test_diff_configs_empty_list() -> None:
    """diff_configs handles an empty input without error."""
    shared, diffs = diff_configs([])
    assert shared == {}
    assert diffs == []


# ---------------------------------------------------------------------------
# load_comparison_data
# ---------------------------------------------------------------------------


def test_load_comparison_data(tmp_path: Path) -> None:
    """load_comparison_data loads two runs and populates ComparisonData."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    cd = load_comparison_data([run1, run2])

    assert isinstance(cd, ComparisonData)
    assert len(cd.runs) == 2
    assert len(cd.labels) == 2
    assert len(cd.config_diffs) == 2
    assert "auroc" in cd.metric_names


def test_load_comparison_data_auto_labels_single_diff(tmp_path: Path) -> None:
    """When runs differ in exactly one field, that value is used as label."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    cd = load_comparison_data([run1, run2])

    assert cd.labels == ["abmil", "clam_sb"]


def test_load_comparison_data_auto_labels_multiple_diffs(tmp_path: Path) -> None:
    """When runs differ in multiple fields, labels fall back to run_id."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", learning_rate=1e-4, run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", learning_rate=1e-3, run_id="run2")

    cd = load_comparison_data([run1, run2])

    assert cd.labels == ["run1", "run2"]


def test_load_comparison_data_custom_labels(tmp_path: Path) -> None:
    """Explicit labels are passed through unchanged."""
    run1 = _make_run_dir(tmp_path / "r1", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    cd = load_comparison_data([run1, run2], labels=["My ABMIL", "My CLAM"])

    assert cd.labels == ["My ABMIL", "My CLAM"]


# ---------------------------------------------------------------------------
# compare_runs (end-to-end HTML)
# ---------------------------------------------------------------------------


def test_compare_runs_creates_html(tmp_path: Path) -> None:
    """compare_runs writes a valid HTML file."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    report_path = compare_runs([run1, run2], output_path=tmp_path / "comparison.html")

    assert report_path.exists()
    html = report_path.read_text()
    assert "<!DOCTYPE html>" in html
    assert "Run Comparison" in html


def test_compare_runs_default_output_path(tmp_path: Path) -> None:
    """Default output_path is parent-of-first-run-dir / comparison.html."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    report_path = compare_runs([run1, run2])

    assert report_path == run1.parent / "comparison.html"
    assert report_path.exists()
