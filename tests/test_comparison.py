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


def _make_regression_predictions(n: int = 8, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": [float(i) for i in range(n)],
        "predicted_value": [float(i) + offset for i in range(n)],
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
    output_root = tmp_path.parent / "output"

    metrics = ["auroc"]
    config = {
        "dataset_csv": "/data/dataset.csv",
        "splits_csv": "/data/splits.csv",
        "output_root": str(output_root),
        "task": {"name": "binary_classification", "params": {}},
        "evaluation": {"metrics": metrics, "subgroups": {"columns": []}},
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


def _make_regression_run_dir(
    tmp_path: Path,
    *,
    run_id: str,
    mae: float,
    offset: float,
) -> Path:
    run_dir = _make_run_dir(tmp_path, run_id=run_id)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    config["task"] = {"name": "regression", "params": {}}
    config["evaluation"]["metrics"] = ["mae"]
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "summary.json").write_text(json.dumps({"test/mae_mean": mae, "test/mae_std": 0.01}))
    fold_dir = run_dir / "fold_0"
    (fold_dir / "metrics.json").write_text(json.dumps({"tune": {"mae": mae}, "test": {"mae": mae}}))
    _make_regression_predictions(offset=offset).to_csv(fold_dir / "predictions_test.csv", index=False)
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

    report_dir = tmp_path / "comparison_reports"
    report_path = compare_runs([run1, run2], output_dir=report_dir)

    assert report_path.exists()
    assert report_path == report_dir / "index.html"
    html = report_path.read_text()
    assert "<!DOCTYPE html>" in html
    assert "Run Comparison" in html


def test_compare_runs_writes_manifest(tmp_path: Path) -> None:
    """compare_runs writes a manifest.json describing the compared runs."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    report_dir = tmp_path / "comparison_reports"
    compare_runs([run1, run2], output_dir=report_dir, labels=["A", "B"])

    manifest_path = report_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["labels"] == ["A", "B"]
    assert len(manifest["runs"]) == 2
    assert manifest["runs"][0]["label"] == "A"
    assert manifest["runs"][0]["run_id"] == "run1"
    assert manifest["runs"][0]["run_dir"] == str(run1.resolve())
    assert manifest["runs"][1]["run_id"] == "run2"
    assert manifest["runs"][1]["run_dir"] == str(run2.resolve())
    # config_digest should differ across runs that differ in config
    assert manifest["runs"][0]["config_digest"] != manifest["runs"][1]["config_digest"]
    assert "generated_at" in manifest


def test_compare_runs_default_output_dir(tmp_path: Path) -> None:
    """Default output_dir is a comparison bundle under the shared output root."""
    run1 = _make_run_dir(tmp_path / "r1", aggregator="abmil", run_id="run1")
    run2 = _make_run_dir(tmp_path / "r2", aggregator="clam_sb", run_id="run2")

    report_path = compare_runs([run1, run2])

    assert report_path.parent.parent == tmp_path / "output" / "comparisons"
    assert report_path.parent.name.startswith("abmil-vs-clam-sb__")
    assert report_path.name == "index.html"
    assert report_path.exists()


def test_compare_runs_overview_ranks_runs_and_metrics(tmp_path: Path) -> None:
    """The overview tab should rank runs and order metrics by best score."""
    metric_names = ["auroc", "balanced_accuracy"]
    run_specs = [
        ("alpha", 0.94, 0.83),
        ("beta", 0.89, 0.79),
        ("gamma", 0.82, 0.77),
        ("delta", 0.75, 0.73),
    ]
    run_dirs = []
    for idx, (run_id, auroc, bal_acc) in enumerate(run_specs):
        run_dir = _make_run_dir(
            tmp_path / run_id,
            aggregator=f"model_{idx}",
            run_id=run_id,
            n_folds=2,
        )
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        config["evaluation"]["metrics"] = metric_names
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        summary = {
            "test/auroc_mean": auroc,
            "test/auroc_std": 0.01,
            "test/balanced_accuracy_mean": bal_acc,
            "test/balanced_accuracy_std": 0.02,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        run_dirs.append(run_dir)

    report_path = compare_runs(
        run_dirs,
        output_dir=tmp_path / "comparison",
        labels=[spec[0] for spec in run_specs],
    )
    html = report_path.read_text()

    assert 'class="tab-group"' in html
    assert "overview-layout" in html
    assert "Cross-run comparison" in html
    assert "Dataset context" in html
    assert "Overview" in html
    assert "Train plots" in html
    assert "Test results" in html
    assert "Statistical analysis" in html
    assert "Configuration" in html
    assert "Dataset CSV" in html
    assert "Splits CSV" in html

    overview_start = html.index('id="overview-tab"')
    overview_end = html.index('id="train-plots-tab"')
    overview_html = html[overview_start:overview_end]

    assert overview_html.index('data-metric="auroc"') < overview_html.index('data-metric="balanced_accuracy"')
    assert overview_html.index("alpha") < overview_html.index("beta") < overview_html.index("gamma") < overview_html.index("delta")
    assert 'podium-gold' in overview_html
    assert 'podium-silver' in overview_html
    assert 'podium-bronze' in overview_html
    assert 'soma-brand-rank' in overview_html
    assert 'Rank</th>' in overview_html
    assert 'rank-link' in overview_html
    assert 'Open report for alpha' in overview_html
    assert 'Metrics' not in overview_html
    assert 'Dataset CSV' in overview_html
    assert 'Splits CSV' in overview_html

    config_start = html.index('id="configuration-tab"')
    config_html = html[config_start:]
    assert "Varying fields" in config_html
    assert "Shared fields" in config_html


def test_compare_runs_treats_error_metrics_as_lower_is_better(tmp_path: Path) -> None:
    good = _make_regression_run_dir(tmp_path / "good", run_id="good", mae=0.1, offset=0.1)
    bad = _make_regression_run_dir(tmp_path / "bad", run_id="bad", mae=0.8, offset=0.8)

    report_path = compare_runs(
        [bad, good],
        output_dir=tmp_path / "comparison-regression",
        labels=["bad", "good"],
    )
    html = report_path.read_text()
    overview_html = html[html.index('id="overview-tab"'):html.index('id="train-plots-tab"')]
    test_results_html = html[html.index('id="test-results-tab"'):html.index('id="statistical-analysis-tab"')]

    assert overview_html.index("good") < overview_html.index("bad")
    assert "<td class=''><strong>0.800</strong>" in test_results_html
    assert "<td class='best-val'><strong>0.100</strong>" in test_results_html
