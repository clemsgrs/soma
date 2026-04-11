"""Tests for subgroup analysis: metrics computation, stats, pipeline integration, HTML."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from soma.config import AggregatorConfig, EvalConfig, PipelineConfig, SubgroupConfig, TaskConfig, TrainingConfig
from soma.evaluation.metrics import (
    _extract_arrays,
    compute_subgroup_metrics,
    compute_subgroup_stats,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.reporting import generate_report, load_run_data
from soma.reporting.data import FoldData, RunData, run_data_from_result
from soma.reporting.html import render_report
from soma.training.trainer import EpochLog, TrainResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_preds_df(n: int = 16, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic binary-classification predictions DataFrame."""
    rng = np.random.default_rng(seed)
    true_labels = [i % 2 for i in range(n)]
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": true_labels,
        "predicted_label": true_labels,
        "prob_0": [0.3 if t == 1 else 0.7 for t in true_labels],
        "prob_1": [0.7 if t == 1 else 0.3 for t in true_labels],
        "sex": (["M"] * (n // 2) + ["F"] * (n // 2)),
        "grade": (["low", "high"] * (n // 2)),
    })


def _make_regression_preds_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": [float(i) for i in range(n)],
        "predicted_value": [float(i) + 0.05 for i in range(n)],
        "group": (["A"] * (n // 2) + ["B"] * (n // 2)),
    })


def _make_run_dir(
    tmp_path: Path,
    *,
    subgroup_columns: list[str] | None = None,
    statistical_testing: bool = False,
    subgroup_metrics_data: dict | None = None,
) -> Path:
    """Build a synthetic run directory for reporting tests."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    config = {
        "dataset_csv": "/data/dataset.csv",
        "splits_csv": "/data/splits.csv",
        "output_root": "/output",
        "task": {"name": "binary_classification", "params": {}},
        "eval": {
            "metrics": ["auroc"],
            "subgroups": {
                "columns": subgroup_columns or [],
                "statistical_testing": statistical_testing,
            },
        },
        "encoder": None,
        "aggregator": {"name": "abmil", "params": {}},
        "training": {"seed": 42, "epochs": 4, "learning_rate": 1e-4,
                     "optimizer": "adam", "scheduler": "cosine", "patience": 10, "batch_size": 1},
        "tags": [],
    }
    (run_dir / "config.yaml").write_text(yaml.dump(config))
    (run_dir / "run.yaml").write_text(yaml.dump({"run_id": "test-run", "status": "completed"}))
    (run_dir / "summary.json").write_text(json.dumps({"auroc_mean": 0.85, "auroc_std": 0.02}))

    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "training_history.json").write_text(json.dumps([]))
    (fold_dir / "metrics.json").write_text(json.dumps({
        "tune": {"auroc": 0.82},
        "test": {"auroc": 0.85},
    }))

    df = _make_binary_preds_df()
    if subgroup_columns:
        # Keep subgroup columns in the predictions CSV (as enriched by pipeline)
        df[subgroup_columns].to_csv.__doc__  # access check
        df.to_csv(fold_dir / "predictions.csv", index=False)
    else:
        df[["sample_id", "true_label", "predicted_label", "prob_0", "prob_1"]].to_csv(
            fold_dir / "predictions.csv", index=False
        )

    if subgroup_metrics_data is not None:
        (fold_dir / "subgroup_metrics.json").write_text(json.dumps(subgroup_metrics_data))

    return run_dir


# ---------------------------------------------------------------------------
# _extract_arrays
# ---------------------------------------------------------------------------


def test_extract_arrays_classification() -> None:
    """_extract_arrays returns correct y_true, y_pred, y_prob for classification."""
    df = pd.DataFrame({
        "true_label": [0, 1, 0, 1],
        "predicted_label": [0, 1, 1, 1],
        "prob_0": [0.8, 0.3, 0.4, 0.2],
        "prob_1": [0.2, 0.7, 0.6, 0.8],
    })
    y_true, y_pred, y_prob = _extract_arrays(df, "binary_classification")

    np.testing.assert_array_equal(y_true, [0, 1, 0, 1])
    np.testing.assert_array_equal(y_pred, [0, 1, 1, 1])
    assert y_prob is not None
    assert y_prob.shape == (4, 2)
    np.testing.assert_allclose(y_prob[:, 1], [0.2, 0.7, 0.6, 0.8])


def test_extract_arrays_regression() -> None:
    """_extract_arrays returns y_pred from predicted_value for regression."""
    df = pd.DataFrame({
        "true_label": [1.0, 2.0, 3.0],
        "predicted_value": [1.1, 1.9, 3.2],
    })
    y_true, y_pred, y_prob = _extract_arrays(df, "regression")

    np.testing.assert_allclose(y_true, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_pred, [1.1, 1.9, 3.2])
    assert y_prob is None


# ---------------------------------------------------------------------------
# compute_subgroup_metrics
# ---------------------------------------------------------------------------


def test_compute_subgroup_metrics_binary() -> None:
    """compute_subgroup_metrics returns metrics for each group value."""
    df = _make_binary_preds_df(n=16)
    result = compute_subgroup_metrics("binary_classification", ["auroc"], df, ["sex"])

    assert "sex" in result
    assert "M" in result["sex"]
    assert "F" in result["sex"]
    assert "auroc" in result["sex"]["M"]
    assert result["sex"]["M"]["n"] == 8
    assert result["sex"]["F"]["n"] == 8


def test_compute_subgroup_metrics_includes_n() -> None:
    """Each group dict has an 'n' key with the sample count."""
    df = _make_binary_preds_df(n=16)
    result = compute_subgroup_metrics("binary_classification", ["auroc"], df, ["sex"])

    for group_data in result["sex"].values():
        assert "n" in group_data
        assert isinstance(group_data["n"], int)


def test_compute_subgroup_metrics_skips_single_sample_group() -> None:
    """Groups with n < 2 are skipped."""
    df = pd.DataFrame({
        "sample_id": ["s0", "s1", "s2"],
        "true_label": [0, 1, 0],
        "predicted_label": [0, 1, 0],
        "prob_0": [0.8, 0.3, 0.7],
        "prob_1": [0.2, 0.7, 0.3],
        "group": ["A", "B", "A"],  # B has only 1 sample
    })
    result = compute_subgroup_metrics("binary_classification", ["auroc"], df, ["group"])

    assert "A" in result["group"]
    assert "B" not in result["group"]  # single sample — skipped


def test_compute_subgroup_metrics_missing_column_skipped() -> None:
    """Columns not present in the DataFrame are silently skipped."""
    df = _make_binary_preds_df(n=8)
    result = compute_subgroup_metrics("binary_classification", ["auroc"], df, ["nonexistent"])

    assert result == {}


def test_compute_subgroup_metrics_multiple_columns() -> None:
    """Multiple subgroup columns each produce their own entry."""
    df = _make_binary_preds_df(n=16)
    result = compute_subgroup_metrics("binary_classification", ["auroc"], df, ["sex", "grade"])

    assert "sex" in result
    assert "grade" in result
    assert set(result["grade"].keys()) == {"low", "high"}


# ---------------------------------------------------------------------------
# compute_subgroup_stats
# ---------------------------------------------------------------------------


def test_compute_subgroup_stats_structure() -> None:
    """compute_subgroup_stats returns p-values in [0, 1] for qualifying groups."""
    df = _make_binary_preds_df(n=32)
    # Use a small number of permutations for speed in tests
    result = compute_subgroup_stats(
        "binary_classification", ["auroc"], df, ["sex"],
        n_permutations=50, min_group_size=5,
    )

    assert "sex" in result
    for group, metric_pvals in result["sex"].items():
        assert "auroc" in metric_pvals
        assert 0.0 <= metric_pvals["auroc"] <= 1.0


def test_compute_subgroup_stats_skips_small_groups() -> None:
    """Groups with n < min_group_size are not included in stats output."""
    df = _make_binary_preds_df(n=16)
    # Set min_group_size > group size (8 per group) → no groups qualify
    result = compute_subgroup_stats(
        "binary_classification", ["auroc"], df, ["sex"],
        n_permutations=10, min_group_size=20,
    )

    # sex groups have n=8, below the threshold
    assert result.get("sex", {}) == {}


def test_compute_subgroup_stats_seed_reproducible() -> None:
    """Same seed produces identical p-values across two calls."""
    df = _make_binary_preds_df(n=32)
    kwargs = dict(n_permutations=100, seed=99, min_group_size=5)
    r1 = compute_subgroup_stats("binary_classification", ["auroc"], df, ["sex"], **kwargs)
    r2 = compute_subgroup_stats("binary_classification", ["auroc"], df, ["sex"], **kwargs)

    assert r1["sex"]["M"]["auroc"] == r2["sex"]["M"]["auroc"]


# ---------------------------------------------------------------------------
# SubgroupConfig validation in pipeline
# ---------------------------------------------------------------------------


def test_subgroup_column_validation_raises_for_missing_column(tmp_path: Path) -> None:
    """Configuring a subgroup column that is not in the dataset raises a clear error."""
    from soma.dataset import Dataset, Splits
    from soma.features import FeatureStore
    from soma.pipeline import train_one_fold
    from soma.dataset import FoldSplit

    # Build a minimal dataset with no extra metadata columns
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("sample_id,image_path,label\ns0,/img/s0.tif,0\ns1,/img/s1.tif,1\n")
    dataset = Dataset(dataset_csv)

    fold_split = FoldSplit(train=["s0"], tune=["s1"], test=["s0"])
    feature_store = MagicMock()
    feature_store.has_feature_manifest = False
    feature_store.is_slide_level = True
    feature_store.is_hierarchical = False
    feature_store.feature_dim = 8
    feature_store.available_samples = {"s0", "s1"}

    with pytest.raises(ValueError, match="nonexistent"):
        train_one_fold(
            feature_store=feature_store,
            dataset=dataset,
            fold_split=fold_split,
            task=TaskConfig(name="binary_classification"),
            eval=EvalConfig(subgroups=SubgroupConfig(columns=["nonexistent"])),
            training=TrainingConfig(seed=0, epochs=1),
            fold_dir=tmp_path / "fold_0",
        )


# ---------------------------------------------------------------------------
# Predictions CSV enrichment (pipeline integration)
# ---------------------------------------------------------------------------


def test_predictions_csv_enriched_with_subgroup_columns(tmp_path: Path) -> None:
    """_save_predictions writes subgroup columns when subgroup_data is provided."""
    from soma.evaluation.report import EvaluationReport, SamplePrediction
    from soma.pipeline import _build_subgroup_data, _save_predictions
    from soma.dataset import Dataset

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label,sex\n"
        "s0,/img/s0.tif,0,M\n"
        "s1,/img/s1.tif,1,F\n"
    )
    dataset = Dataset(dataset_csv)

    predictions = [
        SamplePrediction(sample_id="s0", true_label=0, predicted_label=0, probabilities=[0.8, 0.2]),
        SamplePrediction(sample_id="s1", true_label=1, predicted_label=1, probabilities=[0.3, 0.7]),
    ]
    report = EvaluationReport(split="test", metrics={"auroc": 0.9}, predictions=predictions)

    subgroup_data = _build_subgroup_data(dataset, report, ["sex"])
    path = tmp_path / "predictions.csv"
    _save_predictions(report, path, subgroup_data=subgroup_data)

    df = pd.read_csv(path)
    assert "sex" in df.columns
    assert df.loc[df["sample_id"] == "s0", "sex"].iloc[0] == "M"
    assert df.loc[df["sample_id"] == "s1", "sex"].iloc[0] == "F"


def test_predictions_csv_no_extra_cols_when_no_subgroups(tmp_path: Path) -> None:
    """_save_predictions without subgroup_data produces only prediction columns."""
    from soma.evaluation.report import EvaluationReport, SamplePrediction
    from soma.pipeline import _save_predictions

    predictions = [
        SamplePrediction(sample_id="s0", true_label=0, predicted_label=0, probabilities=[0.8, 0.2]),
    ]
    report = EvaluationReport(split="test", metrics={}, predictions=predictions)
    path = tmp_path / "predictions.csv"
    _save_predictions(report, path)

    df = pd.read_csv(path)
    assert set(df.columns) == {"sample_id", "true_label", "predicted_label", "prob_0", "prob_1"}


# ---------------------------------------------------------------------------
# Subgroup metrics JSON saved to disk
# ---------------------------------------------------------------------------


def test_subgroup_metrics_json_saved_and_loaded(tmp_path: Path) -> None:
    """load_run_data reads subgroup_metrics.json when present."""
    sg_data = {
        "metrics": {
            "sex": {
                "M": {"auroc": 0.82, "n": 8},
                "F": {"auroc": 0.91, "n": 8},
            }
        }
    }
    run_dir = _make_run_dir(
        tmp_path,
        subgroup_columns=["sex"],
        subgroup_metrics_data=sg_data,
    )

    run_data = load_run_data(run_dir)

    assert run_data.subgroup_columns == ["sex"]
    fd = run_data.folds[0]
    assert fd.subgroup_metrics is not None
    assert "metrics" in fd.subgroup_metrics
    assert fd.subgroup_metrics["metrics"]["sex"]["M"]["auroc"] == 0.82


def test_subgroup_metrics_json_absent_gives_none(tmp_path: Path) -> None:
    """FoldData.subgroup_metrics is None when subgroup_metrics.json is absent."""
    run_dir = _make_run_dir(tmp_path)  # no subgroup_metrics_data
    run_data = load_run_data(run_dir)

    assert run_data.folds[0].subgroup_metrics is None


# ---------------------------------------------------------------------------
# HTML report subgroup section
# ---------------------------------------------------------------------------


def test_report_contains_subgroup_section_when_configured(tmp_path: Path) -> None:
    """render_report includes Subgroup Analysis when subgroup columns are configured."""
    run_dir = _make_run_dir(tmp_path, subgroup_columns=["sex"])
    html = generate_report(run_dir).read_text()

    assert "Subgroup Analysis" in html
    assert "sex" in html


def test_report_no_subgroup_section_when_not_configured(tmp_path: Path) -> None:
    """render_report omits Subgroup Analysis when no columns are configured."""
    run_dir = _make_run_dir(tmp_path)
    html = generate_report(run_dir).read_text()

    assert "Subgroup Analysis" not in html


def test_stats_panel_shown_only_when_stats_present(tmp_path: Path) -> None:
    """Statistical testing panel appears only when stats data is present in fold data."""
    sg_data_with_stats = {
        "metrics": {"sex": {"M": {"auroc": 0.82, "n": 8}, "F": {"auroc": 0.91, "n": 8}}},
        "stats": {"sex": {"M": {"auroc": 0.12}, "F": {"auroc": 0.12}}},
    }
    run_dir = _make_run_dir(
        tmp_path, subgroup_columns=["sex"], subgroup_metrics_data=sg_data_with_stats
    )
    html = generate_report(run_dir).read_text()
    assert "permutation test" in html.lower()


def test_stats_panel_absent_without_stats(tmp_path: Path) -> None:
    """Statistical testing panel is absent when subgroup_metrics.json has no 'stats' key."""
    sg_data_no_stats = {
        "metrics": {"sex": {"M": {"auroc": 0.82, "n": 8}, "F": {"auroc": 0.91, "n": 8}}},
    }
    run_dir = _make_run_dir(
        tmp_path, subgroup_columns=["sex"], subgroup_metrics_data=sg_data_no_stats
    )
    html = generate_report(run_dir).read_text()
    # No stats → no permutation test panel
    assert "permutation test" not in html.lower()
