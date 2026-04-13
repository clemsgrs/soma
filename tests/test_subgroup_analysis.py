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
from soma.evaluation.metrics import bh_correct, compare_run_metrics
from soma.evaluation.metrics import (
    _extract_arrays,
    compute_subgroup_metrics,
    compute_subgroup_stats,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.reporting import generate_report, load_run_data
from soma.reporting.data import FoldData, RunData, aggregate_fold_predictions, run_data_from_result
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
        "evaluation": {
            "metrics": ["auroc"],
            "subgroups": {
                "columns": subgroup_columns or [],
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
    (run_dir / "summary.json").write_text(json.dumps({"test/auroc_mean": 0.85, "test/auroc_std": 0.02}))

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
        df.to_csv(fold_dir / "predictions_test.csv", index=False)
    else:
        df[["sample_id", "true_label", "predicted_label", "prob_0", "prob_1"]].to_csv(
            fold_dir / "predictions_test.csv", index=False
        )

    if subgroup_metrics_data is not None:
        (fold_dir / "subgroup_metrics_test.json").write_text(json.dumps(subgroup_metrics_data))

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

    fold_split = FoldSplit(train=("s0",), tune=("s1",), tests={"test": ("s0",)})
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
            evaluation=EvalConfig(subgroups=SubgroupConfig(columns=["nonexistent"])),

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
        },
        "stats": {
            "sex": {"M": {"auroc": 0.12}, "F": {"auroc": 0.12}},
        },
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
    assert "test" in fd.subgroup_metrics
    assert "metrics" in fd.subgroup_metrics["test"]
    assert fd.subgroup_metrics["test"]["metrics"]["sex"]["M"]["auroc"] == 0.82


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


def test_subgroup_section_contains_css_highlight_classes(tmp_path: Path) -> None:
    """When subgroup data is present, the HTML report uses subgroup highlight CSS classes."""
    run_dir = _make_run_dir(tmp_path, subgroup_columns=["sex"])
    html = generate_report(run_dir).read_text()
    # CSS classes for highlighting must be defined in the report
    assert "subgroup-sig" in html or "subgroup-flag" in html or "subgroup-sig-small" in html


# ---------------------------------------------------------------------------
# aggregate_fold_predictions
# ---------------------------------------------------------------------------


def _fold(df: pd.DataFrame, split_name: str = "test") -> FoldData:
    """Wrap a DataFrame as a minimal FoldData for testing."""
    return FoldData(
        fold=0,
        training_history=[],
        tune_metrics={},
        test_metrics={split_name: {}},
        predictions={split_name: df},
    )


def test_aggregate_fold_predictions_no_duplicates_unchanged() -> None:
    """Two folds with disjoint sample_ids are simply concatenated."""
    fold0 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [0], "predicted_label": [0],
        "prob_0": [0.8], "prob_1": [0.2],
    }))
    fold1 = _fold(pd.DataFrame({
        "sample_id": ["s1"], "true_label": [1], "predicted_label": [1],
        "prob_0": [0.3], "prob_1": [0.7],
    }))
    result = aggregate_fold_predictions([fold0, fold1], "test")
    assert len(result) == 2
    assert set(result["sample_id"]) == {"s0", "s1"}


def test_aggregate_fold_predictions_averages_probs() -> None:
    """Same sample in two folds has its probs averaged."""
    fold0 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [0], "predicted_label": [0],
        "prob_0": [0.8], "prob_1": [0.2],
    }))
    fold1 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [0], "predicted_label": [1],
        "prob_0": [0.6], "prob_1": [0.4],
    }))
    result = aggregate_fold_predictions([fold0, fold1], "test")
    assert len(result) == 1
    assert result["prob_0"].iloc[0] == pytest.approx(0.7)
    assert result["prob_1"].iloc[0] == pytest.approx(0.3)


def test_aggregate_fold_predictions_recomputes_predicted_label() -> None:
    """predicted_label is recomputed as argmax of averaged probs."""
    fold0 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [1], "predicted_label": [0],
        "prob_0": [0.6], "prob_1": [0.4],   # fold 0: class 0 wins
    }))
    fold1 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [1], "predicted_label": [1],
        "prob_0": [0.3], "prob_1": [0.7],   # fold 1: class 1 wins
    }))
    result = aggregate_fold_predictions([fold0, fold1], "test")
    # mean prob_0=0.45, mean prob_1=0.55 → argmax = 1
    assert result["predicted_label"].iloc[0] == 1


def test_aggregate_fold_predictions_regression() -> None:
    """Regression predictions (predicted_value) are averaged across folds."""
    fold0 = _fold(pd.DataFrame({"sample_id": ["s0"], "true_label": [3.0], "predicted_value": [2.8]}))
    fold1 = _fold(pd.DataFrame({"sample_id": ["s0"], "true_label": [3.0], "predicted_value": [3.2]}))
    result = aggregate_fold_predictions([fold0, fold1], "test")
    assert len(result) == 1
    assert result["predicted_value"].iloc[0] == pytest.approx(3.0)


def test_aggregate_fold_predictions_preserves_subgroup_columns() -> None:
    """Metadata columns (subgroup) are kept from the first fold."""
    fold0 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [0], "predicted_label": [0],
        "prob_0": [0.7], "prob_1": [0.3], "sex": ["M"],
    }))
    fold1 = _fold(pd.DataFrame({
        "sample_id": ["s0"], "true_label": [0], "predicted_label": [0],
        "prob_0": [0.7], "prob_1": [0.3], "sex": ["M"],
    }))
    result = aggregate_fold_predictions([fold0, fold1], "test")
    assert result["sex"].iloc[0] == "M"


# ---------------------------------------------------------------------------
# bh_correct
# ---------------------------------------------------------------------------


def test_bh_correct_empty() -> None:
    assert bh_correct([]) == []


def test_bh_correct_single() -> None:
    assert bh_correct([0.03]) == [0.03]


def test_bh_correct_all_significant() -> None:
    """Clearly small p-values remain below 0.05 after correction."""
    adjusted = bh_correct([0.001, 0.002, 0.003])
    assert all(p < 0.05 for p in adjusted)


def test_bh_correct_none_significant() -> None:
    """Large p-values remain above 0.05 after correction."""
    adjusted = bh_correct([0.4, 0.5, 0.6])
    assert all(p > 0.05 for p in adjusted)


def test_bh_correct_preserves_order_relationship() -> None:
    """Adjusted p-values maintain the same rank order as raw p-values."""
    raw = [0.01, 0.04, 0.20, 0.50]
    adjusted = bh_correct(raw)
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]


def test_bh_correct_known_values() -> None:
    """BH step-up on a known example matches hand-computed output.

    Input: [0.01, 0.04, 0.10, 0.20], n=4
    Ranks (ascending): p_(1)=0.01, p_(2)=0.04, p_(3)=0.10, p_(4)=0.20
    Adjusted (step-up from largest):
      rank 4: 0.20 * 4/4 = 0.20
      rank 3: min(0.20, 0.10 * 4/3) = min(0.20, 0.133) = 0.133
      rank 2: min(0.133, 0.04 * 4/2) = min(0.133, 0.08) = 0.08
      rank 1: min(0.08, 0.01 * 4/1) = min(0.08, 0.04) = 0.04
    """
    raw = [0.01, 0.04, 0.10, 0.20]
    adjusted = bh_correct(raw)
    assert adjusted == pytest.approx([0.04, 0.08, 0.1333, 0.20], abs=1e-3)


# ---------------------------------------------------------------------------
# compare_run_metrics
# ---------------------------------------------------------------------------


def test_compare_run_metrics_returns_p_values() -> None:
    """compare_run_metrics returns a p-value per run; best run gets 1.0."""
    # Run 0 consistently better than run 1
    run0 = [0.9, 0.88, 0.91, 0.89]
    run1 = [0.7, 0.72, 0.68, 0.71]
    p_values = compare_run_metrics([run0, run1], n_permutations=500, seed=0)

    assert len(p_values) == 2
    assert p_values[0] == 1.0  # best run
    assert p_values[1] is not None
    assert 0.0 <= p_values[1] <= 1.0


def test_compare_run_metrics_significant_difference() -> None:
    """A clearly better run produces a low p-value for the other run.

    With 8 folds the minimum sign-permutation p-value is 2/2^8 ≈ 0.008,
    well below 0.05 for a clear separation.
    """
    run0 = [0.95, 0.94, 0.96, 0.95, 0.94, 0.96, 0.95, 0.93]
    run1 = [0.60, 0.61, 0.59, 0.62, 0.60, 0.58, 0.61, 0.63]
    p_values = compare_run_metrics([run0, run1], n_permutations=500, seed=42)

    assert p_values[1] < 0.05


def test_compare_run_metrics_no_difference() -> None:
    """Runs with identical performance produce a high p-value."""
    vals = [0.80, 0.81, 0.79, 0.80, 0.82]
    p_values = compare_run_metrics([vals, vals[:]], n_permutations=200, seed=42)

    # Both identical → p-value should be high (≥ 0.05)
    assert p_values[1] is not None
    assert p_values[1] >= 0.05


def test_compare_run_metrics_single_fold_returns_none() -> None:
    """Single-fold runs cannot be tested; all p-values are None."""
    p_values = compare_run_metrics([[0.85], [0.80]])
    assert all(p is None for p in p_values)


def test_compare_run_metrics_mismatched_folds_returns_none() -> None:
    """Runs with different fold counts cannot be compared."""
    p_values = compare_run_metrics([[0.8, 0.82], [0.7, 0.72, 0.71]])
    assert all(p is None for p in p_values)


def test_compare_run_metrics_three_runs() -> None:
    """With 3 runs, best gets p=1.0; other two get a p-value."""
    run0 = [0.95, 0.94, 0.96]  # best
    run1 = [0.70, 0.71, 0.69]
    run2 = [0.80, 0.81, 0.79]
    p_values = compare_run_metrics([run0, run1, run2], n_permutations=200, seed=0)

    assert len(p_values) == 3
    assert p_values[0] == 1.0
    assert all(p is None or 0.0 <= p <= 1.0 for p in p_values)
