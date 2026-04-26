"""Tests for soma.evaluation — metrics dispatcher and reports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from soma.evaluation.metrics import (
    DEFAULT_METRICS,
    VALID_METRICS,
    compute_metrics,
    resolve_metrics,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction


# ---------------------------------------------------------------------------
# resolve_metrics
# ---------------------------------------------------------------------------


class TestResolveMetrics:
    def test_empty_list_returns_defaults(self):
        resolved = resolve_metrics("binary_classification", [])
        assert resolved == DEFAULT_METRICS["binary_classification"]

    def test_explicit_list_returned_as_is(self):
        resolved = resolve_metrics("binary_classification", ["auroc", "f1"])
        assert set(resolved) == {"auroc", "f1"}

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            resolve_metrics("binary_classification", ["qwk"])

    def test_multiclass_defaults(self):
        resolved = resolve_metrics("multiclass_classification", [])
        assert resolved == DEFAULT_METRICS["multiclass_classification"]

    def test_ordinal_defaults(self):
        resolved = resolve_metrics("ordinal_classification", [])
        assert resolved == DEFAULT_METRICS["ordinal_classification"]

    def test_regression_defaults(self):
        resolved = resolve_metrics("regression", [])
        assert resolved == DEFAULT_METRICS["regression"]

    def test_multiple_invalid_metrics_listed_in_error(self):
        with pytest.raises(ValueError, match="bad1"):
            resolve_metrics("regression", ["mae", "bad1", "bad2"])


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------


class TestBinaryClassificationMetrics:
    def _probs(self, scores):
        """Build (N, 2) probability array from positive-class scores."""
        p = np.array(scores)
        return np.column_stack([1 - p, p])

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = self._probs([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics("binary_classification", ["accuracy", "balanced_accuracy", "auroc"], y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["auroc"] == 1.0

    def test_all_valid_metrics_computable(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = self._probs([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        metrics = list(VALID_METRICS["binary_classification"])
        m = compute_metrics("binary_classification", metrics, y_true, y_pred, y_prob)
        for key in metrics:
            assert key in m
            assert isinstance(m[key], float)

    def test_sensitivity_and_specificity(self):
        # TP=2, FN=0, TN=2, FP=0 → sens=1, spec=1
        y_true = np.array([0, 0, 1, 1])
        y_prob = self._probs([0.1, 0.2, 0.8, 0.9])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics("binary_classification", ["sensitivity", "specificity"], y_true, y_pred, y_prob)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_single_class_auroc_fallback(self):
        y_true = np.array([1, 1, 1])
        y_prob = self._probs([0.8, 0.9, 0.7])
        y_pred = np.array([1, 1, 1])
        m = compute_metrics("binary_classification", ["auroc"], y_true, y_pred, y_prob)
        assert m["auroc"] == 0.5

    def test_auprc(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = self._probs([0.1, 0.9, 0.2, 0.8])
        y_pred = np.array([0, 1, 0, 1])
        m = compute_metrics("binary_classification", ["auprc"], y_true, y_pred, y_prob)
        assert 0.0 <= m["auprc"] <= 1.0

    def test_single_class_auprc_is_reported_as_undefined(self):
        y_true = np.array([1, 1, 1])
        y_prob = self._probs([0.8, 0.9, 0.7])
        y_pred = np.array([1, 1, 1])
        m = compute_metrics("binary_classification", ["auprc"], y_true, y_pred, y_prob)
        assert np.isnan(m["auprc"])

    def test_mcc(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = self._probs([0.1, 0.2, 0.8, 0.9])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics("binary_classification", ["mcc"], y_true, y_pred, y_prob)
        assert m["mcc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multiclass classification metrics
# ---------------------------------------------------------------------------


class TestMulticlassClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_pred = np.array([0, 1, 2])
        m = compute_metrics("multiclass_classification", ["accuracy", "balanced_accuracy", "auroc_macro"], y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["auroc_macro"] == 1.0

    def test_all_valid_metrics_computable(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])
        y_pred = y_prob.argmax(axis=1)
        metrics = list(VALID_METRICS["multiclass_classification"])
        m = compute_metrics("multiclass_classification", metrics, y_true, y_pred, y_prob)
        for key in metrics:
            assert key in m
            assert isinstance(m[key], float)

    def test_single_class_auroc_macro_fallback(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        y_pred = np.array([1, 1, 1])
        m = compute_metrics("multiclass_classification", ["auroc_macro"], y_true, y_pred, y_prob)
        assert m["auroc_macro"] == 0.5

    def test_f1_weighted_vs_macro(self):
        y_true = np.array([0, 0, 0, 1])  # imbalanced
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
        y_pred = np.array([0, 0, 0, 1])
        m = compute_metrics("multiclass_classification", ["f1_macro", "f1_weighted"], y_true, y_pred, y_prob)
        # With imbalanced data, weighted != macro
        assert isinstance(m["f1_macro"], float)
        assert isinstance(m["f1_weighted"], float)

    def test_qwk_is_supported_for_multiclass(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_metrics("multiclass_classification", ["qwk"], y_true, y_pred, y_prob=None)
        assert m["qwk"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Ordinal classification metrics
# ---------------------------------------------------------------------------


class TestOrdinalClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        m = compute_metrics("ordinal_classification", ["qwk", "accuracy", "balanced_accuracy"], y_true, y_pred)
        assert m["qwk"] == pytest.approx(1.0, abs=1e-6)
        assert m["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert m["balanced_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_all_valid_metrics_computable(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 4])
        metrics = list(VALID_METRICS["ordinal_classification"])
        m = compute_metrics("ordinal_classification", metrics, y_true, y_pred)
        for key in metrics:
            assert key in m
            assert isinstance(m[key], float)

    def test_qwk_penalises_large_errors_more(self):
        y_true = np.array([0, 0, 5, 5])
        y_near = np.array([1, 1, 4, 4])
        y_far = np.array([5, 5, 0, 0])
        qwk_near = compute_metrics("ordinal_classification", ["qwk"], y_true, y_near)["qwk"]
        qwk_far = compute_metrics("ordinal_classification", ["qwk"], y_true, y_far)["qwk"]
        assert qwk_near > qwk_far

    def test_linear_wk(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_metrics("ordinal_classification", ["linear_wk"], y_true, y_pred)
        assert m["linear_wk"] == pytest.approx(1.0, abs=1e-6)

    def test_single_class_qwk_is_safe(self):
        y_true = np.array([2, 2, 2])
        y_pred = np.array([2, 2, 2])
        m = compute_metrics("ordinal_classification", ["qwk"], y_true, y_pred)
        assert isinstance(m["qwk"], float)

    def test_spearman(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        m = compute_metrics("ordinal_classification", ["spearman"], y_true, y_pred)
        assert m["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_constant_input_spearman_uses_finite_fallback(self):
        y_true = np.array([2, 2, 2])
        y_pred = np.array([2, 2, 2])
        m = compute_metrics("ordinal_classification", ["spearman"], y_true, y_pred)
        assert m["spearman"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics("regression", ["mse", "mae", "r2"], y_true, y_pred)
        assert m["mse"] == pytest.approx(0.0, abs=1e-6)
        assert m["mae"] == pytest.approx(0.0, abs=1e-6)
        assert m["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_all_valid_metrics_computable(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        metrics = list(VALID_METRICS["regression"])
        m = compute_metrics("regression", metrics, y_true, y_pred)
        for key in metrics:
            assert key in m
            assert isinstance(m[key], float)

    def test_known_mse_and_mae(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        m = compute_metrics("regression", ["mse", "mae"], y_true, y_pred)
        assert m["mse"] == pytest.approx(1.5, abs=1e-6)
        assert m["mae"] == pytest.approx(1.0, abs=1e-6)

    def test_rmse(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        m = compute_metrics("regression", ["mse", "rmse"], y_true, y_pred)
        assert m["rmse"] == pytest.approx(m["mse"] ** 0.5, abs=1e-6)

    def test_mse_and_mae_non_negative(self):
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(20)
        y_pred = rng.standard_normal(20)
        m = compute_metrics("regression", ["mse", "mae"], y_true, y_pred)
        assert m["mse"] >= 0
        assert m["mae"] >= 0

    def test_pearson_and_spearman_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics("regression", ["pearson", "spearman"], y_true, y_pred)
        assert m["pearson"] == pytest.approx(1.0, abs=1e-6)
        assert m["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_pearson_and_spearman_constant_inputs_use_finite_fallback(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        m = compute_metrics("regression", ["pearson", "spearman"], y_true, y_pred)
        assert m["pearson"] == pytest.approx(0.0)
        assert m["spearman"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestEvaluationReport:
    def test_to_dict_round_trip(self):
        pred = SamplePrediction(
            sample_id="s1",
            true_label=0,
            predicted_label=1,
            probabilities=[0.3, 0.7],
        )
        report = EvaluationReport(
            split="test",
            metrics={"accuracy": 0.85, "auroc": 0.92},
            predictions=[pred],
        )
        d = report.to_dict()
        assert d["split"] == "test"
        assert d["metrics"]["accuracy"] == 0.85
        assert len(d["predictions"]) == 1
        assert d["predictions"][0]["sample_id"] == "s1"

    def test_json_serializable(self):
        pred = SamplePrediction(
            sample_id="s1",
            true_label=0,
            predicted_label=0,
            probabilities=[0.9, 0.1],
        )
        report = EvaluationReport(
            split="tune",
            metrics={"accuracy": 1.0},
            predictions=[pred],
        )
        json_str = json.dumps(report.to_dict())
        loaded = json.loads(json_str)
        assert loaded["split"] == "tune"
