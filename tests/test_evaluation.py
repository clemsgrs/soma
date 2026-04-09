"""Tests for soma.evaluation — metrics and reports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from soma.evaluation.metrics import compute_classification_metrics, compute_ordinal_metrics, compute_regression_metrics
from soma.evaluation.report import EvaluationReport, SamplePrediction


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["auc"] == 1.0

    def test_known_values(self):
        """Hand-computed example: 3 correct, 1 wrong out of 4."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1]])
        y_pred = np.array([0, 0, 1, 0])  # last one is wrong
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        assert metrics["accuracy"] == 0.75
        assert metrics["balanced_accuracy"] == 0.75

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        for key in ["accuracy", "balanced_accuracy", "auc", "f1_macro"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_single_class_validation_split_is_safe(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.array([[0.2, 0.8], [0.1, 0.9], [0.4, 0.6]])
        y_pred = np.array([1, 1, 1])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        assert metrics["auc"] == 0.5
        assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Ordinal metrics
# ---------------------------------------------------------------------------


class TestOrdinalMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        metrics = compute_ordinal_metrics(y_true, y_pred)
        assert metrics["qwk"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 4])
        metrics = compute_ordinal_metrics(y_true, y_pred)
        for key in ["qwk", "accuracy", "balanced_accuracy"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_qwk_penalises_large_errors_more(self):
        # One-off errors vs. large errors on same set of true labels
        y_true = np.array([0, 0, 5, 5])
        y_near = np.array([1, 1, 4, 4])   # off by 1
        y_far  = np.array([5, 5, 0, 0])   # off by 5
        qwk_near = compute_ordinal_metrics(y_true, y_near)["qwk"]
        qwk_far  = compute_ordinal_metrics(y_true, y_far)["qwk"]
        assert qwk_near > qwk_far

    def test_single_class_is_safe(self):
        y_true = np.array([2, 2, 2])
        y_pred = np.array([2, 2, 2])
        metrics = compute_ordinal_metrics(y_true, y_pred)
        assert isinstance(metrics["qwk"], float)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])  # constant prediction = mean of [1,2,3,4] = 2.5... actually 2
        metrics = compute_regression_metrics(y_true, y_pred)
        # MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2 + (4-2)^2) / 4 = (1+0+1+4)/4 = 1.5
        assert metrics["mse"] == pytest.approx(1.5, abs=1e-6)
        # MAE = (1+0+1+2)/4 = 1.0
        assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)

    def test_all_metrics_present(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        metrics = compute_regression_metrics(y_true, y_pred)
        for key in ["mse", "mae", "r2"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_mse_and_mae_non_negative(self):
        y_true = np.random.randn(20)
        y_pred = np.random.randn(20)
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0


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
            metrics={"accuracy": 0.85, "auc": 0.92},
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
