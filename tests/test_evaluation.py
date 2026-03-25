"""Tests for soma.evaluation — metrics and reports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from soma.evaluation.metrics import compute_classification_metrics
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
