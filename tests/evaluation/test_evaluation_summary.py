"""Shared fold/seed aggregation: sample std, nan-aware means."""

from __future__ import annotations

import math

import numpy as np
import pytest

from soma.evaluation.summary import sample_std, summarize_values
from soma.pipeline.aggregate import summarize_fold_metric_dicts


class TestSampleStd:
    def test_uses_ddof_one(self):
        values = [0.8, 0.6, 0.7, 0.9, 0.5]
        assert sample_std(values) == pytest.approx(np.std(values, ddof=1))
        assert sample_std(values) > np.std(values)

    def test_single_value_is_nan(self):
        assert math.isnan(sample_std([0.7]))
        assert math.isnan(sample_std([]))


class TestSummarizeValues:
    def test_mean_over_finite_values_and_counts_nan(self):
        stats = summarize_values([0.8, float("nan"), 0.6])
        assert stats.mean == pytest.approx(0.7)
        assert stats.std == pytest.approx(np.std([0.8, 0.6], ddof=1))
        assert stats.n == 2
        assert stats.n_nan == 1

    def test_all_nan_is_nan(self):
        stats = summarize_values([float("nan"), float("nan")])
        assert math.isnan(stats.mean) and math.isnan(stats.std)
        assert stats.n == 0 and stats.n_nan == 2


class TestFoldSummary:
    def test_nan_fold_is_excluded_and_counted(self):
        per_fold = [
            {"test": {"auroc": 0.8, "accuracy": 0.9}},
            {"test": {"auroc": float("nan"), "accuracy": 0.7}},
            {"test": {"auroc": 0.6, "accuracy": 0.8}},
        ]
        summary = summarize_fold_metric_dicts(per_fold, include_tune=False)
        assert summary["test/auroc_mean"] == pytest.approx(0.7)
        assert summary["test/auroc_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
        assert summary["test/auroc_nan_folds"] == 1
        # Metrics finite on every fold carry no nan counter.
        assert "test/accuracy_nan_folds" not in summary
        assert summary["test/accuracy_std"] == pytest.approx(np.std([0.9, 0.7, 0.8], ddof=1))
