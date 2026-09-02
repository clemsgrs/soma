"""Shared fold/seed aggregation: sample std, nan-aware means."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from soma.atomic_io import atomic_write_json, atomic_write_text
from soma.evaluation.summary import sample_std, summarize_values
from soma.pipeline import _summarize_fold_metric_dicts


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
        summary = _summarize_fold_metric_dicts(per_fold, include_tune=False)
        assert summary["test/auroc_mean"] == pytest.approx(0.7)
        assert summary["test/auroc_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
        assert summary["test/auroc_nan_folds"] == 1
        # Metrics finite on every fold carry no nan counter.
        assert "test/accuracy_nan_folds" not in summary
        assert summary["test/accuracy_std"] == pytest.approx(np.std([0.9, 0.7, 0.8], ddof=1))


class TestAtomicWrites:
    def test_json_lands_without_staging_leftovers(self, tmp_path: Path):
        target = tmp_path / "metrics.json"
        atomic_write_json(target, {"a": 1})
        assert json.loads(target.read_text()) == {"a": 1}
        assert list(tmp_path.iterdir()) == [target]

    def test_failed_write_keeps_previous_content(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "summary.json"
        atomic_write_text(target, "old")

        import soma.atomic_io as atomic_io

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(atomic_io.os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_text(target, "new")
        assert target.read_text() == "old"
