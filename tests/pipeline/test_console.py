"""The run-summary panel a completed run prints."""

from __future__ import annotations

import io

from rich.console import Console

from soma.pipeline.console import build_completed_run_panel


def _rendered(summary: dict[str, float]) -> str:
    out = io.StringIO()
    Console(file=out, width=200).print(build_completed_run_panel(summary_metrics=summary))
    return out.getvalue()


def test_the_completed_run_panel_shows_the_pooled_metric_when_subgroup_keys_come_first():
    summary = {
        "tune/centre=A/auroc": 0.9,
        "tune/mean_over_centre/auroc": 0.8,
        "tune/accuracy": 0.7,
        "tune/coverage": 1.0,
        "tune/num_samples": 8.0,
        "tune/num_predicted": 8.0,
    }

    assert "tune coverage  accuracy=0.7000" in _rendered(summary)


def test_the_completed_run_panel_shows_the_mean_counts_per_fold_over_folds():
    # Folds of 11, 10 and 10 test samples; fold 0 left one out: the rounded means would
    # read 10/10 next to a coverage below 1.
    summary = {
        "test/auroc_mean": 0.8,
        "test/auroc_std": 0.1,
        "test/num_samples_mean": 31 / 3,
        "test/num_samples_std": 0.58,
        "test/num_predicted_mean": 10.0,
        "test/num_predicted_std": 0.0,
        "test/coverage_mean": (10 / 11 + 1 + 1) / 3,
        "test/coverage_std": 0.05,
    }

    assert "test coverage  auroc=0.8000  ·  97.0% (10/10.33 per fold)" in _rendered(summary)
