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
