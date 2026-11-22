"""Subgroup breakdowns across folds: every fold's ``subgroup_metrics_<split>.json`` and
the ``summary.json`` keys aggregated from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk


def _write_fold(fold_dir: Path, *, pooled: float, groups: dict[str, float]) -> None:
    """One fold's tune ``metrics.json`` and ``subgroup_metrics_tune.json`` (AUROC by
    centre), as a run writes them."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics.json").write_text(json.dumps({"tune": {"auroc": pooled}}))
    breakdown = {
        "metrics": {"centre": {group: {"auroc": value, "n": 4} for group, value in groups.items()}},
        "mean_over": {"centre": {"auroc": sum(groups.values()) / len(groups), "auroc_groups": len(groups)}},
        "stats": {"centre": {}},
    }
    (fold_dir / "subgroup_metrics_tune.json").write_text(json.dumps(breakdown))


def test_a_group_absent_from_a_fold_is_left_out_of_its_mean_and_counted(tmp_path: Path):
    # Fold 0's tune split had under 2 samples of centre B, so B is absent there.
    _write_fold(tmp_path / "fold_0", pooled=0.7, groups={"A": 0.8})
    _write_fold(tmp_path / "fold_1", pooled=0.8, groups={"A": 0.6, "B": 0.9})
    _write_fold(tmp_path / "fold_2", pooled=0.9, groups={"A": 0.7, "B": 0.6})

    summary = aggregate_fold_metrics_from_disk(tmp_path, single_fold=False, include_tune=True, num_folds=3)

    # Sample std (ddof=1): A over 0.8, 0.6, 0.7 -> 0.1; B over 0.9, 0.6 -> 0.15 * sqrt(2);
    # the fold means 0.8, 0.75, 0.65 -> sqrt(0.0175 / 3).
    assert summary == pytest.approx(
        {
            "tune/auroc_mean": 0.8,
            "tune/auroc_std": 0.1,
            "tune/centre=A/auroc_mean": 0.7,
            "tune/centre=A/auroc_std": 0.1,
            "tune/centre=B/auroc_mean": 0.75,
            "tune/centre=B/auroc_std": 0.2121320344,
            "tune/centre=B/auroc_nan_folds": 1,
            "tune/mean_over_centre/auroc_mean": 0.7333333333,
            "tune/mean_over_centre/auroc_std": 0.0763762616,
        }
    )
