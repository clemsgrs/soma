"""Subgroup breakdowns across folds: every fold's ``subgroup_metrics_<split>.json`` and
the ``summary.json`` keys aggregated from them."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from soma.data import CachedSetSource
from soma.pipeline import Pipeline
from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk
from tests.experiment_configs import experiment

FOLDS = 3
BLOCK = 8


def _three_fold_cohort(root: Path) -> tuple[Path, Path, Path]:
    """24 samples in three blocks of 8 with pooled feature bags. Fold ``f`` tunes on
    block ``f``, tests on block ``f + 1`` and trains on the third. Every block holds
    two samples of each class in each of the centres A and B."""
    ids = [f"s{i:02d}" for i in range(FOLDS * BLOCK)]
    pd.DataFrame(
        {
            "sample_id": ids,
            "image_path": [f"/slides/{sid}.svs" for sid in ids],
            "label": ["tumor" if i % 2 else "normal" for i in range(len(ids))],
            "centre": ["A" if (i % BLOCK) // 2 % 2 == 0 else "B" for i in range(len(ids))],
        }
    ).to_csv(root / "dataset.csv", index=False)
    roles = ("tune", "test", "train")
    rows = [
        {"fold": fold, "sample_id": sid, "split": roles[(i // BLOCK - fold) % FOLDS]}
        for fold in range(FOLDS)
        for i, sid in enumerate(ids)
    ]
    pd.DataFrame(rows).to_csv(root / "splits.csv", index=False)
    features = root / "features"
    features.mkdir()
    torch.manual_seed(0)
    for i, sid in enumerate(ids):
        torch.save(torch.randn(5 + i % 4, 16), features / f"{sid}.pt")
    return root / "dataset.csv", root / "splits.csv", features


def _run(tmp_path: Path, *, holdout_test: bool):
    dataset_csv, splits_csv, features = _three_fold_cohort(tmp_path)
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "outputs",
        aggregator="mean_pool",
        loaders={"batch_size": 2},
        trainer={"max_epochs": 2},
        evaluation={"metrics": ["auroc"], "subgroups": ["centre"], "holdout_test": holdout_test},
    )
    return Pipeline(config, source=CachedSetSource(features)).run()


@pytest.mark.parametrize("holdout_test", [True, False], ids=["holdout_test", "with_test"])
def test_every_fold_breaks_tune_down_by_subgroup_and_the_summary_aggregates_it(tmp_path: Path, holdout_test):
    result = _run(tmp_path, holdout_test=holdout_test)
    summary = json.loads((result.run_dir / "summary.json").read_text())
    scored = ["tune"] if holdout_test else ["tune", "test"]
    breakdowns = {
        split: [
            json.loads((result.run_dir / f"fold_{fold}" / f"subgroup_metrics_{split}.json").read_text())
            for fold in range(FOLDS)
        ]
        for split in scored
    }

    for split, folds in breakdowns.items():
        for breakdown in folds:
            assert set(breakdown["metrics"]["centre"]) == {"A", "B"}
            assert breakdown["mean_over"]["centre"]["auroc_groups"] == 2
    # The summary holds the splits it always held: tune only when the test is held out.
    summarized = "tune" if holdout_test else "test"
    folds = breakdowns[summarized]
    for group in ("A", "B"):
        values = [breakdown["metrics"]["centre"][group]["auroc"] for breakdown in folds]
        assert summary[f"{summarized}/centre={group}/auroc_mean"] == pytest.approx(sum(values) / FOLDS)
        assert f"{summarized}/centre={group}/auroc_std" in summary
    means = [breakdown["mean_over"]["centre"]["auroc"] for breakdown in folds]
    assert summary[f"{summarized}/mean_over_centre/auroc_mean"] == pytest.approx(sum(means) / FOLDS)
    if not holdout_test:
        assert not any(key.startswith("tune/") for key in summary)
    # Tune predictions are still not written.
    assert not list(result.run_dir.glob("fold_*/predictions_tune*"))


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
