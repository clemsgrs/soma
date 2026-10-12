"""Subgroup breakdowns through ``soma.evaluate``: the same metrics over each group of a
target metadata column, plus a permutation p-value of each group against the rest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from soma.config import EvalConfig, SubgroupConfig, TaskConfig, TrainingConfig
from soma.evaluation import Predictions, evaluate, metric_registry
from soma.evaluation.subgroups import subgroup_stats


def _binary(n: int = 16) -> tuple[Predictions, pd.DataFrame]:
    """Perfect binary predictions over ``n`` samples; ``sex`` splits them in halves and
    ``grade`` alternates."""
    labels = [i % 2 for i in range(n)]
    ids = [f"s{i:02d}" for i in range(n)]
    predictions = Predictions(
        pd.DataFrame(
            {
                "sample_id": ids,
                "prob_0": [0.3 if t else 0.7 for t in labels],
                "prob_1": [0.7 if t else 0.3 for t in labels],
                "predicted_label": labels,
            }
        ),
        task_family="binary_classification",
    )
    targets = pd.DataFrame(
        {
            "label": labels,
            "sex": ["M"] * (n // 2) + ["F"] * (n // 2),
            "grade": ["low", "high"] * (n // 2),
        },
        index=pd.Index(ids, name="sample_id"),
    )
    return predictions, targets


def test_each_group_of_a_column_gets_the_metrics_and_its_size():
    predictions, targets = _binary(16)

    breakdown = evaluate(predictions, targets, metrics=["auroc"], subgroups=["sex"]).subgroups

    assert set(breakdown) == {"metrics", "stats"}
    assert breakdown["metrics"]["sex"] == {
        "F": {"auroc": 1.0, "n": 8},
        "M": {"auroc": 1.0, "n": 8},
    }


def test_every_requested_column_is_broken_down():
    predictions, targets = _binary(16)

    breakdown = evaluate(predictions, targets, metrics=["auroc"], subgroups=["sex", "grade"]).subgroups

    assert set(breakdown["metrics"]) == {"sex", "grade"}
    assert set(breakdown["metrics"]["grade"]) == {"low", "high"}


def test_a_single_sample_group_is_skipped():
    predictions, targets = _binary(4)
    targets["site"] = ["A", "B", "A", "A"]

    breakdown = evaluate(predictions, targets, metrics=["balanced_accuracy"], subgroups=["site"]).subgroups

    assert set(breakdown["metrics"]["site"]) == {"A"}


def test_an_unknown_subgroup_column_is_refused():
    predictions, targets = _binary(8)

    with pytest.raises(ValueError, match="nonexistent"):
        evaluate(predictions, targets, subgroups=["nonexistent"])


def test_no_subgroup_columns_gives_an_empty_breakdown():
    predictions, targets = _binary(8)

    assert evaluate(predictions, targets).subgroups == {}


def test_permutation_p_values_cover_groups_of_at_least_ten_samples():
    predictions, targets = _binary(18)
    targets["site"] = ["A"] * 10 + ["B"] * 8

    stats = evaluate(predictions, targets, metrics=["auroc"], subgroups=["site"]).subgroups["stats"]

    # Site B has 8 samples, below the 10-sample floor of the permutation test.
    assert set(stats["site"]) == {"A"}
    assert 0.0 <= stats["site"]["A"]["auroc"] <= 1.0


def test_the_permutation_test_is_seeded():
    predictions, targets = _binary(32)
    frame = predictions.frame.set_index("sample_id")
    metrics = {"auroc": metric_registry.get("auroc").fn}

    first, second = (
        subgroup_stats(frame, targets, metrics, ["sex"], n_permutations=50, min_group_size=5)
        for _ in range(2)
    )

    assert first == second
    assert set(first["sex"]) == {"F", "M"}


def test_survival_predictions_break_down_by_subgroup():
    ids = [f"s{i}" for i in range(8)]
    predictions = Predictions(
        pd.DataFrame({"sample_id": ids, "risk_score": [8.0, 7, 6, 5, 4, 3, 2, 1]}),
        task_family="survival",
    )
    targets = pd.DataFrame(
        {"time": [1.0, 2, 3, 4, 5, 6, 7, 8], "event": [1] * 8, "site": ["A", "B"] * 4},
        index=pd.Index(ids, name="sample_id"),
    )

    breakdown = evaluate(predictions, targets, subgroups=["site"]).subgroups

    assert breakdown["metrics"]["site"]["A"] == {"c_index": 1.0, "n": 4}


def test_a_subgroup_column_missing_from_the_dataset_is_refused_before_training(tmp_path: Path) -> None:
    import torch

    from soma.config import PipelineConfig
    from soma.data import CachedSetSource
    from soma.pipeline import Pipeline

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("sample_id,image_path,label\ns0,/img/s0.tif,0\ns1,/img/s1.tif,1\ns2,/img/s2.tif,0\n")
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text("sample_id,split\ns0,train\ns1,tune\ns2,test\n")
    features = tmp_path / "features"
    features.mkdir()
    for sid in ("s0", "s1", "s2"):
        torch.save(torch.randn(8), features / f"{sid}.pt")
    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "out",
        dataset_type="slide",
        task=TaskConfig(name="binary_classification"),
        evaluation=EvalConfig(subgroups=SubgroupConfig(columns=["nonexistent"])),
        training=TrainingConfig(seed=0, epochs=1),
    )

    def no_training(*args, **kwargs):  # pragma: no cover - the refusal comes first
        raise AssertionError("trained before refusing the subgroup column")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("soma.learn.trainer.Trainer.fit", no_training)
        with pytest.raises(ValueError, match="nonexistent"):
            Pipeline(config, source=CachedSetSource(features)).run()
