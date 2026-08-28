from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from soma.config import TrainingConfig
from soma.dataset import FoldSplit, SampleRecord
from soma.training.fold_planning import plan_dense_fold


def _record(sample_id: str) -> SampleRecord:
    return SampleRecord(sample_id=sample_id, image_path=f"{sample_id}.png", label=None)


def _dataset(*sample_ids: str):
    return SimpleNamespace(samples={sample_id: _record(sample_id) for sample_id in sample_ids})


def _ids(records: list[SampleRecord]) -> list[str]:
    return [record.sample_id for record in records]


def test_plan_dense_fold_selects_train_tune_and_test_records():
    dataset = _dataset("train_a", "train_b", "tune", "test_a", "test_b")
    fold_split = FoldSplit(
        train=("train_a", "train_b"),
        tune=("tune",),
        tests={"test": ("test_a", "test_b")},
    )

    plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(),
        fold_label="Run",
    )

    assert _ids(plan.train_records) == ["train_a", "train_b"]
    assert _ids(plan.tune_records) == ["tune"]
    assert {name: _ids(records) for name, records in plan.test_records_by_split.items()} == {
        "test": ["test_a", "test_b"],
    }
    assert _ids(plan.all_records) == ["train_a", "train_b", "tune", "test_a", "test_b"]


def test_plan_dense_fold_holdout_test_drops_test_splits():
    dataset = _dataset("train_a", "train_b", "tune", "test_a", "test_b")
    fold_split = FoldSplit(
        train=("train_a", "train_b"),
        tune=("tune",),
        tests={"test": ("test_a", "test_b")},
    )

    plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(),
        fold_label="Run",
        holdout_test=True,
    )

    # Train/tune unchanged; test dropped entirely (also out of all_records, so
    # coverage/geometry checks never touch it).
    assert _ids(plan.train_records) == ["train_a", "train_b"]
    assert _ids(plan.tune_records) == ["tune"]
    assert plan.test_records_by_split == {}
    assert _ids(plan.all_records) == ["train_a", "train_b", "tune"]


def test_plan_dense_fold_holdout_test_keeps_tune_is_test_resolution():
    """holdout_test drops test *after* tune_is_test mirrors it into tune."""
    dataset = _dataset("train", "heldout_a", "heldout_b")
    fold_split = FoldSplit(
        train=("train",),
        tune=(),
        tests={"test": ("heldout_a", "heldout_b")},
    )

    plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(tune_is_test=True),
        fold_label="Run",
        holdout_test=True,
    )

    assert _ids(plan.tune_records) == ["heldout_a", "heldout_b"]
    assert plan.test_records_by_split == {}


def test_plan_dense_fold_uses_single_test_split_as_tune_when_tune_is_test():
    dataset = _dataset("train", "configured_tune", "heldout_a", "heldout_b")
    fold_split = FoldSplit(
        train=("train",),
        tune=("configured_tune",),
        tests={"test": ("heldout_a", "heldout_b")},
    )

    plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(tune_is_test=True),
        fold_label="Run",
    )

    assert _ids(plan.tune_records) == ["heldout_a", "heldout_b"]
    assert {name: _ids(records) for name, records in plan.test_records_by_split.items()} == {
        "test": ["heldout_a", "heldout_b"],
    }
    assert _ids(plan.all_records) == ["train", "heldout_a", "heldout_b"]


def test_plan_dense_fold_rejects_tune_is_test_with_multiple_test_splits():
    dataset = _dataset("train", "test_a", "test_b")
    fold_split = FoldSplit(
        train=("train",),
        tune=(),
        tests={"test": ("test_a",), "test_external": ("test_b",)},
    )

    with pytest.raises(ValueError, match="requires exactly one test split"):
        plan_dense_fold(
            dataset=dataset,
            fold_split=fold_split,
            training=TrainingConfig(tune_is_test=True),
            fold_label="Run",
        )


def test_plan_dense_fold_uses_train_as_tune_when_missing_tune_is_allowed():
    dataset = _dataset("train_a", "train_b", "test")
    fold_split = FoldSplit(
        train=("train_a", "train_b"),
        tune=(),
        tests={"test": ("test",)},
    )

    plan = plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(allow_missing_tune=True),
        fold_label="Run",
    )

    assert _ids(plan.tune_records) == ["train_a", "train_b"]


def test_plan_dense_fold_warns_when_missing_tune_uses_train(
    caplog: pytest.LogCaptureFixture,
):
    dataset = _dataset("train", "test")
    fold_split = FoldSplit(train=("train",), tune=(), tests={"test": ("test",)})
    logger = logging.getLogger("tests.dense_fold_planning")
    caplog.set_level(logging.WARNING, logger=logger.name)

    plan_dense_fold(
        dataset=dataset,
        fold_split=fold_split,
        training=TrainingConfig(allow_missing_tune=True),
        fold_label="Run",
        logger=logger,
    )

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == logger.name and record.levelno == logging.WARNING
    ] == ["Run has no tune samples; using train as tune (allow_missing_tune)"]


def test_plan_dense_fold_rejects_missing_tune_by_default():
    dataset = _dataset("train", "test")
    fold_split = FoldSplit(train=("train",), tune=(), tests={"test": ("test",)})

    with pytest.raises(ValueError, match="no tuning samples"):
        plan_dense_fold(
            dataset=dataset,
            fold_split=fold_split,
            training=TrainingConfig(),
            fold_label="Run",
        )


def test_plan_dense_fold_rejects_empty_train_split():
    dataset = _dataset("tune", "test")
    fold_split = FoldSplit(train=(), tune=("tune",), tests={"test": ("test",)})

    with pytest.raises(ValueError, match="no training samples"):
        plan_dense_fold(
            dataset=dataset,
            fold_split=fold_split,
            training=TrainingConfig(),
            fold_label="Run",
        )


def test_plan_dense_fold_rejects_empty_test_split():
    dataset = _dataset("train", "tune")
    fold_split = FoldSplit(train=("train",), tune=("tune",), tests={"test": ()})

    with pytest.raises(ValueError, match="no samples in split 'test'"):
        plan_dense_fold(
            dataset=dataset,
            fold_split=fold_split,
            training=TrainingConfig(),
            fold_label="Run",
        )
