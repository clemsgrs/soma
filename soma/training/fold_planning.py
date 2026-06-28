"""Shared train/tune/test fold planning helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from soma.config import TrainingConfig
from soma.dataset import FoldSplit, SampleRecord


@dataclass(frozen=True)
class DenseFoldPlan:
    """Selected dense records for one fold."""

    train_records: list[SampleRecord]
    tune_records: list[SampleRecord]
    test_records_by_split: dict[str, list[SampleRecord]]

    @property
    def all_records(self) -> list[SampleRecord]:
        return [
            *self.train_records,
            *self.tune_records,
            *(record for records in self.test_records_by_split.values() for record in records),
        ]


def plan_dense_fold(
    *,
    dataset,
    fold_split: FoldSplit,
    training: TrainingConfig,
    fold_label: str,
    logger: logging.Logger | None = None,
    holdout_test: bool = False,
) -> DenseFoldPlan:
    """Select dense train, tune, and test records for one fold.

    ``holdout_test`` (from ``evaluation.holdout_test``) drops every declared test
    split from the plan so downstream coverage/geometry checks, loaders, inference,
    and metrics never touch it (tune-only model-selection runs). The drop happens
    *after* ``tune_is_test`` has resolved the tune records from the test split, so
    checkpoint selection and training are unaffected.
    """
    train_records = [dataset.samples[sid] for sid in fold_split.train]
    tune_records = [dataset.samples[sid] for sid in fold_split.tune]
    test_records_by_split = {
        split_name: [dataset.samples[sid] for sid in ids]
        for split_name, ids in fold_split.tests.items()
    }
    if not train_records:
        raise ValueError(f"{fold_label} has no training samples")
    if training.tune_is_test:
        if len(fold_split.test_split_names) != 1:
            raise ValueError(
                f"{fold_label} has {len(fold_split.test_split_names)} test splits; "
                "training.tune_is_test=True requires exactly one test split"
            )
        test_split_name = fold_split.test_split_names[0]
        tune_records = list(test_records_by_split[test_split_name])
    if not tune_records:
        if not training.allow_missing_tune:
            raise ValueError(f"{fold_label} has no tuning samples")
        if logger is not None:
            logger.warning(
                "%s has no tune samples; using train as tune (allow_missing_tune)",
                fold_label,
            )
        tune_records = list(train_records)
    if holdout_test:
        # Tune is already resolved (incl. tune_is_test); drop test before validating
        # or returning so the split is never touched (not even coverage-checked).
        test_records_by_split = {}
    for split_name, records in test_records_by_split.items():
        if not records:
            raise ValueError(f"{fold_label} has no samples in split '{split_name}'")
    return DenseFoldPlan(
        train_records=train_records,
        tune_records=tune_records,
        test_records_by_split=test_records_by_split,
    )
