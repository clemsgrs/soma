"""Exact fixtures for dataset-neutral segmentation confusion evidence."""

from __future__ import annotations

import numpy as np
import pytest

from soma.evaluation import (
    SegmentationConfusionRecord,
    aggregate_confusion_records,
    confusion_dice_from_matrices,
    load_confusion_records,
    validate_confusion_records,
    write_confusion_records,
)


def test_arbitrary_k_sample_confusion_record_round_trips(tmp_path) -> None:
    record = SegmentationConfusionRecord(
        sample_id="held_out_slide",
        fold=3,
        class_vocabulary=("background", "epithelium", "stroma"),
        confusion_matrix=((5, 1, 0), (2, 7, 1), (0, 1, 9)),
    )
    path = tmp_path / "confusion_evidence_tune.json"

    write_confusion_records(path, [record])

    assert load_confusion_records(path) == [record]


def test_aggregation_sums_confusions_before_recomputing_dice() -> None:
    records = [
        SegmentationConfusionRecord(
            sample_id="sample_a",
            fold=0,
            class_vocabulary=("negative", "positive"),
            confusion_matrix=((3, 1), (1, 1)),
        ),
        SegmentationConfusionRecord(
            sample_id="sample_b",
            fold=1,
            class_vocabulary=("negative", "positive"),
            confusion_matrix=((1, 0), (0, 3)),
        ),
    ]

    metrics = aggregate_confusion_records(records)

    assert metrics.confusion_matrix == ((4, 1), (1, 4))
    assert metrics.dice_per_class == (0.8, 0.8)
    assert metrics.mean_dice == 0.8


def test_batch_confusion_reducer_recomputes_each_matrix_without_averaging_scores() -> None:
    per_class, mean = confusion_dice_from_matrices(
        (
            ((3, 1), (1, 1)),
            ((4, 1), (1, 4)),
        )
    )

    np.testing.assert_allclose(per_class, [[0.75, 0.5], [0.8, 0.8]])
    np.testing.assert_allclose(mean, [0.625, 0.8])


def _binary_record(sample_id: str, fold: int = 0) -> SegmentationConfusionRecord:
    return SegmentationConfusionRecord(
        sample_id=sample_id,
        fold=fold,
        class_vocabulary=("negative", "positive"),
        confusion_matrix=((1, 0), (0, 1)),
    )


def test_fold_validation_rejects_missing_sample() -> None:
    with pytest.raises(ValueError, match="missing sample.*sample_b"):
        validate_confusion_records(
            [_binary_record("sample_a")],
            expected_sample_ids=("sample_a", "sample_b"),
            fold=0,
        )


def test_fold_validation_rejects_unexpected_sample() -> None:
    with pytest.raises(ValueError, match="unexpected sample.*sample_b"):
        validate_confusion_records(
            [_binary_record("sample_a"), _binary_record("sample_b")],
            expected_sample_ids=("sample_a",),
            fold=0,
        )


def test_fold_validation_rejects_duplicate_sample() -> None:
    with pytest.raises(ValueError, match="duplicate sample.*sample_a"):
        validate_confusion_records(
            [_binary_record("sample_a"), _binary_record("sample_a")],
            expected_sample_ids=("sample_a",),
            fold=0,
        )


def test_fold_validation_rejects_wrong_fold() -> None:
    with pytest.raises(ValueError, match=r"fold 0 contains record fold\(s\) \[1\]"):
        validate_confusion_records(
            [_binary_record("sample_a", fold=1)],
            expected_sample_ids=("sample_a",),
            fold=0,
        )
