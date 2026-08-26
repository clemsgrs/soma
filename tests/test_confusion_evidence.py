"""Exact fixtures for dataset-neutral segmentation confusion evidence."""

from __future__ import annotations

from soma.evaluation import (
    SegmentationConfusionRecord,
    aggregate_confusion_records,
    load_confusion_records,
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
