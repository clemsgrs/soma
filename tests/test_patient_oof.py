"""Exact fixtures for patient-level segmentation OOF evidence."""

from __future__ import annotations

import json

import pytest

from soma.evaluation.patient_oof import (
    PatientConfusionAccumulator,
    PatientConfusionRecord,
    aggregate_patient_oof,
    aggregate_patient_oof_files,
    write_oof_report,
    write_patient_confusion_records,
)


CLASS_MAPPING = {
    0: "other",
    1: "non_invasive_epithelium",
    2: "invasive_epithelium",
    3: "necrosis",
}


def _record(
    patient: str,
    fold: int,
    matrix: list[list[int]],
    *,
    arm: str = "uniform",
) -> PatientConfusionRecord:
    return PatientConfusionRecord(
        arm=arm,
        fold=fold,
        patient_id=patient,
        class_mapping=CLASS_MAPPING,
        contributing_slides=(f"slide_{patient}",),
        contributing_rois=(f"roi_{patient}",),
        annotated_pixel_count=sum(sum(row) for row in matrix),
        confusion_matrix=matrix,
    )


def test_oof_metrics_sum_patient_matrices_before_computing_dice() -> None:
    records = [
        _record(
            "p1",
            0,
            [
                [3, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ],
        ),
        _record(
            "p2",
            0,
            [
                [1, 0, 0, 0],
                [0, 2, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 2],
            ],
        ),
        _record(
            "p3",
            1,
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
    ]

    report = aggregate_patient_oof(
        records,
        expected_patient_ids=("p1", "p2", "p3"),
        bootstrap_draws=0,
    )

    assert report.folds[0].confusion_matrix == (
        (4, 1, 0, 0),
        (1, 3, 1, 0),
        (0, 1, 3, 0),
        (0, 0, 0, 3),
    )
    assert report.folds[0].dice_per_class == pytest.approx((0.8, 0.6, 0.75, 1.0))
    assert report.folds[0].macro_dice == pytest.approx(0.7875)
    assert report.pooled.confusion_matrix == (
        (5, 1, 0, 0),
        (1, 4, 1, 0),
        (0, 1, 4, 0),
        (0, 0, 0, 4),
    )
    assert report.pooled.micro_dice == pytest.approx(17 / 21)
    assert report.pooled.dice_per_class == pytest.approx((5 / 6, 2 / 3, 0.8, 1.0))
    assert report.fold_macro_class_dice == pytest.approx(0.89375)


def test_roi_confusions_add_to_one_exact_patient_record() -> None:
    accumulator = PatientConfusionAccumulator(
        arm="uniform", fold=2, class_mapping=CLASS_MAPPING
    )
    accumulator.add(
        patient_id="p1",
        slide_id="slide_a",
        roi_id="roi_a",
        confusion_matrix=[
            [1, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4],
        ],
    )
    accumulator.add(
        patient_id="p1",
        slide_id="slide_b",
        roi_id="roi_b",
        confusion_matrix=[
            [4, 0, 0, 0],
            [1, 3, 0, 0],
            [0, 1, 2, 0],
            [0, 0, 1, 1],
        ],
    )

    assert accumulator.records() == [
        PatientConfusionRecord(
            arm="uniform",
            fold=2,
            patient_id="p1",
            class_mapping=CLASS_MAPPING,
            contributing_slides=("slide_a", "slide_b"),
            contributing_rois=("roi_a", "roi_b"),
            annotated_pixel_count=24,
            confusion_matrix=(
                (5, 1, 0, 0),
                (1, 5, 0, 0),
                (0, 1, 5, 0),
                (0, 0, 1, 5),
            ),
        )
    ]


def test_patient_oof_rejects_non_four_class_matrix() -> None:
    with pytest.raises(ValueError, match="exactly four classes"):
        PatientConfusionRecord(
            arm="uniform",
            fold=0,
            patient_id="p1",
            class_mapping={0: "negative", 1: "positive"},
            contributing_slides=("s1",),
            contributing_rois=("r1",),
            annotated_pixel_count=2,
            confusion_matrix=((1, 0), (0, 1)),
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record("p1", 0, [[1, 0, 0, 0]] * 4)], "missing.*p2"),
        (
            [
                _record("p1", 0, [[1, 0, 0, 0]] * 4),
                _record("p1", 1, [[1, 0, 0, 0]] * 4),
                _record("p2", 1, [[1, 0, 0, 0]] * 4),
            ],
            "more than once.*p1",
        ),
    ],
)
def test_oof_coverage_requires_every_expected_patient_exactly_once(
    records: list[PatientConfusionRecord], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_patient_oof(
            records,
            expected_patient_ids=("p1", "p2"),
            bootstrap_draws=0,
        )


def test_patient_bootstrap_has_exact_seeded_draws_and_percentile_interval() -> None:
    records = [
        _record(
            "p1",
            0,
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        _record(
            "p2",
            1,
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
        ),
    ]

    report = aggregate_patient_oof(
        records,
        expected_patient_ids=("p1", "p2"),
        bootstrap_draws=4,
        bootstrap_seed=7,
    )

    assert report.bootstrap is not None
    assert report.bootstrap.replicates["micro_dice"] == pytest.approx((0.0, 0.0, 0.0, 0.5))
    for class_index in range(4):
        assert report.bootstrap.replicates[f"dice_class_{class_index}"] == pytest.approx(
            (0.0, 0.0, 0.0, 0.5)
        )
    assert report.bootstrap.percentile_95_ci["micro_dice"] == pytest.approx((0.0, 0.4625))


def test_spacing_sensitivity_is_derived_from_same_oof_predictions() -> None:
    perfect = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    wrong = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    records = [
        _record("ordinary_good", 0, perfect),
        _record("ordinary_bad", 1, wrong),
        _record("coarse_a", 2, perfect),
        _record("coarse_b", 3, perfect),
        _record("coarse_c", 4, perfect),
    ]

    report = aggregate_patient_oof(
        records,
        expected_patient_ids=tuple(record.patient_id for record in records),
        spacing_exception_patient_ids=("coarse_a", "coarse_b", "coarse_c"),
        bootstrap_draws=0,
        expected_patient_count=5,
        expected_spacing_sensitivity_patient_count=2,
    )

    assert report.spacing_sensitivity is not None
    assert report.spacing_sensitivity.label == "evaluation_only_spacing_sensitivity"
    assert report.spacing_sensitivity.evaluation_only is True
    assert report.spacing_sensitivity.source == "same_oof_predictions"
    assert report.spacing_sensitivity.excluded_patient_ids == (
        "coarse_a",
        "coarse_b",
        "coarse_c",
    )
    assert report.spacing_sensitivity.patient_count == 2
    assert report.spacing_sensitivity.pooled.micro_dice == pytest.approx(0.5)
    assert report.pooled.micro_dice == pytest.approx(0.8)


def test_spacing_sensitivity_requires_declared_retained_cohort_size() -> None:
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    records = [_record(f"p{index}", index, matrix) for index in range(5)]

    with pytest.raises(ValueError, match="wrong cohort size.*expected 3, observed 2"):
        aggregate_patient_oof(
            records,
            expected_patient_ids=tuple(record.patient_id for record in records),
            spacing_exception_patient_ids=("p0", "p1", "p2"),
            bootstrap_draws=0,
            expected_patient_count=5,
            expected_spacing_sensitivity_patient_count=3,
        )


def test_sampling_arms_write_independently_recomputable_bootstrap_outputs(tmp_path) -> None:
    matrix = [
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
    ]
    outputs = {}
    for arm in ("uniform", "class_conditioned"):
        records_path = tmp_path / f"{arm}_fold.json"
        report_path = tmp_path / f"{arm}_report.json"
        write_patient_confusion_records(
            records_path,
            [_record("p1", 0, matrix, arm=arm)],
        )
        report = aggregate_patient_oof_files(
            [records_path],
            expected_patient_ids=("p1",),
            bootstrap_draws=10_000,
            bootstrap_seed=0,
        )
        write_oof_report(report_path, report)
        outputs[arm] = json.loads(report_path.read_text())

    assert outputs["uniform"]["arm"] == "uniform"
    assert outputs["class_conditioned"]["arm"] == "class_conditioned"
    for payload in outputs.values():
        assert payload["coverage"] == {
            "status": "complete",
            "expected_patient_count": 1,
            "observed_patient_count": 1,
            "exactly_once": True,
        }
        assert payload["bootstrap"]["seed"] == 0
        assert payload["bootstrap"]["draws"] == 10_000
        assert len(payload["bootstrap"]["replicates"]["micro_dice"]) == 10_000
        assert payload["pooled"]["confusion_matrix"] == matrix
