"""Synthetic project fixtures for the BEETLE OOF publication report."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.beetle.report_oof import (
    BEETLE_ARMS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    BeetleCohort,
    assemble_beetle_oof_report,
    group_sample_confusions,
    patient_bootstrap,
)
from soma.evaluation import SegmentationConfusionRecord, write_confusion_records


VOCABULARY = ("negative", "positive")


def _record(sample_id: str, fold: int, matrix) -> SegmentationConfusionRecord:
    return SegmentationConfusionRecord(
        sample_id=sample_id,
        fold=fold,
        class_vocabulary=VOCABULARY,
        confusion_matrix=matrix,
    )


def _write_five_fold_fixture(root: Path) -> tuple[Path, dict[str, list[Path]]]:
    dataset_csv = root / "dataset.csv"
    rows = ["sample_id,patient_id"]
    for patient_index in range(5):
        rows.extend(
            [
                f"p{patient_index}_a,p{patient_index}",
                f"p{patient_index}_b,p{patient_index}",
            ]
        )
    dataset_csv.write_text("\n".join(rows) + "\n")

    evidence_by_arm: dict[str, list[Path]] = {}
    for arm in BEETLE_ARMS:
        paths = []
        for fold in range(5):
            path = root / f"{arm}_fold_{fold}.json"
            write_confusion_records(
                path,
                [
                    _record(f"p{fold}_a", fold, ((2, 0), (0, 1))),
                    _record(f"p{fold}_b", fold, ((1, 0), (0, 2))),
                ],
            )
            paths.append(path)
        evidence_by_arm[arm] = paths
    return dataset_csv, evidence_by_arm


def test_beetle_report_pools_each_held_out_patient_once_per_arm(tmp_path: Path) -> None:
    dataset_csv, evidence_by_arm = _write_five_fold_fixture(tmp_path)
    cohort = BeetleCohort(
        primary_patient_count=5,
        sensitivity_patient_count=2,
        spacing_exception_patient_ids=("p2", "p3", "p4"),
    )

    report = assemble_beetle_oof_report(
        evidence_by_arm=evidence_by_arm,
        dataset_csv=dataset_csv,
        cohort=cohort,
    )

    assert report["protocol"] == {
        "folds": 5,
        "arms": list(BEETLE_ARMS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "patient",
        "confidence_interval": "percentile_95",
    }
    for arm in BEETLE_ARMS:
        arm_report = report["arms"][arm]
        assert arm_report["coverage"] == {
            "expected_patient_count": 5,
            "observed_patient_count": 5,
            "patient_ids": ["p0", "p1", "p2", "p3", "p4"],
            "folds": [0, 1, 2, 3, 4],
            "exactly_once": True,
        }
        assert arm_report["primary"]["patient_count"] == 5
        assert arm_report["primary"]["fold_macro_class_dice"] == 1.0
        assert arm_report["primary"]["bootstrap"]["percentile_95_ci"] == {
            "mean_dice": [1.0, 1.0],
            "dice_per_class": {
                "negative": [1.0, 1.0],
                "positive": [1.0, 1.0],
            },
        }
        assert arm_report["spacing_sensitivity"]["patient_count"] == 2
        assert arm_report["spacing_sensitivity"]["excluded_patient_ids"] == [
            "p2",
            "p3",
            "p4",
        ]


def test_patient_bootstrap_resamples_whole_grouped_patient_matrices() -> None:
    sample_records = [
        _record("good_a", 0, ((5, 0), (0, 0))),
        _record("good_b", 0, ((0, 0), (0, 5))),
        _record("bad_a", 1, ((0, 5), (0, 0))),
        _record("bad_b", 1, ((0, 0), (5, 0))),
    ]
    patients = group_sample_confusions(
        sample_records,
        {
            "good_a": "a_good",
            "good_b": "a_good",
            "bad_a": "b_bad",
            "bad_b": "b_bad",
        },
    )

    result = patient_bootstrap(patients)

    assert result.seed == 0
    assert result.draws == 10_000
    assert result.replicates["mean_dice"][:6] == pytest.approx(
        (0.0, 0.5, 1.0, 1.0, 0.5, 0.0)
    )
    assert result.percentile_95_ci["mean_dice"] == pytest.approx((0.0, 1.0))
