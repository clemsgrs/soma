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
    sample_patient_csv = root / "sample_patient.csv"
    rows = ["sample_id,patient_id"]
    for patient_index in range(5):
        rows.extend(
            [
                f"p{patient_index}_a,p{patient_index}",
                f"p{patient_index}_b,p{patient_index}",
            ]
        )
    sample_patient_csv.write_text("\n".join(rows) + "\n")

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
    return sample_patient_csv, evidence_by_arm


@pytest.fixture(scope="module")
def beetle_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    sample_patient_csv, evidence_by_arm = _write_five_fold_fixture(
        tmp_path_factory.mktemp("beetle_oof")
    )
    cohort = BeetleCohort(
        primary_patient_count=5,
        sensitivity_patient_count=2,
        spacing_exception_patient_ids=("p2", "p3", "p4"),
    )

    return assemble_beetle_oof_report(
        evidence_by_arm=evidence_by_arm,
        sample_patient_csv=sample_patient_csv,
        cohort=cohort,
    )


def test_beetle_report_records_the_fixed_protocol(beetle_report: dict) -> None:
    assert beetle_report["protocol"] == {
        "folds": 5,
        "arms": list(BEETLE_ARMS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_unit": "patient",
        "confidence_interval": "percentile_95",
    }


def test_beetle_report_pools_each_held_out_patient_once_per_arm(
    beetle_report: dict,
) -> None:
    for arm in BEETLE_ARMS:
        assert beetle_report["arms"][arm]["coverage"] == {
            "expected_patient_count": 5,
            "observed_patient_count": 5,
            "patient_ids": ["p0", "p1", "p2", "p3", "p4"],
            "folds": [0, 1, 2, 3, 4],
            "exactly_once": True,
        }


def test_beetle_report_recomputes_primary_metrics_and_patient_intervals(
    beetle_report: dict,
) -> None:
    for arm in BEETLE_ARMS:
        primary = beetle_report["arms"][arm]["primary"]
        assert primary["patient_count"] == 5
        assert primary["fold_macro_class_dice"] == 1.0
        assert primary["bootstrap"]["percentile_95_ci"] == {
            "mean_dice": [1.0, 1.0],
            "dice_per_class": {
                "negative": [1.0, 1.0],
                "positive": [1.0, 1.0],
            },
        }


def test_beetle_report_owns_the_spacing_sensitivity_subset(
    beetle_report: dict,
) -> None:
    for arm in BEETLE_ARMS:
        sensitivity = beetle_report["arms"][arm]["spacing_sensitivity"]
        assert sensitivity["patient_count"] == 2
        assert sensitivity["excluded_patient_ids"] == ["p2", "p3", "p4"]


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
