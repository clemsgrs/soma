"""Assemble BEETLE patient-level OOF publication evidence from Soma records.

Soma's reusable boundary ends at per-sample confusion matrices.  This module owns
BEETLE's patient grouping, arm/fold coverage, cohort assertions, spacing sensitivity,
fixed patient bootstrap, confidence intervals, and JSON report layout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from examples.beetle.curate import FULL_COHORT_PATIENTS
from examples.beetle.protocol import ARM_NAMES, NUM_FOLDS
from soma.evaluation import (
    ConfusionMetrics,
    SegmentationConfusionRecord,
    aggregate_confusion_matrices,
    confusion_dice_from_matrices,
    load_confusion_records,
)


BEETLE_ARMS = ARM_NAMES
BEETLE_FOLDS = NUM_FOLDS
BOOTSTRAP_SEED = 0
BOOTSTRAP_DRAWS = 10_000
NATIVE_SPACING_EXCEPTION_PATIENT_IDS = (
    "TCGA-OL-A66I",
    "TCGA-OL-A66P",
    "TCGA-OL-A6VO",
)
SENSITIVITY_PATIENT_COUNT = 524


@dataclass(frozen=True)
class BeetleCohort:
    """BEETLE's primary and native-spacing sensitivity cohort assertions."""

    primary_patient_count: int = FULL_COHORT_PATIENTS
    sensitivity_patient_count: int = SENSITIVITY_PATIENT_COUNT
    spacing_exception_patient_ids: tuple[str, ...] = NATIVE_SPACING_EXCEPTION_PATIENT_IDS

    def __post_init__(self) -> None:
        exceptions = tuple(str(value) for value in self.spacing_exception_patient_ids)
        if len(exceptions) != 3 or len(set(exceptions)) != 3:
            raise ValueError("BEETLE requires exactly three unique spacing-exception patients")
        primary = int(self.primary_patient_count)
        sensitivity = int(self.sensitivity_patient_count)
        if primary <= 0 or sensitivity <= 0 or primary - sensitivity != 3:
            raise ValueError(
                "BEETLE primary and sensitivity cohorts must be positive and differ by "
                f"three patients, got {primary} and {sensitivity}"
            )
        object.__setattr__(self, "primary_patient_count", primary)
        object.__setattr__(self, "sensitivity_patient_count", sensitivity)
        object.__setattr__(self, "spacing_exception_patient_ids", exceptions)


@dataclass(frozen=True)
class PatientConfusionRecord:
    """One BEETLE patient's whole held-out confusion matrix."""

    patient_id: str
    fold: int
    class_vocabulary: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BootstrapResult:
    seed: int
    draws: int
    replicates: Mapping[str, tuple[float, ...]]
    percentile_95_ci: Mapping[str, tuple[float, float]]


def _read_sample_to_patient(sample_patient_csv: str | Path) -> dict[str, str]:
    with Path(sample_patient_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0] or "patient_id" not in rows[0]:
        raise ValueError(
            "BEETLE sample-to-patient metadata requires sample_id and patient_id"
        )
    mapping: dict[str, str] = {}
    for row in rows:
        sample_id = str(row["sample_id"]).strip()
        patient_id = str(row["patient_id"]).strip()
        if not sample_id or not patient_id:
            raise ValueError("BEETLE report requires non-empty sample_id and patient_id")
        if sample_id in mapping:
            raise ValueError(
                f"BEETLE sample-to-patient metadata repeats sample_id {sample_id!r}"
            )
        mapping[sample_id] = patient_id
    return mapping


def group_sample_confusions(
    records: Sequence[SegmentationConfusionRecord],
    sample_to_patient: Mapping[str, str],
) -> list[PatientConfusionRecord]:
    """Map samples to patients and sum every patient's held-out sample matrices."""
    if not records:
        raise ValueError("BEETLE confusion evidence is empty")
    sample_counts = Counter(record.sample_id for record in records)
    duplicate_samples = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    if duplicate_samples:
        raise ValueError(
            f"BEETLE held-out sample(s) appear more than once: {duplicate_samples}"
        )
    vocabulary = tuple(records[0].class_vocabulary)
    if any(tuple(record.class_vocabulary) != vocabulary for record in records):
        raise ValueError("BEETLE evidence disagrees on class vocabulary")

    matrices: dict[str, list[tuple[tuple[int, ...], ...]]] = {}
    patient_folds: dict[str, set[int]] = {}
    for record in records:
        patient_id = str(sample_to_patient.get(record.sample_id, "")).strip()
        if not patient_id:
            raise ValueError(
                f"BEETLE sample {record.sample_id!r} has no patient mapping"
            )
        matrices.setdefault(patient_id, []).append(record.confusion_matrix)
        patient_folds.setdefault(patient_id, set()).add(record.fold)

    leaking = {
        patient_id: sorted(folds)
        for patient_id, folds in patient_folds.items()
        if len(folds) != 1
    }
    if leaking:
        raise ValueError(f"BEETLE patient(s) appear in multiple held-out folds: {leaking}")

    return [
        PatientConfusionRecord(
            patient_id=patient_id,
            fold=next(iter(patient_folds[patient_id])),
            class_vocabulary=vocabulary,
            confusion_matrix=aggregate_confusion_matrices(
                matrices[patient_id]
            ).confusion_matrix,
        )
        for patient_id in sorted(matrices)
    ]


def patient_bootstrap(
    records: Sequence[PatientConfusionRecord],
) -> BootstrapResult:
    """Run BEETLE's seed-0, 10,000-draw whole-patient bootstrap."""
    if not records:
        raise ValueError("BEETLE patient bootstrap requires at least one patient")
    matrices = np.asarray([record.confusion_matrix for record in records], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled = np.empty(
        (BOOTSTRAP_DRAWS, matrices.shape[1], matrices.shape[2]), dtype=np.int64
    )
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(records), size=len(records))
        pooled[draw] = matrices[indices].sum(axis=0)
    per_class, mean_dice = confusion_dice_from_matrices(pooled)
    replicates = {"mean_dice": tuple(float(value) for value in mean_dice)}
    replicates.update(
        {
            f"dice_class_{index}": tuple(float(value) for value in per_class[:, index])
            for index in range(matrices.shape[1])
        }
    )
    intervals = {
        name: tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))
        for name, samples in replicates.items()
    }
    return BootstrapResult(
        seed=BOOTSTRAP_SEED,
        draws=BOOTSTRAP_DRAWS,
        replicates=replicates,
        percentile_95_ci=intervals,
    )


def _metrics_payload(metrics: ConfusionMetrics, vocabulary: Sequence[str]) -> dict:
    return {
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
        "dice_per_class": {
            name: metrics.dice_per_class[index]
            for index, name in enumerate(vocabulary)
        },
        "mean_dice": metrics.mean_dice,
    }


def _bootstrap_payload(result: BootstrapResult, vocabulary: Sequence[str]) -> dict:
    intervals = {
        "mean_dice": list(result.percentile_95_ci["mean_dice"]),
        "dice_per_class": {
            name: list(result.percentile_95_ci[f"dice_class_{index}"])
            for index, name in enumerate(vocabulary)
        },
    }
    return {
        "seed": result.seed,
        "draws": result.draws,
        "sampling_unit": "patient",
        "with_replacement": True,
        "percentile_95_ci": intervals,
    }


def _cohort_payload(records: Sequence[PatientConfusionRecord]) -> dict:
    vocabulary = records[0].class_vocabulary
    pooled = aggregate_confusion_matrices(
        [record.confusion_matrix for record in records]
    )
    fold_metrics = {
        fold: aggregate_confusion_matrices(
            [record.confusion_matrix for record in records if record.fold == fold]
        )
        for fold in sorted({record.fold for record in records})
    }
    bootstrap = patient_bootstrap(records)
    return {
        "patient_count": len(records),
        "class_vocabulary": list(vocabulary),
        "folds": {
            str(fold): _metrics_payload(metrics, vocabulary)
            for fold, metrics in fold_metrics.items()
        },
        "pooled": _metrics_payload(pooled, vocabulary),
        "fold_macro_class_dice": float(
            np.mean([metrics.mean_dice for metrics in fold_metrics.values()])
        ),
        "bootstrap": _bootstrap_payload(bootstrap, vocabulary),
    }


def assemble_beetle_oof_report(
    *,
    evidence_by_arm: Mapping[str, Sequence[str | Path]],
    sample_patient_csv: str | Path,
    cohort: BeetleCohort | None = None,
) -> dict:
    """Build BEETLE's two-arm five-fold primary and sensitivity report."""
    cohort = cohort or BeetleCohort()
    if set(evidence_by_arm) != set(BEETLE_ARMS):
        raise ValueError(f"BEETLE report requires arms {list(BEETLE_ARMS)}")
    sample_to_patient = _read_sample_to_patient(sample_patient_csv)
    patients_by_arm: dict[str, list[PatientConfusionRecord]] = {}
    for arm in BEETLE_ARMS:
        paths = list(evidence_by_arm[arm])
        if len(paths) != BEETLE_FOLDS:
            raise ValueError(f"BEETLE arm {arm!r} requires exactly five fold artifacts")
        records = [record for path in paths for record in load_confusion_records(path)]
        patients = group_sample_confusions(records, sample_to_patient)
        if len(patients) != cohort.primary_patient_count:
            raise ValueError(
                f"BEETLE primary cohort requires {cohort.primary_patient_count} patients; "
                f"arm {arm!r} has {len(patients)}"
            )
        folds = {record.fold for record in patients}
        if folds != set(range(BEETLE_FOLDS)):
            raise ValueError(
                f"BEETLE arm {arm!r} must pool held-out folds 0 through 4; got {sorted(folds)}"
            )
        patients_by_arm[arm] = patients

    patient_sets = {
        arm: {record.patient_id for record in records}
        for arm, records in patients_by_arm.items()
    }
    if len({frozenset(values) for values in patient_sets.values()}) != 1:
        raise ValueError(f"BEETLE arms do not contain the same held-out patients: {patient_sets}")

    arm_payloads = {}
    for arm, patients in patients_by_arm.items():
        exception_ids = set(cohort.spacing_exception_patient_ids)
        missing_exceptions = sorted(exception_ids - patient_sets[arm])
        if missing_exceptions:
            raise ValueError(
                f"BEETLE spacing-exception patients missing from arm {arm!r}: "
                f"{missing_exceptions}"
            )
        sensitivity = [
            record for record in patients if record.patient_id not in exception_ids
        ]
        if len(sensitivity) != cohort.sensitivity_patient_count:
            raise ValueError(
                "BEETLE spacing-sensitivity cohort requires "
                f"{cohort.sensitivity_patient_count} patients; arm {arm!r} has "
                f"{len(sensitivity)}"
            )
        arm_payloads[arm] = {
            "coverage": {
                "expected_patient_count": cohort.primary_patient_count,
                "observed_patient_count": len(patients),
                "patient_ids": [record.patient_id for record in patients],
                "folds": sorted({record.fold for record in patients}),
                "exactly_once": True,
            },
            "primary": _cohort_payload(patients),
            "spacing_sensitivity": {
                "evaluation_only": True,
                "source": "same_oof_predictions",
                "excluded_patient_ids": list(cohort.spacing_exception_patient_ids),
                **_cohort_payload(sensitivity),
            },
        }
    return {
        "schema_version": 1,
        "protocol": {
            "folds": BEETLE_FOLDS,
            "arms": list(BEETLE_ARMS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_unit": "patient",
            "confidence_interval": "percentile_95",
        },
        "arms": arm_payloads,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-patient-csv",
        type=Path,
        required=True,
        help="Derived BEETLE metadata with sample_id and patient_id columns",
    )
    parser.add_argument("--uniform", type=Path, nargs=5, required=True)
    parser.add_argument("--class-conditioned", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = assemble_beetle_oof_report(
        evidence_by_arm={
            "uniform": args.uniform,
            "class_conditioned": args.class_conditioned,
        },
        sample_patient_csv=args.sample_patient_csv,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEETLE_ARMS",
    "BEETLE_FOLDS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "BeetleCohort",
    "BootstrapResult",
    "PatientConfusionRecord",
    "assemble_beetle_oof_report",
    "group_sample_confusions",
    "main",
    "patient_bootstrap",
]
