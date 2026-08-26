"""Patient-additive out-of-fold evidence for dense segmentation.

Confusion matrices use the conventional ``[true_class][predicted_class]`` layout.
Every reduction in this module first adds integer patient matrices; region-level
Dice values are never inputs to the scientific aggregation path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from soma.tasks.dense_metrics import reduce_confusion_matrix_dice


ConfusionMatrix = tuple[tuple[int, ...], ...]
OOF_NUM_CLASSES = 4


def _as_confusion_matrix(
    value: Sequence[Sequence[int]], *, num_classes: int | None = None
) -> ConfusionMatrix:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion_matrix must be square, got shape {matrix.shape}")
    if num_classes is not None and matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"confusion_matrix must be {num_classes}x{num_classes}, got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError("confusion_matrix entries must be non-negative integers")
    return tuple(tuple(int(value) for value in row) for row in matrix.tolist())


@dataclass(frozen=True)
class PatientConfusionRecord:
    """One additive held-out-patient confusion record from a selected checkpoint."""

    arm: str
    fold: int
    patient_id: str
    class_mapping: Mapping[int, str]
    contributing_slides: tuple[str, ...]
    contributing_rois: tuple[str, ...]
    annotated_pixel_count: int
    confusion_matrix: Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        mapping = {int(index): str(name) for index, name in self.class_mapping.items()}
        if not self.arm:
            raise ValueError("arm must be a non-empty string")
        if not self.patient_id:
            raise ValueError("patient_id must be a non-empty string")
        if sorted(mapping) != list(range(len(mapping))):
            raise ValueError("class_mapping keys must be contiguous class indices from 0")
        if len(mapping) != OOF_NUM_CLASSES:
            raise ValueError("patient OOF evidence requires exactly four classes")
        matrix = _as_confusion_matrix(
            self.confusion_matrix, num_classes=OOF_NUM_CLASSES
        )
        if int(self.annotated_pixel_count) != sum(sum(row) for row in matrix):
            raise ValueError(
                "annotated_pixel_count must equal the sum of the confusion matrix"
            )
        object.__setattr__(self, "fold", int(self.fold))
        object.__setattr__(self, "class_mapping", mapping)
        object.__setattr__(self, "contributing_slides", tuple(self.contributing_slides))
        object.__setattr__(self, "contributing_rois", tuple(self.contributing_rois))
        object.__setattr__(self, "annotated_pixel_count", int(self.annotated_pixel_count))
        object.__setattr__(self, "confusion_matrix", matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "fold": self.fold,
            "patient_id": self.patient_id,
            "class_mapping": {
                str(index): name for index, name in self.class_mapping.items()
            },
            "contributing_slides": list(self.contributing_slides),
            "contributing_rois": list(self.contributing_rois),
            "annotated_pixel_count": self.annotated_pixel_count,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PatientConfusionRecord":
        return cls(
            arm=str(value["arm"]),
            fold=int(value["fold"]),
            patient_id=str(value["patient_id"]),
            class_mapping={
                int(index): str(name)
                for index, name in dict(value["class_mapping"]).items()  # type: ignore[arg-type]
            },
            contributing_slides=tuple(value["contributing_slides"]),  # type: ignore[arg-type]
            contributing_rois=tuple(value["contributing_rois"]),  # type: ignore[arg-type]
            annotated_pixel_count=int(value["annotated_pixel_count"]),
            confusion_matrix=value["confusion_matrix"],  # type: ignore[arg-type]
        )


class PatientConfusionAccumulator:
    """Streaming ROI-to-patient integer confusion accumulator."""

    def __init__(self, *, arm: str, fold: int, class_mapping: Mapping[int, str]) -> None:
        if not arm:
            raise ValueError("patient OOF export requires a non-empty arm")
        if fold < 0:
            raise ValueError(f"patient OOF export requires a non-negative fold, got {fold}")
        if len(class_mapping) != OOF_NUM_CLASSES:
            raise ValueError("patient OOF export requires exactly four classes")
        self._arm = arm
        self._fold = int(fold)
        self._class_mapping = dict(class_mapping)
        self._matrices: dict[str, np.ndarray] = {}
        self._slides: dict[str, set[str]] = {}
        self._rois: dict[str, set[str]] = {}

    def add(
        self,
        *,
        patient_id: str | None,
        slide_id: str | None,
        roi_id: str | None,
        confusion_matrix: Sequence[Sequence[int]],
    ) -> None:
        if not patient_id:
            raise ValueError("patient OOF export requires patient_id for every tune ROI")
        if not slide_id:
            raise ValueError(
                f"patient OOF export requires contributing slide metadata for patient '{patient_id}'"
            )
        if not roi_id:
            raise ValueError(
                f"patient OOF export requires contributing ROI metadata for patient '{patient_id}'"
            )
        matrix = np.asarray(
            _as_confusion_matrix(confusion_matrix, num_classes=len(self._class_mapping)),
            dtype=np.int64,
        )
        self._matrices.setdefault(
            patient_id,
            np.zeros((len(self._class_mapping), len(self._class_mapping)), dtype=np.int64),
        )
        self._matrices[patient_id] += matrix
        self._slides.setdefault(patient_id, set()).add(slide_id)
        self._rois.setdefault(patient_id, set()).add(roi_id)

    def records(self) -> list[PatientConfusionRecord]:
        return [
            PatientConfusionRecord(
                arm=self._arm,
                fold=self._fold,
                patient_id=patient_id,
                class_mapping=self._class_mapping,
                contributing_slides=tuple(sorted(self._slides[patient_id])),
                contributing_rois=tuple(sorted(self._rois[patient_id])),
                annotated_pixel_count=int(matrix.sum()),
                confusion_matrix=matrix,
            )
            for patient_id, matrix in sorted(self._matrices.items())
        ]


def write_patient_confusion_records(
    path: str | Path, records: Sequence[PatientConfusionRecord]
) -> None:
    """Write one fold's independently recomputable held-out patient matrices."""
    if not records:
        raise ValueError("patient OOF export produced no held-out tune patients")
    arm = records[0].arm
    fold = records[0].fold
    mapping = dict(records[0].class_mapping)
    if any(record.arm != arm or record.fold != fold for record in records):
        raise ValueError("one patient confusion artifact must contain exactly one arm and fold")
    if any(dict(record.class_mapping) != mapping for record in records):
        raise ValueError("patient confusion records disagree on class_mapping")
    payload = {
        "schema_version": 1,
        "arm": arm,
        "fold": fold,
        "class_mapping": {str(index): name for index, name in mapping.items()},
        "patients": [record.to_dict() for record in records],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_patient_confusion_records(path: str | Path) -> list[PatientConfusionRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported patient confusion schema in {path}")
    return [PatientConfusionRecord.from_dict(record) for record in payload["patients"]]


@dataclass(frozen=True)
class ConfusionMetrics:
    confusion_matrix: ConfusionMatrix
    micro_dice: float
    dice_per_class: tuple[float, ...]
    macro_dice: float


@dataclass(frozen=True)
class BootstrapResult:
    """Fixed-prediction patient bootstrap replicates and percentile intervals."""

    seed: int
    draws: int
    replicates: Mapping[str, tuple[float, ...]]
    percentile_95_ci: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class SpacingSensitivityResult:
    """Metrics derived by omitting declared patients from fixed OOF predictions."""

    label: str
    evaluation_only: bool
    source: str
    excluded_patient_ids: tuple[str, ...]
    patient_count: int
    pooled: ConfusionMetrics
    bootstrap: BootstrapResult | None


@dataclass(frozen=True)
class OOFReport:
    arm: str
    class_mapping: Mapping[int, str]
    patient_count: int
    folds: Mapping[int, ConfusionMetrics]
    pooled: ConfusionMetrics
    fold_macro_class_dice: float
    bootstrap: BootstrapResult | None
    spacing_sensitivity: SpacingSensitivityResult | None

    def to_dict(self) -> dict[str, object]:
        def metrics(value: ConfusionMetrics) -> dict[str, object]:
            return {
                "confusion_matrix": [list(row) for row in value.confusion_matrix],
                "micro_dice": value.micro_dice,
                "dice_per_class": list(value.dice_per_class),
                "macro_dice": value.macro_dice,
            }

        def bootstrap(value: BootstrapResult | None) -> dict[str, object] | None:
            if value is None:
                return None
            return {
                "seed": value.seed,
                "draws": value.draws,
                "sampling_unit": "patient",
                "with_replacement": True,
                "rng": "numpy.default_rng(PCG64)",
                "interval": "percentile_95",
                "replicates": {
                    name: list(samples) for name, samples in value.replicates.items()
                },
                "percentile_95_ci": {
                    name: list(interval)
                    for name, interval in value.percentile_95_ci.items()
                },
            }

        spacing = None
        if self.spacing_sensitivity is not None:
            value = self.spacing_sensitivity
            spacing = {
                "label": value.label,
                "evaluation_only": value.evaluation_only,
                "source": value.source,
                "excluded_patient_ids": list(value.excluded_patient_ids),
                "patient_count": value.patient_count,
                "pooled": metrics(value.pooled),
                "bootstrap": bootstrap(value.bootstrap),
            }
        return {
            "schema_version": 1,
            "arm": self.arm,
            "class_mapping": {
                str(index): name for index, name in self.class_mapping.items()
            },
            "coverage": {
                "status": "complete",
                "expected_patient_count": self.patient_count,
                "observed_patient_count": self.patient_count,
                "exactly_once": True,
            },
            "folds": {
                str(fold): metrics(value) for fold, value in self.folds.items()
            },
            "pooled": metrics(self.pooled),
            "fold_macro_class_dice": self.fold_macro_class_dice,
            "bootstrap": bootstrap(self.bootstrap),
            "spacing_sensitivity": spacing,
        }


def _sum_matrices(records: Iterable[PatientConfusionRecord], num_classes: int) -> np.ndarray:
    total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for record in records:
        total += np.asarray(record.confusion_matrix, dtype=np.int64)
    return total


def _metrics_from_confusion_matrix(matrix: np.ndarray) -> ConfusionMetrics:
    micro, per_class = reduce_confusion_matrix_dice(torch.as_tensor(matrix))
    return ConfusionMetrics(
        confusion_matrix=_as_confusion_matrix(matrix),
        micro_dice=micro,
        dice_per_class=per_class,
        macro_dice=float(np.mean(per_class)),
    )


def _bootstrap(
    records: Sequence[PatientConfusionRecord], *, draws: int, seed: int
) -> BootstrapResult | None:
    if draws < 0:
        raise ValueError(f"bootstrap_draws must be >= 0, got {draws}")
    if draws == 0:
        return None
    matrices = np.asarray([record.confusion_matrix for record in records], dtype=np.int64)
    rng = np.random.default_rng(seed)
    metric_names = ["micro_dice"] + [
        f"dice_class_{index}" for index in range(matrices.shape[1])
    ]
    values = {name: [] for name in metric_names}
    for _ in range(draws):
        indices = rng.integers(0, len(records), size=len(records))
        metrics = _metrics_from_confusion_matrix(matrices[indices].sum(axis=0))
        values["micro_dice"].append(metrics.micro_dice)
        for index, dice in enumerate(metrics.dice_per_class):
            values[f"dice_class_{index}"].append(dice)
    replicates = {name: tuple(samples) for name, samples in values.items()}
    intervals = {
        name: tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))
        for name, samples in replicates.items()
    }
    return BootstrapResult(
        seed=int(seed),
        draws=int(draws),
        replicates=replicates,
        percentile_95_ci=intervals,
    )


def aggregate_patient_oof(
    records: Sequence[PatientConfusionRecord],
    *,
    expected_patient_ids: Sequence[str],
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 0,
    spacing_exception_patient_ids: Sequence[str] = (),
    expected_patient_count: int | None = None,
    expected_spacing_sensitivity_patient_count: int | None = None,
) -> OOFReport:
    """Reaggregate one arm's exact patient matrices into fold and pooled metrics."""
    if not records:
        raise ValueError("patient OOF evidence is empty")
    expected = tuple(str(patient) for patient in expected_patient_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("expected_patient_ids contains duplicate patient IDs")
    observed_counts = Counter(record.patient_id for record in records)
    duplicate = sorted(patient for patient, count in observed_counts.items() if count > 1)
    missing = sorted(set(expected) - set(observed_counts))
    unexpected = sorted(set(observed_counts) - set(expected))
    problems: list[str] = []
    if missing:
        problems.append(f"missing expected patient(s): {missing}")
    if duplicate:
        problems.append(f"patient(s) appearing more than once: {duplicate}")
    if unexpected:
        problems.append(f"unexpected patient(s): {unexpected}")
    if problems:
        raise ValueError("patient OOF coverage failed; " + "; ".join(problems))
    if expected_patient_count is not None and len(records) != int(expected_patient_count):
        raise ValueError(
            "patient OOF coverage has the wrong cohort size: "
            f"expected {expected_patient_count}, observed {len(records)}"
        )

    arm = records[0].arm
    mapping = dict(records[0].class_mapping)
    if any(record.arm != arm for record in records):
        raise ValueError("patient OOF records must contain exactly one arm")
    if any(dict(record.class_mapping) != mapping for record in records):
        raise ValueError("patient OOF records disagree on class_mapping")
    records = tuple(sorted(records, key=lambda record: record.patient_id))

    fold_metrics: dict[int, ConfusionMetrics] = {}
    for fold in sorted({record.fold for record in records}):
        matrix = _sum_matrices(
            (record for record in records if record.fold == fold), len(mapping)
        )
        fold_metrics[fold] = _metrics_from_confusion_matrix(matrix)
    pooled = _metrics_from_confusion_matrix(_sum_matrices(records, len(mapping)))
    spacing_sensitivity = None
    if spacing_exception_patient_ids:
        excluded = tuple(str(patient) for patient in spacing_exception_patient_ids)
        if len(excluded) != 3 or len(set(excluded)) != 3:
            raise ValueError(
                "spacing_exception_patient_ids must contain exactly three unique patient IDs"
            )
        unknown = sorted(set(excluded) - set(expected))
        if unknown:
            raise ValueError(f"spacing exception patient(s) are not in OOF coverage: {unknown}")
        subset_records = [record for record in records if record.patient_id not in set(excluded)]
        if not subset_records:
            raise ValueError("spacing sensitivity subset cannot be empty")
        if (
            expected_spacing_sensitivity_patient_count is not None
            and len(subset_records) != int(expected_spacing_sensitivity_patient_count)
        ):
            raise ValueError(
                "spacing sensitivity has the wrong cohort size: expected "
                f"{expected_spacing_sensitivity_patient_count}, observed {len(subset_records)}"
            )
        spacing_sensitivity = SpacingSensitivityResult(
            label="evaluation_only_spacing_sensitivity",
            evaluation_only=True,
            source="same_oof_predictions",
            excluded_patient_ids=excluded,
            patient_count=len(subset_records),
            pooled=_metrics_from_confusion_matrix(
                _sum_matrices(subset_records, len(mapping))
            ),
            bootstrap=_bootstrap(
                subset_records, draws=bootstrap_draws, seed=bootstrap_seed
            ),
        )
    return OOFReport(
        arm=arm,
        class_mapping=mapping,
        patient_count=len(records),
        folds=fold_metrics,
        pooled=pooled,
        fold_macro_class_dice=float(
            np.mean([metrics.macro_dice for metrics in fold_metrics.values()])
        ),
        bootstrap=_bootstrap(records, draws=bootstrap_draws, seed=bootstrap_seed),
        spacing_sensitivity=spacing_sensitivity,
    )


def aggregate_patient_oof_files(
    paths: Sequence[str | Path],
    *,
    expected_patient_ids: Sequence[str],
    spacing_exception_patient_ids: Sequence[str] = (),
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 0,
    expected_patient_count: int | None = None,
    expected_spacing_sensitivity_patient_count: int | None = None,
) -> OOFReport:
    """Load fold artifacts and run the single scientific reaggregation path."""
    records = [record for path in paths for record in load_patient_confusion_records(path)]
    return aggregate_patient_oof(
        records,
        expected_patient_ids=expected_patient_ids,
        spacing_exception_patient_ids=spacing_exception_patient_ids,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        expected_patient_count=expected_patient_count,
        expected_spacing_sensitivity_patient_count=(
            expected_spacing_sensitivity_patient_count
        ),
    )


def write_oof_report(path: str | Path, report: OOFReport) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


__all__ = [
    "BootstrapResult",
    "ConfusionMetrics",
    "OOFReport",
    "PatientConfusionRecord",
    "PatientConfusionAccumulator",
    "SpacingSensitivityResult",
    "aggregate_patient_oof",
    "aggregate_patient_oof_files",
    "load_patient_confusion_records",
    "write_patient_confusion_records",
    "write_oof_report",
]
