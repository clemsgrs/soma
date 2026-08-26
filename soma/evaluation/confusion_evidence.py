"""Dataset-neutral per-sample confusion evidence for segmentation.

Matrices use ``[true_class][predicted_class]`` ordering.  Records contain only
additive sufficient statistics and fold/sample provenance; grouping and report policy
belong to project code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import torch

from soma.tasks.dense_metrics import reduce_confusion_matrix_dice


ConfusionMatrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ConfusionMetrics:
    """Dice values recomputed from one additive confusion matrix."""

    confusion_matrix: ConfusionMatrix
    dice_per_class: tuple[float, ...]
    mean_dice: float


def _validated_matrix(
    value: Sequence[Sequence[int]], *, num_classes: int
) -> ConfusionMatrix:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape != (num_classes, num_classes):
        raise ValueError(
            "confusion_matrix must be square and match class_vocabulary; "
            f"got shape {matrix.shape} for {num_classes} classes"
        )
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError("confusion_matrix entries must be non-negative integers")
    return tuple(tuple(int(entry) for entry in row) for row in matrix.tolist())


@dataclass(frozen=True)
class SegmentationConfusionRecord:
    """One held-out sample's confusion matrix from a fold-selected checkpoint."""

    sample_id: str
    fold: int
    class_vocabulary: Sequence[str]
    confusion_matrix: Sequence[Sequence[int]]

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if isinstance(self.fold, bool) or int(self.fold) < 0:
            raise ValueError(f"fold must be a non-negative integer, got {self.fold!r}")
        vocabulary = tuple(str(name).strip() for name in self.class_vocabulary)
        if not vocabulary or any(not name for name in vocabulary):
            raise ValueError("class_vocabulary must contain non-empty class names")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("class_vocabulary class names must be unique")
        matrix = _validated_matrix(
            self.confusion_matrix, num_classes=len(vocabulary)
        )
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "fold", int(self.fold))
        object.__setattr__(self, "class_vocabulary", vocabulary)
        object.__setattr__(self, "confusion_matrix", matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "fold": self.fold,
            "class_vocabulary": list(self.class_vocabulary),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SegmentationConfusionRecord":
        return cls(
            sample_id=str(value["sample_id"]),
            fold=int(value["fold"]),
            class_vocabulary=value["class_vocabulary"],  # type: ignore[arg-type]
            confusion_matrix=value["confusion_matrix"],  # type: ignore[arg-type]
        )


def write_confusion_records(
    path: str | Path, records: Sequence[SegmentationConfusionRecord]
) -> None:
    """Write one fold's selected-checkpoint sample confusion evidence."""
    if not records:
        raise ValueError("confusion evidence must contain at least one sample record")
    fold = records[0].fold
    vocabulary = tuple(records[0].class_vocabulary)
    if any(record.fold != fold for record in records):
        raise ValueError("one confusion evidence artifact must contain exactly one fold")
    if any(tuple(record.class_vocabulary) != vocabulary for record in records):
        raise ValueError("confusion evidence records disagree on class_vocabulary")
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("confusion evidence contains duplicate sample_id records")
    payload = {
        "schema_version": 1,
        "records": [record.to_dict() for record in records],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_confusion_records(path: str | Path) -> list[SegmentationConfusionRecord]:
    """Load and validate one selected-checkpoint confusion evidence artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported confusion evidence schema in {path}")
    records = [
        SegmentationConfusionRecord.from_dict(value) for value in payload["records"]
    ]
    if not records:
        raise ValueError("confusion evidence must contain at least one sample record")
    # Reuse the cross-record validation without rewriting the artifact.
    fold = records[0].fold
    vocabulary = tuple(records[0].class_vocabulary)
    if any(record.fold != fold for record in records):
        raise ValueError("one confusion evidence artifact must contain exactly one fold")
    if any(tuple(record.class_vocabulary) != vocabulary for record in records):
        raise ValueError("confusion evidence records disagree on class_vocabulary")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("confusion evidence contains duplicate sample_id records")
    return records


def validate_confusion_records(
    records: Sequence[SegmentationConfusionRecord],
    *,
    expected_sample_ids: Sequence[str],
    fold: int,
) -> None:
    """Require complete, exactly-once evidence for one held-out fold."""
    expected = tuple(str(sample_id) for sample_id in expected_sample_ids)
    if not expected:
        raise ValueError(f"confusion evidence fold {fold} has no held-out samples")
    if len(set(expected)) != len(expected):
        raise ValueError(f"confusion evidence fold {fold} expected duplicate sample IDs")
    wrong_folds = sorted({record.fold for record in records if record.fold != fold})
    if wrong_folds:
        raise ValueError(
            f"confusion evidence fold {fold} contains record fold(s) {wrong_folds}"
        )
    counts = Counter(record.sample_id for record in records)
    missing = sorted(set(expected) - set(counts))
    unexpected = sorted(set(counts) - set(expected))
    duplicate = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    problems = []
    if missing:
        problems.append(f"missing sample(s): {missing}")
    if unexpected:
        problems.append(f"unexpected sample(s): {unexpected}")
    if duplicate:
        problems.append(f"duplicate sample(s): {duplicate}")
    if problems:
        raise ValueError(
            f"confusion evidence fold {fold} coverage failed; " + "; ".join(problems)
        )


def aggregate_confusion_records(
    records: Sequence[SegmentationConfusionRecord],
) -> ConfusionMetrics:
    """Sum sample matrices, then compute per-class and mean Dice once."""
    if not records:
        raise ValueError("cannot aggregate empty confusion evidence")
    vocabulary = tuple(records[0].class_vocabulary)
    if any(tuple(record.class_vocabulary) != vocabulary for record in records):
        raise ValueError("confusion evidence records disagree on class_vocabulary")
    return aggregate_confusion_matrices(
        [record.confusion_matrix for record in records]
    )


def aggregate_confusion_matrices(
    matrices: Sequence[Sequence[Sequence[int]]],
) -> ConfusionMetrics:
    """Sum arbitrary same-shaped square matrices before recomputing Dice."""
    if not matrices:
        raise ValueError("cannot aggregate empty confusion matrices")
    first = np.asarray(matrices[0])
    if first.ndim != 2 or first.shape[0] == 0 or first.shape[0] != first.shape[1]:
        raise ValueError(f"confusion_matrix must be non-empty and square, got {first.shape}")
    validated = [
        _validated_matrix(matrix, num_classes=int(first.shape[0])) for matrix in matrices
    ]
    matrix = np.sum(np.asarray(validated, dtype=np.int64), axis=0)
    _, dice_per_class = reduce_confusion_matrix_dice(torch.as_tensor(matrix))
    return ConfusionMetrics(
        confusion_matrix=tuple(
            tuple(int(entry) for entry in row) for row in matrix.tolist()
        ),
        dice_per_class=tuple(float(value) for value in dice_per_class),
        mean_dice=float(np.mean(dice_per_class)),
    )


__all__ = [
    "ConfusionMatrix",
    "ConfusionMetrics",
    "SegmentationConfusionRecord",
    "aggregate_confusion_matrices",
    "aggregate_confusion_records",
    "load_confusion_records",
    "validate_confusion_records",
    "write_confusion_records",
]
