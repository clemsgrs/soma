"""SampleRecord: identity plus targets, nothing else (design §4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SampleRecord", "targets_equal"]


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    return bool(left == right)


def targets_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare two target dicts, treating array-valued targets element-wise."""
    if left.keys() != right.keys():
        return False
    return all(_values_equal(left[key], right[key]) for key in left)


@dataclass(frozen=True, eq=False)
class SampleRecord:
    """One sample: its identity and the targets a head reads.

    ``targets`` holds the values the task head declares in ``target_keys`` (``label``,
    ``value``, ``time`` / ``event``, ``expression`` ...). File paths never live here:
    extraction and curation read them from :class:`~soma.data.ImageManifest` /
    :class:`~soma.data.AnnotationManifest`. Sampling-unit membership is ``patient_id``;
    everything else from the manifest lands in ``metadata``.
    """

    sample_id: str
    targets: dict[str, Any] = field(default_factory=dict)
    patient_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SampleRecord):
            return NotImplemented
        return (
            self.sample_id == other.sample_id
            and self.patient_id == other.patient_id
            and targets_equal(self.targets, other.targets)
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        return hash((self.sample_id, self.patient_id))
