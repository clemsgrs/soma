"""Source protocols: the only way features and dense targets enter soma (design §4.2).

Two shapes. A :class:`SetSource` serves one tensor per sample of declared rank: ``(D,)``
vectors, ``(N, D)`` bags, or ``(M, K, D)`` hierarchical bags (HIPT is rank 3; there is
no hierarchy type). A :class:`GridSource` serves token grids with their geometry and
spacing. A :class:`TargetSource` serves dense supervision in physical units (a label
map or points); the learner maps it onto the grid with the feature source's geometry.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from soma.data.cohort import Cohort
from soma.data.geometry import GridGeometry

__all__ = [
    "GridSource",
    "GroupedSetSource",
    "ImageSource",
    "Points",
    "SetSource",
    "TargetSource",
    "covers",
    "group_by",
    "require_coverage",
]


@runtime_checkable
class SetSource(Protocol):
    """Per-sample feature tensors of one declared rank, row-aligned with ``coords``."""

    @property
    def sample_ids(self) -> list[str]: ...

    @property
    def feature_dim(self) -> int: ...

    @property
    def rank(self) -> Literal[1, 2, 3]: ...

    def load(self, sample_id: str) -> Tensor:
        """``(D,)`` for rank 1, ``(N, D)`` for rank 2, ``(M, K, D)`` for rank 3."""
        ...

    def coords(self, sample_id: str) -> Tensor | None:
        """Level-0 ``(N, 2)`` tile origins aligned with ``load``'s rows, or ``None``."""
        ...


@runtime_checkable
class GridSource(Protocol):
    """Per-sample token grids with their geometry and the spacing they were read at."""

    @property
    def sample_ids(self) -> list[str]: ...

    @property
    def feature_dim(self) -> int: ...

    def load(self, sample_id: str) -> Tensor:
        """A ``(D, H, W)`` channel-first token grid (the layout decoders consume)."""
        ...

    def geometry(self, sample_id: str) -> GridGeometry: ...

    def spacing(self, sample_id: str) -> float:
        """Effective µm/px of the grid's pixel frame."""
        ...


@dataclass(frozen=True)
class Points:
    """Point annotations in level-0 pixels: ``xy`` ``(N, 2)`` float, ``classes`` ``(N,)`` long.

    ``ignore_mask`` is the optional level-0 don't-care raster (255 = ignored) that
    travels with the points so the learner can mask both targets and predictions.
    """

    xy: Tensor
    classes: Tensor
    ignore_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"Points.xy must be (N, 2), got {tuple(self.xy.shape)}")
        if self.classes.ndim != 1 or self.classes.shape[0] != self.xy.shape[0]:
            raise ValueError(
                f"Points.classes must be (N,) aligned with xy, got {tuple(self.classes.shape)} "
                f"for N={self.xy.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.xy.shape[0])


@runtime_checkable
class TargetSource(Protocol):
    """Dense supervision per sample, in physical units: a label map or points."""

    @property
    def sample_ids(self) -> list[str]: ...

    def load(self, sample_id: str) -> Tensor | Points:
        """A ``(H', W')`` long label map (``ignore_index`` folded in) or :class:`Points`."""
        ...


@runtime_checkable
class ImageSource(Protocol):
    """Live path: pixels plus geometry per sample. Declared here, implemented in slice 4."""

    @property
    def sample_ids(self) -> list[str]: ...

    def load(self, sample_id: str) -> Tensor: ...

    def geometry(self, sample_id: str) -> GridGeometry: ...


def covers(source, sample_ids: Iterable[str]) -> bool:
    """True when ``source`` serves every id in ``sample_ids``."""
    return not set(map(str, sample_ids)) - set(source.sample_ids)


def require_coverage(source, sample_ids: Iterable[str], *, what: str = "features") -> None:
    """Raise ``ValueError`` naming the ids ``source`` does not serve."""
    missing = sorted(set(map(str, sample_ids)) - set(source.sample_ids))
    if missing:
        raise ValueError(f"Missing {what} for {len(missing)} samples: {missing}")


class GroupedSetSource:
    """A :class:`SetSource` over unit ids, each built from its members' tensors.

    ``how="concat"`` concatenates members along the bag axis (vectors become an
    ``(M, D)`` bag, bags grow); ``how="stack"`` adds a leading member axis (vectors
    become ``(M, D)``, equal-size bags become ``(M, K, D)``).
    """

    def __init__(
        self,
        source: SetSource,
        members: dict[str, tuple[str, ...]],
        *,
        how: Literal["concat", "stack"],
    ) -> None:
        if how not in ("concat", "stack"):
            raise ValueError(f"how must be 'concat' or 'stack', got {how!r}")
        if source.rank == 3 and how == "stack":
            raise ValueError("Cannot stack rank-3 members: soma has no rank-4 set shape.")
        self._source = source
        self._members = {str(unit): tuple(str(m) for m in ids) for unit, ids in members.items()}
        self._how = how

    @property
    def source(self) -> SetSource:
        return self._source

    @property
    def members(self) -> dict[str, tuple[str, ...]]:
        return dict(self._members)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._members)

    @property
    def feature_dim(self) -> int:
        return int(self._source.feature_dim)

    @property
    def rank(self) -> Literal[1, 2, 3]:
        base = int(self._source.rank)
        if self._how == "concat":
            return 2 if base == 1 else base  # type: ignore[return-value]
        return min(base + 1, 3)  # type: ignore[return-value]

    def _member_ids(self, unit_id: str) -> tuple[str, ...]:
        try:
            return self._members[unit_id]
        except KeyError:
            raise KeyError(
                f"Unit '{unit_id}' not found in grouped source. Available: {sorted(self._members)}"
            ) from None

    def load(self, unit_id: str) -> Tensor:
        tensors = [self._source.load(sid) for sid in self._member_ids(unit_id)]
        if self._how == "stack" or self._source.rank == 1:
            sizes = {tuple(t.shape) for t in tensors}
            if len(sizes) > 1:
                raise ValueError(
                    f"Unit '{unit_id}' cannot be stacked: its members have shapes "
                    f"{sorted(sizes)}; stacking needs equal-size members."
                )
            return torch.stack(tensors)
        return torch.cat(tensors, dim=0)

    def coords(self, unit_id: str) -> Tensor | None:
        if self._how == "stack" or self._source.rank != 2:
            return None
        coords = [self._source.coords(sid) for sid in self._member_ids(unit_id)]
        if any(c is None for c in coords):
            return None
        return torch.cat(coords, dim=0)

    def __len__(self) -> int:
        return len(self._members)


def group_by(
    source: SetSource,
    cohort: Cohort,
    unit: str = "patient_id",
    how: Literal["concat", "stack"] = "concat",
) -> SetSource:
    """Derive a per-``unit`` set source from a per-sample one.

    Members of a unit are the cohort's records sharing that ``patient_id``, in cohort
    order. Pair with :meth:`Cohort.collapse` for the unit-level records. ``unit="sample_id"``
    returns ``source`` unchanged.
    """
    if unit == "sample_id":
        return source
    if unit != "patient_id":
        raise ValueError(f"Unknown sampling unit {unit!r}; expected 'patient_id' or 'sample_id'.")
    members: dict[str, list[str]] = {}
    for record in cohort.records:
        if record.patient_id is None:
            raise ValueError(
                f"Sample {record.sample_id!r} has no patient_id; every record needs one to "
                "group features by patient."
            )
        members.setdefault(record.patient_id, []).append(record.sample_id)
    return GroupedSetSource(
        source, {unit_id: tuple(ids) for unit_id, ids in members.items()}, how=how
    )
