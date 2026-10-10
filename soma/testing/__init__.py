"""Conformance checks for the :mod:`soma.data` source protocols.

Plain functions usable from pytest or a notebook. soma runs them on its own stores and
adapters; users run them on custom classes. Each raises :class:`ConformanceError`
naming the method and the offending sample id.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from soma.data.geometry import DenseGridGeometry
from soma.data.sources import Points

__all__ = [
    "ConformanceError",
    "check_grid_source",
    "check_set_source",
    "check_target_source",
]


class ConformanceError(AssertionError):
    """A source violates its protocol; ``method`` and ``sample_id`` locate the breach."""

    def __init__(self, method: str, message: str, *, sample_id: str | None = None) -> None:
        self.method = method
        self.sample_id = sample_id
        where = f"{method}({sample_id!r})" if sample_id is not None else method
        super().__init__(f"{where}: {message}")


def _check_ids(source, method: str = "sample_ids") -> list[str]:
    try:
        ids = list(source.sample_ids)
    except Exception as error:  # noqa: BLE001 - any failure is a conformance failure
        raise ConformanceError(method, f"raised {error!r}") from error
    if not ids:
        raise ConformanceError(method, "must list at least one sample id")
    if not all(isinstance(sid, str) and sid for sid in ids):
        raise ConformanceError(method, "every sample id must be a non-empty str")
    if len(set(ids)) != len(ids):
        raise ConformanceError(method, "sample ids must be unique")
    return ids


def _check_feature_dim(source) -> int:
    dim = getattr(source, "feature_dim", None)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ConformanceError("feature_dim", f"must be a positive int, got {dim!r}")
    return dim


def _select(ids: list[str], sample_ids: Sequence[str] | None, limit: int | None) -> list[str]:
    chosen = list(sample_ids) if sample_ids is not None else ids
    unknown = sorted(set(chosen) - set(ids))
    if unknown:
        raise ConformanceError("sample_ids", f"requested ids are not served: {unknown}")
    return chosen if limit is None else chosen[:limit]


def _call(source, method: str, sample_id: str):
    try:
        return getattr(source, method)(sample_id)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError(method, f"raised {error!r}", sample_id=sample_id) from error


def _check_unknown_id_raises_key_error(source, method: str) -> None:
    ghost = "__soma_conformance_unknown_sample__"
    try:
        getattr(source, method)(ghost)
    except KeyError:
        return
    except Exception as error:  # noqa: BLE001
        raise ConformanceError(
            method, f"must raise KeyError for an unknown id, raised {error!r}", sample_id=ghost
        ) from error
    raise ConformanceError(method, "must raise KeyError for an unknown id", sample_id=ghost)


def check_set_source(
    source, *, sample_ids: Sequence[str] | None = None, limit: int | None = None
) -> None:
    """Check a :class:`~soma.data.SetSource`: ids, ``feature_dim``, declared ``rank``,
    ``load`` shapes and row-aligned ``coords``. ``limit`` bounds how many samples are loaded."""
    ids = _check_ids(source)
    dim = _check_feature_dim(source)
    rank = getattr(source, "rank", None)
    if rank not in (1, 2, 3) or isinstance(rank, bool):
        raise ConformanceError("rank", f"must be 1, 2 or 3, got {rank!r}")
    for sample_id in _select(ids, sample_ids, limit):
        tensor = _call(source, "load", sample_id)
        if not torch.is_tensor(tensor):
            raise ConformanceError("load", f"must return a Tensor, got {type(tensor)!r}", sample_id=sample_id)
        if tensor.ndim != rank:
            raise ConformanceError(
                "load", f"returned rank {tensor.ndim}, declared rank {rank}", sample_id=sample_id
            )
        if int(tensor.shape[-1]) != dim:
            raise ConformanceError(
                "load", f"last axis is {tensor.shape[-1]}, feature_dim is {dim}", sample_id=sample_id
            )
        coords = _call(source, "coords", sample_id)
        if coords is None:
            continue
        if not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[1] != 2:
            raise ConformanceError(
                "coords", f"must be None or an (N, 2) Tensor, got {getattr(coords, 'shape', coords)!r}",
                sample_id=sample_id,
            )
        if rank == 2 and coords.shape[0] != tensor.shape[0]:
            raise ConformanceError(
                "coords",
                f"{coords.shape[0]} rows do not match the bag's {tensor.shape[0]} tiles",
                sample_id=sample_id,
            )
    _check_unknown_id_raises_key_error(source, "load")


def check_grid_source(
    source, *, sample_ids: Sequence[str] | None = None, limit: int | None = None
) -> None:
    """Check a :class:`~soma.data.GridSource`: ids, ``feature_dim``, ``(D, H, W)`` grids
    matching ``geometry(...).grid_shape``, and a positive ``spacing``."""
    ids = _check_ids(source)
    dim = _check_feature_dim(source)
    for sample_id in _select(ids, sample_ids, limit):
        grid = _call(source, "load", sample_id)
        if not torch.is_tensor(grid) or grid.ndim != 3:
            raise ConformanceError(
                "load", f"must return a (D, H, W) Tensor, got {getattr(grid, 'shape', grid)!r}",
                sample_id=sample_id,
            )
        if int(grid.shape[0]) != dim:
            raise ConformanceError(
                "load", f"channel axis is {grid.shape[0]}, feature_dim is {dim}", sample_id=sample_id
            )
        geometry = _call(source, "geometry", sample_id)
        if not isinstance(geometry, DenseGridGeometry):
            raise ConformanceError(
                "geometry", f"must return a GridGeometry, got {type(geometry)!r}", sample_id=sample_id
            )
        if tuple(int(v) for v in geometry.grid_shape) != tuple(int(v) for v in grid.shape[1:]):
            raise ConformanceError(
                "geometry",
                f"grid_shape {geometry.grid_shape} does not match the grid's {tuple(grid.shape[1:])}",
                sample_id=sample_id,
            )
        spacing = _call(source, "spacing", sample_id)
        value = getattr(spacing, "effective_spacing_um", spacing)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ConformanceError(
                "spacing", f"must be a positive µm/px, got {spacing!r}", sample_id=sample_id
            )
    _check_unknown_id_raises_key_error(source, "load")


def check_target_source(
    source, *, sample_ids: Sequence[str] | None = None, limit: int | None = None
) -> None:
    """Check a :class:`~soma.data.TargetSource`: ids and ``load`` returning a ``(H, W)``
    long label map or :class:`~soma.data.Points`."""
    ids = _check_ids(source)
    for sample_id in _select(ids, sample_ids, limit):
        target = _call(source, "load", sample_id)
        if isinstance(target, Points):
            continue
        if not torch.is_tensor(target) or target.ndim != 2:
            raise ConformanceError(
                "load",
                f"must return a (H, W) label map or Points, got {getattr(target, 'shape', target)!r}",
                sample_id=sample_id,
            )
        if target.dtype != torch.long:
            raise ConformanceError(
                "load", f"label map must be int64 (long), got {target.dtype}", sample_id=sample_id
            )
    _check_unknown_id_raises_key_error(source, "load")
