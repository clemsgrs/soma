"""Reconcile the tiling an extraction samples at with the dense read geometry."""

from __future__ import annotations

from dataclasses import replace

from hs2p import TilingConfig
from slide2vec import DenseOptions, PreprocessingConfig


def reconcile_dense_tiling(preprocessing: PreprocessingConfig, dense: DenseOptions | None) -> PreprocessingConfig:
    """``preprocessing`` with a tiling that names ``dense``'s read geometry.

    A grid run samples ROIs and crops their masks at the tiling, and reads and encodes
    them at ``dense``, so both must agree on the spacing, tile size, tolerance and
    backend. An unset tiling takes them from ``dense``; a set one that disagrees is
    refused. Without ``dense``, ``preprocessing`` is returned as held.
    """
    if dense is None:
        return preprocessing
    geometry = (float(dense.spacing_um), int(dense.target_size), float(dense.tolerance), dense.backend)
    tiling = preprocessing.tiling
    if tiling is None:
        spacing_um, tile_size_px, tolerance, backend = geometry
        return replace(
            preprocessing,
            tiling=TilingConfig(
                requested_spacing_um=spacing_um,
                requested_tile_size_px=tile_size_px,
                tolerance=tolerance,
                overlap=0.0,
                backend=backend,
            ),
        )
    held = (
        float(tiling.requested_spacing_um),
        int(tiling.requested_tile_size_px),
        float(tiling.tolerance),
        tiling.backend,
    )
    if held != geometry:
        raise ValueError(
            "The tiling and the dense options disagree on the read geometry (spacing µm/px, "
            f"tile size px, tolerance, backend): tiling {held}, dense {geometry}. A grid run "
            "samples ROIs at the tiling and reads them at the dense options; leave the "
            "tiling unset or make them agree."
        )
    return preprocessing


__all__ = ["reconcile_dense_tiling"]
