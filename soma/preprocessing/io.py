"""Atomic save/load for tiling artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from soma.preprocessing.tiling import TilingResult, canonicalize_tiling_result


def _validate_tile_index(tile_index: np.ndarray, n_tiles: int) -> np.ndarray:
    tile_index = np.asarray(tile_index, dtype=np.int32)
    if tile_index.ndim != 1 or tile_index.shape[0] != n_tiles:
        raise ValueError("tile_index must be a 1D array aligned with coordinates")
    expected = np.arange(n_tiles, dtype=np.int32)
    if not np.array_equal(tile_index, expected):
        raise ValueError("tile_index must be a contiguous range from 0 to n_tiles-1")
    return tile_index


def save_tiling_result(
    result: TilingResult, output_dir: Path, slide_id: str
) -> dict[str, Path]:
    """Atomically save a TilingResult to disk.

    Writes to .tmp files first, then uses os.replace() for atomicity.

    Produces:
        {slide_id}.coordinates.npz — coordinates + tissue_fractions
        {slide_id}.coordinates.meta.json — scalar metadata

    Returns:
        Dict with 'npz' and 'meta' keys pointing to the saved files.
    """
    result = canonicalize_tiling_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{slide_id}.coordinates.npz"
    meta_path = output_dir / f"{slide_id}.coordinates.meta.json"

    # Save arrays to temp file, then atomic replace
    # np.savez_compressed appends .npz automatically, so use stem-based tmp path
    npz_tmp = output_dir / f"{slide_id}.coordinates.tmp"
    np.savez_compressed(
        str(npz_tmp),
        tile_index=_validate_tile_index(result.tile_index, len(result.coordinates)),
        coordinates=result.coordinates,
        tissue_fractions=result.tissue_fractions,
    )
    # np.savez_compressed creates {npz_tmp}.npz
    npz_tmp_actual = npz_tmp.with_suffix(".tmp.npz")
    os.replace(npz_tmp_actual, npz_path)

    # Save metadata to temp file, then atomic replace
    n_tiles = len(result.coordinates)
    meta: dict[str, Any] = {
        "requested_tile_size_px": result.requested_tile_size_px,
        "requested_spacing_um": result.requested_spacing_um,
        "read_level": result.read_level,
        "effective_tile_size_px": result.effective_tile_size_px,
        "effective_spacing_um": result.effective_spacing_um,
        "tile_size_lv0": result.tile_size_lv0,
        "is_within_tolerance": result.is_within_tolerance,
        "n_tiles": n_tiles,
    }

    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2))
    os.replace(meta_tmp, meta_path)

    return {"npz": npz_path, "meta": meta_path}


def load_tiling_result(npz_path: Path, meta_path: Path) -> TilingResult:
    """Load a TilingResult from disk.

    Args:
        npz_path: Path to the .npz file with coordinates and tissue_fractions.
        meta_path: Path to the .meta.json file with scalar metadata.

    Returns:
        Reconstructed TilingResult (tissue_mask will be None).
    """
    data = np.load(npz_path)
    meta = json.loads(Path(meta_path).read_text())
    coordinates = data["coordinates"]
    tissue_fractions = data["tissue_fractions"]
    n_tiles = len(coordinates)
    if tissue_fractions.ndim != 1 or tissue_fractions.shape[0] != n_tiles:
        raise ValueError("Invalid tiling npz artifact: tissue_fractions length mismatch")
    if "tile_index" not in data:
        raise ValueError("Invalid tiling npz artifact: missing tile_index")
    tile_index = _validate_tile_index(data["tile_index"], n_tiles)

    return TilingResult(
        coordinates=coordinates,
        tissue_fractions=tissue_fractions,
        requested_tile_size_px=meta["requested_tile_size_px"],
        requested_spacing_um=meta["requested_spacing_um"],
        read_level=meta["read_level"],
        effective_tile_size_px=meta["effective_tile_size_px"],
        effective_spacing_um=meta["effective_spacing_um"],
        tile_size_lv0=meta["tile_size_lv0"],
        is_within_tolerance=meta["is_within_tolerance"],
        tile_index=tile_index,
    )
