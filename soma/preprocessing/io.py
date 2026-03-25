"""Atomic save/load for tiling artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from soma.preprocessing.tiling import TilingResult


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{slide_id}.coordinates.npz"
    meta_path = output_dir / f"{slide_id}.coordinates.meta.json"

    # Save arrays to temp file, then atomic replace
    # np.savez_compressed appends .npz automatically, so use stem-based tmp path
    npz_tmp = output_dir / f"{slide_id}.coordinates.tmp"
    np.savez_compressed(
        str(npz_tmp),
        coordinates=result.coordinates,
        tissue_fractions=result.tissue_fractions,
    )
    # np.savez_compressed creates {npz_tmp}.npz
    npz_tmp_actual = npz_tmp.with_suffix(".tmp.npz")
    os.replace(npz_tmp_actual, npz_path)

    # Save metadata to temp file, then atomic replace
    meta: dict[str, Any] = {
        "requested_tile_size_px": result.requested_tile_size_px,
        "requested_spacing_um": result.requested_spacing_um,
        "read_level": result.read_level,
        "effective_tile_size_px": result.effective_tile_size_px,
        "effective_spacing_um": result.effective_spacing_um,
        "tile_size_lv0": result.tile_size_lv0,
        "is_within_tolerance": result.is_within_tolerance,
        "n_tiles": len(result.coordinates),
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

    return TilingResult(
        coordinates=data["coordinates"],
        tissue_fractions=data["tissue_fractions"],
        requested_tile_size_px=meta["requested_tile_size_px"],
        requested_spacing_um=meta["requested_spacing_um"],
        read_level=meta["read_level"],
        effective_tile_size_px=meta["effective_tile_size_px"],
        effective_spacing_um=meta["effective_spacing_um"],
        tile_size_lv0=meta["tile_size_lv0"],
        is_within_tolerance=meta["is_within_tolerance"],
    )
