"""Atomic save/load for tiling artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from soma.preprocessing.tiling import TilingResult, canonicalize_tiling_result


COORDINATE_SPACE = "level0_px"
TILE_ORDER = "x_then_y"
_TOP_LEVEL_META_KEYS = {
    "requested_tile_size_px",
    "requested_spacing_um",
    "read_level",
    "effective_tile_size_px",
    "effective_spacing_um",
    "tile_size_lv0",
    "is_within_tolerance",
    "use_padding",
    "n_tiles",
    "sample_id",
    "image_path",
    "backend",
    "requested_backend",
    "base_spacing_um",
    "slide_dimensions",
    "level_downsamples",
    "overlap",
    "min_tissue_fraction",
    "step_px_lv0",
    "tissue_method",
    "seg_downsample",
    "seg_level",
    "seg_spacing_um",
    "ref_tile_size_px",
    "a_t",
    "tissue_mask_path",
    "tissue_mask_tissue_value",
    "mask_level",
    "mask_spacing_um",
    "coordinate_space",
    "tile_order",
    "provenance",
    "slide",
    "tiling",
    "segmentation",
    "artifact",
}
_PROVENANCE_KEYS = {
    "sample_id",
    "image_path",
    "tissue_mask_path",
    "backend",
    "requested_backend",
}
_SLIDE_KEYS = {
    "dimensions",
    "base_spacing_um",
    "level_downsamples",
    "seg_level",
    "seg_spacing_um",
}
_TILING_KEYS = {
    "requested_tile_size_px",
    "requested_spacing_um",
    "read_level",
    "effective_tile_size_px",
    "effective_spacing_um",
    "tile_size_lv0",
    "use_padding",
    "step_px_lv0",
    "overlap",
    "min_tissue_fraction",
    "is_within_tolerance",
    "n_tiles",
}
_SEGMENTATION_KEYS = {
    "tissue_method",
    "seg_downsample",
    "seg_level",
    "seg_spacing_um",
    "ref_tile_size_px",
    "a_t",
    "tissue_mask_tissue_value",
    "mask_level",
    "mask_spacing_um",
}
_ARTIFACT_KEYS = {"coordinate_space", "tile_order"}


def _build_tiling_metadata(result: TilingResult) -> dict[str, Any]:
    n_tiles = len(result.coordinates)

    provenance = {
        "sample_id": result.sample_id,
        "image_path": result.image_path,
        "tissue_mask_path": result.tissue_mask_path,
        "backend": result.backend,
        "requested_backend": result.requested_backend,
    }
    slide = {
        "dimensions": result.slide_dimensions,
        "base_spacing_um": result.base_spacing_um,
        "level_downsamples": result.level_downsamples,
        "seg_level": result.seg_level,
        "seg_spacing_um": result.seg_spacing_um,
    }
    tiling = {
        "requested_tile_size_px": result.requested_tile_size_px,
        "requested_spacing_um": result.requested_spacing_um,
        "read_level": result.read_level,
        "effective_tile_size_px": result.effective_tile_size_px,
        "effective_spacing_um": result.effective_spacing_um,
        "tile_size_lv0": result.tile_size_lv0,
        "use_padding": result.use_padding,
        "step_px_lv0": result.step_px_lv0,
        "overlap": result.overlap,
        "min_tissue_fraction": result.min_tissue_fraction,
        "is_within_tolerance": result.is_within_tolerance,
        "n_tiles": n_tiles,
    }
    segmentation = {
        "tissue_method": result.tissue_method,
        "seg_downsample": result.seg_downsample,
        "seg_level": result.seg_level,
        "seg_spacing_um": result.seg_spacing_um,
        "ref_tile_size_px": result.ref_tile_size_px,
        "a_t": result.a_t,
        "tissue_mask_tissue_value": result.tissue_mask_tissue_value,
        "mask_level": result.mask_level,
        "mask_spacing_um": result.mask_spacing_um,
    }
    artifact = {
        "coordinate_space": COORDINATE_SPACE,
        "tile_order": TILE_ORDER,
    }

    meta: dict[str, Any] = {
        "requested_tile_size_px": result.requested_tile_size_px,
        "requested_spacing_um": result.requested_spacing_um,
        "read_level": result.read_level,
        "effective_tile_size_px": result.effective_tile_size_px,
        "effective_spacing_um": result.effective_spacing_um,
        "tile_size_lv0": result.tile_size_lv0,
        "is_within_tolerance": result.is_within_tolerance,
        "use_padding": result.use_padding,
        "n_tiles": n_tiles,
        "sample_id": provenance["sample_id"],
        "image_path": provenance["image_path"],
        "backend": provenance["backend"],
        "requested_backend": provenance["requested_backend"],
        "base_spacing_um": slide["base_spacing_um"],
        "slide_dimensions": slide["dimensions"],
        "level_downsamples": slide["level_downsamples"],
        "overlap": tiling["overlap"],
        "min_tissue_fraction": tiling["min_tissue_fraction"],
        "step_px_lv0": tiling["step_px_lv0"],
        "tissue_method": segmentation["tissue_method"],
        "seg_downsample": segmentation["seg_downsample"],
        "seg_level": segmentation["seg_level"],
        "seg_spacing_um": segmentation["seg_spacing_um"],
        "ref_tile_size_px": segmentation["ref_tile_size_px"],
        "a_t": segmentation["a_t"],
        "tissue_mask_path": provenance["tissue_mask_path"],
        "tissue_mask_tissue_value": segmentation["tissue_mask_tissue_value"],
        "mask_level": segmentation["mask_level"],
        "mask_spacing_um": segmentation["mask_spacing_um"],
        "coordinate_space": artifact["coordinate_space"],
        "tile_order": artifact["tile_order"],
        "provenance": provenance,
        "slide": slide,
        "tiling": tiling,
        "segmentation": segmentation,
        "artifact": artifact,
    }
    return meta


def _validate_tile_index(tile_index: np.ndarray, n_tiles: int) -> np.ndarray:
    tile_index = np.asarray(tile_index, dtype=np.int32)
    if tile_index.ndim != 1 or tile_index.shape[0] != n_tiles:
        raise ValueError("tile_index must be a 1D array aligned with coordinates")
    expected = np.arange(n_tiles, dtype=np.int32)
    if not np.array_equal(tile_index, expected):
        raise ValueError("tile_index must be a contiguous range from 0 to n_tiles-1")
    return tile_index


def _validate_metadata_schema(meta: dict[str, Any]) -> None:
    def _raise_key_error(section: str, missing: set[str], extra: set[str]) -> None:
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys {sorted(extra)}")
        raise ValueError(f"Invalid tiling metadata in {section}: " + "; ".join(parts))

    top_keys = set(meta)
    missing_top = _TOP_LEVEL_META_KEYS - top_keys
    extra_top = top_keys - _TOP_LEVEL_META_KEYS
    if missing_top or extra_top:
        _raise_key_error("top-level", missing_top, extra_top)

    sections = {
        "provenance": _PROVENANCE_KEYS,
        "slide": _SLIDE_KEYS,
        "tiling": _TILING_KEYS,
        "segmentation": _SEGMENTATION_KEYS,
        "artifact": _ARTIFACT_KEYS,
    }
    for section_name, expected_keys in sections.items():
        section = meta[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"Invalid tiling metadata in {section_name}: expected object")
        section_keys = set(section)
        missing = expected_keys - section_keys
        extra = section_keys - expected_keys
        if missing or extra:
            _raise_key_error(section_name, missing, extra)


def normalize_artifact_path(path: str | Path | None) -> str | None:
    """Normalize persisted provenance paths for stable equality checks."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def validate_tiling_result_provenance(
    result: TilingResult,
    *,
    sample_id: str,
    image_path: str | Path,
    tissue_mask_path: str | Path | None,
    tissue_mask_tissue_value: int | None,
) -> None:
    """Validate that a tiling artifact matches the requested sample provenance."""
    if result.sample_id != sample_id:
        raise ValueError(
            f"Precomputed tiles sample_id mismatch: expected {sample_id!r}, found {result.sample_id!r}"
        )
    expected_image = normalize_artifact_path(image_path)
    actual_image = normalize_artifact_path(result.image_path)
    if actual_image != expected_image:
        raise ValueError(
            "Precomputed tiles image_path mismatch: "
            f"expected {expected_image!r}, found {actual_image!r}"
        )
    expected_mask = normalize_artifact_path(tissue_mask_path)
    actual_mask = normalize_artifact_path(result.tissue_mask_path)
    if actual_mask != expected_mask:
        raise ValueError(
            "Precomputed tiles tissue_mask_path mismatch: "
            f"expected {expected_mask!r}, found {actual_mask!r}"
        )
    if result.tissue_mask_tissue_value != tissue_mask_tissue_value:
        raise ValueError(
            "Precomputed tiles tissue_mask_tissue_value mismatch: "
            f"expected {tissue_mask_tissue_value!r}, found {result.tissue_mask_tissue_value!r}"
        )


def save_tiling_result(
    result: TilingResult, output_dir: Path, slide_id: str
) -> dict[str, Path]:
    """Atomically save a TilingResult to disk.

    Writes to .tmp files first, then uses os.replace() for atomicity.

    Produces:
        {slide_id}.coordinates.npz — coordinates + tissue_fractions
        {slide_id}.coordinates.meta.json — scalar metadata + provenance payload

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
    meta = _build_tiling_metadata(result)
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
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
    _validate_metadata_schema(meta)
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
        use_padding=meta["use_padding"],
        tile_index=tile_index,
        sample_id=meta["sample_id"],
        image_path=meta["image_path"],
        backend=meta["backend"],
        requested_backend=meta["requested_backend"],
        base_spacing_um=meta["base_spacing_um"],
        slide_dimensions=meta["slide_dimensions"],
        level_downsamples=meta["level_downsamples"],
        overlap=meta["overlap"],
        min_tissue_fraction=meta["min_tissue_fraction"],
        step_px_lv0=meta["step_px_lv0"],
        tissue_method=meta["tissue_method"],
        seg_downsample=meta["seg_downsample"],
        seg_level=meta["seg_level"],
        seg_spacing_um=meta["seg_spacing_um"],
        ref_tile_size_px=meta["ref_tile_size_px"],
        a_t=meta["a_t"],
        tissue_mask_path=meta["tissue_mask_path"],
        tissue_mask_tissue_value=meta["tissue_mask_tissue_value"],
        mask_level=meta["mask_level"],
        mask_spacing_um=meta["mask_spacing_um"],
    )
