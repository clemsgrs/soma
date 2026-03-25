"""Tile coordinate generation from tissue contours.

Coordinates are always returned in level-0 pixel space (following hs2p's convention).
Tissue fraction per tile is computed efficiently via integral images (hs2p's trick).
Per-contour processing enables parallelism via ThreadPoolExecutor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from soma.preprocessing.tissue import ContourResult
from soma.wsi.reader import select_level


@dataclass(frozen=True)
class TilingResult:
    """Result of tile coordinate generation."""

    coordinates: np.ndarray  # (N, 2) — (x, y) top-left in level-0 pixels
    tissue_fractions: np.ndarray  # (N,) tissue coverage per tile

    requested_tile_size_px: int
    requested_spacing_um: float

    read_level: int
    effective_tile_size_px: int
    effective_spacing_um: float
    tile_size_lv0: int
    is_within_tolerance: bool

    tissue_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_tiles(
    slide_dimensions: tuple[int, int],
    contours: ContourResult,
    *,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    base_spacing_um: float,
    level_downsamples: list[float],
    overlap: float = 0.0,
    min_tissue_fraction: float = 0.5,
    tolerance: float = 0.05,
    num_workers: int = 1,
) -> TilingResult:
    """Generate tile coordinates within tissue contours.

    Args:
        slide_dimensions: (width, height) at level 0.
        contours: ContourResult from detect_contours().
        requested_tile_size_px: Desired tile size at requested spacing.
        requested_spacing_um: Desired spacing in µm/px.
        base_spacing_um: Native spacing at level 0 in µm/px.
        level_downsamples: Downsample factor at each pyramid level.
        overlap: Fraction of tile overlap (0.0 = no overlap).
        min_tissue_fraction: Minimum tissue coverage to keep a tile.
        tolerance: Tolerance for level selection.
        num_workers: Number of threads for per-contour processing.

    Returns:
        TilingResult with coordinates in level-0 pixel space.
    """
    # Resolve pyramid level
    level_sel = select_level(
        requested_spacing_um, level_downsamples, base_spacing_um, tolerance=tolerance
    )

    # Compute effective tile size and level-0 footprint
    effective_tile_size_px = round(
        requested_tile_size_px * requested_spacing_um / level_sel.effective_spacing_um
    )
    tile_size_lv0 = round(requested_tile_size_px * requested_spacing_um / base_spacing_um)

    # Step size in level-0 pixels
    step_lv0 = max(1, round(tile_size_lv0 * (1.0 - overlap)))

    slide_w, slide_h = slide_dimensions

    # Empty result template
    def _empty_result() -> TilingResult:
        return TilingResult(
            coordinates=np.empty((0, 2), dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            effective_tile_size_px=effective_tile_size_px,
            effective_spacing_um=level_sel.effective_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            tissue_mask=contours.mask,
        )

    if len(contours.contours) == 0:
        return _empty_result()

    # Tile can't fit in slide
    if tile_size_lv0 > slide_w or tile_size_lv0 > slide_h:
        return _empty_result()

    # Process each contour
    def _process_contour(idx: int) -> tuple[np.ndarray, np.ndarray]:
        contour = contours.contours[idx]
        holes = contours.holes[idx]
        return _tiles_for_contour(
            contour=contour,
            holes=holes,
            tissue_mask=contours.mask,
            slide_dimensions=slide_dimensions,
            tile_size_lv0=tile_size_lv0,
            step_lv0=step_lv0,
            min_tissue_fraction=min_tissue_fraction,
        )

    if num_workers > 1 and len(contours.contours) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_contour, range(len(contours.contours))))
    else:
        results = [_process_contour(i) for i in range(len(contours.contours))]

    # Merge and deduplicate
    all_coords = []
    all_fracs = []
    for coords, fracs in results:
        if len(coords) > 0:
            all_coords.append(coords)
            all_fracs.append(fracs)

    if not all_coords:
        return _empty_result()

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_fracs = np.concatenate(all_fracs, axis=0)

    # Deduplicate by (x, y)
    if len(merged_coords) > 1:
        _, unique_idx = np.unique(merged_coords, axis=0, return_index=True)
        unique_idx.sort()  # preserve original order
        merged_coords = merged_coords[unique_idx]
        merged_fracs = merged_fracs[unique_idx]

    return TilingResult(
        coordinates=merged_coords,
        tissue_fractions=merged_fracs,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        read_level=level_sel.level,
        effective_tile_size_px=effective_tile_size_px,
        effective_spacing_um=level_sel.effective_spacing_um,
        tile_size_lv0=tile_size_lv0,
        is_within_tolerance=level_sel.is_within_tolerance,
        tissue_mask=contours.mask,
    )


def _tiles_for_contour(
    contour: np.ndarray,
    holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    tile_size_lv0: int,
    step_lv0: int,
    min_tissue_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate tiles within a single contour's bounding box."""
    slide_w, slide_h = slide_dimensions

    # Bounding box of contour in level-0 coords
    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)

    # Grid within bounding box, clipped to slide
    x_start = max(x_cont, 0)
    y_start = max(y_cont, 0)
    x_end = min(x_cont + w_cont, slide_w - tile_size_lv0 + 1)
    y_end = min(y_cont + h_cont, slide_h - tile_size_lv0 + 1)

    if x_end <= x_start or y_end <= y_start:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    # Align to step grid
    xs = np.arange(x_start, x_end, step_lv0, dtype=np.int64)
    ys = np.arange(y_start, y_end, step_lv0, dtype=np.int64)

    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Compute tissue fractions via integral image on the mask
    fractions = _compute_tissue_fractions(
        candidates=candidates,
        tissue_mask=tissue_mask,
        tile_size_lv0=tile_size_lv0,
        slide_dimensions=slide_dimensions,
    )

    keep = fractions >= min_tissue_fraction
    return candidates[keep], fractions[keep]


def _compute_tissue_fractions(
    candidates: np.ndarray,
    tissue_mask: np.ndarray,
    tile_size_lv0: int,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    """Compute tissue fraction per tile using an integral image for O(1) queries."""
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions

    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    binary = (tissue_mask > 0).astype(np.float64)
    integral = cv2.integral(binary)  # (mask_h+1, mask_w+1)

    tile_w_mask = max(1, round(tile_size_lv0 * scale_x))
    tile_h_mask = max(1, round(tile_size_lv0 * scale_y))
    tile_area = tile_w_mask * tile_h_mask

    xs_mask = np.round(candidates[:, 0] * scale_x).astype(np.int64)
    ys_mask = np.round(candidates[:, 1] * scale_y).astype(np.int64)

    x1 = np.clip(xs_mask, 0, mask_w)
    y1 = np.clip(ys_mask, 0, mask_h)
    x2 = np.clip(xs_mask + tile_w_mask, 0, mask_w)
    y2 = np.clip(ys_mask + tile_h_mask, 0, mask_h)

    tissue_sum = (
        integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    )

    fractions = (tissue_sum / tile_area).astype(np.float32)
    return np.clip(fractions, 0.0, 1.0)
