"""Tile coordinate generation from tissue contours.

Coordinates are always returned in level-0 pixel space (following hs2p's convention).
Tissue fraction per tile is computed efficiently via integral images (hs2p's trick).
Per-contour processing enables parallelism via ThreadPoolExecutor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
    use_padding: bool = True

    tile_index: np.ndarray | None = None  # (N,) contiguous saved-artifact ids
    tissue_mask: np.ndarray | None = None

    sample_id: str | None = None
    image_path: str | None = None
    backend: str | None = None
    requested_backend: str | None = None
    base_spacing_um: float | None = None
    slide_dimensions: list[int] | None = None
    level_downsamples: list[float] | None = None
    overlap: float | None = None
    min_tissue_fraction: float | None = None
    step_px_lv0: int | None = None
    tissue_method: str | None = None
    seg_downsample: int | None = None
    seg_level: int | None = None
    seg_spacing_um: float | None = None
    ref_tile_size_px: int | None = None
    a_t: float | None = None
    tissue_mask_path: str | None = None
    tissue_mask_tissue_value: int | None = None
    mask_level: int | None = None
    mask_spacing_um: float | None = None

    # Hierarchical (HIPT-style) fields — set by expand_regions_to_subtiles()
    hierarchical: bool = False
    npatch: int | None = None  # grid dim per region (e.g., 16 → 16×16 subtiles/region)
    region_index: np.ndarray | None = None  # (N,) maps each subtile to parent region
    region_coordinates: np.ndarray | None = None  # (M, 2) original region coordinates
    requested_region_size_px: int | None = None  # original region tile size

    def __post_init__(self) -> None:
        n_tiles = int(self.coordinates.shape[0])
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError(
                f"coordinates must have shape (N, 2), got {self.coordinates.shape}"
            )
        if self.tissue_fractions.ndim != 1 or self.tissue_fractions.shape[0] != n_tiles:
            raise ValueError(
                "tissue_fractions must be a 1D array aligned with coordinates"
            )
        if self.tile_index is None:
            object.__setattr__(self, "tile_index", np.arange(n_tiles, dtype=np.int32))
            return
        tile_index = np.asarray(self.tile_index, dtype=np.int32)
        if tile_index.ndim != 1 or tile_index.shape[0] != n_tiles:
            raise ValueError("tile_index must be a 1D array aligned with coordinates")
        object.__setattr__(self, "tile_index", tile_index)


def _context_kwargs(result: TilingResult) -> dict[str, object | None]:
    return {
        "sample_id": result.sample_id,
        "image_path": result.image_path,
        "backend": result.backend,
        "requested_backend": result.requested_backend,
        "base_spacing_um": result.base_spacing_um,
        "slide_dimensions": None if result.slide_dimensions is None else list(result.slide_dimensions),
        "level_downsamples": None
        if result.level_downsamples is None
        else list(result.level_downsamples),
        "overlap": result.overlap,
        "min_tissue_fraction": result.min_tissue_fraction,
        "step_px_lv0": result.step_px_lv0,
        "tissue_method": result.tissue_method,
        "seg_downsample": result.seg_downsample,
        "seg_level": result.seg_level,
        "seg_spacing_um": result.seg_spacing_um,
        "ref_tile_size_px": result.ref_tile_size_px,
        "a_t": result.a_t,
        "tissue_mask_path": result.tissue_mask_path,
        "tissue_mask_tissue_value": result.tissue_mask_tissue_value,
        "mask_level": result.mask_level,
        "mask_spacing_um": result.mask_spacing_um,
    }


def canonicalize_tiling_result(result: TilingResult) -> TilingResult:
    """Deduplicate and sort a tiling result into canonical x-then-y order."""
    coords = result.coordinates
    fracs = result.tissue_fractions

    if len(coords) > 1:
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx.sort()  # preserve first occurrence during deduplication
        coords = coords[unique_idx]
        fracs = fracs[unique_idx]

        order = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[order]
        fracs = fracs[order]

    return TilingResult(
        coordinates=coords,
        tissue_fractions=fracs,
        requested_tile_size_px=result.requested_tile_size_px,
        requested_spacing_um=result.requested_spacing_um,
        read_level=result.read_level,
        effective_tile_size_px=result.effective_tile_size_px,
        effective_spacing_um=result.effective_spacing_um,
        tile_size_lv0=result.tile_size_lv0,
        is_within_tolerance=result.is_within_tolerance,
        use_padding=result.use_padding,
        tile_index=np.arange(len(coords), dtype=np.int32),
        tissue_mask=result.tissue_mask,
        **_context_kwargs(result),
    )


def resolve_base_spacing_um(result: TilingResult) -> float:
    """Resolve level-0 spacing stored alongside a tiling artifact."""
    if result.base_spacing_um is None:
        raise ValueError("TilingResult is missing base_spacing_um metadata")
    return float(result.base_spacing_um)


def expand_regions_to_subtiles(
    tiling_result: TilingResult, npatch: int
) -> TilingResult:
    """Expand region-level coordinates into subtile coordinates for hierarchical MIL.

    Each region of size ``tile_size_lv0`` is subdivided into an ``npatch × npatch``
    grid of subtiles in **row-major order** (matching VisionTransformer4K's
    ``flatten(2, 3)`` on ``(D, npatch, npatch)`` input).

    Args:
        tiling_result: Region-level TilingResult (coordinates are region top-lefts).
        npatch: Grid dimension per region (e.g., 16 → 256 subtiles per region).

    Returns:
        New TilingResult with ``M * npatch²`` subtile coordinates and hierarchical
        metadata (``hierarchical=True``, ``npatch``, ``region_index``,
        ``region_coordinates``, ``requested_region_size_px``).
    """
    if tiling_result.tile_size_lv0 % npatch != 0:
        msg = (
            f"tile_size_lv0 ({tiling_result.tile_size_lv0}) must be divisible "
            f"by npatch ({npatch})"
        )
        raise ValueError(msg)

    subtile_size_lv0 = tiling_result.tile_size_lv0 // npatch
    m = len(tiling_result.coordinates)
    p = npatch * npatch

    if m == 0:
        return TilingResult(
            coordinates=np.empty((0, 2), dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            requested_tile_size_px=tiling_result.requested_tile_size_px // npatch,
            requested_spacing_um=tiling_result.requested_spacing_um,
            read_level=tiling_result.read_level,
            effective_tile_size_px=tiling_result.effective_tile_size_px // npatch,
            effective_spacing_um=tiling_result.effective_spacing_um,
            tile_size_lv0=subtile_size_lv0,
            is_within_tolerance=tiling_result.is_within_tolerance,
            use_padding=tiling_result.use_padding,
            hierarchical=True,
            npatch=npatch,
            region_index=np.empty(0, dtype=np.int32),
            region_coordinates=tiling_result.coordinates.copy(),
            requested_region_size_px=tiling_result.requested_tile_size_px,
            **_context_kwargs(tiling_result),
        )

    # Build subtile offsets: row-major (i=row, j=col)
    # Flat index i*npatch + j → offset (j * subtile_size, i * subtile_size)
    rows, cols = np.divmod(np.arange(p), npatch)
    offsets_x = cols * subtile_size_lv0  # (P,)
    offsets_y = rows * subtile_size_lv0  # (P,)

    # Broadcast: (M, 1) + (1, P) → (M, P) for both x and y
    region_x = tiling_result.coordinates[:, 0:1]  # (M, 1)
    region_y = tiling_result.coordinates[:, 1:2]  # (M, 1)
    subtile_x = region_x + offsets_x[np.newaxis, :]  # (M, P)
    subtile_y = region_y + offsets_y[np.newaxis, :]  # (M, P)

    coords = np.stack([subtile_x.ravel(), subtile_y.ravel()], axis=1).astype(np.int64)
    tissue_fracs = np.repeat(tiling_result.tissue_fractions, p)
    region_index = np.repeat(np.arange(m, dtype=np.int32), p)

    return TilingResult(
        coordinates=coords,
        tissue_fractions=tissue_fracs,
        requested_tile_size_px=tiling_result.requested_tile_size_px // npatch,
        requested_spacing_um=tiling_result.requested_spacing_um,
        read_level=tiling_result.read_level,
        effective_tile_size_px=tiling_result.effective_tile_size_px // npatch,
        effective_spacing_um=tiling_result.effective_spacing_um,
        tile_size_lv0=subtile_size_lv0,
        is_within_tolerance=tiling_result.is_within_tolerance,
        use_padding=tiling_result.use_padding,
        tile_index=np.arange(m * p, dtype=np.int32),
        hierarchical=True,
        npatch=npatch,
        region_index=region_index,
        region_coordinates=tiling_result.coordinates.copy(),
        requested_region_size_px=tiling_result.requested_tile_size_px,
        **_context_kwargs(tiling_result),
    )


def generate_tiles(
    slide_dimensions: tuple[int, int],
    contours: ContourResult,
    *,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
    base_spacing_um: float,
    level_downsamples: list[float],
    overlap: float = 0.0,
    use_padding: bool = True,
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
    if requested_spacing_um < base_spacing_um:
        relative_diff = abs(base_spacing_um - requested_spacing_um) / requested_spacing_um
        if relative_diff > tolerance:
            raise ValueError(
                f"Desired spacing ({requested_spacing_um}) is smaller than the "
                f"whole-slide image starting spacing ({base_spacing_um}) and does not "
                f"fall within tolerance ({tolerance:.0%})"
            )

    # Resolve pyramid level
    level_sel = select_level(
        requested_spacing_um, level_downsamples, base_spacing_um, tolerance=tolerance
    )

    # Compute effective tile size and level-0 footprint
    if level_sel.is_within_tolerance:
        effective_tile_size_px = requested_tile_size_px
    else:
        effective_tile_size_px = round(
            requested_tile_size_px
            * requested_spacing_um
            / level_sel.effective_spacing_um
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
            use_padding=use_padding,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(level_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
        )

    if len(contours.contours) == 0:
        return _empty_result()

    # Tile can't fit in slide
    if not use_padding and (tile_size_lv0 > slide_w or tile_size_lv0 > slide_h):
        return _empty_result()

    # Process each contour
    def _process_contour(idx: int) -> tuple[np.ndarray, np.ndarray]:
        contour = contours.contours[idx]
        return _tiles_for_contour(
            contour=contour,
            contour_holes=contours.holes[idx],
            tissue_mask=contours.mask,
            slide_dimensions=slide_dimensions,
            tile_size_lv0=tile_size_lv0,
            step_lv0=step_lv0,
            use_padding=use_padding,
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

    return canonicalize_tiling_result(
        TilingResult(
            coordinates=merged_coords,
            tissue_fractions=merged_fracs,
            requested_tile_size_px=requested_tile_size_px,
            requested_spacing_um=requested_spacing_um,
            read_level=level_sel.level,
            effective_tile_size_px=effective_tile_size_px,
            effective_spacing_um=level_sel.effective_spacing_um,
            tile_size_lv0=tile_size_lv0,
            is_within_tolerance=level_sel.is_within_tolerance,
            use_padding=use_padding,
            tissue_mask=contours.mask,
            base_spacing_um=base_spacing_um,
            slide_dimensions=list(slide_dimensions),
            level_downsamples=list(level_downsamples),
            overlap=overlap,
            min_tissue_fraction=min_tissue_fraction,
        )
    )


def _tiles_for_contour(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    tile_size_lv0: int,
    step_lv0: int,
    use_padding: bool,
    min_tissue_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate tiles within a single contour's bounding box."""
    slide_w, slide_h = slide_dimensions

    # Bounding box of contour in level-0 coords
    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)

    # Grid within bounding box, clipped to slide
    x_start = max(x_cont, 0)
    y_start = max(y_cont, 0)
    if use_padding:
        x_end = min(x_cont + w_cont, slide_w)
        y_end = min(y_cont + h_cont, slide_h)
    else:
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

    contour_mask = _build_contour_tissue_mask(
        contour=contour,
        contour_holes=contour_holes,
        tissue_mask=tissue_mask,
        slide_dimensions=slide_dimensions,
    )

    # Compute tissue fractions via integral image on the mask
    fractions = _compute_tissue_fractions(
        candidates=candidates,
        tissue_mask=contour_mask,
        tile_size_lv0=tile_size_lv0,
        slide_dimensions=slide_dimensions,
        use_padding=use_padding,
    )

    keep = fractions >= min_tissue_fraction
    return candidates[keep], fractions[keep]


def _build_contour_tissue_mask(
    contour: np.ndarray,
    contour_holes: list[np.ndarray],
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
) -> np.ndarray:
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = mask_w / slide_w
    scale_y = mask_h / slide_h

    contour_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    contour_mask_scaled = contour.copy().astype(np.float64)
    contour_mask_scaled[:, 0, 0] *= scale_x
    contour_mask_scaled[:, 0, 1] *= scale_y
    contour_mask_scaled = np.round(contour_mask_scaled).astype(np.int32)
    cv2.drawContours(contour_mask, [contour_mask_scaled], -1, 1, thickness=-1)

    if contour_holes:
        holes_scaled = []
        for hole in contour_holes:
            hole_scaled = hole.copy().astype(np.float64)
            hole_scaled[:, 0, 0] *= scale_x
            hole_scaled[:, 0, 1] *= scale_y
            holes_scaled.append(np.round(hole_scaled).astype(np.int32))
        cv2.drawContours(contour_mask, holes_scaled, -1, 0, thickness=-1)

    return contour_mask * (tissue_mask > 0).astype(np.uint8)


def _compute_tissue_fractions(
    candidates: np.ndarray,
    tissue_mask: np.ndarray,
    tile_size_lv0: int,
    slide_dimensions: tuple[int, int],
    use_padding: bool,
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

    if use_padding:
        valid_area = np.maximum(1, (x2 - x1) * (y2 - y1))
        fractions = (tissue_sum / valid_area).astype(np.float32)
    else:
        fractions = (tissue_sum / tile_area).astype(np.float32)
    return np.clip(fractions, 0.0, 1.0)
