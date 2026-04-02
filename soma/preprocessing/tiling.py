"""Thin SOMA wrappers over hs2p's shared tile-generation core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hs2p.preprocessing import (
    ContourResult,
    TilingResult as _CoreTilingResult,
    canonicalize_tiling_result as _canonicalize_core_tiling_result,
    generate_tiles as _generate_core_tiles,
    resolve_base_spacing_um as _resolve_core_base_spacing_um,
)


@dataclass(frozen=True)
class TilingResult:
    """Result of tile coordinate generation."""

    coordinates: np.ndarray
    tissue_fractions: np.ndarray

    requested_tile_size_px: int
    requested_spacing_um: float

    read_level: int
    effective_tile_size_px: int
    effective_spacing_um: float
    tile_size_lv0: int
    is_within_tolerance: bool
    use_padding: bool = True

    tile_index: np.ndarray | None = None
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
    config_hash: str | None = None

    hierarchical: bool = False
    npatch: int | None = None
    region_index: np.ndarray | None = None
    region_coordinates: np.ndarray | None = None
    requested_region_size_px: int | None = None

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
        "slide_dimensions": None
        if result.slide_dimensions is None
        else list(result.slide_dimensions),
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
        "config_hash": result.config_hash,
    }


def _to_core_tiling_result(result: TilingResult) -> _CoreTilingResult:
    return _CoreTilingResult(
        coordinates=result.coordinates,
        tissue_fractions=result.tissue_fractions,
        requested_tile_size_px=result.requested_tile_size_px,
        requested_spacing_um=result.requested_spacing_um,
        read_level=result.read_level,
        effective_tile_size_px=result.effective_tile_size_px,
        effective_spacing_um=result.effective_spacing_um,
        tile_size_lv0=result.tile_size_lv0,
        is_within_tolerance=result.is_within_tolerance,
        use_padding=result.use_padding,
        tile_index=result.tile_index,
        tissue_mask=result.tissue_mask,
        sample_id=result.sample_id,
        image_path=result.image_path,
        backend=result.backend,
        requested_backend=result.requested_backend,
        base_spacing_um=result.base_spacing_um,
        slide_dimensions=result.slide_dimensions,
        level_downsamples=result.level_downsamples,
        overlap=result.overlap,
        min_tissue_fraction=result.min_tissue_fraction,
        step_px_lv0=result.step_px_lv0,
        tissue_method=result.tissue_method,
        seg_downsample=result.seg_downsample,
        seg_level=result.seg_level,
        seg_spacing_um=result.seg_spacing_um,
        ref_tile_size_px=result.ref_tile_size_px,
        a_t=result.a_t,
        tissue_mask_path=result.tissue_mask_path,
        tissue_mask_tissue_value=result.tissue_mask_tissue_value,
        mask_level=result.mask_level,
        mask_spacing_um=result.mask_spacing_um,
        config_hash=result.config_hash,
    )


def _from_core_tiling_result(result: _CoreTilingResult) -> TilingResult:
    return TilingResult(
        coordinates=result.coordinates,
        tissue_fractions=result.tissue_fractions,
        requested_tile_size_px=result.requested_tile_size_px,
        requested_spacing_um=result.requested_spacing_um,
        read_level=result.read_level,
        effective_tile_size_px=result.effective_tile_size_px,
        effective_spacing_um=result.effective_spacing_um,
        tile_size_lv0=result.tile_size_lv0,
        is_within_tolerance=result.is_within_tolerance,
        use_padding=result.use_padding,
        tile_index=result.tile_index,
        tissue_mask=result.tissue_mask,
        sample_id=result.sample_id,
        image_path=result.image_path,
        backend=result.backend,
        requested_backend=result.requested_backend,
        base_spacing_um=result.base_spacing_um,
        slide_dimensions=None
        if result.slide_dimensions is None
        else list(result.slide_dimensions),
        level_downsamples=None
        if result.level_downsamples is None
        else list(result.level_downsamples),
        overlap=result.overlap,
        min_tissue_fraction=result.min_tissue_fraction,
        step_px_lv0=result.step_px_lv0,
        tissue_method=result.tissue_method,
        seg_downsample=result.seg_downsample,
        seg_level=result.seg_level,
        seg_spacing_um=result.seg_spacing_um,
        ref_tile_size_px=result.ref_tile_size_px,
        a_t=result.a_t,
        tissue_mask_path=result.tissue_mask_path,
        tissue_mask_tissue_value=result.tissue_mask_tissue_value,
        mask_level=result.mask_level,
        mask_spacing_um=result.mask_spacing_um,
        config_hash=result.config_hash,
    )


def canonicalize_tiling_result(result: TilingResult) -> TilingResult:
    return _from_core_tiling_result(
        _canonicalize_core_tiling_result(_to_core_tiling_result(result))
    )


def resolve_base_spacing_um(result: TilingResult) -> float:
    return _resolve_core_base_spacing_um(_to_core_tiling_result(result))


def expand_regions_to_subtiles(
    tiling_result: TilingResult, npatch: int
) -> TilingResult:
    """Expand region-level coordinates into subtile coordinates for hierarchical MIL."""
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

    rows, cols = np.divmod(np.arange(p), npatch)
    offsets_x = cols * subtile_size_lv0
    offsets_y = rows * subtile_size_lv0

    region_x = tiling_result.coordinates[:, 0:1]
    region_y = tiling_result.coordinates[:, 1:2]
    subtile_x = region_x + offsets_x[np.newaxis, :]
    subtile_y = region_y + offsets_y[np.newaxis, :]

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
    core_result = _generate_core_tiles(
        slide_dimensions=slide_dimensions,
        contours=contours,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        base_spacing_um=base_spacing_um,
        level_downsamples=level_downsamples,
        overlap=overlap,
        use_padding=use_padding,
        min_tissue_fraction=min_tissue_fraction,
        tolerance=tolerance,
        num_workers=num_workers,
    )
    return _from_core_tiling_result(core_result)


__all__ = [
    "ContourResult",
    "TilingResult",
    "canonicalize_tiling_result",
    "expand_regions_to_subtiles",
    "generate_tiles",
    "resolve_base_spacing_um",
]
