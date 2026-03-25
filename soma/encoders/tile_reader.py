"""Super-tile grouping and batch tile reading.

Groups adjacent tiles into NxN blocks (8×8, 4×4, 2×2) so that one large
``read_region`` call replaces many small ones. Ported from hs2p/slide2vec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soma.preprocessing.tiling import TilingResult


@dataclass(frozen=True)
class SuperTile:
    """A rectangular block of adjacent tiles read as one region."""

    x_lv0: int  # top-left x in level-0 pixels
    y_lv0: int  # top-left y in level-0 pixels
    read_size_lv0: int  # pixel size of the read region (square) at level 0
    block_size: int  # NxN grid dimension (8, 4, 2, or 1)


@dataclass(frozen=True)
class SuperTileIndex:
    """Result of grouping tiles into super-tiles.

    Attributes:
        supertiles: List of SuperTile objects.
        tile_to_st: (num_tiles,) array mapping tile_idx → super-tile id.
        tile_crop_x: (num_tiles,) crop x-offset within the super-tile read region (lv0 px).
        tile_crop_y: (num_tiles,) crop y-offset within the super-tile read region (lv0 px).
        ordered_indices: (num_tiles,) tile indices reordered so tiles in the
            same super-tile are contiguous. Use ``np.argsort(ordered_indices)``
            to map features back to original coordinate order.
    """

    supertiles: list[SuperTile]
    tile_to_st: np.ndarray
    tile_crop_x: np.ndarray
    tile_crop_y: np.ndarray
    ordered_indices: np.ndarray


def build_supertile_index(
    tiling_result: TilingResult,
    supertile_sizes: tuple[int, ...] = (8, 4, 2),
) -> SuperTileIndex:
    """Group adjacent tiles into super-tile blocks.

    Greedy algorithm: for each block size (8 → 4 → 2), try to form NxN
    grids of adjacent tiles. Remaining tiles become singletons (block_size=1).

    Args:
        tiling_result: Tiling output with coordinates in level-0 space.
        supertile_sizes: Block sizes to attempt, in descending order.

    Returns:
        SuperTileIndex with grouping, crop offsets, and reordered indices.
    """
    coords = tiling_result.coordinates
    num_tiles = len(coords)
    tile_size_lv0 = tiling_result.tile_size_lv0

    # Step between adjacent tiles in level-0 pixels.
    # For non-overlapping grids this equals tile_size_lv0.
    step_lv0 = _infer_step(coords, tile_size_lv0)

    # Build coordinate → tile index map for O(1) lookup
    coord_to_idx: dict[tuple[int, int], int] = {
        (int(coords[i, 0]), int(coords[i, 1])): i for i in range(num_tiles)
    }

    consumed = np.zeros(num_tiles, dtype=bool)
    supertiles: list[SuperTile] = []
    # Per-tile arrays
    tile_to_st = np.empty(num_tiles, dtype=np.int64)
    tile_crop_x = np.empty(num_tiles, dtype=np.int64)
    tile_crop_y = np.empty(num_tiles, dtype=np.int64)
    ordered_indices: list[int] = []

    for block_size in supertile_sizes:
        if num_tiles < block_size * block_size:
            continue
        for idx in range(num_tiles):
            if consumed[idx]:
                continue
            grouped = _try_build_block(
                idx, block_size, coords, coord_to_idx, consumed, step_lv0
            )
            if grouped is None:
                continue
            # Record the super-tile
            st_id = len(supertiles)
            x0, y0 = int(coords[idx, 0]), int(coords[idx, 1])
            read_size = tile_size_lv0 + (block_size - 1) * step_lv0
            supertiles.append(
                SuperTile(
                    x_lv0=x0,
                    y_lv0=y0,
                    read_size_lv0=read_size,
                    block_size=block_size,
                )
            )
            # Assign tile metadata
            for gy in range(block_size):
                for gx in range(block_size):
                    tile_idx = grouped[gy * block_size + gx]
                    consumed[tile_idx] = True
                    tile_to_st[tile_idx] = st_id
                    tile_crop_x[tile_idx] = gx * step_lv0
                    tile_crop_y[tile_idx] = gy * step_lv0
                    ordered_indices.append(tile_idx)

    # Remaining tiles become singletons
    for idx in range(num_tiles):
        if consumed[idx]:
            continue
        consumed[idx] = True
        st_id = len(supertiles)
        supertiles.append(
            SuperTile(
                x_lv0=int(coords[idx, 0]),
                y_lv0=int(coords[idx, 1]),
                read_size_lv0=tile_size_lv0,
                block_size=1,
            )
        )
        tile_to_st[idx] = st_id
        tile_crop_x[idx] = 0
        tile_crop_y[idx] = 0
        ordered_indices.append(idx)

    return SuperTileIndex(
        supertiles=supertiles,
        tile_to_st=tile_to_st,
        tile_crop_x=tile_crop_x,
        tile_crop_y=tile_crop_y,
        ordered_indices=np.array(ordered_indices, dtype=np.int64),
    )


def _try_build_block(
    start_idx: int,
    block_size: int,
    coords: np.ndarray,
    coord_to_idx: dict[tuple[int, int], int],
    consumed: np.ndarray,
    step_lv0: int,
) -> list[int] | None:
    """Try to form a block_size × block_size grid starting at start_idx.

    Returns list of tile indices (row-major) or None if the block can't be formed.
    """
    x0 = int(coords[start_idx, 0])
    y0 = int(coords[start_idx, 1])
    indices: list[int] = []

    for gy in range(block_size):
        for gx in range(block_size):
            coord = (x0 + gx * step_lv0, y0 + gy * step_lv0)
            tile_idx = coord_to_idx.get(coord)
            if tile_idx is None or consumed[tile_idx]:
                return None
            indices.append(tile_idx)

    return indices


def _infer_step(coords: np.ndarray, tile_size_lv0: int) -> int:
    """Infer the step (stride) between adjacent tiles in level-0 pixels.

    Falls back to tile_size_lv0 (no overlap) if fewer than 2 unique values.
    """
    if len(coords) < 2:
        return tile_size_lv0

    unique_x = np.unique(coords[:, 0])
    if len(unique_x) >= 2:
        diffs = np.diff(unique_x)
        return int(diffs.min())

    unique_y = np.unique(coords[:, 1])
    if len(unique_y) >= 2:
        diffs = np.diff(unique_y)
        return int(diffs.min())

    return tile_size_lv0
