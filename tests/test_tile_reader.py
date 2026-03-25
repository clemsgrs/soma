"""Tests for soma.encoders.tile_reader — super-tile grouping + batch reading."""

from __future__ import annotations

import numpy as np
import pytest

from soma.encoders.tile_reader import SuperTile, SuperTileIndex, build_supertile_index
from soma.preprocessing.tiling import TilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_tiling_result(
    nx: int, ny: int, tile_size_lv0: int = 512
) -> TilingResult:
    """Create a perfect nx × ny grid of tile coordinates."""
    coords = []
    for y_idx in range(ny):
        for x_idx in range(nx):
            coords.append([x_idx * tile_size_lv0, y_idx * tile_size_lv0])
    return TilingResult(
        coordinates=np.array(coords, dtype=np.int64),
        tissue_fractions=np.ones(len(coords), dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=tile_size_lv0,
        is_within_tolerance=True,
    )


# ---------------------------------------------------------------------------
# build_supertile_index — grouping algorithm
# ---------------------------------------------------------------------------


class TestBuildSupertileIndex:
    def test_perfect_8x8_grid(self):
        """An 8×8 grid should produce exactly one 8×8 super-tile."""
        tiling = _make_grid_tiling_result(8, 8)
        index = build_supertile_index(tiling)
        assert len(index.supertiles) == 1
        assert index.supertiles[0].block_size == 8

    def test_perfect_4x4_grid(self):
        """A 4×4 grid should produce one 4×4 super-tile."""
        tiling = _make_grid_tiling_result(4, 4)
        index = build_supertile_index(tiling)
        assert len(index.supertiles) == 1
        assert index.supertiles[0].block_size == 4

    def test_perfect_2x2_grid(self):
        """A 2×2 grid should produce one 2×2 super-tile."""
        tiling = _make_grid_tiling_result(2, 2)
        index = build_supertile_index(tiling)
        assert len(index.supertiles) == 1
        assert index.supertiles[0].block_size == 2

    def test_single_tile_becomes_singleton(self):
        """One tile → one singleton super-tile."""
        tiling = _make_grid_tiling_result(1, 1)
        index = build_supertile_index(tiling)
        assert len(index.supertiles) == 1
        assert index.supertiles[0].block_size == 1

    def test_9x9_grid_decomposition(self):
        """A 9×9 grid (81 tiles) should produce 1×8x8 + remaining tiles."""
        tiling = _make_grid_tiling_result(9, 9)
        index = build_supertile_index(tiling)
        # All 81 tiles must be covered
        assert len(index.ordered_indices) == 81
        assert set(index.ordered_indices.tolist()) == set(range(81))

    def test_all_tiles_covered(self):
        """Every tile index appears exactly once in ordered_indices."""
        tiling = _make_grid_tiling_result(10, 10)
        index = build_supertile_index(tiling)
        assert len(index.ordered_indices) == 100
        assert sorted(index.ordered_indices.tolist()) == list(range(100))

    def test_tile_to_st_mapping(self):
        """tile_to_st maps every tile to a valid super-tile id."""
        tiling = _make_grid_tiling_result(4, 4)
        index = build_supertile_index(tiling)
        assert index.tile_to_st.shape == (16,)
        assert all(0 <= st_id < len(index.supertiles) for st_id in index.tile_to_st)

    def test_crop_offsets_within_supertile(self):
        """Crop offsets should be non-negative and within the read region."""
        tiling = _make_grid_tiling_result(4, 4, tile_size_lv0=512)
        index = build_supertile_index(tiling)
        for tile_idx in range(16):
            st_id = index.tile_to_st[tile_idx]
            st = index.supertiles[st_id]
            assert index.tile_crop_x[tile_idx] >= 0
            assert index.tile_crop_y[tile_idx] >= 0
            assert index.tile_crop_x[tile_idx] + tiling.tile_size_lv0 <= st.read_size_lv0
            assert index.tile_crop_y[tile_idx] + tiling.tile_size_lv0 <= st.read_size_lv0

    def test_supertile_coordinates_in_level0(self):
        """Super-tile (x_lv0, y_lv0) should match the top-left tile's coordinates."""
        tiling = _make_grid_tiling_result(4, 4, tile_size_lv0=512)
        index = build_supertile_index(tiling)
        st = index.supertiles[0]
        # For a perfect 4x4 grid starting at (0,0):
        assert st.x_lv0 == 0
        assert st.y_lv0 == 0

    def test_read_size_formula(self):
        """read_size_lv0 = tile_size_lv0 + (block_size - 1) * step_lv0."""
        tile_size_lv0 = 512
        tiling = _make_grid_tiling_result(4, 4, tile_size_lv0=tile_size_lv0)
        index = build_supertile_index(tiling)
        st = index.supertiles[0]
        expected = tile_size_lv0 + (st.block_size - 1) * tile_size_lv0
        assert st.read_size_lv0 == expected

    def test_custom_supertile_sizes(self):
        """Can restrict to smaller super-tile sizes."""
        tiling = _make_grid_tiling_result(8, 8)
        index = build_supertile_index(tiling, supertile_sizes=(4, 2))
        # No 8×8 blocks, so should decompose into 4×4 blocks
        for st in index.supertiles:
            assert st.block_size <= 4

    def test_irregular_grid(self):
        """Non-rectangular tile set still groups what it can."""
        # L-shaped: 4×4 minus top-right 2×2
        coords = []
        for y in range(4):
            for x in range(4):
                if x >= 2 and y < 2:
                    continue
                coords.append([x * 512, y * 512])
        tiling = TilingResult(
            coordinates=np.array(coords, dtype=np.int64),
            tissue_fractions=np.ones(len(coords), dtype=np.float32),
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
            read_level=1,
            effective_tile_size_px=256,
            effective_spacing_um=0.5,
            tile_size_lv0=512,
            is_within_tolerance=True,
        )
        index = build_supertile_index(tiling)
        # All 12 tiles covered
        assert len(index.ordered_indices) == 12
        assert sorted(index.ordered_indices.tolist()) == list(range(12))


# ---------------------------------------------------------------------------
# SuperTile dataclass
# ---------------------------------------------------------------------------


class TestSuperTile:
    def test_frozen(self):
        st = SuperTile(x_lv0=0, y_lv0=0, read_size_lv0=512, block_size=1)
        with pytest.raises(AttributeError):
            st.x_lv0 = 100  # type: ignore[misc]
