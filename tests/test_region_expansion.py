"""Tests for expand_regions_to_subtiles()."""

from __future__ import annotations

import numpy as np
import pytest

from soma.preprocessing.tiling import TilingResult, expand_regions_to_subtiles


def _make_tiling(
    coordinates: np.ndarray,
    tissue_fractions: np.ndarray | None = None,
    tile_size_lv0: int = 1024,
    requested_tile_size_px: int = 4096,
    effective_tile_size_px: int = 4096,
    base_spacing_um: float | None = None,
) -> TilingResult:
    """Helper to build a minimal TilingResult for testing."""
    n = len(coordinates)
    if tissue_fractions is None:
        tissue_fractions = np.ones(n, dtype=np.float32)
    return TilingResult(
        coordinates=np.asarray(coordinates, dtype=np.int64),
        tissue_fractions=tissue_fractions,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=effective_tile_size_px,
        effective_spacing_um=0.5,
        tile_size_lv0=tile_size_lv0,
        is_within_tolerance=True,
        base_spacing_um=base_spacing_um,
    )


class TestExpandRegionsToSubtiles:
    def test_basic_expansion(self):
        """2 regions, npatch=2 → 8 subtile coords at exact expected positions."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0], [1024, 0]], dtype=np.int64),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=2)

        # subtile_size_lv0 = 1024 // 2 = 512
        # Region (0, 0): row-major → (0,0), (512,0), (0,512), (512,512)
        # Region (1024, 0): → (1024,0), (1536,0), (1024,512), (1536,512)
        expected = np.array(
            [
                [0, 0], [512, 0], [0, 512], [512, 512],
                [1024, 0], [1536, 0], [1024, 512], [1536, 512],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result.coordinates, expected)
        assert len(result.coordinates) == 8

    def test_row_major_order(self):
        """Subtiles are in row-major order matching VisionTransformer4K's flatten(2,3)."""
        tiling = _make_tiling(
            coordinates=np.array([[100, 200]], dtype=np.int64),
            tile_size_lv0=900,
            requested_tile_size_px=900,
            effective_tile_size_px=900,
        )
        result = expand_regions_to_subtiles(tiling, npatch=3)

        # subtile_size = 900 // 3 = 300
        # Row-major: row i, col j → (x + j*300, y + i*300)
        sub = 300
        expected = np.array(
            [
                [100 + 0 * sub, 200 + 0 * sub],  # (0,0)
                [100 + 1 * sub, 200 + 0 * sub],  # (0,1)
                [100 + 2 * sub, 200 + 0 * sub],  # (0,2)
                [100 + 0 * sub, 200 + 1 * sub],  # (1,0)
                [100 + 1 * sub, 200 + 1 * sub],  # (1,1)
                [100 + 2 * sub, 200 + 1 * sub],  # (1,2)
                [100 + 0 * sub, 200 + 2 * sub],  # (2,0)
                [100 + 1 * sub, 200 + 2 * sub],  # (2,1)
                [100 + 2 * sub, 200 + 2 * sub],  # (2,2)
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result.coordinates, expected)

    def test_tissue_fractions_repeated(self):
        """Parent fraction copied to all subtiles."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0], [1024, 0]], dtype=np.int64),
            tissue_fractions=np.array([0.8, 0.3], dtype=np.float32),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=2)

        expected = np.array(
            [0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result.tissue_fractions, expected)

    def test_tile_size_fields_updated(self):
        """Size fields reflect subtile dimensions after expansion."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=1024,
            requested_tile_size_px=4096,
            effective_tile_size_px=4096,
        )
        result = expand_regions_to_subtiles(tiling, npatch=4)

        assert result.tile_size_lv0 == 256  # 1024 // 4
        assert result.effective_tile_size_px == 1024  # 4096 // 4
        assert result.requested_tile_size_px == 1024  # 4096 // 4

    def test_hierarchical_flag_set(self):
        """Returned TilingResult has hierarchical=True and correct npatch."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=4)

        assert result.hierarchical is True
        assert result.npatch == 4

    def test_region_index(self):
        """region_index maps each subtile to its parent region."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0], [1024, 0], [2048, 0]], dtype=np.int64),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=2)

        # 3 regions × 4 subtiles each = 12 subtiles
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(result.region_index, expected)

    def test_region_coordinates_preserved(self):
        """region_coordinates stores original (M, 2) region coordinates."""
        coords = np.array([[0, 0], [1024, 0]], dtype=np.int64)
        tiling = _make_tiling(coordinates=coords, tile_size_lv0=1024)
        result = expand_regions_to_subtiles(tiling, npatch=2)

        np.testing.assert_array_equal(result.region_coordinates, coords)

    def test_requested_region_size_preserved(self):
        """requested_region_size_px stores original region size."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=1024,
            requested_tile_size_px=4096,
        )
        result = expand_regions_to_subtiles(tiling, npatch=4)

        assert result.requested_region_size_px == 4096

    def test_empty_input(self):
        """0 regions → 0 subtiles."""
        tiling = _make_tiling(
            coordinates=np.empty((0, 2), dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=2)

        assert len(result.coordinates) == 0
        assert result.hierarchical is True
        assert result.region_coordinates is not None
        assert len(result.region_coordinates) == 0

    def test_npatch_not_dividing_raises(self):
        """Raise ValueError when tile_size_lv0 is not divisible by npatch."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=1000,  # Not divisible by 3
        )
        with pytest.raises(ValueError, match="divisible"):
            expand_regions_to_subtiles(tiling, npatch=3)

    def test_tile_index_is_identity(self):
        """Expanded tile_index is a contiguous identity [0, 1, ..., N-1]."""
        tiling = _make_tiling(
            coordinates=np.array([[0, 0], [1024, 0]], dtype=np.int64),
            tile_size_lv0=1024,
        )
        result = expand_regions_to_subtiles(tiling, npatch=2)

        expected = np.arange(8, dtype=np.int32)
        np.testing.assert_array_equal(result.tile_index, expected)

    def test_metadata_is_preserved(self):
        tiling = _make_tiling(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            base_spacing_um=0.25,
        )

        result = expand_regions_to_subtiles(tiling, npatch=2)

        assert result.base_spacing_um == 0.25
