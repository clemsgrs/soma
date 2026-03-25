"""Tests for soma.preprocessing.io — atomic save/load of tiling artifacts."""

import json

import numpy as np
import pytest

from soma.preprocessing.io import load_tiling_result, save_tiling_result
from soma.preprocessing.tiling import TilingResult


def _make_tiling_result(n_tiles: int = 10) -> TilingResult:
    """Minimal TilingResult for testing I/O."""
    rng = np.random.RandomState(42)
    return TilingResult(
        coordinates=rng.randint(0, 10000, size=(n_tiles, 2)).astype(np.int64),
        tissue_fractions=rng.uniform(0.5, 1.0, size=n_tiles).astype(np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
        tissue_mask=np.zeros((100, 100), dtype=np.uint8),
    )


# --- Save produces expected files ---


def test_save_creates_npz_and_meta(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    assert paths["npz"].exists()
    assert paths["meta"].exists()
    assert paths["npz"].name == "slide_001.coordinates.npz"
    assert paths["meta"].name == "slide_001.coordinates.meta.json"


def test_save_npz_contains_arrays(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    data = np.load(paths["npz"])
    assert "coordinates" in data
    assert "tissue_fractions" in data
    np.testing.assert_array_equal(data["coordinates"], result.coordinates)
    np.testing.assert_array_almost_equal(data["tissue_fractions"], result.tissue_fractions)


def test_save_meta_contains_fields(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    meta = json.loads(paths["meta"].read_text())
    assert meta["requested_tile_size_px"] == 256
    assert meta["requested_spacing_um"] == 0.5
    assert meta["read_level"] == 1
    assert meta["effective_tile_size_px"] == 256
    assert meta["effective_spacing_um"] == 0.5
    assert meta["tile_size_lv0"] == 512
    assert meta["is_within_tolerance"] is True
    assert meta["n_tiles"] == 10


# --- Roundtrip ---


def test_roundtrip(tmp_path):
    original = _make_tiling_result()
    paths = save_tiling_result(original, tmp_path, "slide_001")
    loaded = load_tiling_result(paths["npz"], paths["meta"])

    np.testing.assert_array_equal(loaded.coordinates, original.coordinates)
    np.testing.assert_array_almost_equal(loaded.tissue_fractions, original.tissue_fractions)
    assert loaded.requested_tile_size_px == original.requested_tile_size_px
    assert loaded.requested_spacing_um == original.requested_spacing_um
    assert loaded.read_level == original.read_level
    assert loaded.effective_tile_size_px == original.effective_tile_size_px
    assert loaded.effective_spacing_um == original.effective_spacing_um
    assert loaded.tile_size_lv0 == original.tile_size_lv0
    assert loaded.is_within_tolerance == original.is_within_tolerance


def test_roundtrip_empty(tmp_path):
    """Empty TilingResult should roundtrip correctly."""
    result = TilingResult(
        coordinates=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=256,
        effective_spacing_um=0.25,
        tile_size_lv0=512,
        is_within_tolerance=False,
    )
    paths = save_tiling_result(result, tmp_path, "empty_slide")
    loaded = load_tiling_result(paths["npz"], paths["meta"])
    assert len(loaded.coordinates) == 0
    assert len(loaded.tissue_fractions) == 0


# --- Atomicity ---


def test_save_does_not_leave_tmp_files(tmp_path):
    """After save, no .tmp files should remain."""
    result = _make_tiling_result()
    save_tiling_result(result, tmp_path, "slide_001")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
