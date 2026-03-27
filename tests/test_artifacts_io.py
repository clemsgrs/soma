"""Tests for soma.preprocessing.io — atomic save/load of tiling artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest

from soma.preprocessing.io import (
    load_tiling_result,
    save_tiling_result,
    validate_tiling_result_provenance,
)
from soma.preprocessing.tiling import TilingResult, canonicalize_tiling_result


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
        sample_id="slide_001",
        image_path="/tmp/slide_001.svs",
        backend="openslide",
        requested_backend="auto",
        base_spacing_um=0.25,
        slide_dimensions=[10000, 8000],
        level_downsamples=[1.0, 2.0, 4.0],
        overlap=0.25,
        min_tissue_fraction=0.5,
        step_px_lv0=384,
        tissue_method="precomputed_mask",
        seg_downsample=64,
        seg_level=2,
        seg_spacing_um=1.0,
        ref_tile_size_px=256,
        a_t=4,
        tissue_mask_path="/tmp/slide_001-mask.tif",
        tissue_mask_tissue_value=1,
        mask_level=1,
        mask_spacing_um=0.5,
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
    assert "tile_index" in data
    assert "coordinates" in data
    assert "tissue_fractions" in data
    np.testing.assert_array_equal(data["tile_index"], np.arange(len(result.coordinates), dtype=np.int32))
    np.testing.assert_array_equal(data["coordinates"], canonicalize_tiling_result(result).coordinates)
    np.testing.assert_array_almost_equal(
        data["tissue_fractions"], canonicalize_tiling_result(result).tissue_fractions
    )


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
    assert meta["use_padding"] is True
    assert meta["n_tiles"] == 10
    assert meta["sample_id"] == "slide_001"
    assert meta["image_path"] == "/tmp/slide_001.svs"
    assert meta["backend"] == "openslide"
    assert meta["requested_backend"] == "auto"
    assert meta["base_spacing_um"] == 0.25
    assert meta["slide_dimensions"] == [10000, 8000]
    assert meta["level_downsamples"] == [1.0, 2.0, 4.0]
    assert meta["overlap"] == 0.25
    assert meta["min_tissue_fraction"] == 0.5
    assert meta["step_px_lv0"] == 384
    assert meta["tissue_method"] == "precomputed_mask"
    assert meta["seg_downsample"] == 64
    assert meta["seg_level"] == 2
    assert meta["seg_spacing_um"] == 1.0
    assert meta["ref_tile_size_px"] == 256
    assert meta["a_t"] == 4
    assert meta["tissue_mask_path"] == "/tmp/slide_001-mask.tif"
    assert meta["tissue_mask_tissue_value"] == 1
    assert meta["mask_level"] == 1
    assert meta["mask_spacing_um"] == 0.5
    assert meta["coordinate_space"] == "level0_px"
    assert meta["tile_order"] == "x_then_y"
    assert meta["provenance"]["sample_id"] == "slide_001"
    assert meta["provenance"]["tissue_mask_path"] == "/tmp/slide_001-mask.tif"
    assert meta["slide"]["dimensions"] == [10000, 8000]
    assert meta["slide"]["seg_level"] == 2
    assert meta["slide"]["seg_spacing_um"] == 1.0
    assert meta["tiling"]["step_px_lv0"] == 384
    assert meta["segmentation"]["tissue_method"] == "precomputed_mask"
    assert meta["segmentation"]["seg_level"] == 2
    assert meta["segmentation"]["seg_spacing_um"] == 1.0
    assert meta["segmentation"]["tissue_mask_tissue_value"] == 1
    assert meta["segmentation"]["mask_level"] == 1
    assert meta["segmentation"]["mask_spacing_um"] == 0.5
    assert meta["artifact"]["coordinate_space"] == "level0_px"
    assert meta["tiling"]["use_padding"] is True


# --- Roundtrip ---


def test_roundtrip(tmp_path):
    original = _make_tiling_result()
    paths = save_tiling_result(original, tmp_path, "slide_001")
    loaded = load_tiling_result(paths["npz"], paths["meta"])
    canonical = canonicalize_tiling_result(original)

    np.testing.assert_array_equal(loaded.tile_index, canonical.tile_index)
    np.testing.assert_array_equal(loaded.coordinates, canonical.coordinates)
    np.testing.assert_array_almost_equal(loaded.tissue_fractions, canonical.tissue_fractions)
    assert loaded.requested_tile_size_px == canonical.requested_tile_size_px
    assert loaded.requested_spacing_um == canonical.requested_spacing_um
    assert loaded.read_level == canonical.read_level
    assert loaded.effective_tile_size_px == canonical.effective_tile_size_px
    assert loaded.effective_spacing_um == canonical.effective_spacing_um
    assert loaded.tile_size_lv0 == canonical.tile_size_lv0
    assert loaded.is_within_tolerance == canonical.is_within_tolerance
    assert loaded.use_padding == canonical.use_padding
    assert loaded.base_spacing_um == 0.25
    assert loaded.slide_dimensions == [10000, 8000]
    assert loaded.level_downsamples == [1.0, 2.0, 4.0]
    assert loaded.seg_level == 2
    assert loaded.seg_spacing_um == 1.0
    assert loaded.tissue_mask_path == "/tmp/slide_001-mask.tif"
    assert loaded.tissue_mask_tissue_value == 1
    assert loaded.mask_level == 1
    assert loaded.mask_spacing_um == 0.5


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
    assert len(loaded.tile_index) == 0


def test_load_rejects_additional_metadata_fields(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    meta = json.loads(paths["meta"].read_text())
    meta["extra_field"] = {"kept": True}
    meta["tiling"]["future_flag"] = "ok"
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="unexpected keys"):
        load_tiling_result(paths["npz"], paths["meta"])

def test_load_requires_all_metadata_fields(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    meta = json.loads(paths["meta"].read_text())
    del meta["base_spacing_um"]
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="missing keys"):
        load_tiling_result(paths["npz"], paths["meta"])


def test_save_canonicalizes_coordinates_and_tissue_fractions(tmp_path):
    result = TilingResult(
        coordinates=np.array([[10, 5], [1, 9], [10, 5], [2, 2]], dtype=np.int64),
        tissue_fractions=np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
    )

    paths = save_tiling_result(result, tmp_path, "slide_001")
    data = np.load(paths["npz"])

    np.testing.assert_array_equal(
        data["coordinates"],
        np.array([[1, 9], [2, 2], [10, 5]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        data["tissue_fractions"],
        np.array([0.4, 0.2, 0.1], dtype=np.float32),
    )
    np.testing.assert_array_equal(data["tile_index"], np.array([0, 1, 2], dtype=np.int32))

def test_load_requires_tile_index(tmp_path):
    npz_path = tmp_path / "legacy.coordinates.npz"
    meta_path = tmp_path / "legacy.coordinates.meta.json"
    result = _make_tiling_result(n_tiles=2)
    np.savez_compressed(
        npz_path,
        coordinates=np.array([[1, 2], [3, 4]], dtype=np.int64),
        tissue_fractions=np.array([0.5, 0.75], dtype=np.float32),
    )
    meta = json.loads(
        save_tiling_result(result, tmp_path, "reference")["meta"].read_text()
    )
    meta["n_tiles"] = 2
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="missing tile_index"):
        load_tiling_result(npz_path, meta_path)


def test_validate_tiling_result_provenance_rejects_mask_path_mismatch(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    loaded = load_tiling_result(paths["npz"], paths["meta"])

    with pytest.raises(ValueError, match="tissue_mask_path mismatch"):
        validate_tiling_result_provenance(
            loaded,
            sample_id="slide_001",
            image_path=Path("/tmp/slide_001.svs"),
            tissue_mask_path=Path("/tmp/other-mask.tif"),
            tissue_mask_tissue_value=1,
        )


def test_validate_tiling_result_provenance_rejects_mask_value_mismatch(tmp_path):
    result = _make_tiling_result()
    paths = save_tiling_result(result, tmp_path, "slide_001")
    loaded = load_tiling_result(paths["npz"], paths["meta"])

    with pytest.raises(ValueError, match="tissue_mask_tissue_value mismatch"):
        validate_tiling_result_provenance(
            loaded,
            sample_id="slide_001",
            image_path=Path("/tmp/slide_001.svs"),
            tissue_mask_path=Path("/tmp/slide_001-mask.tif"),
            tissue_mask_tissue_value=2,
        )


# --- Atomicity ---


def test_save_does_not_leave_tmp_files(tmp_path):
    """After save, no .tmp files should remain."""
    result = _make_tiling_result()
    save_tiling_result(result, tmp_path, "slide_001")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
