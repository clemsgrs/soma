"""Tests for soma.preprocessing.tiling — tile coordinate generation."""

import numpy as np
import pytest

from soma.preprocessing.tissue import ContourResult, detect_contours
from soma.preprocessing.tiling import TilingResult, generate_tiles


# --- Helpers ---


def _make_contour_result(
    mask_h: int = 100,
    mask_w: int = 100,
    slide_w: int = 1000,
    slide_h: int = 1000,
) -> ContourResult:
    """ContourResult with a large tissue blob in the center."""
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    return detect_contours(mask, slide_dimensions=(slide_w, slide_h), a_t=0, a_h=0)


def _make_full_tissue_contour(
    mask_h: int = 100,
    mask_w: int = 100,
    slide_w: int = 1000,
    slide_h: int = 1000,
) -> ContourResult:
    """ContourResult where the entire slide is tissue."""
    mask = np.full((mask_h, mask_w), 255, dtype=np.uint8)
    return detect_contours(mask, slide_dimensions=(slide_w, slide_h), a_t=0, a_h=0)


# Standard pyramid: 0.25 µm/px base, 4 levels
BASE_SPACING = 0.25
DOWNSAMPLES = [1.0, 2.0, 4.0, 16.0]


# --- Basic functionality ---


def test_returns_tiling_result():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert isinstance(result, TilingResult)


def test_coordinates_shape():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.coordinates.ndim == 2
    assert result.coordinates.shape[1] == 2


def test_coordinates_are_level0():
    contours = _make_contour_result(slide_w=2000, slide_h=2000)
    result = generate_tiles(
        slide_dimensions=(2000, 2000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    if len(result.coordinates) > 0:
        assert result.coordinates[:, 0].max() < 2000
        assert result.coordinates[:, 1].max() < 2000


# --- Requested vs effective fields ---


def test_requested_fields_stored():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.requested_tile_size_px == 256
    assert result.requested_spacing_um == 0.5


def test_effective_fields_from_level_selection():
    """Effective spacing should come from select_level()."""
    contours = _make_contour_result()
    # base=0.25, downsamples=[1,2,4,16] → level 1 has 0.5 µm/px (exact match)
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.read_level == 1
    assert result.effective_spacing_um == pytest.approx(0.5)
    assert result.is_within_tolerance is True


def test_effective_tile_size_when_exact():
    """When effective == requested spacing, effective_tile_size == requested."""
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.effective_tile_size_px == 256


def test_effective_tile_size_when_not_exact():
    """When effective != requested, effective_tile_size is adjusted."""
    contours = _make_contour_result()
    # Request 0.75 µm/px → level 1 (0.5) is best → effective_tile_size = round(256 * 0.75/0.5) = 384
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.75,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.read_level == 1
    assert result.effective_spacing_um == pytest.approx(0.5)
    assert result.effective_tile_size_px == 384  # round(256 * 0.75 / 0.5)
    assert result.is_within_tolerance is False


def test_tile_size_lv0():
    """tile_size_lv0 should be the tile footprint in level-0 pixels."""
    contours = _make_contour_result()
    # requested=256 at 0.5 µm/px, base=0.25 → tile_size_lv0 = 256 * 0.5/0.25 = 512
    result = generate_tiles(
        slide_dimensions=(2000, 2000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.tile_size_lv0 == 512


# --- Tissue fraction filtering ---


def test_tissue_fractions_computed():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.tissue_fractions is not None
    assert len(result.tissue_fractions) == len(result.coordinates)


def test_tissue_fractions_above_threshold():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        min_tissue_fraction=0.5,
    )
    if len(result.tissue_fractions) > 0:
        assert np.all(result.tissue_fractions >= 0.5)


def test_strict_threshold_fewer_tiles():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :50] = 255  # tissue only in top-left quadrant
    contours = detect_contours(mask, slide_dimensions=(1000, 1000), a_t=0, a_h=0)

    result_strict = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=128,
        requested_spacing_um=0.25,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        min_tissue_fraction=0.9,
    )
    result_loose = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=128,
        requested_spacing_um=0.25,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        min_tissue_fraction=0.1,
    )
    assert len(result_strict.coordinates) <= len(result_loose.coordinates)


# --- Overlap ---


def test_overlap_produces_more_tiles():
    contours = _make_full_tissue_contour()
    result_no = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.25,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        overlap=0.0,
    )
    result_yes = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.25,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        overlap=0.5,
    )
    assert len(result_yes.coordinates) > len(result_no.coordinates)


def test_no_duplicate_coordinates():
    contours = _make_full_tissue_contour()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.25,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    if len(result.coordinates) > 1:
        unique = np.unique(result.coordinates, axis=0)
        assert len(unique) == len(result.coordinates)


# --- Edge cases ---


def test_empty_contours():
    mask = np.zeros((100, 100), dtype=np.uint8)
    contours = ContourResult(contours=[], holes=[], mask=mask)
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert len(result.coordinates) == 0
    assert len(result.tissue_fractions) == 0


def test_tile_larger_than_slide():
    contours = _make_full_tissue_contour(slide_w=100, slide_h=100)
    # tile_size_lv0 = 256 * 0.5/0.25 = 512 > slide 100x100
    result = generate_tiles(
        slide_dimensions=(100, 100),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert len(result.coordinates) == 0


# --- Parallel processing ---


def test_num_workers_produces_same_result():
    """Multi-worker should produce the same tiles as single-worker."""
    contours = _make_contour_result(slide_w=2000, slide_h=2000)
    kwargs = dict(
        slide_dimensions=(2000, 2000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    result_1 = generate_tiles(**kwargs, num_workers=1)
    result_2 = generate_tiles(**kwargs, num_workers=2)

    # Sort for deterministic comparison
    idx_1 = np.lexsort((result_1.coordinates[:, 1], result_1.coordinates[:, 0]))
    idx_2 = np.lexsort((result_2.coordinates[:, 1], result_2.coordinates[:, 0]))
    np.testing.assert_array_equal(result_1.coordinates[idx_1], result_2.coordinates[idx_2])


# --- Tissue mask stored ---


def test_tissue_mask_stored():
    contours = _make_contour_result()
    result = generate_tiles(
        slide_dimensions=(1000, 1000),
        contours=contours,
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert result.tissue_mask is not None
