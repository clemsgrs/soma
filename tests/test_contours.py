"""Tests for soma.preprocessing.tissue — contour detection + area filtering."""

import numpy as np

from soma.preprocessing.tissue import ContourResult, detect_contours


BASE_SPACING = 0.25
DOWNSAMPLES = [1.0, 2.0, 4.0, 16.0]


def _make_mask_with_blob(
    mask_h: int = 100,
    mask_w: int = 100,
    y0: int = 20,
    y1: int = 80,
    x0: int = 20,
    x1: int = 80,
) -> np.ndarray:
    """Binary mask with a single rectangular tissue blob."""
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


# --- Basic functionality ---


def test_returns_contour_result():
    mask = _make_mask_with_blob()
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert isinstance(result, ContourResult)


def test_detects_single_contour():
    mask = _make_mask_with_blob()
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert len(result.contours) == 1


def test_contour_result_has_mask():
    mask = _make_mask_with_blob()
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    np.testing.assert_array_equal(result.mask, mask)


def test_empty_mask_returns_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )
    assert len(result.contours) == 0


# --- Coordinate scaling ---


def test_contours_scaled_to_level0():
    """Contour coords should be scaled from mask space to level-0 space."""
    # 100x100 mask for a 1000x1000 slide → scale factor = 10
    mask = _make_mask_with_blob(mask_h=100, mask_w=100)
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
    )

    contour = result.contours[0]
    # The blob spans x=[20,80), y=[20,80) in mask space
    # In level-0 space: x=[200,800), y=[200,800)
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    assert xs.min() >= 190  # allow slight contour approximation
    assert xs.max() <= 810
    assert ys.min() >= 190
    assert ys.max() <= 810


# --- Area filtering ---


def test_small_contour_filtered_out():
    """Contours smaller than a_t * ref_area should be removed."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large blob
    mask[10:90, 10:90] = 255
    # Tiny blob (4x4 = 16 pixels)
    mask[2:6, 2:6] = 255

    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_t=4,  # min area = 4 * (16 * scale)^2 in mask pixels
    )
    # Only the large blob should survive
    assert len(result.contours) == 1


def test_multiple_contours():
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Two separate blobs
    mask[20:80, 10:80] = 255
    mask[20:80, 120:190] = 255

    result = detect_contours(
        mask,
        slide_dimensions=(2000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        a_t=0,  # don't filter by area
    )
    assert len(result.contours) == 2


def test_area_threshold_scales_with_selected_level_geometry():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 20:30] = 255  # 10x10 blob -> area 100 mask px

    fine = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_t=4,
    )
    coarse = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        ref_tile_size_px=16,
        requested_spacing_um=1.0,
        a_t=4,
    )

    assert len(fine.contours) == 1
    assert len(coarse.contours) == 0


def test_hole_area_is_subtracted_before_applying_a_t():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[15:85, 15:85] = 0

    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        base_spacing_um=BASE_SPACING,
        level_downsamples=DOWNSAMPLES,
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_t=200,
    )

    assert len(result.contours) == 0
