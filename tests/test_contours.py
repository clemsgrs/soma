"""Tests for soma.preprocessing.tissue — contour detection + area filtering."""

import numpy as np
import pytest

from soma.preprocessing.tissue import ContourResult, detect_contours


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


def _make_mask_with_hole(
    mask_h: int = 100,
    mask_w: int = 100,
) -> np.ndarray:
    """Binary mask with a large blob containing a hole."""
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    # Outer tissue region
    mask[10:90, 10:90] = 255
    # Inner hole
    mask[30:60, 30:60] = 0
    return mask


# --- Basic functionality ---


def test_returns_contour_result():
    mask = _make_mask_with_blob()
    result = detect_contours(mask, slide_dimensions=(1000, 1000))
    assert isinstance(result, ContourResult)


def test_detects_single_contour():
    mask = _make_mask_with_blob()
    result = detect_contours(mask, slide_dimensions=(1000, 1000))
    assert len(result.contours) == 1


def test_contour_result_has_mask():
    mask = _make_mask_with_blob()
    result = detect_contours(mask, slide_dimensions=(1000, 1000))
    np.testing.assert_array_equal(result.mask, mask)


# --- Coordinate scaling ---


def test_contours_scaled_to_level0():
    """Contour coords should be scaled from mask space to level-0 space."""
    # 100x100 mask for a 1000x1000 slide → scale factor = 10
    mask = _make_mask_with_blob(mask_h=100, mask_w=100)
    result = detect_contours(mask, slide_dimensions=(1000, 1000))

    contour = result.contours[0]
    # The blob spans x=[20,80), y=[20,80) in mask space
    # In level-0 space: x=[200,800), y=[200,800)
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    assert xs.min() >= 190  # allow slight contour approximation
    assert xs.max() <= 810
    assert ys.min() >= 190
    assert ys.max() <= 810


# --- Hole detection ---


def test_detects_hole_in_contour():
    mask = _make_mask_with_hole()
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        a_h=0,  # don't filter holes by area
    )
    assert len(result.contours) == 1
    assert len(result.holes) == 1
    assert len(result.holes[0]) >= 1  # at least one hole for the first contour


def test_holes_scaled_to_level0():
    mask = _make_mask_with_hole()
    result = detect_contours(
        mask,
        slide_dimensions=(1000, 1000),
        a_h=0,
    )
    hole = result.holes[0][0]
    xs = hole[:, 0, 0]
    ys = hole[:, 0, 1]
    # Hole spans x=[30,60), y=[30,60) in mask space → [300,600) in level-0
    assert xs.min() >= 290
    assert xs.max() <= 610
    assert ys.min() >= 290
    assert ys.max() <= 610


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
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_t=4,  # min area = 4 * (16 * scale)^2 in mask pixels
    )
    # Only the large blob should survive
    assert len(result.contours) == 1


def test_small_hole_filtered_out():
    """Holes smaller than a_h * ref_area should be removed."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:190, 10:190] = 255
    # Large hole (80x80 = 6400 px)
    mask[40:120, 40:120] = 0
    # Small hole (5x5 = 25 px)
    mask[150:155, 150:155] = 0

    # Use a_h high enough to filter the small hole but keep the large one
    # ref_area ≈ 2.56, so a_h=20 → min_hole_area ≈ 51.2
    # Small hole area ~25 < 51.2 → filtered
    # Large hole area ~6400 > 51.2 → kept
    result = detect_contours(
        mask,
        slide_dimensions=(2000, 2000),
        ref_tile_size_px=16,
        requested_spacing_um=0.5,
        a_h=20,
    )
    assert len(result.contours) == 1
    # Only the large hole should survive
    assert len(result.holes[0]) == 1


def test_max_holes_per_contour():
    """Should keep at most max_holes_per_contour holes, largest first."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[5:195, 5:195] = 255
    # Create 3 holes of different sizes
    mask[20:50, 20:50] = 0    # 30x30 = 900 px (largest)
    mask[60:80, 60:80] = 0    # 20x20 = 400 px (medium)
    mask[100:110, 100:110] = 0  # 10x10 = 100 px (smallest)

    result = detect_contours(
        mask,
        slide_dimensions=(2000, 2000),
        a_h=0,  # don't filter by area
        max_holes_per_contour=2,
    )
    assert len(result.contours) == 1
    assert len(result.holes[0]) <= 2


# --- Edge cases ---


def test_empty_mask_returns_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = detect_contours(mask, slide_dimensions=(1000, 1000))
    assert len(result.contours) == 0
    assert len(result.holes) == 0


def test_multiple_contours():
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Two separate blobs
    mask[20:80, 10:80] = 255
    mask[20:80, 120:190] = 255

    result = detect_contours(
        mask,
        slide_dimensions=(2000, 1000),
        a_t=0,  # don't filter by area
    )
    assert len(result.contours) == 2
    assert len(result.holes) == 2


def test_holes_list_matches_contours():
    """holes list should have same length as contours list."""
    mask = _make_mask_with_blob()
    result = detect_contours(mask, slide_dimensions=(1000, 1000))
    assert len(result.holes) == len(result.contours)
