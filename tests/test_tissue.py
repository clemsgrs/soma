"""Tests for soma.preprocessing.tissue — tissue segmentation."""

import numpy as np
import pytest

from soma.preprocessing.tissue import segment_tissue


def _make_he_thumbnail(width: int = 200, height: int = 150) -> np.ndarray:
    """Synthetic thumbnail mimicking H&E-stained tissue on white background.

    Tissue region: pinkish (R=200, G=130, B=170) — falls in HSV tissue range.
    Background: near-white (240, 240, 240) — low saturation, high value.
    """
    img = np.full((height, width, 3), 240, dtype=np.uint8)  # white background
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    img[y0:y1, x0:x1] = [200, 130, 170]  # pinkish tissue
    return img


# --- HSV method (default) ---


def test_hsv_default_method():
    """HSV should be the default method."""
    thumbnail = _make_he_thumbnail()
    mask = segment_tissue(thumbnail)  # no method= argument
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_hsv_returns_binary_mask():
    thumbnail = _make_he_thumbnail()
    mask = segment_tissue(thumbnail, method="hsv")
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_hsv_shape_matches_input():
    thumbnail = _make_he_thumbnail(width=200, height=150)
    mask = segment_tissue(thumbnail, method="hsv")
    assert mask.shape == (150, 200)


def test_hsv_detects_tissue_region():
    thumbnail = _make_he_thumbnail()
    mask = segment_tissue(thumbnail, method="hsv")
    h, w = thumbnail.shape[:2]

    # Center (tissue) should be mostly detected
    center = mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    assert np.mean(center) > 200, "Center should be mostly tissue"

    # Corners (background) should be empty
    corner = mask[: h // 6, : w // 6]
    assert np.mean(corner) < 50, "Corners should be mostly background"


def test_hsv_rejects_white_background():
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = segment_tissue(white, method="hsv")
    assert np.mean(mask > 0) < 0.05


# --- Otsu method ---


def test_otsu_returns_binary_mask():
    thumbnail = _make_he_thumbnail()
    mask = segment_tissue(thumbnail, method="otsu")
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_otsu_detects_tissue_region():
    thumbnail = _make_he_thumbnail()
    mask = segment_tissue(thumbnail, method="otsu")
    h, w = thumbnail.shape[:2]

    center = mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    assert np.mean(center) > 200, "Center should be mostly tissue"


def test_otsu_rejects_white_background():
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = segment_tissue(white, method="otsu")
    assert np.mean(mask > 0) < 0.1


# --- Error handling ---


def test_unknown_method_raises():
    thumbnail = _make_he_thumbnail()
    with pytest.raises(ValueError, match="method"):
        segment_tissue(thumbnail, method="nonexistent")
