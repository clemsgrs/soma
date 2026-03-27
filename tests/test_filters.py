"""Tests for soma.preprocessing.filters — tile QC filters.

Combines best ideas from:
- slideflow: whitespace/grayspace filtering at downsampled level, Laplacian blur
- hs2p: black/white pixel fraction filtering
"""

import numpy as np
import pytest

from soma.preprocessing.filters import (
    detect_blur,
    filter_grayspace,
    filter_whitespace,
)


# --- Whitespace filtering ---


def test_whitespace_filter_rejects_white_tile():
    white = np.full((256, 256, 3), 245, dtype=np.uint8)
    assert filter_whitespace(white, threshold=230, max_fraction=0.8) is False


def test_whitespace_filter_accepts_tissue_tile():
    tissue = np.full((256, 256, 3), 180, dtype=np.uint8)
    assert filter_whitespace(tissue, threshold=230, max_fraction=0.8) is True


def test_whitespace_filter_mixed_tile():
    tile = np.full((256, 256, 3), 180, dtype=np.uint8)
    # Make top half white
    tile[:128, :, :] = 250
    # 50% white — should pass with max_fraction=0.8
    assert filter_whitespace(tile, threshold=230, max_fraction=0.8) is True
    # But fail with strict threshold
    assert filter_whitespace(tile, threshold=230, max_fraction=0.3) is False


def test_whitespace_filter_ignores_masked_padding():
    tile = np.full((256, 256, 3), 180, dtype=np.uint8)
    tile[:128, :, :] = 250
    valid_mask = np.zeros((256, 256), dtype=bool)
    valid_mask[128:, :] = True
    assert (
        filter_whitespace(
            tile,
            threshold=230,
            max_fraction=0.1,
            valid_mask=valid_mask,
        )
        is True
    )


# --- Grayspace filtering ---


def test_grayspace_filter_rejects_gray_tile():
    # Gray tile: low saturation in HSV
    gray = np.full((256, 256, 3), 150, dtype=np.uint8)
    assert filter_grayspace(gray, saturation_threshold=0.05, max_fraction=0.6) is False


def test_grayspace_filter_accepts_colored_tile():
    # Create a colored (pinkish) tile — typical H&E stain
    tile = np.zeros((256, 256, 3), dtype=np.uint8)
    tile[:, :, 0] = 200  # R
    tile[:, :, 1] = 130  # G
    tile[:, :, 2] = 170  # B
    assert filter_grayspace(tile, saturation_threshold=0.05, max_fraction=0.6) is True


def test_grayspace_filter_ignores_masked_padding():
    tile = np.zeros((256, 256, 3), dtype=np.uint8)
    tile[:, :, 0] = 200
    tile[:, :, 1] = 130
    tile[:, :, 2] = 170
    tile[:128, :, :] = 150
    valid_mask = np.zeros((256, 256), dtype=bool)
    valid_mask[128:, :] = True
    assert (
        filter_grayspace(
            tile,
            saturation_threshold=0.05,
            max_fraction=0.1,
            valid_mask=valid_mask,
        )
        is True
    )


# --- Blur detection ---


def test_blur_detects_blurry_region():
    # Uniform region = blurry (no edges)
    uniform = np.full((256, 256, 3), 150, dtype=np.uint8)
    # Add tiny noise to avoid degenerate case
    rng = np.random.RandomState(0)
    uniform = uniform + rng.randint(-2, 3, uniform.shape, dtype=np.int16)
    uniform = np.clip(uniform, 0, 255).astype(np.uint8)
    score = detect_blur(uniform)
    assert score < 50  # Low Laplacian variance = blurry


def test_blur_detects_sharp_region():
    # Image with strong edges
    rng = np.random.RandomState(42)
    sharp = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    score = detect_blur(sharp)
    assert score > 100  # High Laplacian variance = sharp


def test_blur_score_monotonic():
    """Progressively blurring an image should decrease the score."""
    import cv2

    rng = np.random.RandomState(42)
    base = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    scores = []
    for ksize in [1, 5, 15, 31]:
        if ksize == 1:
            blurred = base
        else:
            blurred = cv2.GaussianBlur(base, (ksize, ksize), 0)
        scores.append(detect_blur(blurred))

    # Scores should generally decrease with more blur
    assert scores[0] > scores[-1]
