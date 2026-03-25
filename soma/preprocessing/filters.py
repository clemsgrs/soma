"""Tile-level QC filters.

Combines best ideas from:
- slideflow: whitespace/grayspace filtering (designed for downsampled level), Laplacian blur
- hs2p: black/white pixel fraction checks

These filters operate on individual tile images (numpy arrays).
For maximum efficiency, call them on downsampled tile reads first
(slideflow's trick: filter at a coarser pyramid level to reject bad tiles
before reading at full resolution).
"""

from __future__ import annotations

import cv2
import numpy as np


def filter_whitespace(
    tile: np.ndarray,
    *,
    threshold: int = 230,
    max_fraction: float = 0.8,
) -> bool:
    """Check if a tile has acceptable whitespace levels.

    Args:
        tile: RGB image (H, W, 3), uint8.
        threshold: RGB value above which a pixel is considered white.
        max_fraction: Maximum allowed fraction of white pixels.

    Returns:
        True if the tile passes (acceptable whitespace), False if rejected.
    """
    # A pixel is "white" if its mean RGB value exceeds the threshold
    mean_rgb = np.mean(tile, axis=-1)
    white_fraction = np.mean(mean_rgb > threshold)
    return bool(white_fraction <= max_fraction)


def filter_grayspace(
    tile: np.ndarray,
    *,
    saturation_threshold: float = 0.05,
    max_fraction: float = 0.6,
) -> bool:
    """Check if a tile has acceptable grayspace levels.

    Gray pixels have very low saturation in HSV space. Tissue (H&E stained)
    typically has higher saturation due to eosin/hematoxylin color.

    Args:
        tile: RGB image (H, W, 3), uint8.
        saturation_threshold: HSV saturation below which a pixel is "gray" (0-1 scale).
        max_fraction: Maximum allowed fraction of gray pixels.

    Returns:
        True if the tile passes, False if rejected.
    """
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    # OpenCV HSV saturation is 0-255, normalize to 0-1
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    gray_fraction = np.mean(saturation < saturation_threshold)
    return bool(gray_fraction <= max_fraction)


def detect_blur(tile: np.ndarray) -> float:
    """Compute a blur score for a tile using Laplacian variance.

    Higher scores indicate sharper images. Typical thresholds:
    - < 50: likely blurry / out of focus
    - > 100: likely sharp

    Inspired by slideflow's Gaussian QC filter (Laplacian edge detection).

    Args:
        tile: RGB image (H, W, 3), uint8.

    Returns:
        Laplacian variance (float). Higher = sharper.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())
