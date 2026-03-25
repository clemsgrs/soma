"""Tissue segmentation from WSI thumbnails.

Default method is HSV thresholding (following hs2p), which works better than
Otsu for H&E-stained pathology images by targeting the hue/saturation/value
ranges characteristic of tissue.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def segment_tissue(
    thumbnail: np.ndarray,
    *,
    method: str = "hsv",
    kernel_size: int = 5,
    morph_iterations: int = 2,
) -> np.ndarray:
    """Segment tissue from a thumbnail image.

    Args:
        thumbnail: RGB image of shape (H, W, 3), dtype uint8.
        method: Segmentation method.
            - "hsv" (default): HSV color-space thresholding, robust for H&E slides.
            - "otsu": Otsu's method on grayscale (median blur on saturation channel).
        kernel_size: Size of the morphological kernel for cleanup.
        morph_iterations: Number of morphological close iterations.

    Returns:
        Binary mask of shape (H, W), dtype uint8, where 255 = tissue, 0 = background.
    """
    if method == "hsv":
        return _segment_hsv(
            thumbnail,
            kernel_size=kernel_size,
            morph_iterations=morph_iterations,
        )
    elif method == "otsu":
        return _segment_otsu(
            thumbnail,
            kernel_size=kernel_size,
            morph_iterations=morph_iterations,
        )
    else:
        msg = f"Unknown tissue segmentation method: '{method}'. Available: hsv, otsu"
        raise ValueError(msg)


def _segment_hsv(
    thumbnail: np.ndarray,
    *,
    kernel_size: int,
    morph_iterations: int,
    hsv_lower: tuple[int, int, int] = (90, 8, 103),
    hsv_upper: tuple[int, int, int] = (180, 255, 255),
) -> np.ndarray:
    """HSV thresholding for tissue segmentation.

    Targets the hue/saturation/value ranges characteristic of H&E-stained tissue.
    This is more robust than Otsu for pathology images because it directly selects
    for tissue color rather than relying on intensity contrast alone.

    Default thresholds from hs2p: lower=(90, 8, 103), upper=(180, 255, 255).
    These capture the purple-pink hues of hematoxylin/eosin while rejecting
    white background and common artifacts.
    """
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # Morphological closing to fill small holes in tissue regions
    if kernel_size > 0 and morph_iterations > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
        )

    return mask


def _segment_otsu(
    thumbnail: np.ndarray,
    *,
    kernel_size: int,
    morph_iterations: int,
    median_blur_size: int = 7,
) -> np.ndarray:
    """Otsu's method on the saturation channel (following hs2p/slideflow pattern).

    Converts to HSV, applies median blur on the saturation channel,
    then uses Otsu's automatic thresholding.
    """
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.medianBlur(saturation, median_blur_size)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing
    if kernel_size > 0 and morph_iterations > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
        )

    return mask


# --- Contour detection ---


@dataclass(frozen=True)
class ContourResult:
    """Result of contour detection on a tissue mask.

    All contour coordinates are in level-0 pixel space.
    """

    contours: list[np.ndarray]
    mask: np.ndarray


def detect_contours(
    tissue_mask: np.ndarray,
    *,
    slide_dimensions: tuple[int, int],
    ref_tile_size_px: int = 16,
    requested_spacing_um: float = 0.5,
    a_t: int = 4,
) -> ContourResult:
    """Detect tissue contours from a binary mask.

    Uses cv2.findContours with RETR_CCOMP to identify foreground contours.
    Contours are filtered by area relative to a reference tile size
    (following hs2p).

    Args:
        tissue_mask: Binary mask (H, W), dtype uint8, 255=tissue.
        slide_dimensions: (width, height) at level 0.
        ref_tile_size_px: Reference tile size in pixels for area filtering.
        requested_spacing_um: Requested spacing for ref_area computation.
        a_t: Minimum foreground contour area as multiple of ref_area.

    Returns:
        ContourResult with contours in level-0 coordinates.
    """
    if tissue_mask.max() == 0:
        return ContourResult(contours=[], mask=tissue_mask)

    # Scale factors: mask space → level-0 space
    mask_h, mask_w = tissue_mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = slide_w / mask_w
    scale_y = slide_h / mask_h

    # Reference area in mask pixels for filtering
    # ref_tile_size_px is in requested-spacing pixels; convert to mask pixels
    mask_spacing_um_x = (slide_w * 0.5) / mask_w  # approximate
    ref_tile_mask_px = ref_tile_size_px * (requested_spacing_um / max(mask_spacing_um_x, 1e-9))
    ref_area = ref_tile_mask_px * ref_tile_mask_px

    # Find contours with two-level hierarchy (RETR_CCOMP)
    raw_contours, hierarchy = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None or len(raw_contours) == 0:
        return ContourResult(contours=[], mask=tissue_mask)

    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, child, parent]

    # Separate foreground contours (parent == -1)
    foreground_indices = []

    for i, h in enumerate(hierarchy):
        if h[3] == -1:  # no parent → foreground contour
            foreground_indices.append(i)

    # Filter foreground contours by area
    min_fg_area = a_t * ref_area
    filtered_contours = []

    for fg_idx in foreground_indices:
        area = cv2.contourArea(raw_contours[fg_idx])
        if area < min_fg_area:
            continue

        # Scale contour to level-0
        contour_lv0 = raw_contours[fg_idx].copy().astype(np.float64)
        contour_lv0[:, 0, 0] *= scale_x
        contour_lv0[:, 0, 1] *= scale_y
        contour_lv0 = contour_lv0.astype(np.int32)
        filtered_contours.append(contour_lv0)

    return ContourResult(
        contours=filtered_contours,
        mask=tissue_mask,
    )
