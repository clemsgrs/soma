"""Combined tissue mask overlay + tiling grid visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from soma.preprocessing.tiling import TilingResult


def render_preview(
    slide_thumbnail: np.ndarray,
    tissue_mask: np.ndarray,
    tiling_result: TilingResult | None = None,
    *,
    mask_alpha: float = 0.3,
    mask_color: tuple[int, int, int] = (0, 200, 0),
    grid_color: tuple[int, int, int] = (0, 0, 0),
    grid_thickness: int = 1,
) -> np.ndarray:
    """Combined preview: tissue mask overlay + tiling grid.

    Args:
        slide_thumbnail: RGB image (H, W, 3), uint8.
        tissue_mask: Binary mask (H, W), uint8, 255=tissue.
        tiling_result: If provided, draw tile grid on the preview.
        mask_alpha: Opacity of the mask overlay (0=invisible, 1=opaque).
        mask_color: RGB color for the tissue mask overlay.
        grid_color: RGB color for tile grid lines.
        grid_thickness: Thickness of grid lines in pixels.

    Returns:
        RGB preview image (H, W, 3), uint8.
    """
    preview = slide_thumbnail.copy()
    thumb_h, thumb_w = preview.shape[:2]

    # Resize mask to match thumbnail if needed
    if tissue_mask.shape[:2] != (thumb_h, thumb_w):
        mask_resized = cv2.resize(
            tissue_mask, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_resized = tissue_mask

    # Tissue mask overlay
    overlay = preview.copy()
    overlay[mask_resized > 0] = mask_color
    cv2.addWeighted(overlay, mask_alpha, preview, 1 - mask_alpha, 0, preview)

    # Tiling grid
    if tiling_result is not None and len(tiling_result.coordinates) > 0:
        _draw_tile_grid(
            preview,
            tiling_result=tiling_result,
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            grid_color=grid_color,
            grid_thickness=grid_thickness,
        )

    return preview


def _draw_tile_grid(
    preview: np.ndarray,
    tiling_result: TilingResult,
    thumb_w: int,
    thumb_h: int,
    grid_color: tuple[int, int, int],
    grid_thickness: int,
) -> None:
    """Draw tile rectangles on the preview image."""
    # We need the slide dimensions to compute scale
    # Use tile_size_lv0 and coordinates (in level-0 space)
    # Estimate slide dimensions from coordinate range + tile_size_lv0
    coords = tiling_result.coordinates
    tile_lv0 = tiling_result.tile_size_lv0

    # Estimate slide dimensions (max coord + tile_size)
    slide_w_est = int(coords[:, 0].max()) + tile_lv0
    slide_h_est = int(coords[:, 1].max()) + tile_lv0

    scale_x = thumb_w / max(slide_w_est, 1)
    scale_y = thumb_h / max(slide_h_est, 1)

    for x_lv0, y_lv0 in coords:
        x1 = int(x_lv0 * scale_x)
        y1 = int(y_lv0 * scale_y)
        x2 = int((x_lv0 + tile_lv0) * scale_x)
        y2 = int((y_lv0 + tile_lv0) * scale_y)
        cv2.rectangle(preview, (x1, y1), (x2, y2), grid_color, grid_thickness)


def save_preview(preview: np.ndarray, path: Path) -> None:
    """Save a preview image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGB to BGR for cv2
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
