"""Combined tissue mask overlay + tiling grid visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hs2p.preprocessing import TilingResult
from hs2p.wsi.geometry import select_level_for_downsample
from hs2p.wsi.reader import SlideReader


def render_preview(
    slide: SlideReader,
    tissue_mask: np.ndarray,
    tiling_result: TilingResult,
    *,
    preview_downsample: int = 32,
    mask_alpha: float = 0.3,
    mask_color: tuple[int, int, int] = (0, 200, 0),
    grid_color: tuple[int, int, int] = (0, 0, 0),
    grid_thickness: int = 1,
) -> np.ndarray:
    """Combined preview: tissue mask overlay + tiling grid.

    Args:
        slide: Whole-slide reader used to render the true visualization level.
        tissue_mask: Binary mask (H, W), uint8, 255=tissue.
        tiling_result: Tiling metadata used for segmentation geometry and grid drawing.
        preview_downsample: Requested downsample for visualization rendering.
        mask_alpha: Opacity of the mask overlay (0=invisible, 1=opaque).
        mask_color: RGB color for the tissue mask overlay.
        grid_color: RGB color for tile grid lines.
        grid_thickness: Thickness of grid lines in pixels.

        Returns:
        RGB preview image (H, W, 3), uint8.
    """
    vis_level = select_level_for_downsample(
        preview_downsample, slide.level_downsamples
    )
    vis_size = slide.level_dimensions[vis_level]
    preview = slide.read_region((0, 0), vis_level, vis_size).copy()
    vis_h, vis_w = preview.shape[:2]

    mask_level = int(tiling_result.seg_level)
    mask_size = slide.level_dimensions[mask_level]
    if tissue_mask.shape[:2] != (mask_size[1], mask_size[0]):
        msg = (
            "tissue_mask shape does not match tiling_result.seg_level "
            f"dimensions: got {tissue_mask.shape[:2]}, expected {(mask_size[1], mask_size[0])}"
        )
        raise ValueError(msg)

    if tissue_mask.shape[:2] != (vis_h, vis_w):
        mask_resized = cv2.resize(
            tissue_mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_resized = tissue_mask

    # Tissue mask overlay
    overlay = preview.copy()
    overlay[mask_resized > 0] = mask_color
    cv2.addWeighted(overlay, mask_alpha, preview, 1 - mask_alpha, 0, preview)

    # Tiling grid
    if tiling_result.num_tiles > 0:
        _draw_tile_grid(
            preview,
            tiling_result=tiling_result,
            preview_width=vis_w,
            preview_height=vis_h,
            slide_dimensions=slide.dimensions,
            grid_color=grid_color,
            grid_thickness=grid_thickness,
        )

    return preview


def _draw_tile_grid(
    preview: np.ndarray,
    tiling_result: TilingResult,
    preview_width: int,
    preview_height: int,
    slide_dimensions: tuple[int, int],
    grid_color: tuple[int, int, int],
    grid_thickness: int,
) -> None:
    """Draw tile rectangles on the preview image."""
    tile_lv0 = tiling_result.tile_size_lv0
    slide_w, slide_h = slide_dimensions
    scale_x = preview_width / max(slide_w, 1)
    scale_y = preview_height / max(slide_h, 1)

    for x_lv0, y_lv0 in zip(tiling_result.x, tiling_result.y):
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
