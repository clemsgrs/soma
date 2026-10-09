"""Point-annotation target encoding for detection (design §4, §5).

Two pure, weight-free steps, isolated so they unit-test against hand-computed
values without an encoder or a torch head:

1. :func:`transform_points_to_target` — map annotation coordinates from the
   **level-0** frame they are persisted in (the pathology convention: ASAP / QuPath
   / hs2p author points in base full-resolution pixels, invariant to the experiment)
   into the run's **target_size** frame, where the heatmap and predictions live.
   The map is ``x_t = x_l0 * (source_spacing_um / effective_spacing_um) - crop_left`` (``y``
   analogously with ``crop_top``); identity for OCELOT-as-shipped (read at native
   spacing, no crop).

2. :func:`render_peak_heatmap` — render the target-frame points as **peak** Gaussians
   (peak value 1, overlaps merged by element-wise max) into a ``(C, H, W)`` map, one
   channel per object class, background implicit. This is a keypoint heatmap
   (CenterNet-style), *not* a count-preserving density map: the F1@δ metric needs
   discrete peaks recovered cleanly, and max-merge keeps adjacent cells separable.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "transform_points_to_target",
    "transform_points_to_level0",
    "render_peak_heatmap",
    "ignore_mask_to_valid",
    "points_on_valid",
]

# The ignore-mask value for "not annotated" (segmentation's ``ignore_index``); 0 = supervised.
IGNORE_VALUE = 255


def _validate_spacings(source_spacing_um: float, effective_spacing_um: float) -> None:
    values = (source_spacing_um, effective_spacing_um)
    if any(
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise ValueError(
            "spacings must be positive and finite, got "
            f"source_spacing_um={source_spacing_um}, "
            f"effective_spacing_um={effective_spacing_um}."
        )


def transform_points_to_target(
    points_xy: np.ndarray,
    *,
    source_spacing_um: float,
    effective_spacing_um: float,
    crop_top: int = 0,
    crop_left: int = 0,
) -> np.ndarray:
    """Map ``(N, 2)`` ``(x, y)`` points from the level-0 frame to the target frame.

    ``x`` indexes columns (width), ``y`` indexes rows (height), matching the
    ``crop_box = (top, left, height, width)`` convention. ``points_xy`` is returned
    as float (sub-pixel positions are kept; the renderer rounds to the nearest
    pixel). An empty ``(0, 2)`` array passes through unchanged.
    """
    _validate_spacings(source_spacing_um, effective_spacing_um)
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must be (N, 2), got shape {pts.shape}.")
    scale = float(source_spacing_um) / float(effective_spacing_um)
    out = pts * scale
    out[:, 0] -= float(crop_left)
    out[:, 1] -= float(crop_top)
    return out


def transform_points_to_level0(
    points_xy: np.ndarray,
    *,
    source_spacing_um: float,
    effective_spacing_um: float,
    crop_top: int = 0,
    crop_left: int = 0,
) -> np.ndarray:
    """Inverse of :func:`transform_points_to_target` — target frame -> level-0 px.

    ``x_l0 = (x_t + crop_left) * effective_spacing_um / source_spacing_um`` (``y`` analogously).
    Used to persist predicted points in the level-0 frame (stitch-ready, design §4).
    """
    _validate_spacings(source_spacing_um, effective_spacing_um)
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must be (N, 2), got shape {pts.shape}.")
    scale = float(effective_spacing_um) / float(source_spacing_um)
    out = pts.copy()
    out[:, 0] = (out[:, 0] + float(crop_left)) * scale
    out[:, 1] = (out[:, 1] + float(crop_top)) * scale
    return out


def render_peak_heatmap(
    points_xy: np.ndarray,
    classes: np.ndarray,
    *,
    target_size: tuple[int, int],
    num_classes: int,
    sigma: float,
    truncate: float = 3.0,
) -> Tensor:
    """Render target-frame points as per-class peak Gaussians ``(C, H, W)``.

    Args:
        points_xy: ``(N, 2)`` ``(x, y)`` point centres in the target frame.
        classes: ``(N,)`` integer class id per point, each in ``[0, num_classes)``.
        target_size: ``(H, W)`` of the output map (the supervision frame).
        num_classes: Number of object classes ``C`` (one channel each; no background
            channel — background is the absence of a peak).
        sigma: Gaussian standard deviation in target-frame pixels (``> 0``).
        truncate: Render each Gaussian within ``truncate * sigma`` of its centre.

    Points whose centre rounds outside ``[0, W) x [0, H)`` are dropped (cropped out
    of this tile); a Gaussian near the border is clipped to the canvas. Overlapping
    Gaussians of the same class are merged by element-wise max (peak stays 1). An
    empty point set yields an all-zero map.
    """
    height, width = int(target_size[0]), int(target_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    if float(sigma) <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    heatmap = torch.zeros(int(num_classes), height, width, dtype=torch.float32)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    cls = np.asarray(classes).reshape(-1).astype(np.int64)
    if pts.shape[0] != cls.shape[0]:
        raise ValueError(
            f"points_xy has {pts.shape[0]} rows but classes has {cls.shape[0]} entries."
        )
    if pts.shape[0] == 0:
        return heatmap

    invalid = sorted({int(c) for c in cls if not 0 <= int(c) < int(num_classes)})
    if invalid:
        raise ValueError(
            f"point class id(s) {invalid} outside [0, num_classes={num_classes})."
        )

    sigma = float(sigma)
    radius = int(math.ceil(float(truncate) * sigma))
    two_sigma_sq = 2.0 * sigma * sigma
    for (x, y), c in zip(pts, cls):
        cx, cy = int(round(float(x))), int(round(float(y)))
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        x0, x1 = max(0, cx - radius), min(width - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(height - 1, cy + radius)
        ys = torch.arange(y0, y1 + 1, dtype=torch.float32).unsqueeze(1)
        xs = torch.arange(x0, x1 + 1, dtype=torch.float32).unsqueeze(0)
        patch = torch.exp(-(((xs - x) ** 2) + ((ys - y) ** 2)) / two_sigma_sq)
        region = heatmap[c, y0 : y1 + 1, x0 : x1 + 1]
        torch.maximum(region, patch, out=region)
    return heatmap


def ignore_mask_to_valid(
    ignore_mask: np.ndarray,
    *,
    source_spacing_um: float,
    effective_spacing_um: float,
    target_size: tuple[int, int],
    crop_top: int = 0,
    crop_left: int = 0,
) -> np.ndarray:
    """Map a level-0 ignore mask to a boolean ``(H, W)`` valid map in the target frame.

    ``ignore_mask`` holds :data:`IGNORE_VALUE` where nothing is annotated and 0 elsewhere.
    Target pixel ``(u, v)`` takes the nearest level-0 pixel under the inverse of
    :func:`transform_points_to_target`, so a point and the pixel it renders on agree.
    Target pixels that fall outside the mask are not valid.
    """
    _validate_spacings(source_spacing_um, effective_spacing_um)
    mask = np.asarray(ignore_mask)
    if mask.ndim != 2:
        raise ValueError(f"ignore_mask must be 2-D, got shape {mask.shape}.")
    height, width = int(target_size[0]), int(target_size[1])
    scale = float(source_spacing_um) / float(effective_spacing_um)

    def _sample_axis(n: int, offset: int, extent: int) -> tuple[np.ndarray, np.ndarray]:
        position = (np.arange(n) + float(offset)) / scale
        inside = (position >= 0) & (position < extent)
        return inside, np.clip(np.rint(position[inside]).astype(np.int64), 0, extent - 1)

    row_in, rows = _sample_axis(height, crop_top, mask.shape[0])
    col_in, cols = _sample_axis(width, crop_left, mask.shape[1])
    valid = np.zeros((height, width), dtype=bool)
    valid[np.ix_(row_in, col_in)] = mask[np.ix_(rows, cols)] != IGNORE_VALUE
    return valid


def points_on_valid(points_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: does each target-frame point land on a valid pixel?

    A point lands on its nearest pixel (as in :func:`render_peak_heatmap`), clamped to the
    canvas so an in-frame point just short of the far edge reads the edge pixel. A point
    outside ``[0, W) x [0, H)`` is not on a valid pixel.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool)
    height, width = valid.shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    cols = np.clip(np.rint(pts[inside, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(pts[inside, 1]).astype(np.int64), 0, height - 1)
    keep = np.zeros(pts.shape[0], dtype=bool)
    keep[inside] = valid[rows, cols]
    return keep
