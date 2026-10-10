"""Token-grid geometry: the padded encoder frame and its mapping back to level-0 pixels.

The grid source records this geometry so decoders can crop outputs back to the
supervision frame, and so stitching, heatmaps and point conversion share one
``token_to_level0``. For example, a 512-pixel tile needs no padding for patch size 16,
but is padded to 518 pixels for patch size 14 to keep grid-to-mask alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DenseGridGeometry", "GridGeometry", "compute_dense_geometry", "normalize_hw"]


def normalize_hw(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    """Coerce an ``int`` or ``(h, w)`` pair to a validated ``(h, w)`` tuple."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value, value
    try:
        h, w = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an int or an (h, w) pair, got {value!r}") from exc
    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must be positive, got {(h, w)}")
    return h, w


@dataclass(frozen=True)
class DenseGridGeometry:
    """Resolved spatial layout of one token grid, in the encoder's pixel frame.

    All sizes are ``(height, width)`` in pixels except ``grid_shape`` which is
    ``(grid_h, grid_w)`` in tokens. ``crop_box`` is ``(top, left, height, width)`` in
    *encoded-pixel* space — the region of the (padded) encoded tile that corresponds to
    the original ``target_size``, so decoded logits can be cropped back to the
    supervision frame. Padding is applied on the bottom/right only, so the tile origin
    ``(0, 0)`` maps to grid cell ``(0, 0)`` and ``crop_box`` is top-left anchored.
    """

    target_size: tuple[int, int]
    patch_size: tuple[int, int]
    encoded_size: tuple[int, int]
    grid_shape: tuple[int, int]
    pad: tuple[int, int]  # (pad_bottom, pad_right)
    crop_box: tuple[int, int, int, int]  # (top, left, height, width)

    @property
    def is_padded(self) -> bool:
        return self.pad != (0, 0)

    @property
    def layout(self) -> "DenseGridGeometry":
        """The frame layout alone: sizes, token grid, padding and crop box.

        This is what a decoder built from one reference sample needs every other grid
        to share. The level-0 anchor a :class:`GridGeometry` adds (origin, scale) is
        per sample, so cohort uniformity checks compare ``layout``, not the geometry.
        """
        return DenseGridGeometry(
            target_size=self.target_size,
            patch_size=self.patch_size,
            encoded_size=self.encoded_size,
            grid_shape=self.grid_shape,
            pad=self.pad,
            crop_box=self.crop_box,
        )


@dataclass(frozen=True)
class GridGeometry(DenseGridGeometry):
    """A :class:`DenseGridGeometry` anchored in the slide's level-0 pixel frame.

    ``origin_level0`` is the ``(x, y)`` level-0 pixel of the grid's top-left corner and
    ``level0_px_per_token_px`` the number of level-0 pixels per pixel of the encoder
    frame (``effective_spacing / source_spacing``). Together they give
    :meth:`token_to_level0`, the one conversion stitching, heatmaps and point
    matching share.
    """

    origin_level0: tuple[float, float] = (0.0, 0.0)
    level0_px_per_token_px: float = 1.0

    @classmethod
    def from_sizes(
        cls,
        *,
        target_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        origin_level0: tuple[float, float] = (0.0, 0.0),
        level0_px_per_token_px: float = 1.0,
    ) -> "GridGeometry":
        return cls.from_dense(
            compute_dense_geometry(target_size=target_size, patch_size=patch_size),
            origin_level0=origin_level0,
            level0_px_per_token_px=level0_px_per_token_px,
        )

    @classmethod
    def from_dense(
        cls,
        geometry: DenseGridGeometry,
        *,
        origin_level0: tuple[float, float] = (0.0, 0.0),
        level0_px_per_token_px: float = 1.0,
    ) -> "GridGeometry":
        if not level0_px_per_token_px > 0:
            raise ValueError(
                f"level0_px_per_token_px must be positive, got {level0_px_per_token_px!r}"
            )
        return cls(
            target_size=geometry.target_size,
            patch_size=geometry.patch_size,
            encoded_size=geometry.encoded_size,
            grid_shape=geometry.grid_shape,
            pad=geometry.pad,
            crop_box=geometry.crop_box,
            origin_level0=(float(origin_level0[0]), float(origin_level0[1])),
            level0_px_per_token_px=float(level0_px_per_token_px),
        )

    def token_to_level0(self, ij):
        """Map token indices ``(i, j)`` (row, column) to level-0 ``(x, y)`` token centres.

        Accepts one ``(i, j)`` pair (returns an ``(x, y)`` tuple) or an ``(N, 2)`` array
        (returns an ``(N, 2)`` float array).
        """
        patch_h, patch_w = self.patch_size
        origin_x, origin_y = self.origin_level0
        scale = self.level0_px_per_token_px
        array = np.asarray(ij, dtype=np.float64)
        rows, cols = array[..., 0], array[..., 1]
        x = origin_x + (cols + 0.5) * patch_w * scale
        y = origin_y + (rows + 0.5) * patch_h * scale
        if array.ndim == 1:
            return float(x), float(y)
        return np.stack([x, y], axis=-1)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def compute_dense_geometry(
    *,
    target_size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
) -> DenseGridGeometry:
    """Resolve the encoded size, token grid, padding, and crop box for a tile.

    ``encoded_size`` is ``target_size`` rounded **up** to the next multiple of the
    patch size (pad on bottom/right). The token grid is ``encoded_size /
    patch_size``. Cropping the decoded logits to ``crop_box`` recovers the
    ``target_size`` region.
    """
    target_h, target_w = normalize_hw(target_size, name="target_size")
    patch_h, patch_w = normalize_hw(patch_size, name="patch_size")

    encoded_h = _round_up(target_h, patch_h)
    encoded_w = _round_up(target_w, patch_w)
    grid_h = encoded_h // patch_h
    grid_w = encoded_w // patch_w
    pad_bottom = encoded_h - target_h
    pad_right = encoded_w - target_w

    return DenseGridGeometry(
        target_size=(target_h, target_w),
        patch_size=(patch_h, patch_w),
        encoded_size=(encoded_h, encoded_w),
        grid_shape=(grid_h, grid_w),
        pad=(pad_bottom, pad_right),
        crop_box=(0, 0, target_h, target_w),
    )
