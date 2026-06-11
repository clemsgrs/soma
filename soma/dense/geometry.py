"""Pure spatial geometry for dense (segmentation) feature extraction.

This module owns the pad-to-patch-multiple math that maps a supervision-sized
tile/mask (``target_size``) to the patch-divisible tensor actually fed to a ViT
encoder (``encoded_size``) and the resulting token grid (``grid_shape``). It is
deliberately weight-free and side-effect-free: the returned :class:`DenseGridGeometry`
is exactly what the dense cache persists as per-sample metadata, and what the
decoder/head later use to crop logits back to the mask.

Why a dedicated function (not inlined in the extractor): this is the
highest-bug-density code in the dense pipeline — the patch-16 vs patch-14
asymmetry (512 is clean for P=16 → 32×32, but P=14 needs 518 → 37×37; see
segmentation-design §5c) silently misregisters the grid against the mask if it is
off by a patch. Isolating it makes it unit-testable without an encoder.

Scope: the **output** geometry — ``target_size`` padded to the patch multiple and the
resulting token grid. Sliding-window extraction (design §5, window-as-knob) reuses this
exact geometry as its stitched output and *derives* its per-window tiling from it in
:mod:`soma.dense.sliding` (``resolve_window_geometry``); it does not need a separate
output layout here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DenseGridGeometry", "compute_dense_geometry", "normalize_hw"]


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
    """Resolved spatial layout for one dense extraction.

    All sizes are ``(height, width)`` in pixels except ``grid_shape`` which is
    ``(grid_h, grid_w)`` in tokens. ``crop_box`` is ``(top, left, height, width)``
    in *encoded-pixel* space — the region of the (padded) encoded tile that
    corresponds to the original ``target_size``, so decoded logits can be cropped
    back to the mask. Padding is applied on the bottom/right only, so the tile
    origin ``(0, 0)`` maps to grid cell ``(0, 0)`` and ``crop_box`` is top-left
    anchored.
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
