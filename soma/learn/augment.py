"""Joint ``(pixels, target)`` augmentations for the live grid path.

An augmentation is any callable ``(pixels, target) -> (pixels, target)``: ``pixels`` is
one sample's ``(3, H, W)`` uint8 tensor, ``target`` its raw supervision (a ``(H, W)``
long label map, or :class:`~soma.data.Points` in the pixel frame of ``pixels``: ``xy`` in
pixels, with an optional ``(H, W)`` ignore raster on the same frame).
:func:`soma.learn.loaders` applies it to the train split only, before the encoder stage's
``preprocess``, and maps level-0 points into the pixel frame and back around it; cached
grids cannot be augmented. ``augment_registry`` resolves the ``augmentation: {name,
params}`` config leaf.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from soma.data import Points
from soma._internal.params import run_decided
from soma.registry import Registry

__all__ = ["SegmentationAugment", "SegmentationAugmentParams", "augment_registry"]

augment_registry = Registry("augmentations")

#: Ignore-raster value of a pixel that carries no supervision.
_IGNORED = 255


def _outside_to_nan(xy: Tensor, canvas_size: tuple[int, int]) -> Tensor:
    """``(N, 2)`` points with every row outside ``[0, W) x [0, H)`` set to NaN."""
    height, width = canvas_size
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return torch.where(inside.unsqueeze(1), xy, torch.full_like(xy, float("nan")))


@functools.lru_cache(maxsize=None)
def _unclamped_keypoints() -> type:
    """A ``tv_tensors.KeyPoints`` whose geometric kernels let points leave the canvas.

    torchvision 0.23 clamps keypoints onto the canvas edge after every geometric op, so a
    point cropped or rotated out of the image would come back supervising the border.
    This subclass registers its own kernels for the ops :class:`SegmentationAugment`
    builds: they move a point exactly as the image kernels move the pixel under it (pixel
    centres at integer coordinates) and never clamp. A point that leaves the canvas
    becomes NaN, so a later op cannot bring it back onto pixels it no longer has.
    """
    from torchvision import tv_tensors
    from torchvision.transforms.v2 import functional as F

    class UnclampedKeyPoints(tv_tensors.KeyPoints):
        pass

    def wrap(xy: Tensor, like, canvas_size=None):
        # Not ``tv_tensors.wrap``: it returns a plain ``KeyPoints``, whose kernels clamp.
        canvas_size = canvas_size or like.canvas_size
        return UnclampedKeyPoints(_outside_to_nan(xy, canvas_size), canvas_size=canvas_size)

    @F.register_kernel(F.horizontal_flip, UnclampedKeyPoints)
    def horizontal_flip(points, **kwargs):
        xy = points.as_subclass(Tensor).clone()
        xy[:, 0] = points.canvas_size[1] - 1 - xy[:, 0]
        return wrap(xy, points)

    @F.register_kernel(F.vertical_flip, UnclampedKeyPoints)
    def vertical_flip(points, **kwargs):
        xy = points.as_subclass(Tensor).clone()
        xy[:, 1] = points.canvas_size[0] - 1 - xy[:, 1]
        return wrap(xy, points)

    @F.register_kernel(F.resized_crop, UnclampedKeyPoints)
    def resized_crop(points, top, left, height, width, size, **kwargs):
        out_h, out_w = (int(v) for v in size)
        xy = points.as_subclass(Tensor)
        moved = torch.stack(
            [
                (xy[:, 0] - left + 0.5) * (out_w / width) - 0.5,
                (xy[:, 1] - top + 0.5) * (out_h / height) - 0.5,
            ],
            dim=1,
        )
        return wrap(moved.to(xy.dtype), points, (out_h, out_w))

    @F.register_kernel(F.affine, UnclampedKeyPoints)
    def affine(points, angle, translate, scale, shear, center=None, **kwargs):
        height, width = points.canvas_size
        xy = points.as_subclass(Tensor).to(torch.float64)
        if center is None:  # the image kernel rotates about the centre pixel
            center = [(width - 1) * 0.5, (height - 1) * 0.5]
        # torchvision's keypoint kernel clamps to its canvas: run it on one padded so wide
        # that no point reaches the edge, then shift back (the map commutes with a shift).
        reach = float(torch.nan_to_num(xy).abs().max()) if xy.numel() else 0.0
        margin = 4 * int(reach + height + width + sum(abs(float(t)) for t in translate) + 1)
        moved, _ = F.affine_keypoints(
            xy + margin,
            canvas_size=(height + 2 * margin, width + 2 * margin),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=[float(center[0]) + margin, float(center[1]) + margin],
        )
        return wrap((moved - margin).to(points.dtype), points)

    return UnclampedKeyPoints


def _check_strengths(
    *,
    crop_scale: Any,
    horizontal_flip: float,
    vertical_flip: float,
    rotation_degrees: float,
    translate: float,
    scale: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> None:
    for name, value in (("horizontal_flip", horizontal_flip), ("vertical_flip", vertical_flip)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1], got {value}")
    for name, value in (
        ("rotation_degrees", rotation_degrees),
        ("translate", translate),
        ("scale", scale),
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if not 0.0 <= float(translate) <= 1.0:
        raise ValueError(f"translate must be in [0, 1], got {translate}")
    if not 0.0 <= float(scale) < 1.0:
        raise ValueError(f"scale must be in [0, 1), got {scale}")
    if not 0.0 <= float(hue) <= 0.5:
        raise ValueError(f"hue must be in [0, 0.5], got {hue}")
    if crop_scale is not None:
        low, high = (float(v) for v in crop_scale)
        if not 0.0 < low <= high <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < low <= high <= 1, got {crop_scale}")


@dataclass
class SegmentationAugmentParams:
    """:class:`SegmentationAugment`'s params. A run fills ``ignore_index`` and
    ``target_size`` from its head and geometry when they are not set."""

    crop_scale: tuple[float, float] | None = None
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    rotation_degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    ignore_index: int | None = run_decided()
    target_size: int | tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _check_strengths(
            crop_scale=self.crop_scale,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            rotation_degrees=self.rotation_degrees,
            translate=self.translate,
            scale=self.scale,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class SegmentationAugment:
    """Joint torchvision-v2 image + label-map augmentation.

    Geometric ops transform the pixels and the label map together (the map resampled
    nearest-neighbour); photometric ops touch the pixels only. Out-of-canvas affine
    pixels fill the image with 0 and the label map with ``ignore_index``, so synthesised
    borders carry no supervision. An optional ``RandomResizedCrop`` to ``target_size``
    runs first, so images of any size reach the encoder at the supervision size.

    Args:
        crop_scale: ``RandomResizedCrop`` area range (``None`` = no crop); needs
            ``target_size``.
        horizontal_flip / vertical_flip: Flip probabilities.
        rotation_degrees: ``RandomAffine`` symmetric rotation range (±value).
        translate: ``RandomAffine`` max translation fraction (both axes).
        scale: ``RandomAffine`` scale jitter, ``scale ∈ (1 - value, 1 + value)``.
        brightness / contrast / saturation / hue: ``ColorJitter`` strengths.
        ignore_index: Label-map fill for synthesised pixels.
        target_size: ``(H, W)`` output of ``crop_scale``.

    A :class:`~soma.data.Points` target (in the pixel frame) rides as
    ``tv_tensors.KeyPoints``: its points move with the pixels, points that leave the
    canvas are dropped (never clamped onto the edge), and its ignore raster moves like a
    label map, filled with 255 outside the canvas. Points without an ignore raster get
    one when the transform synthesises border pixels, so those carry no supervision.
    """

    def __init__(
        self,
        *,
        crop_scale: tuple[float, float] | None = None,
        horizontal_flip: float = 0.0,
        vertical_flip: float = 0.0,
        rotation_degrees: float = 0.0,
        translate: float = 0.0,
        scale: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        ignore_index: int = 255,
        target_size: int | tuple[int, int] | None = None,
    ) -> None:
        _check_strengths(
            crop_scale=crop_scale,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotation_degrees=rotation_degrees,
            translate=translate,
            scale=scale,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        if crop_scale is not None:
            crop_scale = (float(crop_scale[0]), float(crop_scale[1]))
            if target_size is None:
                raise ValueError("crop_scale crops to target_size; pass target_size")
        self.crop_scale = crop_scale
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.rotation_degrees = float(rotation_degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.ignore_index = int(ignore_index)
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = None if target_size is None else tuple(int(v) for v in target_size)
        self._transform = self._build(mask_fill=self.ignore_index)
        self._point_transform = self._build(mask_fill=_IGNORED)

    def _build(self, *, mask_fill: int) -> Any:
        from torchvision import tv_tensors
        from torchvision.transforms import v2

        transforms: list[Any] = []
        if self.crop_scale is not None:
            transforms.append(v2.RandomResizedCrop(size=self.target_size, scale=self.crop_scale))
        if self.horizontal_flip > 0.0:
            transforms.append(v2.RandomHorizontalFlip(p=self.horizontal_flip))
        if self.vertical_flip > 0.0:
            transforms.append(v2.RandomVerticalFlip(p=self.vertical_flip))
        if self.rotation_degrees > 0.0 or self.translate > 0.0 or self.scale > 0.0:
            transforms.append(
                v2.RandomAffine(
                    degrees=self.rotation_degrees,
                    translate=(self.translate, self.translate) if self.translate > 0.0 else None,
                    scale=(1.0 - self.scale, 1.0 + self.scale) if self.scale > 0.0 else None,
                    fill={tv_tensors.Image: 0.0, tv_tensors.Mask: float(mask_fill)},
                )
            )
        if any(v > 0.0 for v in (self.brightness, self.contrast, self.saturation, self.hue)):
            transforms.append(
                v2.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                )
            )
        # ``v2.Compose([])`` raises (torchvision 0.22); no enabled op is the identity.
        return v2.Compose(transforms) if transforms else v2.Identity()

    def __call__(self, pixels: Tensor, target: Any) -> tuple[Tensor, Any]:
        from torchvision import tv_tensors

        if isinstance(target, Points):
            return self._augment_points(pixels, target)
        if not torch.is_tensor(target) or target.ndim != 2:
            raise ValueError(f"SegmentationAugment augments a (H, W) label map, got {type(target).__name__}")
        image, mask = self._transform(
            tv_tensors.Image(pixels), tv_tensors.Mask(target.to(torch.long))
        )
        return image.as_subclass(Tensor).to(torch.uint8), mask.as_subclass(Tensor).to(torch.long)

    def _augment_points(self, pixels: Tensor, target: Points) -> tuple[Tensor, Points]:
        from torchvision import tv_tensors

        height, width = (int(v) for v in pixels.shape[1:])
        if target.ignore_mask is None:
            ignore = np.zeros((height, width), dtype=np.uint8)
        else:
            ignore = np.where(np.asarray(target.ignore_mask) == _IGNORED, _IGNORED, 0).astype(np.uint8)
            if ignore.shape != (height, width):
                raise ValueError(
                    f"the points' ignore mask is {ignore.shape} but the pixels are "
                    f"{(height, width)}; an augmentation takes the ignore raster on the pixel frame"
                )
        keypoints = _unclamped_keypoints()(
            target.xy.to(torch.float32).reshape(-1, 2), canvas_size=(height, width)
        )
        image, mask, moved = self._point_transform(
            tv_tensors.Image(pixels), tv_tensors.Mask(torch.from_numpy(ignore)), keypoints
        )
        xy = moved.as_subclass(Tensor).to(torch.float32).reshape(-1, 2)
        inside = ~torch.isnan(_outside_to_nan(xy, tuple(image.shape[1:]))).any(dim=1)
        ignore_out = mask.as_subclass(Tensor).numpy().astype(np.uint8)
        if target.ignore_mask is None and not (ignore_out == _IGNORED).any():
            ignore_out = None
        return image.as_subclass(Tensor).to(torch.uint8), Points(
            xy=xy[inside], classes=target.classes.to(torch.long)[inside], ignore_mask=ignore_out
        )

    def __repr__(self) -> str:
        return f"SegmentationAugment({self._transform})"


augment_registry.register("segmentation", SegmentationAugment, params=SegmentationAugmentParams)
