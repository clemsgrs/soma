"""Joint ``(pixels, target)`` augmentations for the live grid path.

An augmentation is any callable ``(pixels, target) -> (pixels, target)``: ``pixels`` is
one sample's ``(3, H, W)`` uint8 tensor, ``target`` its raw supervision (a ``(H, W)``
long label map, or :class:`~soma.data.Points`). :func:`soma.learn.loaders` applies it to
the train split only, before the encoder stage's ``preprocess``; cached grids cannot be
augmented. ``augment_registry`` resolves the ``augmentation: {name, params}`` config leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from soma.data import Points
from soma._internal.params import run_decided
from soma.registry import Registry

__all__ = ["SegmentationAugment", "SegmentationAugmentParams", "augment_registry"]

augment_registry = Registry("augmentations")


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

    Point targets are refused until torchvision 0.23 brings ``tv_tensors.KeyPoints``
    (soma#586).
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
        self._transform = self._build()

    def _build(self) -> Any:
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
                    fill={tv_tensors.Image: 0.0, tv_tensors.Mask: float(self.ignore_index)},
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
            raise ValueError(
                "point targets cannot be augmented yet: joint point augmentation waits for "
                "torchvision 0.23 tv_tensors.KeyPoints (soma#586)"
            )
        if not torch.is_tensor(target) or target.ndim != 2:
            raise ValueError(f"SegmentationAugment augments a (H, W) label map, got {type(target).__name__}")
        image, mask = self._transform(
            tv_tensors.Image(pixels), tv_tensors.Mask(target.to(torch.long))
        )
        return image.as_subclass(Tensor).to(torch.uint8), mask.as_subclass(Tensor).to(torch.long)

    def __repr__(self) -> str:
        return f"SegmentationAugment({self._transform})"


augment_registry.register("segmentation", SegmentationAugment, params=SegmentationAugmentParams)
