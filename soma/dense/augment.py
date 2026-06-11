"""Joint image+mask augmentation for the live segmentation path (``feature_mode='live'``).

Builds a single ``torchvision.transforms.v2`` pipeline that operates on a
``(tv_tensors.Image, tv_tensors.Mask)`` pair. v2 dispatches by tensor *type*:
geometric ops (flips, affine) transform **both** the image and the mask (the mask is
resampled nearest-neighbor automatically, preserving class indices), while photometric
ops (``ColorJitter``) transform the **image only** and pass the mask through untouched.

Affine out-of-canvas pixels fill the **image with 0** and the **mask with
``ignore_index``** (per-type ``fill`` dict), so synthesized border pixels are excluded
from loss/metrics rather than mislabeled as a real class.

The transform set mirrors the paper's "affine + color jitter": ``RandomHorizontalFlip``,
``RandomVerticalFlip``, ``RandomAffine`` (rotation/translate/scale), ``ColorJitter``.
Only the ops the config actually enables are added, so an all-default
:class:`~soma.config.AugmentationConfig` yields ``None`` (no-op = the live-no-aug parity
case). A registry-backed/swept transform list (stain/HED jitter, elastic) is deferred
until augmentation becomes a tuned axis.
"""

from __future__ import annotations

from typing import Callable

from soma.config import AugmentationConfig


def build_segmentation_augmentation(
    augmentation: AugmentationConfig,
    *,
    ignore_index: int,
) -> Callable | None:
    """Build a joint image+mask v2 transform, or ``None`` when augmentation is disabled.

    The returned callable takes ``(image, mask)`` as a ``tv_tensors.Image`` /
    ``tv_tensors.Mask`` pair and returns the transformed pair.
    """
    if not augmentation.is_enabled():
        return None

    from torchvision import tv_tensors
    from torchvision.transforms import v2

    transforms: list = []
    if augmentation.horizontal_flip > 0.0:
        transforms.append(v2.RandomHorizontalFlip(p=float(augmentation.horizontal_flip)))
    if augmentation.vertical_flip > 0.0:
        transforms.append(v2.RandomVerticalFlip(p=float(augmentation.vertical_flip)))

    if augmentation.rotation_degrees > 0.0 or augmentation.translate > 0.0 or augmentation.scale > 0.0:
        translate = (
            (float(augmentation.translate), float(augmentation.translate))
            if augmentation.translate > 0.0
            else None
        )
        scale = (
            (1.0 - float(augmentation.scale), 1.0 + float(augmentation.scale))
            if augmentation.scale > 0.0
            else None
        )
        transforms.append(
            v2.RandomAffine(
                degrees=float(augmentation.rotation_degrees),
                translate=translate,
                scale=scale,
                # Per-type fill: image border -> 0, mask border -> ignore_index
                # (excluded from loss/metrics). Keys are the tv_tensor types v2 routes on.
                fill={tv_tensors.Image: 0.0, tv_tensors.Mask: float(ignore_index)},
            )
        )

    if any(
        getattr(augmentation, name) > 0.0
        for name in ("brightness", "contrast", "saturation", "hue")
    ):
        # ColorJitter only transforms images; the Mask passes through unchanged.
        transforms.append(
            v2.ColorJitter(
                brightness=float(augmentation.brightness),
                contrast=float(augmentation.contrast),
                saturation=float(augmentation.saturation),
                hue=float(augmentation.hue),
            )
        )

    return v2.Compose(transforms)
