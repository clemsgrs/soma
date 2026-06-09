"""Pure dense (segmentation) loss and metric functions.

Operate on **already-aligned** logits ``(B, C, H, W)`` and integer masks
``(B, H, W)`` — no geometry, no resize — so they unit-test against hand-computed
values on tiny explicit masks. Two distinct families, deliberately NOT sharing a
code path (different definitions):

- **loss**: differentiable — ``CE(ignore_index)`` + soft Dice over softmax probs.
- **metric**: hard — Dice/IoU over the argmax prediction.

``ignore_index`` pixels are dropped from BOTH numerator and denominator everywhere.
An undefined per-class score (class absent from both prediction and target in an
image → 0/0) is reported as NaN and **excluded** from the means (skip-don't-average),
rather than counted as a perfect or zero score, which would materially move the
number on small cohorts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "soft_dice_loss",
    "cross_entropy_dice_loss",
    "dense_confusion_counts",
    "reduce_dice_iou",
]


def _validate_dense_shapes(logits: Tensor, mask: Tensor, num_classes: int) -> None:
    if logits.ndim != 4:
        raise ValueError(f"logits must be (B, C, H, W), got {tuple(logits.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be (B, H, W), got {tuple(mask.shape)}")
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"logits channel dim {logits.shape[1]} != num_classes {num_classes}"
        )
    if logits.shape[0] != mask.shape[0] or logits.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"logits {tuple(logits.shape)} and mask {tuple(mask.shape)} disagree on "
            "batch/spatial dims"
        )


def soft_dice_loss(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> Tensor:
    """Soft (differentiable) multi-class Dice loss over softmax probabilities.

    ``1 - mean_c Dice_c`` where Dice is computed over the batch with ``ignore_index``
    pixels masked out of probs and one-hot targets. ``eps`` smooths the 0/0 case
    (an absent class contributes ~perfect Dice → ~0 loss).
    """
    _validate_dense_shapes(logits, mask, num_classes)
    probs = logits.softmax(dim=1)  # (B, C, H, W)
    valid = (mask != ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
    safe_mask = mask.clone()
    safe_mask[mask == ignore_index] = 0  # placeholder class; zeroed by `valid`
    onehot = F.one_hot(safe_mask, num_classes).permute(0, 3, 1, 2).float()
    probs = probs * valid
    onehot = onehot * valid
    dims = (0, 2, 3)  # reduce batch + spatial -> per-class
    intersection = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice_per_class = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


def cross_entropy_dice_loss(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """``CE(ignore_index) + dice_weight * soft_dice``.

    If a batch has *no* supervised pixels (every pixel is ``ignore_index`` — e.g. an
    all-background tile at ``batch_size=1``), ``F.cross_entropy`` divides by zero and
    returns ``nan``, silently poisoning training. Guard with a graph-connected zero
    so the step is a no-op rather than a poison.
    """
    if not bool((mask != ignore_index).any()):
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, mask, ignore_index=ignore_index)
    dice = soft_dice_loss(
        logits, mask, num_classes=num_classes, ignore_index=ignore_index, eps=eps
    )
    return ce + dice_weight * dice


def dense_confusion_counts(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> Tensor:
    """Per-image, per-class ``(intersection, pred_area, target_area)`` counts.

    Returns a ``(B, num_classes, 3)`` long tensor over the hard argmax prediction,
    with ``ignore_index`` pixels excluded. These accumulators let a streaming
    evaluator compute either per-image-macro or dataset-global Dice/IoU exactly,
    without holding all logits in memory.
    """
    _validate_dense_shapes(logits, mask, num_classes)
    pred = logits.argmax(dim=1)  # (B, H, W)
    valid = mask != ignore_index
    counts = torch.zeros(logits.shape[0], num_classes, 3, dtype=torch.long, device=logits.device)
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (mask == c) & valid
        counts[:, c, 0] = (pred_c & target_c).sum(dim=(1, 2))
        counts[:, c, 1] = pred_c.sum(dim=(1, 2))
        counts[:, c, 2] = target_c.sum(dim=(1, 2))
    return counts


def _nanmean_to_float(x: Tensor) -> float:
    """Mean over non-NaN entries; 0.0 if all entries are NaN (fully undefined)."""
    value = torch.nanmean(x)
    return 0.0 if torch.isnan(value) else float(value)


def reduce_dice_iou(
    counts: Tensor,
    *,
    num_classes: int,
    aggregation: str = "per_image_macro",
) -> dict[str, float]:
    """Reduce ``(N, C, 3)`` confusion counts to Dice/IoU scalars.

    ``per_image_macro`` (default, the design §9 monitor convention): Dice/IoU per
    image averaged over its defined classes, then averaged over images.
    ``dataset_global``: sum counts over all images, one Dice/IoU per class, averaged.
    Per-class Dice (``dice_class_{c}``) is always the dataset-global per-class value.
    """
    inter = counts[..., 0].float()
    pred = counts[..., 1].float()
    target = counts[..., 2].float()
    denom = pred + target
    union = pred + target - inter
    dice = torch.where(denom > 0, 2.0 * inter / denom, torch.nan)  # (N, C)
    iou = torch.where(union > 0, inter / union, torch.nan)

    # Dataset-global per class: counts summed over images, then one Dice/IoU per
    # class (the paper's "DSC computed separately for each category"). NOT a
    # per-image average of per-image Dice — those diverge when image sizes vary.
    g_inter, g_denom, g_union = inter.sum(0), denom.sum(0), union.sum(0)
    per_class_dice = torch.where(g_denom > 0, 2.0 * g_inter / g_denom, torch.nan)

    if aggregation == "per_image_macro":
        mean_dice = _nanmean_to_float(torch.nanmean(dice, dim=1))
        mean_iou = _nanmean_to_float(torch.nanmean(iou, dim=1))
    elif aggregation == "dataset_global":
        giou = torch.where(g_union > 0, g_inter / g_union, torch.nan)
        mean_dice = _nanmean_to_float(per_class_dice)
        mean_iou = _nanmean_to_float(giou)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    result = {"mean_dice": mean_dice, "mean_iou": mean_iou}
    for c in range(num_classes):
        result[f"dice_class_{c}"] = (
            0.0 if torch.isnan(per_class_dice[c]) else float(per_class_dice[c])
        )
    return result
