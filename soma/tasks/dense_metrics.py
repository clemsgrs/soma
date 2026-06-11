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
    "focal_tversky_loss",
    "focal_cross_entropy",
    "cross_entropy_dice_loss",
    "segmentation_loss",
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


def focal_tversky_loss(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Focal-Tversky loss ``mean_c (1 - TI_c)**gamma`` over softmax probabilities.

    The Tversky index ``TI_c = TP / (TP + alpha*FP + beta*FN)`` generalizes Dice:
    ``alpha = beta = 0.5`` recovers Dice, while ``beta > alpha`` weights false
    negatives more (recall-oriented — useful for small/rare structures). ``gamma > 1``
    focuses the loss on hard, low-overlap classes. ``ignore_index`` pixels are dropped
    from TP/FP/FN. With ``alpha = beta = 0.5, gamma = 1`` this equals
    :func:`soft_dice_loss` up to an ``eps`` term (which is why the default loss path
    keeps calling ``soft_dice_loss`` directly — the parity anchor).
    """
    _validate_dense_shapes(logits, mask, num_classes)
    probs = logits.softmax(dim=1)  # (B, C, H, W)
    valid = (mask != ignore_index).unsqueeze(1).float()
    safe_mask = mask.clone()
    safe_mask[mask == ignore_index] = 0
    onehot = F.one_hot(safe_mask, num_classes).permute(0, 3, 1, 2).float()
    probs = probs * valid
    onehot = onehot * valid
    dims = (0, 2, 3)  # reduce batch + spatial -> per-class
    tp = (probs * onehot).sum(dims)
    fp = (probs * (1.0 - onehot)).sum(dims)
    fn = ((1.0 - probs) * onehot).sum(dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tversky).clamp(min=0.0) ** gamma).mean()


def focal_cross_entropy(
    logits: Tensor,
    mask: Tensor,
    *,
    ignore_index: int,
    gamma: float,
    weight: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Focal cross-entropy ``-(1 - p_t)**gamma log p_t``, averaged over valid pixels.

    Down-weights well-classified pixels (large ``p_t``) so training focuses on the
    hard, typically rare-class pixels. ``gamma = 0`` reduces to plain (optionally
    class-weighted) cross-entropy; the head only routes here when ``gamma > 0``.
    ``weight`` is an optional per-class ``(C,)`` tensor; the normalizer is the
    (weighted) count of valid pixels, mirroring ``F.cross_entropy``'s weighted mean.
    """
    log_prob = F.log_softmax(logits, dim=1)
    valid = mask != ignore_index
    safe_mask = mask.clone()
    safe_mask[~valid] = 0
    log_pt = log_prob.gather(1, safe_mask.unsqueeze(1)).squeeze(1)  # (B, H, W)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt
    valid_f = valid.float()
    if weight is not None:
        per_pixel_weight = weight[safe_mask]  # (B, H, W)
        loss = loss * per_pixel_weight
        norm = (per_pixel_weight * valid_f).sum()
    else:
        norm = valid_f.sum()
    loss = loss * valid_f
    return loss.sum() / norm.clamp(min=eps)


def segmentation_loss(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    dice_weight: float = 1.0,
    class_weights: list[float] | None = None,
    ce_gamma: float = 0.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_gamma: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Composite dense loss: a region term (CE) + ``dice_weight`` * an overlap term.

    The two imbalance knobs (all opt-in; the defaults reproduce ``CE + soft_dice``
    byte-for-byte):

    * **region term** — class-weighted cross-entropy, or *focal* cross-entropy when
      ``ce_gamma > 0`` (down-weights easy pixels). ``class_weights`` is a per-class
      ``(C,)`` list (e.g. inverse frequency) up-weighting rare classes.
    * **overlap term** — soft-Dice by default; *focal-Tversky* when ``tversky_alpha``/
      ``tversky_beta``/``tversky_gamma`` depart from ``(0.5, 0.5, 1.0)``, trading
      false-negatives vs false-positives on rare structures.

    If a batch has *no* supervised pixels (every pixel is ``ignore_index`` — e.g. an
    all-background tile at ``batch_size=1``), the region term divides by zero and
    returns ``nan``, silently poisoning training. Guard with a graph-connected zero so
    the step is a no-op rather than a poison.
    """
    _validate_dense_shapes(logits, mask, num_classes)
    if not bool((mask != ignore_index).any()):
        return logits.sum() * 0.0

    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)

    if float(ce_gamma) > 0.0:
        region = focal_cross_entropy(
            logits, mask, ignore_index=ignore_index, gamma=float(ce_gamma), weight=weight
        )
    else:
        region = F.cross_entropy(logits, mask, weight=weight, ignore_index=ignore_index)

    # Soft-Dice is Tversky(0.5, 0.5) with focal exponent 1: keep the existing,
    # parity-anchored soft_dice_loss on the default path; only take the general
    # focal-Tversky branch when the user tilts alpha/beta/gamma.
    if float(tversky_alpha) == 0.5 and float(tversky_beta) == 0.5 and float(tversky_gamma) == 1.0:
        overlap = soft_dice_loss(
            logits, mask, num_classes=num_classes, ignore_index=ignore_index, eps=eps
        )
    else:
        overlap = focal_tversky_loss(
            logits,
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=float(tversky_alpha),
            beta=float(tversky_beta),
            gamma=float(tversky_gamma),
            eps=eps,
        )
    return region + dice_weight * overlap


def cross_entropy_dice_loss(
    logits: Tensor,
    mask: Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """``CE(ignore_index) + dice_weight * soft_dice`` — the default-knobs special case.

    Thin wrapper over :func:`segmentation_loss` with no class weights, no focal CE,
    and Dice (not Tversky) overlap, so it stays the numerically-identical baseline.
    """
    return segmentation_loss(
        logits,
        mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
        dice_weight=dice_weight,
        eps=eps,
    )


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
