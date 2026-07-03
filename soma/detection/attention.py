"""Attention-map detection — the decoder-free rung 0 of the complexity ladder.

The analogue of arXiv:2602.18747's "attention-maps-as-features" idea, taken to its
zero-parameter limit for detection: the frozen encoder's own CLS/rollout attention grid
``(K, gh, gw)`` (``feature_kind="cls_attention"``) *is* the detection heatmap — no
decoder is trained, so **no parameter touches the features**. A CLS-attention map
highlights the tokens the encoder attends to; where objects are salient (single-class
MIDOG mitoses), its peaks are the detections. This is deliberately class-agnostic
(saliency), so it is **MIDOG-scoped** (design §2.4): strong on one rare class, weak where
detection is multi-class (OCELOT/Monkey) — itself a benchmark finding.

The map is turned into a heatmap purely geometrically: reduce the ``K`` attention
channels to one saliency map, upsample it to the supervision frame with the run's
:class:`~soma.dense.geometry.DenseGridGeometry` (bilinear to ``encoded_size``, crop
``crop_box`` to ``target_size`` — the same map a decoder's logits take), and min-max
normalise to ``[0, 1]``. The existing detection back half (``extract_peaks`` ->
``match_points`` -> F1@δ, or :class:`~soma.tasks.detection.DetectionHead`) scores it
unchanged, so the rung is swappable with no change to the peak/matching components.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from soma.dense.geometry import DenseGridGeometry

__all__ = ["attention_to_detection_heatmap"]

_VALID_REDUCTIONS = {"mean", "max"}


def attention_to_detection_heatmap(
    attention_grid: Tensor,
    *,
    geometry: DenseGridGeometry,
    num_classes: int = 1,
    reduction: str = "mean",
    normalize: bool = True,
) -> Tensor:
    """Turn a ``(K, gh, gw)`` attention grid into a ``(num_classes, H, W)`` detection heatmap.

    Zero trained parameters: the encoder's attention is used verbatim as saliency.

    Args:
        attention_grid: The cached ``cls_attention`` grid ``(K, gh, gw)`` — ``K``
            attention channels (blocks x heads, optionally register rows) over the token
            grid.
        geometry: The sample's :class:`DenseGridGeometry` (supplies ``encoded_size`` and
            ``crop_box``), so the saliency lands on the supervision pixel frame exactly
            where a decoder's logits would.
        num_classes: Heatmap channels ``C``. The saliency is class-agnostic, so it is
            broadcast to every channel (MIDOG uses ``C = 1``); multi-class only makes
            sense as the "attention is class-blind" control the design calls out.
        reduction: How to collapse the ``K`` attention channels to one saliency map —
            ``"mean"`` (default) or ``"max"``.
        normalize: Min-max normalise the saliency to ``[0, 1]`` (matching the ``[0, 1]``
            heatmap the peak extractor / tune-split threshold sweep expect). A constant
            map collapses to all-zeros (no peaks), the correct "nothing salient" answer.

    Returns:
        ``(num_classes, H, W)`` heatmap in ``[0, 1]`` at the mask ``target_size``.
    """
    if attention_grid.ndim != 3:
        raise ValueError(
            f"attention_grid must be (K, gh, gw), got shape {tuple(attention_grid.shape)}"
        )
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_VALID_REDUCTIONS)}, got {reduction!r}")

    grid = attention_grid.detach().float()
    saliency = grid.mean(dim=0) if reduction == "mean" else grid.amax(dim=0)  # (gh, gw)

    # Upsample to the padded encoded frame, then crop to target_size — the head's geometry,
    # so the attention saliency registers against GT points the same way logits would.
    up = F.interpolate(
        saliency[None, None], size=geometry.encoded_size, mode="bilinear", align_corners=False
    )
    top, left, height, width = geometry.crop_box
    cropped = up[0, 0, top : top + height, left : left + width]  # (H, W)

    if normalize:
        lo = float(cropped.min())
        hi = float(cropped.max())
        # Flat map -> all-zeros (no salient peak). The tolerance is relative so bilinear
        # fp noise on a constant grid is not stretched to [0, 1]; a real bump clears it.
        spread = hi - lo
        if spread > 1e-6 * (abs(hi) + abs(lo) + 1e-8):
            cropped = (cropped - lo) / spread
        else:
            cropped = torch.zeros_like(cropped)

    return cropped.unsqueeze(0).expand(int(num_classes), -1, -1).contiguous()
