"""Rung 0 of the detection decoder ladder: the encoder's own attention as the heatmap.

``attention_map`` is a decoder with no parameters. It reads a ``cls_attention`` grid
``(B, K, h, w)`` (``features.dense.feature_kind: cls_attention``), reduces its ``K`` head
maps (all of them, or the ``heads`` it is given) to one map per sample, min-max normalises
that map to ``[0, 1]`` and returns its logit. The detection head's interpolation, crop and
sigmoid then turn it back into a ``[0, 1]`` heatmap on the supervision frame, as they do
for a trained decoder's logits. Every class channel gets the same map: attention says
where the cells are, not which class they belong to.

A run with this decoder trains nothing (:class:`~soma.learn.Trainer` returns a learner
with no trainable parameter as built). Its tune split still freezes the detection
thresholds, so the run is scored like every other rung of the ladder. The decoder
registers when :mod:`soma.benchmark` is imported, which the soma CLI does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from soma.models.decoders.base import Decoder
from soma.models.decoders.registry import decoder_registry
from soma.registry import check_choice, check_non_negative

__all__ = ["DECODER_NAME", "AttentionMapDecoder", "AttentionMapParams"]

DECODER_NAME = "attention_map"

#: Bounds of the normalised map before the logit: a flat map (nothing salient) becomes
#: ``EPS`` everywhere, below the detection head's ``min_score``, so it yields no peak.
EPS = 1e-6


@dataclass
class AttentionMapParams:
    """:class:`AttentionMapDecoder`'s params.

    ``reduction`` collapses the head maps (``mean`` or ``max``); ``heads`` keeps only
    those channels of the grid (``null``: every channel), so ``heads: [k]`` scores head
    ``k`` alone.
    """

    reduction: str = "mean"
    heads: list[int] | None = None

    def __post_init__(self) -> None:
        check_choice("reduction", self.reduction, ("mean", "max"))
        if self.heads is not None:
            if not self.heads:
                raise ValueError("heads must name at least one channel, or be null for every channel")
            for head in self.heads:
                check_non_negative("heads", head, integer=True)


class AttentionMapDecoder(Decoder):
    """``(B, K, h, w)`` attention grid -> ``(B, C, h, w)`` logits, with no parameters."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        reduction: str = "mean",
        heads: list[int] | None = None,
    ) -> None:
        super().__init__()
        params = AttentionMapParams(reduction=reduction, heads=heads)
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if params.heads is not None and max(params.heads) >= input_dim:
            raise ValueError(f"heads {params.heads} out of range for a grid of {input_dim} channels")
        self._num_classes = int(num_classes)
        self.reduction = params.reduction
        self.heads = None if params.heads is None else [int(head) for head in params.heads]

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 4:
            raise ValueError(f"decoder expects a (B, K, h, w) grid, got shape {tuple(X.shape)}")
        grid = X if self.heads is None else X[:, self.heads]
        saliency = grid.mean(dim=1) if self.reduction == "mean" else grid.amax(dim=1)  # (B, h, w)
        low = saliency.amin(dim=(1, 2), keepdim=True)
        spread = saliency.amax(dim=(1, 2), keepdim=True) - low
        # A flat map, up to fp noise, has nothing salient: it normalises to all-zeros.
        flat = spread <= 1e-6 * (saliency.abs().amax(dim=(1, 2), keepdim=True) + 1e-8)
        normalised = torch.where(flat, torch.zeros_like(saliency), (saliency - low) / spread.clamp_min(1e-12))
        logits = torch.logit(normalised.clamp(EPS, 1.0 - EPS))
        return logits.unsqueeze(1).expand(-1, self._num_classes, -1, -1)

    @property
    def num_classes(self) -> int:
        return self._num_classes


if DECODER_NAME not in decoder_registry:
    decoder_registry.register(
        DECODER_NAME,
        AttentionMapDecoder,
        params=AttentionMapParams,
        metadata={
            "description": "CLS attention as the detection heatmap, no parameters "
            "(rung 0 of the decoder ladder, benchmark-private)"
        },
    )
