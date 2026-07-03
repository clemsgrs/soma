"""Heavy dense decoder — pyramid-pooling context fusion + learned upsampling (rung 3).

The third rung of the decoder-complexity ladder (the linear probe and
``lightweight_conv`` are rungs 1-2). It answers the ladder's "does more machinery
help?" question with a UPerNet/DPT-lite decoder: real multi-scale context fusion (a
pyramid-pooling module) followed by **learned** upsampling (``ConvTranspose2d`` blocks),
where ``lightweight_conv`` uses only parameter-free bilinear upsampling.

Fairness invariant (Vitoria et al., design §2.4): like ``lightweight_conv``, it opens
with a single ``1x1`` ``d->D`` projection (``proj``) so **every trainable parameter
below the projection is independent of the encoder's channel dim ``d``**. Two encoders
with different ``d`` therefore train decoders with identical downstream capacity — the
projection is the only ``d``-dependent module, so the ladder isolates decoder machinery
from encoder width. This holds for the multi-FM ensemble rung too: a composite grid of
width ``Σdᵢ`` is just a wider ``d`` into the same projection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.decoders.base import Decoder
from soma.decoders.lightweight_conv import _group_norm
from soma.decoders.registry import decoder_registry


def _pooled_group_norm(channels: int, num_groups: int) -> nn.GroupNorm:
    """GroupNorm that keeps >= 2 channels per group.

    The PPM's global branch (``pool_scale=1``) produces a ``1x1`` feature map, so a group
    holding a single channel would have one element and GroupNorm's training-time variance
    is undefined (``nn.GroupNorm`` raises). Capping the group count at ``channels // 2``
    guarantees >= 2 values per group even at ``1x1``, while staying a divisor of
    ``channels`` — otherwise it matches ``_group_norm``.
    """
    groups = min(num_groups, max(1, channels // 2))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class HeavyConvDecoder(Decoder):
    """``d->D`` projection -> pyramid-pooling fusion -> learned-upsample blocks -> 1x1 classifier.

    The projected ``(B, D, h, w)`` features are enriched with global/multi-scale context
    by a pyramid-pooling module (adaptive-avg-pool each feature to ``s x s`` for
    ``s in pool_scales``, ``1x1`` conv, upsample back to ``(h, w)``), concatenated with
    the un-pooled features, and fused by a ``3x3`` conv. ``num_upsample_blocks``
    transposed-conv blocks then upsample by ``2`` each (learned, unlike the lightweight
    rung's bilinear), so the output grid is ``(h * 2**k, w * 2**k)`` for
    ``k = num_upsample_blocks`` — the same contract the head interpolates+crops to the
    mask. ``num_upsample_blocks=0`` degenerates to a context-fused conv head at grid
    resolution.

    Only ``proj`` depends on ``input_dim``; ``ppm``, ``fuse``, ``upsample``, and
    ``classifier`` are sized purely by ``hidden_dim``/``num_classes`` — the ``d``-invariance
    the ladder's fairness rests on.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        num_upsample_blocks: int = 2,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if num_upsample_blocks < 0:
            raise ValueError(f"num_upsample_blocks must be >= 0, got {num_upsample_blocks}")
        pool_scales = tuple(int(s) for s in pool_scales)
        if not pool_scales or any(s < 1 for s in pool_scales):
            raise ValueError(f"pool_scales must be non-empty positive ints, got {pool_scales}")
        self._num_classes = int(num_classes)

        # d -> D projection: the ONLY input_dim-dependent module (Vitoria fairness).
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            _group_norm(hidden_dim, num_groups),
            nn.ReLU(inplace=True),
        )
        # Pyramid-pooling module: one context branch per scale (UPerNet's PPM head).
        self.ppm = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                _pooled_group_norm(hidden_dim, num_groups),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        )
        # Fuse the un-pooled features + every upsampled context branch back to hidden_dim.
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * (1 + len(pool_scales)), hidden_dim, kernel_size=3, padding=1),
            _group_norm(hidden_dim, num_groups),
            nn.ReLU(inplace=True),
        )
        # Learned upsampling: each block doubles resolution (transposed conv, not bilinear).
        blocks: list[nn.Module] = []
        for _ in range(num_upsample_blocks):
            blocks += [
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                _group_norm(hidden_dim, num_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                _group_norm(hidden_dim, num_groups),
                nn.ReLU(inplace=True),
            ]
        self.upsample = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 4:
            raise ValueError(f"decoder expects a (B, d, h, w) grid, got shape {tuple(X.shape)}")
        feat = self.proj(X)
        grid_h, grid_w = feat.shape[-2], feat.shape[-1]
        context = [feat]
        for branch in self.ppm:
            pooled = branch(feat)
            context.append(
                F.interpolate(pooled, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            )
        fused = self.fuse(torch.cat(context, dim=1))
        return self.classifier(self.upsample(fused))

    @property
    def num_classes(self) -> int:
        return self._num_classes


decoder_registry.register("heavy_conv", HeavyConvDecoder)
