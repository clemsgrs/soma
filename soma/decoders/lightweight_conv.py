"""Lightweight dense decoders — a linear-probe baseline and a small conv decoder.

These keep with the design's "frozen encoder + lightweight decoder" recipe
(Vitoria et al.): the heavy representation work is done by the frozen foundation
model, and the decoder is deliberately small. Two are registered to make the
``decoder`` axis genuinely swappable:

- ``linear``: a single 1x1 conv at grid resolution (the simplest probe); all
  upsampling to the mask is left to the head's interpolation.
- ``lightweight_conv``: a 1x1 projection + a few bilinear-upsample/conv blocks +
  a 1x1 classifier (learned upsampling, fewer interpolation artifacts).
"""

from __future__ import annotations

from torch import Tensor, nn

from soma.decoders.base import Decoder
from soma.decoders.registry import decoder_registry


def _group_norm(channels: int, num_groups: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= ``num_groups`` that divides ``channels``."""
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LinearDecoder(Decoder):
    """1x1 conv at grid resolution; the head upsamples logits to the mask size.

    The minimal linear-probe baseline for the frozen-encoder dense setting.
    """

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self._num_classes = int(num_classes)
        self.classifier = nn.Conv2d(input_dim, num_classes, kernel_size=1)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 4:
            raise ValueError(f"decoder expects a (B, d, h, w) grid, got shape {tuple(X.shape)}")
        return self.classifier(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes


class LightweightConvDecoder(Decoder):
    """1x1 projection -> ``num_upsample_blocks`` bilinear-upsample/conv blocks -> 1x1 classifier.

    Output grid is ``(h * 2**k, w * 2**k)`` for ``k = num_upsample_blocks``; the
    head interpolates that to the mask's target size and crops. ``num_upsample_blocks=0``
    degenerates to a conv head at grid resolution.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
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
        self._num_classes = int(num_classes)

        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            _group_norm(hidden_dim, num_groups),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for _ in range(num_upsample_blocks):
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                _group_norm(hidden_dim, num_groups),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 4:
            raise ValueError(f"decoder expects a (B, d, h, w) grid, got shape {tuple(X.shape)}")
        return self.classifier(self.blocks(self.proj(X)))

    @property
    def num_classes(self) -> int:
        return self._num_classes


decoder_registry.register("linear", LinearDecoder)
decoder_registry.register("lightweight_conv", LightweightConvDecoder)
