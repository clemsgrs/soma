"""Virchow and Virchow2 encoder implementations."""

from __future__ import annotations

import timm.layers
import torch
from torch import Tensor

from soma.encoders.base import TimmTileEncoder
from soma.encoders.registry import register_encoder


class _VirchowBase(TimmTileEncoder):
    """Base for Virchow models that concat CLS + mean-pooled patch tokens."""

    _num_prefix_tokens: int = 1  # Override in subclass if needed

    def encode_tiles(self, batch: Tensor) -> Tensor:
        output = self._model.forward_features(batch)
        cls_token = output[:, 0]
        patch_tokens = output[:, self._num_prefix_tokens :]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


@register_encoder(
    "virchow",
    encode_dim=2560,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="paige-ai/Virchow",
)
class Virchow(_VirchowBase):
    _num_prefix_tokens = 1

    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:paige-ai/Virchow",
            token=token,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )


@register_encoder(
    "virchow2",
    encode_dim=2560,
    input_size=224,
    recommended_spacing_um=[0.5, 1.0, 2.0],
    precision="fp16",
    source="paige-ai/Virchow2",
)
class Virchow2(_VirchowBase):
    _num_prefix_tokens = 5  # 1 CLS + 4 register tokens

    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:paige-ai/Virchow2",
            token=token,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
