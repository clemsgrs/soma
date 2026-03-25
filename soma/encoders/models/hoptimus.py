"""H-Optimus-0, H-Optimus-1, and H0-mini encoder implementations."""

from __future__ import annotations

from typing import Callable

import timm.layers
import torch
from torch import Tensor
from torchvision.transforms import v2

from soma.encoders.base import TimmEncoder
from soma.encoders.registry import register_encoder

# Shared normalization for H-Optimus models
_HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
_HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def _hoptimus_transform(input_size: int = 224) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(input_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(input_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_HOPTIMUS_MEAN, std=_HOPTIMUS_STD),
    ])


@register_encoder(
    "h-optimus-0",
    encode_dim=1536,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H-optimus-0",
)
class HOptimus0(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-0",
            token=token,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    def get_transform(self) -> Callable:
        return _hoptimus_transform()


@register_encoder(
    "h-optimus-1",
    encode_dim=1536,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H-optimus-1",
)
class HOptimus1(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-1",
            token=token,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    def get_transform(self) -> Callable:
        return _hoptimus_transform()


@register_encoder(
    "h0-mini",
    encode_dim=1536,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="bioptimus/H0-mini",
)
class H0Mini(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:bioptimus/H0-mini",
            token=token,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def get_transform(self) -> Callable:
        return _hoptimus_transform()

    def encode(self, batch: Tensor) -> Tensor:
        output = self._model.forward_features(batch)
        cls_token = output[:, 0]
        patch_tokens = output[:, self._model.num_prefix_tokens :]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
