"""Midnight encoder implementation.

Requires the ``transformers`` package.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2

from soma.encoders.base import Encoder
from soma.encoders.registry import register_encoder


@register_encoder(
    "midnight",
    encode_dim=3072,
    input_size=224,
    recommended_spacing_um=[0.25, 0.5, 1.0, 2.0],
    precision="fp16",
    source="kaiko-ai/midnight",
)
class Midnight(Encoder):
    def __init__(self, *, token: str | None = None):
        from transformers import AutoModel

        kwargs = {}
        if token is not None:
            kwargs["token"] = token
        self._model = AutoModel.from_pretrained("kaiko-ai/midnight", **kwargs).eval()
        self._device = torch.device("cpu")

    def get_transform(self) -> Callable:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def encode(self, batch: Tensor) -> Tensor:
        output = self._model(batch).last_hidden_state
        cls_token = output[:, 0, :]
        patch_tokens = output[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_tokens], dim=-1)

    @property
    def encode_dim(self) -> int:
        return 3072

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> Midnight:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
