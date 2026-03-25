"""Phikon and Phikon-v2 encoder implementations.

Both require the ``transformers`` package.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from soma.encoders.base import Encoder
from soma.encoders.registry import register_encoder


class _PhikonBase(Encoder):
    """Base for Phikon models using HuggingFace transformers."""

    _encode_dim: int

    def __init__(self, model_name: str, *, token: str | None = None):
        from transformers import AutoImageProcessor, AutoModel

        kwargs = {}
        if token is not None:
            kwargs["token"] = token
        self._model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        self._processor = AutoImageProcessor.from_pretrained(
            model_name, use_fast=True, **kwargs
        )
        self._device = torch.device("cpu")

    def get_transform(self) -> Callable:
        processor = self._processor

        def _transform(img):
            inputs = processor(images=img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _transform

    def encode(self, batch: Tensor) -> Tensor:
        output = self._model(pixel_values=batch)
        return output.last_hidden_state[:, 0, :]  # CLS token

    @property
    def encode_dim(self) -> int:
        return self._encode_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> _PhikonBase:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self


@register_encoder(
    "phikon",
    encode_dim=768,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp32",
    source="owkin/phikon",
)
class Phikon(_PhikonBase):
    _encode_dim = 768

    def __init__(self, *, token: str | None = None):
        super().__init__("owkin/phikon", token=token)


@register_encoder(
    "phikonv2",
    encode_dim=1024,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp32",
    source="owkin/phikon-v2",
)
class PhikonV2(_PhikonBase):
    _encode_dim = 1024

    def __init__(self, *, token: str | None = None):
        super().__init__("owkin/phikon-v2", token=token)
