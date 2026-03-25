"""Encoder ABC and TimmEncoder base class for tile-level feature extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class Encoder(ABC):
    """Base class for tile-level encoders."""

    @abstractmethod
    def get_transform(self) -> Callable:
        """Image transform pipeline (PIL Image -> Tensor)."""
        ...

    @abstractmethod
    def encode(self, batch: Tensor) -> Tensor:
        """Encode a batch of images. (B, C, H, W) -> (B, D)."""
        ...

    @property
    @abstractmethod
    def encode_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current device of the encoder."""
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> Encoder:
        """Move encoder to the given device. Returns self."""
        ...


class TimmEncoder(Encoder):
    """Convenience base for timm-backed encoders.

    Handles model creation, default transforms via ``resolve_data_config``,
    and the standard encode path. Most foundation models subclass this
    with only ``__init__`` overrides (5-10 lines).
    """

    def __init__(self, model_name: str, *, token: str | None = None, **timm_kwargs):
        import timm

        defaults = {"pretrained": True, "num_classes": 0}
        defaults.update(timm_kwargs)
        if token is not None:
            defaults["hf_token"] = token
        self._model = timm.create_model(model_name, **defaults).eval()
        self._device = torch.device("cpu")

    def get_transform(self) -> Callable:
        from timm.data import create_transform, resolve_data_config

        data_config = resolve_data_config(self._model.pretrained_cfg, model=self._model)
        return create_transform(**data_config)

    def encode(self, batch: Tensor) -> Tensor:
        return self._model(batch)

    @property
    def encode_dim(self) -> int:
        return self._model.num_features

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> TimmEncoder:
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self
