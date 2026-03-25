"""Tests for soma.encoders.base — Encoder ABC + TimmEncoder."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from torch import Tensor

from soma.encoders.base import Encoder, TimmEncoder


# ---------------------------------------------------------------------------
# MockEncoder — minimal concrete implementation of the ABC
# ---------------------------------------------------------------------------


class MockEncoder(Encoder):
    """Trivial encoder that returns a learned-free linear projection."""

    def __init__(self, input_channels: int = 3, dim: int = 64):
        self._dim = dim
        self._device = torch.device("cpu")

    def get_transform(self) -> Callable:
        def _transform(img):
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

        return _transform

    def encode(self, batch: Tensor) -> Tensor:
        b = batch.shape[0]
        return torch.ones(b, self._dim)

    @property
    def encode_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> MockEncoder:
        self._device = torch.device(device)
        return self


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestEncoderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Encoder()  # type: ignore[abstract]

    def test_mock_encoder_encode_shape(self):
        enc = MockEncoder(dim=128)
        batch = torch.randn(4, 3, 224, 224)
        out = enc.encode(batch)
        assert out.shape == (4, 128)

    def test_mock_encoder_encode_dim(self):
        enc = MockEncoder(dim=64)
        assert enc.encode_dim == 64

    def test_mock_encoder_device_default_cpu(self):
        enc = MockEncoder()
        assert enc.device == torch.device("cpu")

    def test_mock_encoder_to_returns_self(self):
        enc = MockEncoder()
        result = enc.to("cpu")
        assert result is enc

    def test_mock_encoder_get_transform_callable(self):
        enc = MockEncoder()
        transform = enc.get_transform()
        assert callable(transform)


# ---------------------------------------------------------------------------
# TimmEncoder tests (using a tiny model from timm)
# ---------------------------------------------------------------------------


class TestTimmEncoder:
    """Test TimmEncoder with resnet18 (small, always available in timm)."""

    @pytest.fixture()
    def encoder(self) -> TimmEncoder:
        return TimmEncoder("resnet18")

    def test_encode_dim(self, encoder: TimmEncoder):
        assert encoder.encode_dim == 512  # resnet18 feature dim

    def test_encode_shape(self, encoder: TimmEncoder):
        batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder.encode(batch)
        assert out.shape == (2, 512)

    def test_get_transform_returns_callable(self, encoder: TimmEncoder):
        transform = encoder.get_transform()
        assert callable(transform)

    def test_device_default_cpu(self, encoder: TimmEncoder):
        assert encoder.device == torch.device("cpu")

    def test_to_returns_self(self, encoder: TimmEncoder):
        result = encoder.to("cpu")
        assert result is encoder

    def test_to_updates_device(self, encoder: TimmEncoder):
        encoder.to("cpu")
        assert encoder.device == torch.device("cpu")

    def test_model_in_eval_mode(self, encoder: TimmEncoder):
        assert not encoder._model.training

    def test_pretrained_false_override(self):
        enc = TimmEncoder("resnet18", pretrained=False)
        assert enc.encode_dim == 512
