"""Tests for soma.encoders.base — Encoder, TileEncoder, SlideEncoder."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from torch import Tensor

from soma.encoders.base import Encoder, SlideEncoder, TileEncoder, TimmTileEncoder


class MockTileEncoder(TileEncoder):
    def __init__(self, dim: int = 64):
        self._dim = dim
        self._device = torch.device("cpu")

    def get_transform(self) -> Callable:
        def _transform(img):
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        return _transform

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return torch.ones(batch.shape[0], self._dim)

    @property
    def encode_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> MockTileEncoder:
        self._device = torch.device(device)
        return self


class MockSlideEncoder(SlideEncoder):
    def __init__(self, dim: int = 32):
        self._dim = dim
        self._device = torch.device("cpu")

    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        return tile_features.mean(dim=0)[: self._dim]

    @property
    def encode_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> MockSlideEncoder:
        self._device = torch.device(device)
        return self


class TestEncoderABC:
    def test_cannot_instantiate_encoder_abc(self):
        with pytest.raises(TypeError):
            Encoder()  # type: ignore[abstract]

    def test_cannot_instantiate_tile_encoder_abc(self):
        with pytest.raises(TypeError):
            TileEncoder()  # type: ignore[abstract]

    def test_cannot_instantiate_slide_encoder_abc(self):
        with pytest.raises(TypeError):
            SlideEncoder()  # type: ignore[abstract]


class TestTileEncoder:
    def test_encode_tiles_shape(self):
        enc = MockTileEncoder(dim=128)
        out = enc.encode_tiles(torch.randn(4, 3, 224, 224))
        assert out.shape == (4, 128)

    def test_get_transform_callable(self):
        assert callable(MockTileEncoder().get_transform())

    def test_device_default_cpu(self):
        assert MockTileEncoder().device == torch.device("cpu")


class TestSlideEncoder:
    def test_encode_slide_shape(self):
        enc = MockSlideEncoder(dim=16)
        out = enc.encode_slide(
            torch.randn(8, 16),
            torch.randint(0, 10, (8, 2)),
            tile_size_lv0=256,
        )
        assert out.shape == (16,)

    def test_encode_slide_can_ignore_optional_coordinates(self):
        enc = MockSlideEncoder(dim=16)
        out = enc.encode_slide(torch.randn(8, 16))
        assert out.shape == (16,)

    def test_prepare_coordinates_identity(self):
        enc = MockSlideEncoder()
        coords = torch.tensor([[1, 2], [3, 4]])
        prepared = enc.prepare_coordinates(
            coords,
            base_spacing_um=0.25,
            target_spacing_um=0.5,
        )
        assert torch.equal(prepared, coords)


class TestTimmTileEncoder:
    @pytest.fixture()
    def encoder(self) -> TimmTileEncoder:
        return TimmTileEncoder("resnet18", pretrained=False)

    def test_encode_dim(self, encoder: TimmTileEncoder):
        assert encoder.encode_dim == 512

    def test_encode_tiles_shape(self, encoder: TimmTileEncoder):
        out = encoder.encode_tiles(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 512)

    def test_get_transform_returns_callable(self, encoder: TimmTileEncoder):
        assert callable(encoder.get_transform())

    def test_to_returns_self(self, encoder: TimmTileEncoder):
        assert encoder.to("cpu") is encoder

    def test_model_in_eval_mode(self, encoder: TimmTileEncoder):
        assert not encoder._model.training

    def test_pretrained_false_override(self):
        enc = TimmTileEncoder("resnet18", pretrained=False)
        assert enc.encode_dim == 512
