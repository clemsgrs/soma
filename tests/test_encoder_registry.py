"""Tests for soma.encoders.registry — encoder registration with metadata."""

from __future__ import annotations

import torch
import pytest

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.registry import encoder_registry, register_encoder


class _DummyTileEncoder(TileEncoder):
    def get_transform(self):
        return lambda x: x

    def encode_tiles(self, batch):
        return batch

    @property
    def encode_dim(self):
        return 64

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self


class _DummySlideEncoder(SlideEncoder):
    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0: int | None = None):
        return tile_features.mean(dim=0)

    @property
    def encode_dim(self):
        return 64

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, device):
        return self


def test_register_and_retrieve():
    @register_encoder(
        "test-model-a",
        encode_dim=512,
        input_size=224,
        recommended_spacing_um=0.5,
    )
    class ModelA(_DummyTileEncoder):
        pass

    assert encoder_registry.get("test-model-a") is ModelA


def test_tile_metadata_stored():
    @register_encoder(
        "test-model-b",
        encode_dim=1024,
        input_size=256,
        recommended_spacing_um=1.0,
        precision="fp32",
        source="org/model-b",
    )
    class ModelB(_DummyTileEncoder):
        pass

    info = encoder_registry.info("test-model-b")
    assert info["level"] == "tile"
    assert info["input_size"] == 256
    assert info["recommended_tile_size_px"] == 256
    assert info["recommended_spacing_um"] == 1.0
    assert info["precision"] == "fp32"
    assert info["source"] == "org/model-b"


def test_slide_metadata_stored():
    @register_encoder(
        "test-slide-model",
        level="slide",
        tile_encoder="test-model-a",
        encode_dim=256,
        recommended_tile_size_px=224,
        recommended_spacing_um=0.5,
        source="org/slide-model",
    )
    class ModelSlide(_DummySlideEncoder):
        pass

    info = encoder_registry.info("test-slide-model")
    assert info["level"] == "slide"
    assert info["tile_encoder"] == "test-model-a"
    assert info["recommended_tile_size_px"] == 224


def test_default_precision_fp16():
    @register_encoder(
        "test-model-c",
        encode_dim=768,
        input_size=224,
        recommended_spacing_um=0.5,
    )
    class ModelC(_DummyTileEncoder):
        pass

    assert encoder_registry.info("test-model-c")["precision"] == "fp16"


def test_duplicate_name_raises():
    @register_encoder(
        "test-model-dup",
        encode_dim=512,
        input_size=224,
        recommended_spacing_um=0.5,
    )
    class ModelD(_DummyTileEncoder):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_encoder(
            "test-model-dup",
            encode_dim=512,
            input_size=224,
            recommended_spacing_um=0.5,
        )
        class ModelD2(_DummyTileEncoder):
            pass


def test_unknown_name_raises():
    with pytest.raises(KeyError, match="not found"):
        encoder_registry.get("nonexistent-model-xyz")


def test_list_includes_registered():
    @register_encoder(
        "test-model-listed",
        encode_dim=256,
        input_size=224,
        recommended_spacing_um=0.5,
    )
    class ModelE(_DummyTileEncoder):
        pass

    assert "test-model-listed" in encoder_registry.list()


def test_multiple_recommended_spacings():
    @register_encoder(
        "test-model-multi-spacing",
        encode_dim=512,
        input_size=224,
        recommended_spacing_um=[0.5, 1.0],
    )
    class ModelF(_DummyTileEncoder):
        pass

    assert encoder_registry.info("test-model-multi-spacing")["recommended_spacing_um"] == [
        0.5,
        1.0,
    ]
