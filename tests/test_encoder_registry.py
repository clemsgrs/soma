"""Tests for soma.encoders.registry — encoder registration with metadata."""

from __future__ import annotations

import torch
import pytest

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.registry import (
    encoder_registry,
    register_encoder,
    resolve_encoder_output,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)


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
        output_variants={"default": {"encode_dim": 512}},
        default_output_variant="default",
        input_size=224,
        supported_spacing_um=0.5,
    )
    class ModelA(_DummyTileEncoder):
        pass

    assert encoder_registry.get("test-model-a") is ModelA


def test_tile_metadata_stored():
    @register_encoder(
        "test-model-b",
        output_variants={"default": {"encode_dim": 1024}},
        default_output_variant="default",
        input_size=256,
        supported_spacing_um=1.0,
        precision="fp32",
        source="org/model-b",
    )
    class ModelB(_DummyTileEncoder):
        pass

    info = encoder_registry.info("test-model-b")
    assert info["level"] == "tile"
    assert info["input_size"] == 256
    assert info["default_output_variant"] == "default"
    assert info["output_variants"]["default"]["encode_dim"] == 1024
    assert info["supported_spacing_um"] == 1.0
    assert info["precision"] == "fp32"
    assert info["source"] == "org/model-b"


def test_slide_metadata_stored():
    @register_encoder(
        "test-slide-model",
        level="slide",
        tile_encoder="test-model-a",
        tile_encoder_output_variant="default",
        output_variants={"default": {"encode_dim": 256}},
        default_output_variant="default",
        supported_spacing_um=0.5,
        source="org/slide-model",
    )
    class ModelSlide(_DummySlideEncoder):
        pass

    info = encoder_registry.info("test-slide-model")
    assert info["level"] == "slide"
    assert info["tile_encoder"] == "test-model-a"
    assert info["tile_encoder_output_variant"] == "default"


def test_resolve_tile_preprocessing_metadata_for_tile_encoder():
    @register_encoder(
        "test-model-resolve-tile",
        output_variants={"default": {"encode_dim": 256}},
        default_output_variant="default",
        input_size=384,
        supported_spacing_um=0.75,
    )
    class ModelResolveTile(_DummyTileEncoder):
        pass

    requirements = resolve_preprocessing_requirements("test-model-resolve-tile")
    assert requirements["tile_size_px"] == 384
    assert requirements["spacing_um"] == 0.75


def test_resolve_tile_preprocessing_metadata_for_slide_encoder():
    @register_encoder(
        "test-model-resolve-base",
        output_variants={"default": {"encode_dim": 128}},
        default_output_variant="default",
        input_size=320,
        supported_spacing_um=0.5,
    )
    class ModelResolveBase(_DummyTileEncoder):
        pass

    @register_encoder(
        "test-model-resolve-slide",
        level="slide",
        tile_encoder="test-model-resolve-base",
        tile_encoder_output_variant="default",
        output_variants={"default": {"encode_dim": 64}},
        default_output_variant="default",
        supported_spacing_um=0.5,
    )
    class ModelResolveSlide(_DummySlideEncoder):
        pass

    requirements = resolve_preprocessing_requirements("test-model-resolve-slide")
    assert requirements["tile_size_px"] == 320
    assert requirements["spacing_um"] == 0.5


def test_default_precision_fp16():
    @register_encoder(
        "test-model-c",
        output_variants={"default": {"encode_dim": 768}},
        default_output_variant="default",
        input_size=224,
        supported_spacing_um=0.5,
    )
    class ModelC(_DummyTileEncoder):
        pass

    assert encoder_registry.info("test-model-c")["precision"] == "fp16"


def test_duplicate_name_raises():
    @register_encoder(
        "test-model-dup",
        output_variants={"default": {"encode_dim": 512}},
        default_output_variant="default",
        input_size=224,
        supported_spacing_um=0.5,
    )
    class ModelD(_DummyTileEncoder):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_encoder(
            "test-model-dup",
            output_variants={"default": {"encode_dim": 512}},
            default_output_variant="default",
            input_size=224,
            supported_spacing_um=0.5,
        )
        class ModelD2(_DummyTileEncoder):
            pass


def test_unknown_name_raises():
    with pytest.raises(KeyError, match="not found"):
        encoder_registry.get("nonexistent-model-xyz")


def test_list_includes_registered():
    @register_encoder(
        "test-model-listed",
        output_variants={"default": {"encode_dim": 256}},
        default_output_variant="default",
        input_size=224,
        supported_spacing_um=0.5,
    )
    class ModelE(_DummyTileEncoder):
        pass

    assert "test-model-listed" in encoder_registry.list()


def test_multiple_recommended_spacings():
    @register_encoder(
        "test-model-multi-spacing",
        output_variants={"default": {"encode_dim": 512}},
        default_output_variant="default",
        input_size=224,
        supported_spacing_um=[0.5, 1.0],
    )
    class ModelF(_DummyTileEncoder):
        pass

    assert encoder_registry.info("test-model-multi-spacing")["supported_spacing_um"] == [
        0.5,
        1.0,
    ]


def test_resolve_encoder_output_uses_default_variant():
    @register_encoder(
        "test-model-default-variant",
        output_variants={
            "cls": {"encode_dim": 32},
            "cls_patch_mean": {"encode_dim": 64},
        },
        default_output_variant="cls_patch_mean",
        input_size=224,
        supported_spacing_um=0.5,
    )
    class ModelDefaultVariant(_DummyTileEncoder):
        pass

    resolved = resolve_encoder_output("test-model-default-variant")
    assert resolved["output_variant"] == "cls_patch_mean"
    assert resolved["encode_dim"] == 64


def test_resolve_encoder_output_accepts_explicit_variant():
    resolved = resolve_encoder_output(
        "test-model-default-variant",
        requested_output_variant="cls",
    )
    assert resolved["output_variant"] == "cls"
    assert resolved["encode_dim"] == 32


def test_resolve_preprocessing_requirements_requires_level_metadata():
    with pytest.raises(ValueError, match="level metadata"):
        resolve_preprocessing_requirements(
            "strict-missing-level",
            metadata={
                "input_size": 224,
                "supported_spacing_um": 0.5,
                "output_variants": {"default": {"encode_dim": 1}},
                "default_output_variant": "default",
            },
        )


def test_resolve_encoder_output_requires_level_metadata():
    with pytest.raises(ValueError, match="level metadata"):
        resolve_encoder_output(
            "strict-missing-level-output",
            metadata={
                "output_variants": {"default": {"encode_dim": 1}},
                "default_output_variant": "default",
            },
        )


def test_resolve_tile_dependency_output_requires_fixed_tile_output_variant_for_slide():
    with pytest.raises(ValueError, match="tile_encoder_output_variant"):
        resolve_tile_dependency_output(
            "strict-slide-missing-tile-output",
            metadata={
                "level": "slide",
                "tile_encoder": "test-model-a",
                "output_variants": {"default": {"encode_dim": 1}},
                "default_output_variant": "default",
                "precision": "fp16",
                "source": "test",
            },
        )
