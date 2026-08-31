"""Tests for resolving fixed MIL recipes from encoder metadata."""

from __future__ import annotations

import pytest

from soma.config import AggregatorConfig, EncoderConfig, PipelineConfig, TaskConfig
from soma.encoders import resolve_aggregator


class _TileMetadataRegistry:
    def info(self, name: str) -> dict[str, str]:
        assert name == "tile-encoder"
        return {"level": "tile"}

    def require(self, name: str):
        raise AssertionError(f"must not resolve encoder class or weights for {name}")


class _SlideMetadataRegistry:
    def info(self, name: str) -> dict[str, str]:
        assert name == "prism"
        return {"level": "slide"}


class _PatientMetadataRegistry:
    def info(self, name: str) -> dict[str, str]:
        assert name == "moozy"
        return {"level": "patient"}


class _UnknownMetadataRegistry:
    def info(self, name: str) -> dict[str, str]:
        assert name == "missing-encoder"
        raise KeyError(name)

    def names(self) -> list[str]:
        return ["tile-z", "slide-a"]


def test_tile_encoder_returns_exact_recipe_without_loading_weights(monkeypatch):
    monkeypatch.setattr("soma.encoders.encoder_registry", _TileMetadataRegistry())
    recipe = AggregatorConfig(name="abmil", params={"hidden_dim": 128})

    resolved = resolve_aggregator("tile-encoder", recipe)

    assert resolved is recipe


def test_slide_encoder_omits_aggregator_in_valid_slide_pipeline(monkeypatch):
    monkeypatch.setattr("soma.encoders.encoder_registry", _SlideMetadataRegistry())
    recipe = AggregatorConfig(name="abmil", params={"hidden_dim": 128})

    resolved = resolve_aggregator("prism", recipe)
    config = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="slide",
        encoder=EncoderConfig(name="prism"),
        aggregator=resolved,
        task=TaskConfig(name="binary_classification"),
    )

    assert config.aggregator is None


def test_patient_encoder_rejected_for_slide_benchmark(monkeypatch):
    monkeypatch.setattr("soma.encoders.encoder_registry", _PatientMetadataRegistry())
    recipe = AggregatorConfig(name="abmil")

    with pytest.raises(
        ValueError,
        match=(
            "Slide-level benchmarks cannot consume patient-level representations.*"
            "choose a tile- or slide-level encoder"
        ),
    ):
        resolve_aggregator("moozy", recipe)


def test_unknown_encoder_lists_available_names(monkeypatch):
    monkeypatch.setattr("soma.encoders.encoder_registry", _UnknownMetadataRegistry())
    recipe = AggregatorConfig(name="abmil")

    with pytest.raises(ValueError) as error:
        resolve_aggregator("missing-encoder", recipe)

    assert str(error.value) == (
        "Unknown encoder name 'missing-encoder'. "
        "Available encoders: slide-a, tile-z"
    )
