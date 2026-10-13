"""Tests for resolving fixed MIL recipes from encoder metadata."""

from __future__ import annotations

import pytest

from soma.config import Component, FeaturesConfig
from soma.plan import check
from tests.experiment_configs import experiment
from soma.models.encoders import resolve_aggregator


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

    def __contains__(self, name: str) -> bool:
        return name == "prism"

    def require(self, name: str):
        return object


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
    monkeypatch.setattr("soma.models.encoders._slide2vec_encoders", _TileMetadataRegistry())
    recipe = Component("abmil", {"hidden_dim": 128})

    resolved = resolve_aggregator("tile-encoder", recipe)

    assert resolved is recipe


def test_slide_encoder_omits_aggregator_in_valid_slide_pipeline(monkeypatch):
    monkeypatch.setattr("soma.models.encoders._slide2vec_encoders", _SlideMetadataRegistry())
    recipe = Component("abmil", {"hidden_dim": 128})

    resolved = resolve_aggregator("prism", recipe)
    config = experiment(features=FeaturesConfig(encoder="prism"), aggregator=resolved)

    assert resolved is None
    check(config)


def test_patient_encoder_rejected_for_slide_benchmark(monkeypatch):
    monkeypatch.setattr("soma.models.encoders._slide2vec_encoders", _PatientMetadataRegistry())
    recipe = Component("abmil")

    with pytest.raises(ValueError) as error:
        resolve_aggregator("moozy", recipe)

    assert str(error.value) == (
        "Slide-level benchmarks cannot consume patient-level representations "
        "from encoder 'moozy'; choose a tile- or slide-level encoder."
    )


def test_unknown_encoder_lists_available_names(monkeypatch):
    monkeypatch.setattr("soma.models.encoders._slide2vec_encoders", _UnknownMetadataRegistry())
    recipe = Component("abmil")

    with pytest.raises(ValueError) as error:
        resolve_aggregator("missing-encoder", recipe)

    assert str(error.value) == (
        "Unknown encoder name 'missing-encoder'. "
        "Available encoders: slide-a, tile-z"
    )
