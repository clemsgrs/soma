"""Tests for public component discovery helpers."""

import pytest

from slide2vec.encoders.registry import encoder_registry

from soma.aggregators import list_aggregators
from soma.aggregators.registry import aggregator_registry
from soma.encoders import list_models
from soma.tasks import list_task_heads
from soma.tasks.registry import task_registry


def test_list_models_matches_encoder_registry():
    assert list_models() == sorted(encoder_registry.names())


def test_list_models_filters_by_level():
    for level in ("tile", "slide", "patient"):
        expected = sorted(
            name
            for name in encoder_registry.names()
            if encoder_registry.info(name)["level"] == level
        )
        assert list_models(level) == expected


def test_list_models_rejects_invalid_level():
    with pytest.raises(ValueError, match="tile, slide, patient"):
        list_models("bogus")


def test_list_aggregators_returns_sorted_names():
    assert list_aggregators() == sorted(aggregator_registry.list())


def test_list_task_heads_returns_sorted_names():
    assert list_task_heads() == sorted(task_registry.list())
