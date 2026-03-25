"""Tests for soma.aggregators — base classes, registry, and simple pooling."""

from __future__ import annotations

import torch
import pytest

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.registry import aggregator_registry
from soma.aggregators.pooling import MeanPool, MaxPool


# ---------------------------------------------------------------------------
# AggregatorOutput
# ---------------------------------------------------------------------------


class TestAggregatorOutput:
    def test_fields(self):
        bag_rep = torch.randn(2, 8)
        out = AggregatorOutput(bag_representation=bag_rep)
        assert torch.equal(out.bag_representation, bag_rep)
        assert out.tile_attention is None

    def test_with_attention(self):
        bag_rep = torch.randn(2, 8)
        attn = torch.randn(2, 10)
        out = AggregatorOutput(bag_representation=bag_rep, tile_attention=attn)
        assert torch.equal(out.tile_attention, attn)


# ---------------------------------------------------------------------------
# MeanPool
# ---------------------------------------------------------------------------


class TestMeanPool:
    def test_shape(self):
        pool = MeanPool(input_dim=8)
        X = torch.randn(2, 10, 8)
        out = pool(X)
        assert out.bag_representation.shape == (2, 8)
        assert out.tile_attention is None

    def test_output_dim(self):
        pool = MeanPool(input_dim=16)
        assert pool.output_dim == 16

    def test_masked_mean(self):
        """Masked tiles should be excluded from the mean."""
        pool = MeanPool(input_dim=2)
        # Bag with 3 tiles, but tile 2 is masked out
        X = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])  # (1, 3, 2)
        mask = torch.tensor([[True, True, False]])  # (1, 3)
        out = pool(X, mask=mask)
        # Mean of tiles 0 and 1 only: (1+3)/2=2.0, (2+4)/2=3.0
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(out.bag_representation, expected)

    def test_no_mask_is_global_mean(self):
        pool = MeanPool(input_dim=2)
        X = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        out = pool(X)
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(out.bag_representation, expected)


# ---------------------------------------------------------------------------
# MaxPool
# ---------------------------------------------------------------------------


class TestMaxPool:
    def test_shape(self):
        pool = MaxPool(input_dim=8)
        X = torch.randn(2, 10, 8)
        out = pool(X)
        assert out.bag_representation.shape == (2, 8)
        assert out.tile_attention is None

    def test_output_dim(self):
        pool = MaxPool(input_dim=16)
        assert pool.output_dim == 16

    def test_masked_max(self):
        """Masked tiles should never be selected by max."""
        pool = MaxPool(input_dim=2)
        # Tile 2 has highest values but is masked out
        X = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])  # (1, 3, 2)
        mask = torch.tensor([[True, True, False]])  # (1, 3)
        out = pool(X, mask=mask)
        # Max of tiles 0 and 1 only: max(1,3)=3.0, max(2,4)=4.0
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(out.bag_representation, expected)

    def test_no_mask_is_global_max(self):
        pool = MaxPool(input_dim=2)
        X = torch.tensor([[[1.0, 5.0], [3.0, 2.0]]])  # (1, 2, 2)
        out = pool(X)
        expected = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(out.bag_representation, expected)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestAggregatorRegistry:
    def test_mean_pool_registered(self):
        cls = aggregator_registry.get("mean_pool")
        assert cls is MeanPool

    def test_max_pool_registered(self):
        cls = aggregator_registry.get("max_pool")
        assert cls is MaxPool

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            aggregator_registry.get("nonexistent_aggregator")
