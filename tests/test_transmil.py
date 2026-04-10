"""Tests for soma.aggregators.mil.transmil — TransMIL aggregator."""

from __future__ import annotations

import torch
import pytest

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.registry import aggregator_registry


class TestTransMIL:
    def test_returns_aggregator_output(self):
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(input_dim=32, att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(2, 10, 32)
        out = model(X)
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 16)

    def test_output_dim(self):
        from soma.aggregators.mil.transmil import TransMIL

        model = TransMIL(input_dim=32, att_dim=64, n_heads=4)
        assert model.output_dim == 64

    def test_tile_attention_shape(self):
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(input_dim=16, att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 10)

    def test_varying_bag_sizes(self):
        """TransMIL pads to perfect square — should work for any bag size."""
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(input_dim=16, att_dim=16, n_heads=2, n_landmarks=4)
        for bag_size in [1, 4, 7, 16, 25, 30]:
            X = torch.randn(1, bag_size, 16)
            out = model(X)
            assert out.bag_representation.shape == (1, 16)
            assert out.tile_attention.shape == (1, bag_size)

    def test_with_mask(self):
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(42)
        model = TransMIL(input_dim=16, att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        out = model(X, mask=mask)
        assert out.bag_representation.shape == (1, 16)
        assert out.tile_attention.shape == (1, 8)

    def test_n_layers_minimum(self):
        from soma.aggregators.mil.transmil import TransMIL

        with pytest.raises(ValueError, match="at least 2"):
            TransMIL(input_dim=16, att_dim=16, n_layers=1)

    def test_gradient_flows(self):
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(input_dim=16, att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 10, 16, requires_grad=True)
        out = model(X)
        # LayerNorm makes the plain sum degenerate; use a loss that still
        # depends on the representation so we can verify gradients flow.
        loss = out.bag_representation.pow(2).sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = aggregator_registry.get("transmil")
        from soma.aggregators.mil.transmil import TransMIL

        assert cls is TransMIL

    def test_input_dim_equals_att_dim(self):
        """When input_dim == att_dim, no projection needed."""
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(input_dim=16, att_dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(2, 5, 16)
        out = model(X)
        assert out.bag_representation.shape == (2, 16)

    def test_more_layers(self):
        from soma.aggregators.mil.transmil import TransMIL

        torch.manual_seed(0)
        model = TransMIL(
            input_dim=16, att_dim=16, n_heads=2, n_landmarks=4, n_layers=4
        )
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.bag_representation.shape == (2, 16)
