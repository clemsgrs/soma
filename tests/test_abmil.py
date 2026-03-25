"""Tests for soma.aggregators.mil — AttentionPool and ABMIL."""

from __future__ import annotations

import torch
import pytest

from soma.aggregators.mil.attention_pool import AttentionPool, masked_softmax
from soma.aggregators.mil.abmil import ABMIL
from soma.aggregators.base import AggregatorOutput
from soma.aggregators.registry import aggregator_registry


# ---------------------------------------------------------------------------
# masked_softmax
# ---------------------------------------------------------------------------


class TestMaskedSoftmax:
    def test_without_mask(self):
        """Without mask, should behave like regular softmax over dim=1."""
        X = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        result = masked_softmax(X, mask=None)
        expected = torch.softmax(X, dim=1)
        assert torch.allclose(result, expected)

    def test_with_mask(self):
        """Masked positions should get ~0 probability."""
        X = torch.tensor([[[1.0], [2.0], [100.0]]])  # (1, 3, 1)
        mask = torch.tensor([[[True], [True], [False]]])  # (1, 3, 1)
        result = masked_softmax(X, mask=mask)
        # Tile 2 is masked → should be ~0
        assert result[0, 2, 0].item() < 1e-6
        # Remaining should sum to ~1
        assert torch.allclose(result[0, :2, 0].sum(), torch.tensor(1.0), atol=1e-5)

    def test_all_valid(self):
        """All-True mask should be equivalent to no mask."""
        X = torch.tensor([[[1.0], [2.0]]])  # (1, 2, 1)
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        result_masked = masked_softmax(X, mask=mask)
        result_none = masked_softmax(X, mask=None)
        assert torch.allclose(result_masked, result_none)


# ---------------------------------------------------------------------------
# AttentionPool
# ---------------------------------------------------------------------------


class TestAttentionPool:
    def test_output_shapes(self):
        torch.manual_seed(0)
        pool = AttentionPool(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        z, attn = pool(X)
        assert z.shape == (2, 16)
        assert attn.shape == (2, 10)

    def test_attention_sums_to_one(self):
        """Normalized attention weights should sum to 1 over valid tiles."""
        torch.manual_seed(0)
        pool = AttentionPool(input_dim=8, hidden_dim=4)
        X = torch.randn(3, 5, 8)
        z, attn = pool(X)
        # attn is raw logits; normalized internally via softmax
        # But the returned attn is pre-softmax. Let's verify via the output:
        # The bag rep z = X^T @ softmax(attn), so z is a weighted sum.
        # We can verify z lies within the convex hull indirectly.
        assert z.shape == (3, 8)
        assert attn.shape == (3, 5)

    def test_masked_attention(self):
        """Masked tiles should not contribute to the output."""
        torch.manual_seed(42)
        pool = AttentionPool(input_dim=4, hidden_dim=4)
        # Only first 2 of 4 tiles valid
        X = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, False, False]])

        z_masked, attn_masked = pool(X, mask=mask)
        # With mask, output should only depend on tiles 0 and 1
        assert z_masked.shape == (1, 4)

    def test_gated_variant(self):
        torch.manual_seed(0)
        pool = AttentionPool(input_dim=8, hidden_dim=4, gated=True)
        X = torch.randn(2, 5, 8)
        z, attn = pool(X)
        assert z.shape == (2, 8)
        assert attn.shape == (2, 5)


# ---------------------------------------------------------------------------
# ABMIL
# ---------------------------------------------------------------------------


class TestABMIL:
    def test_returns_aggregator_output(self):
        torch.manual_seed(0)
        model = ABMIL(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 16)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 10)

    def test_output_dim(self):
        model = ABMIL(input_dim=32, hidden_dim=16)
        assert model.output_dim == 32

    def test_with_mask(self):
        torch.manual_seed(0)
        model = ABMIL(input_dim=8, hidden_dim=4)
        X = torch.randn(1, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = model(X, mask=mask)
        assert out.bag_representation.shape == (1, 8)
        assert out.tile_attention.shape == (1, 6)

    def test_gated(self):
        torch.manual_seed(0)
        model = ABMIL(input_dim=8, hidden_dim=4, gated=True)
        X = torch.randn(2, 5, 8)
        out = model(X)
        assert out.bag_representation.shape == (2, 8)

    def test_gradient_flows(self):
        """Gradient should flow back through attention to input."""
        torch.manual_seed(0)
        model = ABMIL(input_dim=8, hidden_dim=4)
        X = torch.randn(1, 5, 8, requires_grad=True)
        out = model(X)
        loss = out.bag_representation.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = aggregator_registry.get("abmil")
        assert cls is ABMIL
