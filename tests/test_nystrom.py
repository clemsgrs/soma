"""Tests for soma.aggregators.mil.nystrom — Nystrom attention internals."""

from __future__ import annotations

import torch


class TestNystromAttention:
    def test_output_shape(self):
        from soma.aggregators.mil.nystrom import NystromAttention

        torch.manual_seed(0)
        attn = NystromAttention(dim=32, n_heads=4, n_landmarks=8)
        X = torch.randn(2, 20, 32)
        out = attn(X)
        assert out.shape == (2, 20, 32)

    def test_output_shape_non_divisible(self):
        """Seq len not divisible by n_landmarks should still work (padding)."""
        from soma.aggregators.mil.nystrom import NystromAttention

        torch.manual_seed(0)
        attn = NystromAttention(dim=16, n_heads=2, n_landmarks=8)
        X = torch.randn(2, 13, 16)  # 13 not divisible by 8
        out = attn(X)
        assert out.shape == (2, 13, 16)

    def test_return_att(self):
        from soma.aggregators.mil.nystrom import NystromAttention

        torch.manual_seed(0)
        attn = NystromAttention(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16)
        out, att_weights = attn(X, return_att=True)
        assert out.shape == (1, 8, 16)
        assert att_weights.shape == (1, 2, 8, 8)

    def test_with_mask(self):
        from soma.aggregators.mil.nystrom import NystromAttention

        torch.manual_seed(0)
        attn = NystromAttention(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        out = attn(X, mask=mask)
        assert out.shape == (1, 8, 16)

    def test_gradient_flows(self):
        from soma.aggregators.mil.nystrom import NystromAttention

        torch.manual_seed(0)
        attn = NystromAttention(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16, requires_grad=True)
        out = attn(X)
        out.sum().backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0


class TestNystromTransformerLayer:
    def test_output_shape(self):
        from soma.aggregators.mil.nystrom import NystromTransformerLayer

        torch.manual_seed(0)
        layer = NystromTransformerLayer(dim=32, n_heads=4, n_landmarks=8)
        X = torch.randn(2, 10, 32)
        out = layer(X)
        assert out.shape == (2, 10, 32)

    def test_with_mlp(self):
        from soma.aggregators.mil.nystrom import NystromTransformerLayer

        torch.manual_seed(0)
        layer = NystromTransformerLayer(dim=16, n_heads=2, n_landmarks=4, use_mlp=True)
        X = torch.randn(2, 8, 16)
        out = layer(X)
        assert out.shape == (2, 8, 16)

    def test_return_att(self):
        from soma.aggregators.mil.nystrom import NystromTransformerLayer

        torch.manual_seed(0)
        layer = NystromTransformerLayer(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16)
        out, att = layer(X, return_att=True)
        assert out.shape == (1, 8, 16)
        assert att.shape == (1, 2, 8, 8)

    def test_with_mask(self):
        from soma.aggregators.mil.nystrom import NystromTransformerLayer

        torch.manual_seed(0)
        layer = NystromTransformerLayer(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        out = layer(X, mask=mask)
        assert out.shape == (1, 8, 16)

    def test_gradient_flows(self):
        from soma.aggregators.mil.nystrom import NystromTransformerLayer

        torch.manual_seed(0)
        layer = NystromTransformerLayer(dim=16, n_heads=2, n_landmarks=4)
        X = torch.randn(1, 8, 16, requires_grad=True)
        out = layer(X)
        out.sum().backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0
