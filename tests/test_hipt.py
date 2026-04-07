"""Tests for HIPT aggregator (VisionTransformer4K + HIPT)."""

from __future__ import annotations

import torch
import pytest

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.registry import aggregator_registry


class TestVisionTransformer4K:
    def test_output_shape(self):
        """Tiny dims: verify output shape (M, embed_dim)."""
        from soma.aggregators.mil.hipt import VisionTransformer4K

        vit = VisionTransformer4K(
            input_embed_dim=16,
            output_embed_dim=8,
            npatch=4,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
        )
        # Input: (M, D_in, npatch, npatch)
        x = torch.randn(3, 16, 4, 4)
        out = vit(x)
        assert out.shape == (3, 8)

    def test_positional_encoding_interpolation(self):
        """Create with one npatch, run with another — no errors."""
        from soma.aggregators.mil.hipt import VisionTransformer4K

        vit = VisionTransformer4K(
            input_embed_dim=16,
            output_embed_dim=8,
            npatch=4,
            num_heads=2,
            depth=2,
        )
        # Run with different npatch (6 instead of 4)
        x = torch.randn(2, 16, 6, 6)
        out = vit(x)
        assert out.shape == (2, 8)


class TestHIPT:
    def _make_hipt(self, **kwargs):
        from soma.aggregators.mil.hipt import HIPT

        defaults = dict(
            input_dim=16,
            region_size=8,
            patch_size=4,
            embed_dim_region=12,
            embed_dim_slide=12,  # must be divisible by global transformer nhead=3
            num_heads=2,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return HIPT(**defaults)

    def test_returns_aggregator_output(self):
        """(B=2, N=32, D=16) with region_size=8, patch_size=4 → P=4, M=8."""
        hipt = self._make_hipt()
        X = torch.randn(2, 32, 16)
        mask = torch.ones(2, 32, dtype=torch.bool)
        out = hipt(X, mask=mask)

        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 12)

    def test_returns_aggregator_output_from_native_hierarchical_input(self):
        """(B=2, M=8, P=4, D=16) hierarchical input should run directly."""
        hipt = self._make_hipt()
        X = torch.randn(2, 8, 4, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 6:] = False
        out = hipt(X, mask=mask)

        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 12)

    def test_output_dim(self):
        """output_dim matches embed_dim_slide."""
        hipt = self._make_hipt(embed_dim_slide=18)
        assert hipt.output_dim == 18

    def test_with_mask(self):
        """Partially masked input runs without error."""
        hipt = self._make_hipt()
        X = torch.randn(1, 16, 16)
        mask = torch.ones(1, 16, dtype=torch.bool)
        mask[0, 8:] = False  # mask out second half
        out = hipt(X, mask=mask)
        assert out.bag_representation.shape == (1, 12)

    def test_no_mask(self):
        """Works without explicit mask."""
        hipt = self._make_hipt()
        X = torch.randn(1, 8, 16)
        out = hipt(X)
        assert out.bag_representation.shape == (1, 12)

    def test_padding_non_multiple(self):
        """N=5 with P=4 → HIPT pads to 8 → 2 regions."""
        hipt = self._make_hipt()
        X = torch.randn(1, 5, 16)
        mask = torch.ones(1, 5, dtype=torch.bool)
        out = hipt(X, mask=mask)
        assert out.bag_representation.shape == (1, 12)

    def test_gradient_flows(self):
        """loss.backward() produces non-zero input gradients."""
        hipt = self._make_hipt()
        X = torch.randn(1, 8, 16, requires_grad=True)
        out = hipt(X)
        loss = out.bag_representation.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        """HIPT is registered in aggregator_registry."""
        from soma.aggregators.mil.hipt import HIPT

        assert aggregator_registry.get("hipt") is HIPT

    def test_invalid_region_size(self):
        """region_size < 2 * patch_size → ValueError."""
        from soma.aggregators.mil.hipt import HIPT

        with pytest.raises(ValueError, match="region_size.*patch_size"):
            HIPT(input_dim=16, region_size=4, patch_size=4, embed_dim_region=12, embed_dim_slide=12)

    def test_region_size_not_divisible(self):
        """region_size % patch_size != 0 → ValueError."""
        from soma.aggregators.mil.hipt import HIPT

        with pytest.raises(ValueError, match="divisible"):
            HIPT(input_dim=16, region_size=7, patch_size=4, embed_dim_region=12, embed_dim_slide=12)
