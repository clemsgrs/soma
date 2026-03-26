"""Tests for soma.aggregators.mil.dtfdmil — DTFD-MIL aggregator."""

from __future__ import annotations

import torch
import pytest

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.registry import aggregator_registry


class TestDTFDMIL:
    def test_returns_aggregator_output(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=16, hidden_dim=8, n_groups=4)
        X = torch.randn(2, 12, 16)
        out = model(X)
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 16)

    def test_tile_attention_shape(self):
        """Tile attention (instance CAM) should cover all tiles in original order."""
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=16, hidden_dim=8, n_groups=4)
        X = torch.randn(2, 12, 16)
        out = model(X)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 12)

    def test_auxiliary_pseudo_predictions(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=16, hidden_dim=8, n_groups=4)
        X = torch.randn(2, 12, 16)
        out = model(X)
        assert out.auxiliary is not None
        assert "pseudo_predictions" in out.auxiliary
        assert out.auxiliary["pseudo_predictions"].shape == (2, 4)

    def test_output_dim(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        model = DTFDMIL(input_dim=32, hidden_dim=16)
        assert model.output_dim == 32

    def test_with_mask(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(42)
        model = DTFDMIL(input_dim=8, hidden_dim=4, n_groups=3)
        X = torch.randn(1, 9, 8)
        mask = torch.tensor([[True] * 6 + [False] * 3])
        out = model(X, mask=mask)
        assert out.bag_representation.shape == (1, 8)

    def test_n_groups_clipping(self):
        """n_groups should be clipped to bag_size when bag is small."""
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=8, hidden_dim=4, n_groups=100)
        X = torch.randn(1, 3, 8)
        out = model(X)
        assert out.bag_representation.shape == (1, 8)
        # pseudo_predictions should have n_groups clipped to bag_size
        assert out.auxiliary["pseudo_predictions"].shape[1] <= 3

    def test_distill_mode_maxmin(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=8, hidden_dim=4, distill_mode="maxmin")
        X = torch.randn(2, 8, 8)
        out = model(X)
        assert out.bag_representation.shape == (2, 8)

    def test_distill_mode_max(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=8, hidden_dim=4, distill_mode="max")
        X = torch.randn(2, 8, 8)
        out = model(X)
        assert out.bag_representation.shape == (2, 8)

    def test_distill_mode_afs(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=8, hidden_dim=4, distill_mode="afs")
        X = torch.randn(2, 8, 8)
        out = model(X)
        assert out.bag_representation.shape == (2, 8)

    def test_invalid_distill_mode(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        with pytest.raises(ValueError, match="distill_mode"):
            DTFDMIL(input_dim=8, hidden_dim=4, distill_mode="invalid")

    def test_gradient_flows(self):
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        model = DTFDMIL(input_dim=8, hidden_dim=4, n_groups=2)
        X = torch.randn(1, 6, 8, requires_grad=True)
        out = model(X)
        loss = out.bag_representation.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = aggregator_registry.get("dtfdmil")
        from soma.aggregators.mil.dtfdmil import DTFDMIL

        assert cls is DTFDMIL
