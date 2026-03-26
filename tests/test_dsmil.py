"""Tests for soma.aggregators.mil.dsmil — DSMIL aggregator."""

from __future__ import annotations

import torch

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.registry import aggregator_registry


class TestDSMIL:
    def test_returns_aggregator_output(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=16, att_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 16)

    def test_tile_attention_shape(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=16, att_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 10)

    def test_auxiliary_instance_logits(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=16, att_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.auxiliary is not None
        assert "instance_logits" in out.auxiliary
        assert out.auxiliary["instance_logits"].shape == (2, 10)

    def test_output_dim(self):
        from soma.aggregators.mil.dsmil import DSMIL

        model = DSMIL(input_dim=32, att_dim=16)
        assert model.output_dim == 32

    def test_with_mask(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(42)
        model = DSMIL(input_dim=8, att_dim=4)
        X = torch.randn(1, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = model(X, mask=mask)
        assert out.bag_representation.shape == (1, 8)
        assert out.tile_attention.shape == (1, 6)

    def test_nonlinear_q(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=16, att_dim=8, nonlinear_q=True)
        X = torch.randn(2, 5, 16)
        out = model(X)
        assert out.bag_representation.shape == (2, 16)

    def test_nonlinear_v(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=16, att_dim=8, nonlinear_v=True, dropout=0.1)
        X = torch.randn(2, 5, 16)
        out = model(X)
        assert out.bag_representation.shape == (2, 16)

    def test_gradient_flows(self):
        from soma.aggregators.mil.dsmil import DSMIL

        torch.manual_seed(0)
        model = DSMIL(input_dim=8, att_dim=4)
        X = torch.randn(1, 5, 8, requires_grad=True)
        out = model(X)
        loss = out.bag_representation.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = aggregator_registry.get("dsmil")
        from soma.aggregators.mil.dsmil import DSMIL

        assert cls is DSMIL
