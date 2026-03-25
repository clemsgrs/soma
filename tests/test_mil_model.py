"""Tests for soma.training.model — MILModel composition."""

from __future__ import annotations

import torch

from soma.aggregators.pooling import MeanPool
from soma.aggregators.mil.abmil import ABMIL
from soma.tasks.classification import ClassificationHead
from soma.training.model import MILModel, MILModelOutput


class TestMILModel:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = MILModel(
            aggregator=MeanPool(input_dim=16),
            task_head=ClassificationHead(input_dim=16, num_classes=3),
        )
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert isinstance(out, MILModelOutput)
        assert out.logits.shape == (2, 3)

    def test_attention_passthrough_abmil(self):
        """ABMIL should pass tile_attention through MILModel."""
        torch.manual_seed(0)
        model = MILModel(
            aggregator=ABMIL(input_dim=8, hidden_dim=4),
            task_head=ClassificationHead(input_dim=8, num_classes=2),
        )
        X = torch.randn(2, 5, 8)
        out = model(X)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 5)

    def test_attention_none_for_mean_pool(self):
        """MeanPool has no attention — tile_attention should be None."""
        model = MILModel(
            aggregator=MeanPool(input_dim=8),
            task_head=ClassificationHead(input_dim=8, num_classes=2),
        )
        X = torch.randn(1, 5, 8)
        out = model(X)
        assert out.tile_attention is None

    def test_gradient_flows_end_to_end(self):
        """Loss gradient should flow back to input through aggregator."""
        torch.manual_seed(0)
        model = MILModel(
            aggregator=ABMIL(input_dim=8, hidden_dim=4),
            task_head=ClassificationHead(input_dim=8, num_classes=2),
        )
        X = torch.randn(2, 5, 8, requires_grad=True)
        out = model(X)
        targets = torch.tensor([0, 1])
        loss = model.task_head.compute_loss(out.logits, targets)
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0
