"""Tests for soma.training.model — MILModel composition."""

from __future__ import annotations

import torch

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.pooling import MeanPool
from soma.aggregators.mil.abmil import ABMIL
from soma.tasks.classification import BranchAwareClassificationHead, ClassificationHead
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead
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

    def test_auxiliary_passthrough(self):
        """Auxiliary dict from aggregator should be forwarded through MILModel."""

        class DummyAggregator(Aggregator):
            def __init__(self):
                super().__init__()
                self._dim = 8

            def forward(self, X, mask=None):
                bag_rep = X.mean(dim=1)
                aux = {"instance_logits": X.sum(dim=-1)}
                return AggregatorOutput(
                    bag_representation=bag_rep, auxiliary=aux
                )

            @property
            def output_dim(self):
                return self._dim

        model = MILModel(
            aggregator=DummyAggregator(),
            task_head=ClassificationHead(input_dim=8, num_classes=2),
        )
        X = torch.randn(2, 5, 8)
        out = model(X)
        assert out.auxiliary is not None
        assert "instance_logits" in out.auxiliary
        assert out.auxiliary["instance_logits"].shape == (2, 5)

    def test_auxiliary_none_for_abmil(self):
        """ABMIL has no auxiliary — should be None."""
        torch.manual_seed(0)
        model = MILModel(
            aggregator=ABMIL(input_dim=8, hidden_dim=4),
            task_head=ClassificationHead(input_dim=8, num_classes=2),
        )
        X = torch.randn(1, 5, 8)
        out = model(X)
        assert out.auxiliary is None

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

    def test_branch_aware_classification_supported(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = MILModel(
            aggregator=CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3),
            task_head=BranchAwareClassificationHead(input_dim=4, num_classes=3),
        )
        out = model(torch.randn(2, 5, 8))
        assert out.logits.shape == (2, 3)

    def test_plain_classification_rejected_for_branch_aware_input(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = MILModel(
            aggregator=CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3),
            task_head=ClassificationHead(input_dim=4, num_classes=3),
        )
        try:
            model(torch.randn(2, 5, 8))
        except ValueError as exc:
            assert "does not support branch-aware" in str(exc)
        else:
            raise AssertionError("Expected plain classification head to be rejected for CLAM_MB")

    def test_branch_aware_representation_rejected_for_regression(self):
        from soma.aggregators.mil.clam import CLAM_MB

        try:
            MILModel(
                aggregator=CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3),
                task_head=RegressionHead(input_dim=4),
            )
        except ValueError as exc:
            assert "classification tasks" in str(exc)
        else:
            raise AssertionError("Expected branch-aware CLAM_MB to be rejected for regression")

    def test_clam_mb_rejected_for_ordinal(self):
        from soma.aggregators.mil.clam import CLAM_MB

        try:
            MILModel(
                aggregator=CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3),
                task_head=OrdinalClassificationHead(input_dim=4, num_classes=6),
            )
        except ValueError as exc:
            assert "classification tasks" in str(exc)
        else:
            raise AssertionError("Expected clam_mb to be rejected for ordinal")

    def test_clam_sb_accepts_regression(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = MILModel(
            aggregator=CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3),
            task_head=RegressionHead(input_dim=4),
        )
        out = model(torch.randn(2, 5, 8))
        assert out.logits.shape == (2, 1)

    def test_clam_sb_rejects_multitarget_regression(self):
        from soma.aggregators.mil.clam import CLAM_SB

        try:
            MILModel(
                aggregator=CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3),
                task_head=RegressionHead(input_dim=4, num_targets=2),
            )
        except ValueError as exc:
            assert "single-target regression" in str(exc)
        else:
            raise AssertionError("Expected clam_sb to reject multi-target regression")
