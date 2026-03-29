"""Tests for soma.aggregators.mil.clam — CLAM aggregator and SmoothTop1SVM."""

from __future__ import annotations

import torch

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.mil.losses import SmoothTop1SVM
from soma.aggregators.registry import aggregator_registry


# ---------------------------------------------------------------------------
# SmoothTop1SVM
# ---------------------------------------------------------------------------


class TestSmoothTop1SVM:
    def test_correct_prediction_low_loss(self):
        """Correct predictions should yield lower loss than incorrect ones."""
        loss_fn = SmoothTop1SVM(n_classes=2)
        # Correct: high logit on true class
        x_correct = torch.tensor([[5.0, -5.0]])
        y = torch.tensor([0])
        loss_correct = loss_fn(x_correct, y)

        # Incorrect: high logit on wrong class
        x_wrong = torch.tensor([[-5.0, 5.0]])
        loss_wrong = loss_fn(x_wrong, y)

        assert loss_correct < loss_wrong

    def test_scalar_output(self):
        loss_fn = SmoothTop1SVM(n_classes=3)
        x = torch.randn(4, 3)
        y = torch.randint(0, 3, (4,))
        loss = loss_fn(x, y)
        assert loss.dim() == 0  # scalar

    def test_non_negative(self):
        loss_fn = SmoothTop1SVM(n_classes=2)
        x = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        loss = loss_fn(x, y)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# CLAM
# ---------------------------------------------------------------------------


class TestCLAM:
    def test_returns_aggregator_output(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 16)

    def test_tile_attention_shape(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 10)

    def test_auxiliary_contains_embeddings_and_attention(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        out = model(X)
        assert out.auxiliary is not None
        assert "embeddings" in out.auxiliary
        assert "attention" in out.auxiliary
        assert out.auxiliary["embeddings"].shape == (2, 10, 16)
        assert out.auxiliary["attention"].shape == (2, 10)

    def test_output_dim(self):
        from soma.aggregators.mil.clam import CLAM

        model = CLAM(input_dim=32, hidden_dim=16)
        assert model.output_dim == 32

    def test_with_mask(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(42)
        model = CLAM(input_dim=8, hidden_dim=4)
        X = torch.randn(1, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = model(X, mask=mask)
        assert out.bag_representation.shape == (1, 8)

    def test_gated(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=8, hidden_dim=4, gated=True)
        X = torch.randn(2, 5, 8)
        out = model(X)
        assert out.bag_representation.shape == (2, 8)

    def test_gradient_flows(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=8, hidden_dim=4)
        X = torch.randn(1, 5, 8, requires_grad=True)
        out = model(X)
        loss = out.bag_representation.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = aggregator_registry.get("clam")
        from soma.aggregators.mil.clam import CLAM

        assert cls is CLAM

    def test_compute_instance_loss(self):
        """Instance clustering loss should return a scalar."""
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=8, hidden_dim=4, k_sample=3)
        X = torch.randn(2, 10, 8)
        out = model(X)
        labels = torch.tensor([0, 1])
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            labels,
        )
        assert inst_loss.dim() == 0  # scalar
        assert inst_loss.item() >= 0

    def test_compute_instance_loss_with_mask(self):
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=8, hidden_dim=4, k_sample=2)
        X = torch.randn(1, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = model(X, mask=mask)
        labels = torch.tensor([1])
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            labels,
            mask=mask,
        )
        assert inst_loss.dim() == 0

    def test_k_sample_clipping(self):
        """k_sample should be clipped to bag_size when bag is small."""
        from soma.aggregators.mil.clam import CLAM

        torch.manual_seed(0)
        model = CLAM(input_dim=8, hidden_dim=4, k_sample=100)
        X = torch.randn(1, 3, 8)
        out = model(X)
        labels = torch.tensor([0])
        # Should not crash even though k_sample=100 > bag_size=3
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            labels,
        )
        assert inst_loss.dim() == 0
