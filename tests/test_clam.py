"""Tests for soma.aggregators.mil.clam — CLAM aggregators and SmoothTop1SVM."""

from __future__ import annotations

import torch

from soma.aggregators.base import AggregatorOutput
from soma.aggregators.mil.losses import SmoothTop1SVM
from soma.aggregators.registry import aggregator_registry
from soma.tasks.classification import BranchAwareClassificationHead, ClassificationHead
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead


class TestSmoothTop1SVM:
    def test_correct_prediction_low_loss(self):
        loss_fn = SmoothTop1SVM(n_classes=2)
        x_correct = torch.tensor([[5.0, -5.0]])
        x_wrong = torch.tensor([[-5.0, 5.0]])
        y = torch.tensor([0])
        assert loss_fn(x_correct, y) < loss_fn(x_wrong, y)

    def test_scalar_output(self):
        loss_fn = SmoothTop1SVM(n_classes=3)
        x = torch.randn(4, 3)
        y = torch.randint(0, 3, (4,))
        assert loss_fn(x, y).dim() == 0

    def test_non_negative(self):
        loss_fn = SmoothTop1SVM(n_classes=2)
        x = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        assert loss_fn(x, y).item() >= 0


class TestCLAMSB:
    def test_returns_aggregator_output(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=16, hidden_dim=8, attn_dim=4)
        out = model(torch.randn(2, 10, 16))
        assert isinstance(out, AggregatorOutput)
        assert out.bag_representation.shape == (2, 8)

    def test_attention_shape(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=16, hidden_dim=8, attn_dim=4)
        out = model(torch.randn(2, 10, 16))
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 10)

    def test_auxiliary_contains_embeddings_and_attention(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=16, hidden_dim=8, attn_dim=4)
        out = model(torch.randn(2, 10, 16))
        assert out.auxiliary is not None
        assert out.auxiliary["embeddings"].shape == (2, 10, 8)
        assert out.auxiliary["attention"].shape == (2, 1, 10)

    def test_output_dim(self):
        from soma.aggregators.mil.clam import CLAM_SB

        assert CLAM_SB(input_dim=32, hidden_dim=12, attn_dim=6).output_dim == 12

    def test_gated_default(self):
        from soma.aggregators.mil.clam import CLAM_SB, AttnNetGated

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3)
        assert isinstance(model.attention_net[-1], AttnNetGated)

    def test_configure_for_classification(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3)
        model.configure_for_task(ClassificationHead(input_dim=4, num_classes=2))
        assert model._resolved_instance_loss_mode == "classification"

    def test_configure_for_ordinal(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3)
        model.configure_for_task(OrdinalClassificationHead(input_dim=4, num_classes=6))
        assert model._resolved_instance_loss_mode == "ordinal"

    def test_configure_for_regression(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3)
        model.configure_for_task(RegressionHead(input_dim=4))
        assert model._resolved_instance_loss_mode == "regression"

    def test_multitarget_regression_rejected(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3)
        try:
            model.configure_for_task(RegressionHead(input_dim=4, num_targets=2))
        except ValueError as exc:
            assert "single-target regression" in str(exc)
        else:
            raise AssertionError("Expected multi-target regression to be rejected for CLAM_SB")

    def test_inconsistent_instance_loss_mode_rejected(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, instance_loss_mode="classification")
        try:
            model.configure_for_task(RegressionHead(input_dim=4))
        except ValueError as exc:
            assert "incompatible" in str(exc)
        else:
            raise AssertionError("Expected inconsistent CLAM instance loss mode to be rejected")

    def test_negative_class_instance_loss_rejected_for_ordinal(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(
            input_dim=8,
            hidden_dim=4,
            attn_dim=3,
            use_negative_class_instance_loss=True,
        )
        try:
            model.configure_for_task(OrdinalClassificationHead(input_dim=4, num_classes=6))
        except ValueError as exc:
            assert "only supported for classification" in str(exc)
        else:
            raise AssertionError("Expected negative-class instance loss to be rejected for ordinal")

    def test_compute_instance_loss_classification(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, k_sample=3)
        model.configure_for_task(ClassificationHead(input_dim=4, num_classes=2))
        out = model(torch.randn(2, 10, 8))
        labels = torch.tensor([0, 1])
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            labels,
        )
        assert inst_loss.dim() == 0
        assert inst_loss.item() >= 0

    def test_compute_instance_loss_with_mask_and_clipping(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, k_sample=100)
        model.configure_for_task(ClassificationHead(input_dim=4, num_classes=2))
        X = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, True, False]])
        out = model(X, mask=mask)
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            torch.tensor([0]),
            mask=mask,
        )
        assert inst_loss.dim() == 0

    def test_ordinal_auxiliary_loss_is_scalar(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, k_sample=2)
        model.configure_for_task(OrdinalClassificationHead(input_dim=4, num_classes=6))
        X = torch.randn(2, 7, 8)
        out = model(X)
        labels = torch.tensor([1, 4])
        aux_loss = model.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss.dim() == 0
        assert aux_loss.item() >= 0

    def test_regression_auxiliary_loss_is_scalar(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, k_sample=2)
        model.configure_for_task(RegressionHead(input_dim=4))
        X = torch.randn(2, 7, 8)
        out = model(X)
        labels = torch.tensor([0.5, -1.25])
        aux_loss = model.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss.dim() == 0
        assert aux_loss.item() >= 0

    def test_regression_bottom_k_regularization_uses_variance_not_fake_labels(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, k_sample=2, low_attention_weight=0.5)
        model.configure_for_task(RegressionHead(input_dim=4))
        X = torch.randn(1, 6, 8)
        out = model(X)
        labels = torch.tensor([2.0])
        aux_loss = model.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss.dim() == 0
        assert aux_loss.item() >= 0

    def test_combine_losses_uses_bag_weight(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, bag_weight=0.25)
        model.compute_auxiliary_loss = lambda auxiliary, labels, mask=None: torch.tensor(2.0)  # type: ignore[method-assign]
        task_loss = torch.tensor(6.0)
        combined = model.combine_losses(task_loss, {"x": torch.tensor(0.0)}, torch.tensor([0]))
        assert torch.isclose(combined, torch.tensor(3.0))

    def test_combine_losses_skips_instance_loss_when_disabled(self):
        from soma.aggregators.mil.clam import CLAM_SB

        model = CLAM_SB(input_dim=8, hidden_dim=4, attn_dim=3, bag_weight=1.0)
        task_loss = torch.tensor(6.0)
        combined = model.combine_losses(task_loss, {"x": torch.tensor(0.0)}, torch.tensor([0]))
        assert torch.isclose(combined, task_loss)


class TestCLAMMB:
    def test_returns_branch_aware_representation_and_attention(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = CLAM_MB(input_dim=16, hidden_dim=8, attn_dim=4, n_classes=3)
        model.configure_for_task(BranchAwareClassificationHead(input_dim=8, num_classes=3))
        out = model(torch.randn(2, 10, 16))
        assert out.bag_representation.shape == (2, 3, 8)
        assert out.tile_attention is not None
        assert out.tile_attention.shape == (2, 3, 10)
        assert out.auxiliary["attention"].shape == (2, 3, 10)

    def test_instance_loss_multiclass(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = CLAM_MB(
            input_dim=8,
            hidden_dim=4,
            attn_dim=3,
            n_classes=3,
            k_sample=2,
            use_negative_class_instance_loss=True,
        )
        model.configure_for_task(BranchAwareClassificationHead(input_dim=4, num_classes=3))
        out = model(torch.randn(2, 9, 8))
        inst_loss = model.compute_instance_loss(
            out.auxiliary["attention"],
            out.auxiliary["embeddings"],
            torch.tensor([0, 2]),
        )
        assert inst_loss.dim() == 0
        assert inst_loss.item() >= 0

    def test_out_of_class_branch_uses_negative_only_helper(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = CLAM_MB(
            input_dim=8,
            hidden_dim=4,
            attn_dim=3,
            n_classes=3,
            use_negative_class_instance_loss=True,
        )
        model.configure_for_task(BranchAwareClassificationHead(input_dim=4, num_classes=3))
        calls: list[tuple[str, int]] = []
        model._classification_inst_eval = lambda att, emb, classifier, branch_idx, mask: (  # type: ignore[method-assign]
            calls.append(("in", branch_idx)) or emb.new_zeros(())
        )
        model._classification_inst_eval_out = lambda att, emb, classifier, branch_idx, mask: (  # type: ignore[method-assign]
            calls.append(("out", branch_idx)) or emb.new_zeros(())
        )
        out = model(torch.randn(1, 5, 8))
        model.compute_instance_loss(
            out.auxiliary["attention"], out.auxiliary["embeddings"], torch.tensor([1])
        )
        assert ("out", 0) in calls and ("out", 2) in calls
        assert ("in", 1) in calls

    def test_rejected_for_ordinal(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3)
        try:
            model.configure_for_task(OrdinalClassificationHead(input_dim=4, num_classes=6))
        except ValueError as exc:
            assert "classification tasks" in str(exc)
        else:
            raise AssertionError("Expected clam_mb to reject ordinal tasks")

    def test_rejected_for_regression(self):
        from soma.aggregators.mil.clam import CLAM_MB

        model = CLAM_MB(input_dim=8, hidden_dim=4, attn_dim=3, n_classes=3)
        try:
            model.configure_for_task(RegressionHead(input_dim=4))
        except ValueError as exc:
            assert "classification tasks" in str(exc)
        else:
            raise AssertionError("Expected clam_mb to reject regression tasks")

    def test_registered(self):
        from soma.aggregators.mil.clam import CLAM_MB, CLAM_SB

        assert aggregator_registry.get("clam_sb") is CLAM_SB
        assert aggregator_registry.get("clam_mb") is CLAM_MB
