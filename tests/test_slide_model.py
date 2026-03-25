"""Tests for soma.training.slide_model — SlideModel (TaskHead-only wrapper)."""

from __future__ import annotations

import torch

from soma.tasks.classification import ClassificationHead
from soma.training.slide_model import SlideModel, SlideModelOutput


class TestSlideModel:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = SlideModel(task_head=ClassificationHead(input_dim=512, num_classes=3))
        X = torch.randn(4, 512)
        out = model(X)
        assert isinstance(out, SlideModelOutput)
        assert out.logits.shape == (4, 3)

    def test_exposes_task_head(self):
        head = ClassificationHead(input_dim=128, num_classes=2)
        model = SlideModel(task_head=head)
        assert model.task_head is head

    def test_gradient_flows(self):
        torch.manual_seed(0)
        model = SlideModel(task_head=ClassificationHead(input_dim=8, num_classes=2))
        X = torch.randn(2, 8, requires_grad=True)
        out = model(X)
        loss = model.task_head.compute_loss(out.logits, torch.tensor([0, 1]))
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0
