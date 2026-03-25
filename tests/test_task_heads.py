"""Tests for soma.tasks — TaskHead ABC and ClassificationHead."""

from __future__ import annotations

import torch
import pytest

from soma.tasks.base import TaskHead
from soma.tasks.classification import ClassificationHead
from soma.tasks.registry import task_registry


class TestClassificationHead:
    def test_output_shape_binary(self):
        head = ClassificationHead(input_dim=16, num_classes=2)
        X = torch.randn(4, 16)
        logits = head(X)
        assert logits.shape == (4, 2)

    def test_output_shape_multiclass(self):
        head = ClassificationHead(input_dim=32, num_classes=5)
        X = torch.randn(3, 32)
        logits = head(X)
        assert logits.shape == (3, 5)

    def test_loss_is_scalar(self):
        head = ClassificationHead(input_dim=8, num_classes=3)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        loss = head.compute_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows(self):
        head = ClassificationHead(input_dim=8, num_classes=2)
        X = torch.randn(2, 8, requires_grad=True)
        logits = head(X)
        targets = torch.tensor([0, 1])
        loss = head.compute_loss(logits, targets)
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = task_registry.get("classification")
        assert cls is ClassificationHead
