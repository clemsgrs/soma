"""Tests for soma.tasks — TaskHead ABC, ClassificationHead, and RegressionHead."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from soma.tasks.base import TaskHead
from soma.tasks.classification import ClassificationHead
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead
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

    def test_label_dtype(self):
        assert ClassificationHead.label_dtype == torch.long

    def test_postprocess_returns_probs_and_preds(self):
        head = ClassificationHead(input_dim=8, num_classes=3)
        logits = torch.randn(4, 3)
        out = head.postprocess(logits)
        assert "probabilities" in out
        assert "predicted_labels" in out
        probs = out["probabilities"]
        preds = out["predicted_labels"]
        assert probs.shape == (4, 3)
        assert preds.shape == (4,)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)
        assert (preds == probs.argmax(axis=1)).all()

    def test_compute_metrics_returns_classification_keys(self):
        head = ClassificationHead(input_dim=8, num_classes=2)
        logits = torch.randn(6, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 1])
        metrics = head.compute_metrics(logits, targets)
        for key in ["accuracy", "balanced_accuracy", "f1_macro", "auc"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_auto_params(self):
        class _FakeDataset:
            num_classes = 7

        params = ClassificationHead.auto_params(_FakeDataset())
        assert params == {"num_classes": 7}


class TestRegressionHead:
    def test_output_shape_single_target(self):
        head = RegressionHead(input_dim=16)
        X = torch.randn(4, 16)
        out = head(X)
        assert out.shape == (4, 1)

    def test_output_shape_multi_target(self):
        head = RegressionHead(input_dim=16, num_targets=3)
        X = torch.randn(4, 16)
        out = head(X)
        assert out.shape == (4, 3)

    def test_loss_is_scalar(self):
        head = RegressionHead(input_dim=8)
        preds = torch.randn(4, 1)
        targets = torch.randn(4)
        loss = head.compute_loss(preds, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self):
        head = RegressionHead(input_dim=8)
        X = torch.randn(2, 8, requires_grad=True)
        preds = head(X)
        targets = torch.randn(2)
        loss = head.compute_loss(preds, targets)
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = task_registry.get("regression")
        assert cls is RegressionHead

    def test_label_dtype(self):
        assert RegressionHead.label_dtype == torch.float

    def test_postprocess_returns_predictions(self):
        head = RegressionHead(input_dim=8)
        raw = torch.randn(4, 1)
        out = head.postprocess(raw)
        assert "predictions" in out
        assert out["predictions"].shape == (4,)

    def test_compute_metrics_returns_regression_keys(self):
        head = RegressionHead(input_dim=8)
        raw = torch.randn(6, 1)
        targets = torch.randn(6)
        metrics = head.compute_metrics(raw, targets)
        for key in ["mse", "mae", "r2"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_auto_params_is_empty(self):
        class _FakeDataset:
            num_classes = 5

        params = RegressionHead.auto_params(_FakeDataset())
        assert params == {}


class TestOrdinalClassificationHead:
    def test_output_shape(self):
        head = OrdinalClassificationHead(input_dim=16, num_classes=6)
        X = torch.randn(4, 16)
        out = head(X)
        assert out.shape == (4, 1)

    def test_loss_is_scalar(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        preds = torch.randn(4, 1)
        targets = torch.tensor([0, 2, 4, 5])
        loss = head.compute_loss(preds, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        X = torch.randn(3, 8, requires_grad=True)
        preds = head(X)
        targets = torch.tensor([0, 3, 5])
        loss = head.compute_loss(preds, targets)
        loss.backward()
        assert X.grad is not None
        assert X.grad.abs().sum() > 0

    def test_registered(self):
        cls = task_registry.get("ordinal_classification")
        assert cls is OrdinalClassificationHead

    def test_label_dtype(self):
        assert OrdinalClassificationHead.label_dtype == torch.long

    def test_auto_params(self):
        class _FakeDataset:
            num_classes = 6

        params = OrdinalClassificationHead.auto_params(_FakeDataset())
        assert params == {"num_classes": 6}

    def test_postprocess_returns_integer_labels_and_raw_scores(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        # raw output: continuous values that should round to specific classes
        raw = torch.tensor([[0.4], [1.6], [3.1], [5.8]])
        out = head.postprocess(raw)
        assert "predicted_labels" in out
        assert "raw_scores" in out
        np.testing.assert_array_equal(out["predicted_labels"], [0, 2, 3, 5])
        np.testing.assert_allclose(out["raw_scores"], [0.4, 1.6, 3.1, 5.8], atol=1e-5)

    def test_postprocess_clips_to_class_range(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        raw = torch.tensor([[-2.0], [7.5]])  # out of [0, 5] range
        out = head.postprocess(raw)
        np.testing.assert_array_equal(out["predicted_labels"], [0, 5])

    def test_compute_metrics_returns_ordinal_keys(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        raw = torch.randn(8, 1)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 3])
        metrics = head.compute_metrics(raw, targets)
        for key in ["qwk", "accuracy", "balanced_accuracy"]:
            assert key in metrics
            assert isinstance(metrics[key], float)
