"""Tests for soma.tasks — TaskHead ABC and all task head implementations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from soma.evaluation.metrics import DEFAULT_METRICS
from soma.tasks.base import TaskHead
from soma.tasks.classification import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    MulticlassClassificationHead,
)
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead
from soma.tasks.registry import task_registry


class _FakeDataset:
    """Minimal dataset stub exposing ``samples`` with ``.label`` records."""

    def __init__(self, num_classes):
        self.samples = {
            str(i): SimpleNamespace(label=i) for i in range(num_classes)
        }


def _labels(values) -> dict[str, torch.Tensor]:
    return {"label": torch.tensor(values)}


def _values(tensor) -> dict[str, torch.Tensor]:
    return {"value": tensor}


# ---------------------------------------------------------------------------
# BinaryClassificationHead
# ---------------------------------------------------------------------------


class TestBinaryClassificationHead:
    def test_output_shape(self):
        head = BinaryClassificationHead(input_dim=16, num_classes=2)
        X = torch.randn(4, 16)
        assert head(X).shape == (4, 2)

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError, match="num_classes=2"):
            BinaryClassificationHead(input_dim=16, num_classes=3)

    def test_loss_is_scalar(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2)
        logits = torch.randn(4, 2)
        loss = head.compute_loss(logits, _labels([0, 1, 0, 1]))
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2)
        X = torch.randn(2, 8, requires_grad=True)
        loss = head.compute_loss(head(X), _labels([0, 1]))
        loss.backward()
        assert X.grad is not None and X.grad.abs().sum() > 0

    def test_registered(self):
        assert task_registry.get("binary_classification") is BinaryClassificationHead

    def test_target_dtypes(self):
        assert BinaryClassificationHead.target_dtypes == {"label": torch.long}

    def test_postprocess(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2)
        out = head.postprocess(torch.randn(4, 2))
        assert out["probabilities"].shape == (4, 2)
        assert out["predicted_labels"].shape == (4,)
        np.testing.assert_allclose(out["probabilities"].sum(axis=1), np.ones(4), atol=1e-5)

    def test_default_metrics(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2)
        assert set(head.metrics) == set(DEFAULT_METRICS["binary_classification"])

    def test_custom_metrics(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2, metrics=["auroc", "f1"])
        assert set(head.metrics) == {"auroc", "f1"}

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            BinaryClassificationHead(input_dim=8, num_classes=2, metrics=["qwk"])

    def test_compute_metrics_returns_requested_keys(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2, metrics=["auroc", "f1"])
        logits = torch.randn(6, 2)
        m = head.compute_metrics(logits, _labels([0, 1, 0, 1, 0, 1]))
        assert set(m.keys()) == {"auroc", "f1"}
        assert all(isinstance(v, float) for v in m.values())

    def test_compute_metrics_default_keys(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2)
        logits = torch.randn(6, 2)
        m = head.compute_metrics(logits, _labels([0, 1, 0, 1, 0, 1]))
        assert set(m.keys()) == set(DEFAULT_METRICS["binary_classification"])

    def test_auto_params(self):
        params = BinaryClassificationHead.auto_params(_FakeDataset(2))
        assert params["num_classes"] == 2
        assert params["label_map"] == {0: 0, 1: 1}

    def test_extract_targets_uses_label_map(self):
        head = BinaryClassificationHead(input_dim=8, num_classes=2, label_map={"neg": 0, "pos": 1})
        record = SimpleNamespace(label="pos")
        assert head.extract_targets(record) == {"label": 1}


# ---------------------------------------------------------------------------
# MulticlassClassificationHead
# ---------------------------------------------------------------------------


class TestMulticlassClassificationHead:
    def test_output_shape(self):
        head = MulticlassClassificationHead(input_dim=32, num_classes=5)
        assert head(torch.randn(3, 32)).shape == (3, 5)

    def test_rejects_fewer_than_two_classes(self):
        with pytest.raises(ValueError, match="num_classes >= 2"):
            MulticlassClassificationHead(input_dim=8, num_classes=1)

    def test_loss_is_scalar(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3)
        loss = head.compute_loss(torch.randn(4, 3), _labels([0, 1, 2, 1]))
        assert loss.shape == () and loss.item() > 0

    def test_gradient_flows(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3)
        X = torch.randn(2, 8, requires_grad=True)
        head.compute_loss(head(X), _labels([0, 2])).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0

    def test_registered(self):
        assert task_registry.get("multiclass_classification") is MulticlassClassificationHead

    def test_target_dtypes(self):
        assert MulticlassClassificationHead.target_dtypes == {"label": torch.long}

    def test_postprocess(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3)
        out = head.postprocess(torch.randn(4, 3))
        assert out["probabilities"].shape == (4, 3)
        assert out["predicted_labels"].shape == (4,)
        np.testing.assert_allclose(out["probabilities"].sum(axis=1), np.ones(4), atol=1e-5)

    def test_default_metrics(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3)
        assert set(head.metrics) == set(DEFAULT_METRICS["multiclass_classification"])

    def test_custom_metrics(self):
        head = MulticlassClassificationHead(
            input_dim=8, num_classes=3, metrics=["accuracy", "f1_macro", "qwk"]
        )
        assert set(head.metrics) == {"accuracy", "f1_macro", "qwk"}

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            MulticlassClassificationHead(input_dim=8, num_classes=3, metrics=["auroc"])

    def test_qwk_metric_is_accepted(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3, metrics=["qwk"])
        assert head.metrics == ["qwk"]

    def test_compute_metrics_returns_requested_keys(self):
        head = MulticlassClassificationHead(input_dim=8, num_classes=3, metrics=["accuracy", "f1_macro"])
        m = head.compute_metrics(torch.randn(6, 3), _labels([0, 1, 2, 0, 1, 2]))
        assert set(m.keys()) == {"accuracy", "f1_macro"}

    def test_auto_params(self):
        params = MulticlassClassificationHead.auto_params(_FakeDataset(5))
        assert params["num_classes"] == 5
        assert params["label_map"] == {i: i for i in range(5)}


# ---------------------------------------------------------------------------
# BranchAwareClassificationHead
# ---------------------------------------------------------------------------


class TestBranchAwareClassificationHead:
    def test_registered(self):
        assert (
            task_registry.get("branch_aware_classification")
            is BranchAwareClassificationHead
        )

    def test_default_metrics_are_multiclass(self):
        head = BranchAwareClassificationHead(input_dim=8, num_classes=3)
        assert set(head.metrics) == set(DEFAULT_METRICS["multiclass_classification"])

    def test_custom_metrics(self):
        head = BranchAwareClassificationHead(input_dim=8, num_classes=3, metrics=["f1_macro"])
        assert head.metrics == ["f1_macro"]

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            BranchAwareClassificationHead(input_dim=8, num_classes=3, metrics=["auroc"])


# ---------------------------------------------------------------------------
# RegressionHead
# ---------------------------------------------------------------------------


class TestRegressionHead:
    def test_output_shape_single_target(self):
        head = RegressionHead(input_dim=16)
        assert head(torch.randn(4, 16)).shape == (4, 1)

    def test_output_shape_multi_target(self):
        head = RegressionHead(input_dim=16, num_targets=3)
        assert head(torch.randn(4, 16)).shape == (4, 3)

    def test_loss_is_scalar(self):
        head = RegressionHead(input_dim=8)
        loss = head.compute_loss(torch.randn(4, 1), _values(torch.randn(4)))
        assert loss.shape == () and loss.item() >= 0

    def test_gradient_flows(self):
        head = RegressionHead(input_dim=8)
        X = torch.randn(2, 8, requires_grad=True)
        head.compute_loss(head(X), _values(torch.randn(2))).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0

    def test_registered(self):
        assert task_registry.get("regression") is RegressionHead

    def test_target_dtypes(self):
        assert RegressionHead.target_dtypes == {"value": torch.float}

    def test_postprocess_returns_predictions(self):
        head = RegressionHead(input_dim=8)
        out = head.postprocess(torch.randn(4, 1))
        assert "predictions" in out and out["predictions"].shape == (4,)

    def test_default_metrics(self):
        head = RegressionHead(input_dim=8)
        assert set(head.metrics) == set(DEFAULT_METRICS["regression"])

    def test_custom_metrics(self):
        head = RegressionHead(input_dim=8, metrics=["mse", "r2", "pearson"])
        assert set(head.metrics) == {"mse", "r2", "pearson"}

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            RegressionHead(input_dim=8, metrics=["auroc"])

    def test_compute_metrics_returns_requested_keys(self):
        head = RegressionHead(input_dim=8, metrics=["mse", "mae"])
        m = head.compute_metrics(torch.randn(6, 1), _values(torch.randn(6)))
        assert set(m.keys()) == {"mse", "mae"}
        assert all(isinstance(v, float) for v in m.values())

    def test_compute_metrics_default_keys(self):
        head = RegressionHead(input_dim=8)
        m = head.compute_metrics(torch.randn(6, 1), _values(torch.randn(6)))
        assert set(m.keys()) == set(DEFAULT_METRICS["regression"])

    def test_auto_params_is_empty(self):
        assert RegressionHead.auto_params(_FakeDataset(5)) == {}

    def test_extract_targets_casts_label_to_float(self):
        head = RegressionHead(input_dim=8)
        assert head.extract_targets(SimpleNamespace(label="2.5")) == {"value": 2.5}


# ---------------------------------------------------------------------------
# OrdinalClassificationHead
# ---------------------------------------------------------------------------


class TestOrdinalClassificationHead:
    def test_output_shape(self):
        head = OrdinalClassificationHead(input_dim=16, num_classes=6)
        assert head(torch.randn(4, 16)).shape == (4, 1)

    def test_loss_is_scalar(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        loss = head.compute_loss(torch.randn(4, 1), _labels([0, 2, 4, 5]))
        assert loss.shape == () and loss.item() >= 0

    def test_gradient_flows(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        X = torch.randn(3, 8, requires_grad=True)
        head.compute_loss(head(X), _labels([0, 3, 5])).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0

    def test_registered(self):
        assert task_registry.get("ordinal_classification") is OrdinalClassificationHead

    def test_target_dtypes(self):
        assert OrdinalClassificationHead.target_dtypes == {"label": torch.long}

    def test_auto_params(self):
        params = OrdinalClassificationHead.auto_params(_FakeDataset(6))
        assert params["num_classes"] == 6
        assert params["label_map"] == {i: i for i in range(6)}

    def test_postprocess_returns_integer_labels_and_raw_scores(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        raw = torch.tensor([[0.4], [1.6], [3.1], [5.8]])
        out = head.postprocess(raw)
        assert "predicted_labels" in out and "raw_scores" in out
        np.testing.assert_array_equal(out["predicted_labels"], [0, 2, 3, 5])
        np.testing.assert_allclose(out["raw_scores"], [0.4, 1.6, 3.1, 5.8], atol=1e-5)

    def test_postprocess_clips_to_class_range(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        out = head.postprocess(torch.tensor([[-2.0], [7.5]]))
        np.testing.assert_array_equal(out["predicted_labels"], [0, 5])

    def test_default_metrics(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        assert set(head.metrics) == set(DEFAULT_METRICS["ordinal_classification"])

    def test_custom_metrics(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6, metrics=["qwk", "mae"])
        assert set(head.metrics) == {"qwk", "mae"}

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics"):
            OrdinalClassificationHead(input_dim=8, num_classes=6, metrics=["auroc"])

    def test_compute_metrics_returns_requested_keys(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6, metrics=["qwk", "accuracy"])
        m = head.compute_metrics(torch.randn(8, 1), _labels([0, 1, 2, 3, 4, 5, 0, 3]))
        assert set(m.keys()) == {"qwk", "accuracy"}
        assert all(isinstance(v, float) for v in m.values())

    def test_compute_metrics_default_keys(self):
        head = OrdinalClassificationHead(input_dim=8, num_classes=6)
        m = head.compute_metrics(torch.randn(8, 1), _labels([0, 1, 2, 3, 4, 5, 0, 3]))
        assert set(m.keys()) == set(DEFAULT_METRICS["ordinal_classification"])
