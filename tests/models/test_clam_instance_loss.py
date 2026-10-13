"""Tests for hot-path invariants in CLAM instance loss and trainer eval.

Phase 2.1: ``compute_instance_loss`` must not call ``Tensor.item()`` inside
its per-sample / per-class loop. Each ``.item()`` triggers a CUDA stream
sync; with B=8 and n_classes=4 that's ≥40 syncs per training step.

Phase 1.2 (the eval-side CPU transfer) is covered by ``MILLearner.validation_step``,
which returns CPU tensors; the old spy on ``Tensor.cpu`` pinned the loop and is gone. Was:
``Trainer.evaluate_tune`` must move accumulated logits/labels to
CPU before appending to its prediction list. Otherwise GPU memory grows
linearly with the tune split size.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from soma.models.aggregators.mil.clam import CLAM_MB, CLAM_SB
from soma.models.heads.classification import (
    BranchAwareClassificationHead,
    MulticlassClassificationHead,
)


def _count_item_calls():
    """Monkey-patch ``Tensor.item`` to count invocations during the wrapped block."""
    original = torch.Tensor.item
    counter = [0]

    def counting_item(self):  # type: ignore[no-untyped-def]
        counter[0] += 1
        return original(self)

    torch.Tensor.item = counting_item  # type: ignore[method-assign]

    def restore() -> None:
        torch.Tensor.item = original  # type: ignore[method-assign]

    return counter, restore


# ---------------------------------------------------------------------------
# Phase 2.1 — CLAM .item() syncs
# ---------------------------------------------------------------------------


class TestCLAMNoItemSync:
    def test_clam_sb_classification_no_item_calls(self):
        torch.manual_seed(0)
        model = CLAM_SB(input_dim=16, hidden_dim=8, attn_dim=4, k_sample=2, n_classes=2)
        model.configure_for_task(MulticlassClassificationHead(input_dim=8, num_classes=2))
        x = torch.randn(3, 7, 16)
        mask = torch.ones(3, 7, dtype=torch.bool)
        mask[2, 5:] = False
        out = model(x, mask=mask)
        labels = torch.tensor([0, 1, 1])

        counter, restore = _count_item_calls()
        try:
            model.compute_instance_loss(
                out.auxiliary["attention"], out.auxiliary["embeddings"], labels, mask=mask
            )
        finally:
            restore()

        assert counter[0] == 0, (
            f".item() was called {counter[0]} times inside compute_instance_loss; "
            "expected 0 (these calls force GPU sync stalls in the inner loop)"
        )

    def test_clam_mb_classification_no_item_calls(self):
        torch.manual_seed(0)
        model = CLAM_MB(
            input_dim=16,
            hidden_dim=8,
            attn_dim=4,
            k_sample=2,
            n_classes=3,
            use_negative_class_instance_loss=True,
        )
        model.configure_for_task(BranchAwareClassificationHead(input_dim=8, num_classes=3))
        x = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 4:] = False
        out = model(x, mask=mask)
        labels = torch.tensor([0, 2])

        counter, restore = _count_item_calls()
        try:
            model.compute_instance_loss(
                out.auxiliary["attention"], out.auxiliary["embeddings"], labels, mask=mask
            )
        finally:
            restore()

        assert counter[0] == 0

    def test_clam_sb_regression_no_item_calls(self):
        torch.manual_seed(0)
        from soma.models.heads.regression import RegressionHead

        model = CLAM_SB(input_dim=16, hidden_dim=8, attn_dim=4, k_sample=2, n_classes=2)
        model.configure_for_task(RegressionHead(input_dim=8))
        x = torch.randn(2, 6, 16)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 3:] = False
        out = model(x, mask=mask)
        labels = torch.tensor([0.4, 0.8])

        counter, restore = _count_item_calls()
        try:
            model.compute_instance_loss(
                out.auxiliary["attention"], out.auxiliary["embeddings"], labels, mask=mask
            )
        finally:
            restore()

        assert counter[0] == 0
