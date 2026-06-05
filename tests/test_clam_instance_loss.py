"""Tests for hot-path invariants in CLAM instance loss and trainer eval.

Phase 2.1: ``compute_instance_loss`` must not call ``Tensor.item()`` inside
its per-sample / per-class loop. Each ``.item()`` triggers a CUDA stream
sync; with B=8 and n_classes=4 that's ≥40 syncs per training step.

Phase 1.2: ``Trainer.evaluate_tune`` must move accumulated logits/labels to
CPU before appending to its prediction list. Otherwise GPU memory grows
linearly with the tune split size.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from soma.aggregators.mil.clam import CLAM_MB, CLAM_SB
from soma.tasks.classification import (
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
        from soma.tasks.regression import RegressionHead

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


# ---------------------------------------------------------------------------
# Phase 1.2 — Trainer evaluate_tune moves logits/labels to CPU
# ---------------------------------------------------------------------------


def _make_tune_trainer(tmp_path) -> "tuple[object, int]":
    from pathlib import Path

    from soma.aggregators.pooling import MeanPool
    from soma.config import TrainingConfig
    from soma.tasks.classification import BinaryClassificationHead
    from soma.training.collate import bag_collate_fn
    from soma.training.model import MILModel
    from soma.training.trainer import Trainer

    D = 8
    model = MILModel(
        aggregator=MeanPool(input_dim=D),
        task_head=BinaryClassificationHead(input_dim=D, num_classes=2),
    )

    import functools

    torch.manual_seed(0)
    bags = [(torch.randn(5 + i, D), {"label": i % 2}, f"s{i}") for i in range(6)]
    _collate = functools.partial(bag_collate_fn, target_dtypes={"label": torch.long})
    tune_loader = DataLoader(bags, batch_size=2, collate_fn=_collate)
    train_loader = DataLoader(bags, batch_size=2, collate_fn=_collate)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        tune_loader=tune_loader,
        config=TrainingConfig(epochs=1, patience=10),
        fold_dir=Path(tmp_path),
        device=torch.device("cpu"),
    )
    return trainer, len(tune_loader)


class TestTrainerEvalLogitsCPU:
    def test_evaluate_tune_calls_cpu_on_outputs(self, tmp_path):
        """``evaluate_tune`` must call ``.cpu()`` on logits/labels each batch.

        On the current code, ``all_logits.append(out.logits)`` keeps the tensor
        on its source device. On a CUDA training run this leaks GPU memory.
        Fix: ``.cpu()`` before append.
        """
        trainer, n_batches = _make_tune_trainer(tmp_path)

        # Spy on torch.Tensor.cpu — record the call count from the moment the
        # tune loop starts. Note: the metric-computation pass after the loop may
        # also call .cpu(), so we lower-bound by 2 * n_batches (logits + labels
        # per batch).
        original_cpu = torch.Tensor.cpu
        cpu_calls = [0]

        def counting_cpu(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            cpu_calls[0] += 1
            return original_cpu(self, *args, **kwargs)

        torch.Tensor.cpu = counting_cpu  # type: ignore[method-assign]
        try:
            trainer._tune()
        finally:
            torch.Tensor.cpu = original_cpu  # type: ignore[method-assign]

        # 2 .cpu() calls per batch (logits + labels), at minimum.
        assert cpu_calls[0] >= 2 * n_batches, (
            f"evaluate_tune called .cpu() only {cpu_calls[0]} times over "
            f"{n_batches} batches; expected ≥ {2 * n_batches} (logits + labels "
            "must be moved off the accelerator before accumulating)"
        )
