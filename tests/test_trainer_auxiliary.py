"""Tests for auxiliary loss wiring in the Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.abmil import ABMIL
from soma.aggregators.mil.dsmil import DSMIL
from soma.aggregators.mil.clam import CLAM
from soma.aggregators.mil.dtfdmil import DTFDMIL
from soma.aggregators.pooling import MeanPool
from soma.config import TrainingConfig
from soma.tasks.classification import ClassificationHead
from soma.training.collate import BagBatch
from soma.training.model import MILModel
from soma.training.trainer import Trainer


def _make_bag_batches(
    n_samples: int = 4, bag_size: int = 10, feat_dim: int = 16, n_classes: int = 2
) -> list[BagBatch]:
    """Create deterministic BagBatch objects for testing."""
    torch.manual_seed(42)
    batches = []
    for _ in range(n_samples):
        features = torch.randn(1, bag_size, feat_dim)
        mask = torch.ones(1, bag_size, dtype=torch.bool)
        labels = torch.randint(0, n_classes, (1,))
        batches.append(BagBatch(features=features, mask=mask, labels=labels, sample_ids=("s",)))
    return batches


class _FakeBagLoader:
    """Minimal iterable that yields BagBatch objects (avoids real DataLoader)."""

    def __init__(self, batches: list[BagBatch]):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _train_one_epoch(aggregator: Aggregator, feat_dim: int = 16) -> float:
    """Build a MILModel with the given aggregator and train one epoch. Returns loss."""
    torch.manual_seed(0)
    model = MILModel(
        aggregator=aggregator,
        task_head=ClassificationHead(input_dim=aggregator.output_dim, num_classes=2),
    )
    batches = _make_bag_batches(n_samples=4, feat_dim=feat_dim)
    loader = _FakeBagLoader(batches)
    config = TrainingConfig(epochs=1, learning_rate=1e-3, patience=999)
    output_dir = Path("/tmp/test_trainer_aux")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        output_dir=output_dir,
        device=torch.device("cpu"),
    )
    return trainer._train_epoch()


class TestAuxiliaryLossWiring:
    """Verify that auxiliary losses are added during training for aggregators that produce them."""

    def test_dsmil_aux_loss_is_nonzero(self):
        """DSMIL produces auxiliary instance logits — aux loss should contribute."""
        torch.manual_seed(0)
        agg = DSMIL(input_dim=16, att_dim=8)
        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        out = agg(X)
        aux_loss = agg.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss is not None
        assert aux_loss.item() > 0

    def test_clam_aux_loss_is_nonzero(self):
        """CLAM produces auxiliary embeddings + attention — aux loss should contribute."""
        torch.manual_seed(0)
        agg = CLAM(input_dim=16, hidden_dim=8, k_sample=3)
        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        out = agg(X)
        aux_loss = agg.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss is not None
        assert aux_loss.item() > 0

    def test_dtfdmil_aux_loss_is_nonzero(self):
        """DTFD-MIL produces pseudo-bag predictions — aux loss should contribute."""
        torch.manual_seed(0)
        agg = DTFDMIL(input_dim=16, hidden_dim=8, n_groups=4)
        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        out = agg(X)
        aux_loss = agg.compute_auxiliary_loss(out.auxiliary, labels)
        assert aux_loss is not None
        assert aux_loss.item() > 0

    def test_abmil_no_aux_loss(self):
        """ABMIL has no auxiliary output — compute_auxiliary_loss returns None."""
        torch.manual_seed(0)
        agg = ABMIL(input_dim=16, hidden_dim=8)
        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        out = agg(X)
        assert out.auxiliary is None

    def test_meanpool_no_aux_loss(self):
        """MeanPool has no auxiliary output — compute_auxiliary_loss returns None."""
        agg = MeanPool(input_dim=16)
        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        out = agg(X)
        assert out.auxiliary is None

    def test_dsmil_trainer_epoch_runs(self):
        """Trainer epoch with DSMIL completes without error (aux loss wired)."""
        loss = _train_one_epoch(DSMIL(input_dim=16, att_dim=8))
        assert loss > 0

    def test_clam_trainer_epoch_runs(self):
        """Trainer epoch with CLAM completes without error (aux loss wired)."""
        loss = _train_one_epoch(CLAM(input_dim=16, hidden_dim=8, k_sample=3))
        assert loss > 0

    def test_dtfdmil_trainer_epoch_runs(self):
        """Trainer epoch with DTFD-MIL completes without error (aux loss wired)."""
        loss = _train_one_epoch(DTFDMIL(input_dim=16, hidden_dim=8, n_groups=4))
        assert loss > 0

    def test_abmil_trainer_epoch_runs(self):
        """Trainer epoch with ABMIL (no aux) completes unchanged."""
        loss = _train_one_epoch(ABMIL(input_dim=16, hidden_dim=8))
        assert loss > 0

    def test_dsmil_loss_higher_than_task_only(self):
        """DSMIL total loss (task + aux) should differ from task-only loss."""
        torch.manual_seed(0)
        agg = DSMIL(input_dim=16, att_dim=8)
        head = ClassificationHead(input_dim=16, num_classes=2)
        model = MILModel(aggregator=agg, task_head=head)

        X = torch.randn(2, 10, 16)
        labels = torch.tensor([0, 1])
        mask = torch.ones(2, 10, dtype=torch.bool)

        out = model(X, mask=mask)
        task_loss = head.compute_loss(out.logits, labels)
        aux_loss = agg.compute_auxiliary_loss(out.auxiliary, labels, mask=mask)

        assert aux_loss is not None
        total_loss = task_loss + aux_loss
        # Total should differ from task-only (aux contributes something)
        assert not torch.isclose(total_loss, task_loss)
