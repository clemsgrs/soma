"""Tests for soma.training.trainer — training loop with early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.aggregators.pooling import MeanPool
from soma.config import TrainingConfig
from soma.tasks.classification import ClassificationHead
from soma.training.collate import BagBatch, bag_collate_fn
from soma.training.model import MILModel
from soma.training.trainer import Trainer, TrainResult
from soma.training.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_CLASSES = 2


def _make_model() -> MILModel:
    return MILModel(
        aggregator=MeanPool(input_dim=D),
        task_head=ClassificationHead(input_dim=D, num_classes=NUM_CLASSES),
    )


def _make_synthetic_loader(num_slides: int, seed: int = 0):
    """Create a DataLoader from synthetic bags."""
    torch.manual_seed(seed)
    bags = []
    for i in range(num_slides):
        n_tiles = 5 + i * 2
        features = torch.randn(n_tiles, D)
        label = i % NUM_CLASSES
        bags.append((features, label, f"slide_{i}"))
    from torch.utils.data import DataLoader

    return DataLoader(bags, batch_size=2, collate_fn=bag_collate_fn)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_fit_returns_train_result(self, tmp_path: Path):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(6, seed=0)
        val_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=3, learning_rate=1e-3, patience=10)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert isinstance(result, TrainResult)
        assert len(result.history) == 3
        assert result.best_epoch >= 0
        assert result.best_val_loss >= 0

    def test_loss_decreases(self, tmp_path: Path):
        """Training loss should decrease over epochs on synthetic data."""
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(8, seed=0)
        val_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=10, learning_rate=1e-2, patience=20)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        first_loss = result.history[0].train_loss
        last_loss = result.history[-1].train_loss
        assert last_loss < first_loss

    def test_early_stopping(self, tmp_path: Path):
        """Training should stop early if val_loss doesn't improve."""
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(4, seed=0)
        val_loader = _make_synthetic_loader(4, seed=1)
        # Patience=2, many epochs → should stop early
        config = TrainingConfig(epochs=100, learning_rate=1e-5, patience=2)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        # Should have stopped well before 100 epochs
        assert len(result.history) < 100

    def test_checkpoint_saved(self, tmp_path: Path):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(4, seed=0)
        val_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=3, patience=10)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert result.checkpoint_path.exists()
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        assert "model_state_dict" in checkpoint


class TestSeedEverything:
    def test_deterministic(self):
        """Same seed should produce same random numbers."""
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert torch.equal(a, b)
