"""Tests for soma.training.trainer — training loop with early stopping."""

from __future__ import annotations

import io
from pathlib import Path

import functools

import torch
import pytest
from rich.console import Console
from rich.panel import Panel

from soma.aggregators.pooling import MeanPool
from soma.config import TrainingConfig
from soma.tasks.classification import BinaryClassificationHead
from soma.tasks.regression import RegressionHead
from soma.training.collate import bag_collate_fn
from soma.training.model import MILModel
from soma.training.trainer import Trainer, TrainResult, _build_training_panel, _format_batch_progress
from soma.training.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_CLASSES = 2


def _make_model() -> MILModel:
    return MILModel(
        aggregator=MeanPool(input_dim=D),
        task_head=BinaryClassificationHead(input_dim=D, num_classes=NUM_CLASSES),
    )


def _make_epoch_log(
    epoch: int,
    train_loss: float,
    tune_loss: float,
    auroc: float,
    balanced_accuracy: float,
    f1: float,
    auprc: float,
    lr: float,
):
    from soma.training.trainer import EpochLog

    return EpochLog(
        epoch=epoch,
        train_loss=train_loss,
        tune_loss=tune_loss,
        tune_metrics={
            "auroc": auroc,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "auprc": auprc,
        },
        lr=lr,
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
    def test_format_batch_progress_returns_spinner_and_bar(self):
        progress = _format_batch_progress(3, 10, phase="train")
        assert "train 03/10 [###-------]" in progress
        assert progress[0] in "|/-\\"

    def test_build_training_panel_returns_rich_panel(self):
        log = _make_epoch_log(0, 1.2345, 0.9876, 0.75, 0.5, 0.25, 0.125, 1e-4)
        panel = _build_training_panel(
            title="Training progress",
            subtitle="epoch 1/50",
            log=log,
            total_epochs=50,
            best_epoch=0,
            best_tune_loss=0.9876,
            best_tune_metrics=log.tune_metrics,
            patience_counter=0,
            patience_limit=10,
            checkpoint_path=Path("/tmp/best_model.pt"),
            status="new best checkpoint saved at epoch 1",
            batch_progress=_format_batch_progress(3, 10, phase="train"),
        )

        assert isinstance(panel, Panel)
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, color_system=None, width=120, record=True)
        console.print(panel)
        rendered = console.export_text(clear=False)
        assert "epoch" in rendered
        assert "train" in rendered
        assert "tune" in rendered
        assert "batch" in rendered
        assert "03/10" in rendered

    def test_fit_returns_train_result(self, tmp_path: Path):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(6, seed=0)
        tune_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=3, learning_rate=1e-3, patience=10)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert isinstance(result, TrainResult)
        assert len(result.history) == 3
        assert result.best_epoch >= 0
        assert result.best_tune_loss >= 0
        assert set(result.best_tune_metrics) >= {"auroc", "balanced_accuracy", "auprc", "f1"}

    def test_fit_renders_epoch_progress_with_rich_console(
        self, tmp_path: Path
    ):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(6, seed=0)
        tune_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=2, learning_rate=1e-3, patience=10)

        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, color_system=None, width=120)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
            console=console,
        )
        result = trainer.fit()

        output = buffer.getvalue()
        assert "Training progress" in output
        assert "epoch" in output
        assert "train" in output and "tune" in output
        assert "batch" in output
        assert "auroc" in output and "bal" in output and "f1" in output
        assert "new best checkpoint" in output or "training complete" in output
        assert result.history[-1].epoch == len(result.history) - 1

    def test_loss_decreases(self, tmp_path: Path):
        """Training loss should decrease over epochs on synthetic data."""
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(8, seed=0)
        tune_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=10, learning_rate=1e-2, patience=20)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        first_loss = result.history[0].train_loss
        last_loss = result.history[-1].train_loss
        assert last_loss < first_loss

    def test_early_stopping(self, tmp_path: Path):
        """Training should stop early if tune_loss doesn't improve."""
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(4, seed=0)
        tune_loader = _make_synthetic_loader(4, seed=1)
        # Patience=2, many epochs → should stop early
        config = TrainingConfig(epochs=100, learning_rate=1e-5, patience=2)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        # Should have stopped well before 100 epochs
        assert len(result.history) < 100

    def test_checkpoint_saved(self, tmp_path: Path):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(4, seed=0)
        tune_loader = _make_synthetic_loader(4, seed=1)
        config = TrainingConfig(epochs=3, patience=10)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert result.checkpoint_path.exists()
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        assert "model_state_dict" in checkpoint


class TestTrainerWithSlideModel:
    def test_fit_with_slide_model(self, tmp_path: Path):
        """Trainer should work with SlideModel and SlideBatch (no mask)."""
        from torch.utils.data import DataLoader
        from soma.training.slide_dataset import slide_collate_fn
        from soma.training.slide_model import SlideModel

        seed_everything(42)
        D = 16
        model = SlideModel(task_head=BinaryClassificationHead(input_dim=D, num_classes=2))

        # Build slide-level batches: (D,) tensors
        slides = [(torch.randn(D), i % 2, f"s{i}") for i in range(8)]
        train_loader = DataLoader(slides[:6], batch_size=2, collate_fn=slide_collate_fn)
        tune_loader = DataLoader(slides[6:], batch_size=2, collate_fn=slide_collate_fn)

        config = TrainingConfig(epochs=3, learning_rate=1e-3, patience=10)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert isinstance(result, TrainResult)
        assert len(result.history) == 3
        assert result.checkpoint_path.exists()


class TestTrainerWithRegressionHead:
    def test_fit_with_regression_model(self, tmp_path: Path):
        """Trainer should work with RegressionHead and float labels."""
        from torch.utils.data import DataLoader

        seed_everything(42)
        D = 16
        model = MILModel(
            aggregator=MeanPool(input_dim=D),
            task_head=RegressionHead(input_dim=D),
        )

        # Build bags with continuous float labels
        torch.manual_seed(0)
        bags = [(torch.randn(5, D), float(i) * 0.5, f"slide_{i}") for i in range(8)]
        _reg_collate = functools.partial(bag_collate_fn, label_dtype=torch.float)
        train_loader = DataLoader(bags[:6], batch_size=2, collate_fn=_reg_collate)
        tune_loader = DataLoader(bags[6:], batch_size=2, collate_fn=_reg_collate)

        config = TrainingConfig(epochs=3, learning_rate=1e-3, patience=10)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert isinstance(result, TrainResult)
        assert len(result.history) == 3
        assert result.checkpoint_path.exists()
        for key in ["mae", "r2"]:
            assert key in result.best_tune_metrics


class TestSeedEverything:
    def test_deterministic(self):
        """Same seed should produce same random numbers."""
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert torch.equal(a, b)
