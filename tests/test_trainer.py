"""Tests for soma.training.trainer — training loop with early stopping."""

from __future__ import annotations

from pathlib import Path

import functools

import torch

from soma.aggregators.pooling import MeanPool
from soma.config import TrainingConfig
from soma.tasks.classification import BinaryClassificationHead
from soma.tasks.regression import RegressionHead
from soma.training.collate import bag_collate_fn
from soma.training.model import MILModel
from soma.training.trainer import Trainer, TrainResult, _format_batch_progress
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

    def test_fit_with_half_precision_slide_features(self, tmp_path: Path):
        """Trainer should accept cached slide features stored in fp16."""
        from torch.utils.data import DataLoader
        from soma.features import FeatureStore
        from soma.training.slide_dataset import slide_collate_fn
        from soma.training.slide_model import SlideModel

        seed_everything(42)
        D = 16
        model = SlideModel(task_head=BinaryClassificationHead(input_dim=D, num_classes=2))

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        for i in range(8):
            torch.save(torch.randn(D, dtype=torch.float16), feature_dir / f"s{i}.pt")
        store = FeatureStore(feature_dir)
        train_loader = DataLoader(
            [(store.load(f"s{i}"), i % 2, f"s{i}") for i in range(6)],
            batch_size=2,
            collate_fn=slide_collate_fn,
        )
        tune_loader = DataLoader(
            [(store.load(f"s{i}"), i % 2, f"s{i}") for i in range(6, 8)],
            batch_size=2,
            collate_fn=slide_collate_fn,
        )

        config = TrainingConfig(epochs=1, learning_rate=1e-3, patience=1)
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


class TestTrainingProgressFormatting:
    def test_format_batch_progress_uses_item_counts(self):
        text = _format_batch_progress(87, 10000, phase="train")
        assert "train 87/10000" in text

    def test_train_epoch_progress_reports_processed_items(self, tmp_path: Path):
        seed_everything(42)
        model = _make_model()
        train_loader = _make_synthetic_loader(5, seed=0)  # batch_size=2 => 2, 2, 1
        tune_loader = _make_synthetic_loader(4, seed=1)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=TrainingConfig(epochs=1, patience=10),
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )

        updates: list[tuple[str, int, int]] = []
        trainer._train_epoch(on_batch_progress=lambda phase, current, total: updates.append((phase, current, total)))

        assert updates == [
            ("train", 2, 5),
            ("train", 4, 5),
            ("train", 5, 5),
        ]
