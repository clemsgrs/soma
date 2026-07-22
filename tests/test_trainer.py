"""Tests for soma.training.trainer — training loop with early stopping."""

from __future__ import annotations

from pathlib import Path

import functools

import torch
from rich.console import Console

from soma.aggregators.pooling import MeanPool
from soma.config import TrainingConfig
from soma.tasks.classification import BinaryClassificationHead
from soma.tasks.regression import RegressionHead
from soma.training.collate import bag_collate_fn
from soma.training.model import MILModel
from soma.training.trainer import (
    EpochLog,
    Trainer,
    TrainResult,
    _build_training_panel,
    _format_batch_progress,
    _is_monitor_improvement,
    _resolve_monitor_value,
    peak_per_metric,
)
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


_CLS_COLLATE = functools.partial(bag_collate_fn, target_dtypes={"label": torch.long})


def _make_synthetic_loader(num_slides: int, seed: int = 0):
    """Create a DataLoader from synthetic bags."""
    torch.manual_seed(seed)
    bags = []
    for i in range(num_slides):
        n_tiles = 5 + i * 2
        features = torch.randn(n_tiles, D)
        label = i % NUM_CLASSES
        bags.append((features, {"label": label}, f"slide_{i}"))
    from torch.utils.data import DataLoader

    return DataLoader(bags, batch_size=2, collate_fn=_CLS_COLLATE)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_default_monitor_resolves_tune_loss(self):
        config = TrainingConfig()

        value = _resolve_monitor_value(config, tune_loss=0.25, tune_metrics={"balanced_accuracy": 0.8})

        assert value == 0.25

    def test_metric_monitor_resolves_tune_metric(self):
        config = TrainingConfig(monitor="balanced_accuracy", monitor_mode="max")

        value = _resolve_monitor_value(config, tune_loss=0.25, tune_metrics={"balanced_accuracy": 0.8})

        assert value == 0.8

    def test_metric_monitor_rejects_missing_metric(self):
        config = TrainingConfig(monitor="balanced_accuracy", monitor_mode="max")

        try:
            _resolve_monitor_value(config, tune_loss=0.25, tune_metrics={"auroc": 0.7})
        except ValueError as exc:
            assert "balanced_accuracy" in str(exc)
            assert "auroc" in str(exc)
        else:
            raise AssertionError("Expected missing monitor metric to be rejected")

    def test_monitor_improvement_supports_min_and_max(self):
        assert _is_monitor_improvement(0.2, 0.3, "min")
        assert not _is_monitor_improvement(0.4, 0.3, "min")
        assert _is_monitor_improvement(0.8, 0.7, "max")
        assert not _is_monitor_improvement(0.6, 0.7, "max")

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
        assert result.selected_epoch >= 0
        assert result.selected_tune_loss >= 0
        assert set(result.selected_tune_metrics) >= {"auroc", "balanced_accuracy", "auprc", "f1"}

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


def _anticorrelated_bags(*, flip: bool, seed: int = 0, num_slides: int = 6):
    """Train/tune bags over identical features with mirror-image labels.

    Fitting the train mapping therefore *worsens* the tune monitor monotonically, so
    the best-monitor epoch is deterministically the first and never the last — the
    only setup in which `best` and `last` selection are distinguishable.
    """
    torch.manual_seed(seed)
    return [
        (torch.randn(5 + i * 2, D), {"label": (i + 1) % 2 if flip else i % 2}, f"slide_{i}")
        for i in range(num_slides)
    ]


def _states_match(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


class TestCheckpointSelection:
    """`checkpoint_selection` governs WHICH epoch's weights are evaluated (#282)."""

    EPOCHS = 5

    def _fit(self, tmp_path: Path, **overrides):
        from torch.utils.data import DataLoader

        seed_everything(42)
        model = _make_model()
        train_loader = DataLoader(
            _anticorrelated_bags(flip=False), batch_size=2, collate_fn=_CLS_COLLATE
        )
        tune_loader = DataLoader(
            _anticorrelated_bags(flip=True), batch_size=2, collate_fn=_CLS_COLLATE
        )
        config = TrainingConfig(epochs=self.EPOCHS, learning_rate=1e-1, **overrides)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            tune_loader=tune_loader,
            config=config,
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        return model, trainer.fit()

    def test_best_selection_keeps_the_best_monitor_epoch(self, tmp_path: Path):
        model, result = self._fit(tmp_path, patience=None)

        assert result.selected_epoch == 0
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        assert checkpoint["epoch"] == 0
        assert not _states_match(checkpoint["model_state_dict"], model.state_dict())

    def test_best_selection_early_stops_on_a_worsening_monitor(self, tmp_path: Path):
        _, result = self._fit(tmp_path, patience=2)

        assert len(result.history) < self.EPOCHS

    def test_last_selection_never_early_stops(self, tmp_path: Path):
        _, result = self._fit(tmp_path, patience=None, checkpoint_selection="last")

        assert len(result.history) == self.EPOCHS

    def test_last_selection_rejects_unknown_monitor(self, tmp_path: Path):
        """Direct Trainer callers get the same monitor validation as PipelineConfig."""
        try:
            self._fit(
                tmp_path,
                patience=None,
                checkpoint_selection="last",
                monitor="not_a_metric",
            )
        except ValueError as exc:
            assert "not_a_metric" in str(exc)
            assert "balanced_accuracy" in str(exc)
        else:
            raise AssertionError("Expected last-checkpoint training to validate its monitor")

    def test_last_selection_still_logs_per_epoch_tune_metrics(self, tmp_path: Path):
        """Tune metrics stay diagnostics under `last` — computed and logged, never used
        to pick the checkpoint."""
        _, result = self._fit(tmp_path, patience=None, checkpoint_selection="last")

        assert len(result.history) == self.EPOCHS
        for log in result.history:
            assert set(log.tune_metrics) >= {"auroc", "balanced_accuracy", "auprc", "f1"}
        # The monitor worsens every epoch, so a best-selected run would report epoch 0.
        assert result.selected_tune_loss == result.history[-1].tune_loss
        assert result.selected_tune_metrics == result.history[-1].tune_metrics

    def test_last_selection_saves_final_epoch_weights(self, tmp_path: Path):
        model, result = self._fit(tmp_path, patience=None, checkpoint_selection="last")

        assert result.selected_epoch == self.EPOCHS - 1
        checkpoint = torch.load(result.checkpoint_path, weights_only=True)
        assert checkpoint["epoch"] == self.EPOCHS - 1
        assert _states_match(checkpoint["model_state_dict"], model.state_dict())


class TestTrainerWithEmbeddingModel:
    def test_fit_with_embedding_model(self, tmp_path: Path):
        """Trainer should work with EmbeddingModel and SampleBatch (no mask)."""
        from torch.utils.data import DataLoader
        from soma.training.sample_dataset import sample_collate_fn
        from soma.training.model import EmbeddingModel

        seed_everything(42)
        D = 16
        model = EmbeddingModel(task_head=BinaryClassificationHead(input_dim=D, num_classes=2))
        _collate = functools.partial(sample_collate_fn, target_dtypes={"label": torch.long})

        # Build single-embedding batches: (D,) tensors
        slides = [(torch.randn(D), {"label": i % 2}, f"s{i}") for i in range(8)]
        train_loader = DataLoader(slides[:6], batch_size=2, collate_fn=_collate)
        tune_loader = DataLoader(slides[6:], batch_size=2, collate_fn=_collate)

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

    def test_fit_with_half_precision_features(self, tmp_path: Path):
        """Trainer should accept cached features stored in fp16."""
        from torch.utils.data import DataLoader
        from soma.features import FeatureStore
        from soma.training.sample_dataset import sample_collate_fn
        from soma.training.model import EmbeddingModel

        seed_everything(42)
        D = 16
        model = EmbeddingModel(task_head=BinaryClassificationHead(input_dim=D, num_classes=2))
        _collate = functools.partial(sample_collate_fn, target_dtypes={"label": torch.long})

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        for i in range(8):
            torch.save(torch.randn(D, dtype=torch.float16), feature_dir / f"s{i}.pt")
        store = FeatureStore(feature_dir)
        train_loader = DataLoader(
            [(store.load(f"s{i}"), {"label": i % 2}, f"s{i}") for i in range(6)],
            batch_size=2,
            collate_fn=_collate,
        )
        tune_loader = DataLoader(
            [(store.load(f"s{i}"), {"label": i % 2}, f"s{i}") for i in range(6, 8)],
            batch_size=2,
            collate_fn=_collate,
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
        bags = [(torch.randn(5, D), {"value": float(i) * 0.5}, f"slide_{i}") for i in range(8)]
        _reg_collate = functools.partial(bag_collate_fn, target_dtypes={"value": torch.float})
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
            assert key in result.selected_tune_metrics


class _TwoKeyHead(BinaryClassificationHead):
    """A head with two target keys to exercise the multi-key target contract.

    PR-1 ships only single-key heads, so this fake head is the only coverage of
    ``stack_targets`` / per-key concatenation / the multi-key ``_auxiliary_target``
    branch that survival (PR-2) will rely on.
    """

    target_dtypes = {"label": torch.long, "weight": torch.float}

    def extract_targets(self, record):  # pragma: no cover - not used in this test
        return {"label": int(record.label), "weight": 1.0}

    def compute_loss(self, predictions, targets):
        # Touch both keys so a missing/mis-typed key would raise.
        base = super().compute_loss(predictions, {"label": targets["label"]})
        return base * targets["weight"].mean()

    def compute_metrics(self, raw_output, targets):
        assert set(targets) == {"label", "weight"}
        return super().compute_metrics(raw_output, {"label": targets["label"]})


class TestMultiKeyTargets:
    def test_two_key_targets_train_and_tune(self, tmp_path: Path):
        """Multi-key targets flow through collate, _train_epoch, and per-key cat in _tune."""
        seed_everything(0)
        model = MILModel(aggregator=MeanPool(input_dim=D), task_head=_TwoKeyHead(input_dim=D, num_classes=2))
        collate = functools.partial(
            bag_collate_fn, target_dtypes={"label": torch.long, "weight": torch.float}
        )
        from torch.utils.data import DataLoader

        bags = [
            (torch.randn(5 + i, D), {"label": i % 2, "weight": 0.5 + i}, f"s{i}")
            for i in range(6)
        ]
        loader = DataLoader(bags, batch_size=2, collate_fn=collate)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            tune_loader=loader,
            config=TrainingConfig(epochs=1, patience=10),
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        result = trainer.fit()

        assert isinstance(result, TrainResult)
        assert len(result.history) == 1


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

    def test_training_panel_labels_selected_checkpoint_by_monitor_value(self):
        panel = _build_training_panel(
            title="Training progress",
            subtitle="epoch 2/10 | tune",
            log=None,
            total_epochs=10,
            selected_epoch=1,
            selected_tune_loss=0.95,
            selected_tune_metrics={"balanced_accuracy": 0.8, "auroc": 0.7},
            monitor_name="balanced_accuracy",
            selected_monitor_value=0.8,
            patience_counter=0,
            patience_limit=10,
            status="new selected checkpoint saved at epoch 2",
            trainable_param_count=1234,
            fold=None,
            num_folds=1,
            elapsed_seconds=0.0,
            avg_epoch_seconds=None,
            eta_seconds=None,
            batch_progress=None,
        )
        console = Console(record=True, width=120)
        console.print(panel)
        rendered = console.export_text()

        assert "selected" in rendered
        assert "balanced_accuracy=0.8000 @ 02" in rendered
        assert "0.9500 @ 02" not in rendered
        assert "best metrics" not in rendered

    def test_training_panel_shows_fold_position_for_multifold_runs(self):
        panel = _build_training_panel(
            title="Training progress",
            subtitle="epoch 1/10 | train",
            log=None,
            total_epochs=10,
            selected_epoch=0,
            selected_tune_loss=float("inf"),
            selected_tune_metrics={},
            monitor_name="tune_loss",
            selected_monitor_value=float("inf"),
            patience_counter=0,
            patience_limit=10,
            status="waiting for epoch 1",
            trainable_param_count=1234,
            fold=1,
            num_folds=3,
            elapsed_seconds=0.0,
            avg_epoch_seconds=None,
            eta_seconds=None,
            batch_progress=None,
        )
        console = Console(record=True, width=100)
        console.print(panel)
        rendered = console.export_text()

        assert "fold" in rendered.lower()
        assert "2/3" in rendered

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


class TestPeakPerMetric:
    """peak_per_metric: diagnostic per-metric best across epochs (1-based epoch)."""

    def _log(self, epoch: int, metrics: dict[str, float]) -> EpochLog:
        return EpochLog(
            epoch=epoch,
            train_loss=0.5,
            tune_loss=0.5,
            tune_metrics=metrics,
            lr=1e-4,
        )

    def test_reports_per_metric_peak_with_one_based_epoch(self):
        history = [
            self._log(0, {"auroc": 0.55, "rare_dice": 0.40}),
            self._log(1, {"auroc": 0.70, "rare_dice": 0.35}),
            self._log(2, {"auroc": 0.65, "rare_dice": 0.50}),
        ]

        peaks = peak_per_metric(history)

        # 1-based epochs: auroc peaks at the 2nd epoch, rare_dice at the 3rd.
        assert peaks == {
            "auroc": {"epoch": 2, "value": 0.70},
            "rare_dice": {"epoch": 3, "value": 0.50},
        }

    def test_excludes_tune_loss_and_skips_non_finite(self):
        history = [
            self._log(0, {"auroc": float("nan")}),
            self._log(1, {"auroc": 0.60}),
        ]

        peaks = peak_per_metric(history)

        # tune_loss is not a tune_metric, so it never appears; the NaN epoch is skipped.
        assert "tune_loss" not in peaks
        assert peaks == {"auroc": {"epoch": 2, "value": 0.60}}

    def test_empty_history_is_empty(self):
        assert peak_per_metric([]) == {}
