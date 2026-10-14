"""The learning seam: ``Learner`` contracts, the generic ``Trainer`` and its callbacks.

Every test goes through the public API of :mod:`soma.learn` (and ``soma.fit``); none
reaches into the loop.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from soma.models.aggregators.mil.abmil import ABMIL
from soma.models.aggregators.mil.clam import CLAM_SB
from soma.models.aggregators.pooling import MeanPool
from soma.learn import (
    Budget,
    Checkpoint,
    EarlyStopping,
    FitResult,
    History,
    MILLearner,
    SchedulerSpec,
    StepOutput,
    Trainer,
)
from soma.models.heads.classification import BinaryClassificationHead
from soma.models.heads.regression import RegressionHead
from soma.models.heads.survival import CoxSurvivalHead, cox_breslow_loss
from soma.learn.collate import BagBatch, CoxWindowBatch, bag_collate_fn
from soma.models.model import EmbeddingModel, MILModel
from soma.learn.sample_dataset import SampleBatch

D = 16


def _bags(n: int = 8, seed: int = 0) -> list[tuple[Tensor, dict[str, int], str]]:
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        label = i % 2
        features = torch.randn(5 + i, D, generator=g) + (2.0 if label else -2.0)
        items.append((features, {"label": label}, f"s{i}"))
    return items


def _bag_loader(n: int = 8, batch_size: int = 2, seed: int = 0) -> DataLoader:
    return DataLoader(
        _bags(n, seed),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(bag_collate_fn, target_dtypes={"label": torch.long}),
    )


def _learner(aggregator=None, **kwargs) -> MILLearner:
    torch.manual_seed(0)
    aggregator = aggregator or MeanPool(input_dim=D)
    head = BinaryClassificationHead(input_dim=aggregator.output_dim, num_classes=2)
    return MILLearner(MILModel(aggregator=aggregator, task_head=head), **kwargs)


# ---------------------------------------------------------------------------
# Learner contract
# ---------------------------------------------------------------------------


class TestMILLearnerContract:
    def test_training_step_returns_a_scalar_loss_with_gradients(self):
        learner = _learner()
        batch = next(iter(_bag_loader()))
        loss = learner.training_step(batch)
        assert loss.ndim == 0 and loss.requires_grad

    def test_validation_summary_reduces_over_the_explicit_list(self):
        learner = _learner()
        outputs = [learner.validation_step(batch) for batch in _bag_loader()]
        assert all(isinstance(o, StepOutput) for o in outputs)
        summary = learner.validation_summary(outputs)
        assert "loss" in summary and math.isfinite(summary["loss"])
        assert {"balanced_accuracy", "auroc"} <= set(summary)

    def test_configure_optimizers_defaults_to_adam_without_a_scheduler(self):
        optimizer, schedulers = _learner().configure_optimizers(
            Budget(max_epochs=7, max_steps=None, updates_per_epoch=4)
        )
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
        assert schedulers == []

    @pytest.mark.parametrize("leaf", [None, "none", {"name": "none"}])
    def test_a_disabled_scheduler_leaf_installs_nothing_in_either_form(self, leaf):
        _, schedulers = _learner(schedulers=[leaf, "cosine"]).configure_optimizers(
            Budget(max_epochs=7, max_steps=None, updates_per_epoch=4)
        )
        assert len(schedulers) == 1
        assert isinstance(schedulers[0].scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_derives_t_max_from_the_epoch_budget(self):
        learner = _learner(schedulers=["cosine"])
        _, schedulers = learner.configure_optimizers(
            Budget(max_epochs=7, max_steps=None, updates_per_epoch=4)
        )
        (spec,) = schedulers
        assert isinstance(spec, SchedulerSpec)
        assert isinstance(spec.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert spec.scheduler.T_max == 7 and spec.interval == "epoch"

    def test_cosine_derives_t_max_and_step_interval_from_a_step_budget(self):
        learner = _learner(schedulers=["cosine"])
        _, (spec,) = learner.configure_optimizers(
            Budget(max_epochs=None, max_steps=12, updates_per_epoch=5)
        )
        assert spec.scheduler.T_max == 12 and spec.interval == "step"

    def test_cosine_step_interval_under_an_epoch_budget_spans_every_update(self):
        learner = _learner(schedulers=[{"name": "cosine", "interval": "step"}])
        _, (spec,) = learner.configure_optimizers(
            Budget(max_epochs=4, max_steps=None, updates_per_epoch=10)
        )
        assert spec.scheduler.T_max == 40 and spec.interval == "step"

    def test_cosine_with_an_explicit_t_max_keeps_the_step_cadence_of_a_step_budget(self):
        learner = _learner(schedulers=[{"name": "cosine", "params": {"T_max": 40}}])
        _, (spec,) = learner.configure_optimizers(
            Budget(max_epochs=None, max_steps=40, updates_per_epoch=10)
        )
        assert spec.scheduler.T_max == 40 and spec.interval == "step"

    def test_optimizer_and_scheduler_leaves_resolve_through_the_registry(self):
        learner = _learner(
            optimizer={"name": "sgd", "params": {"lr": 0.5, "momentum": 0.9}},
            schedulers=[{"name": "constant", "params": {"factor": 0.25, "total_iters": 2}}],
        )
        optimizer, (spec,) = learner.configure_optimizers(
            Budget(max_epochs=3, max_steps=None, updates_per_epoch=1)
        )
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert isinstance(spec.scheduler, torch.optim.lr_scheduler.ConstantLR)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.125)

    def test_string_leaf_is_the_name(self):
        learner = _learner(optimizer="adamw", schedulers=["cosine"])
        optimizer, (spec,) = learner.configure_optimizers(
            Budget(max_epochs=3, max_steps=None, updates_per_epoch=1)
        )
        assert isinstance(optimizer, torch.optim.AdamW)
        assert isinstance(spec.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_a_dotted_path_leaf_imports_the_class(self):
        learner = _learner(optimizer={"name": "torch.optim.RMSprop", "params": {"lr": 0.01}})
        optimizer, _ = learner.configure_optimizers(
            Budget(max_epochs=1, max_steps=None, updates_per_epoch=1)
        )
        assert isinstance(optimizer, torch.optim.RMSprop)

    def test_transfer_batch_walks_the_dataclass_and_keeps_its_type(self):
        learner = _learner()
        batch = next(iter(_bag_loader()))
        moved = learner.transfer_batch(batch, torch.device("cpu"))
        assert isinstance(moved, BagBatch)
        assert torch.equal(moved.features, batch.features)
        assert moved.sample_ids == batch.sample_ids

    def test_transfer_batch_moves_tensors_inside_named_tuples(self):
        from typing import NamedTuple

        class Pair(NamedTuple):
            x: Tensor
            y: dict[str, Tensor]

        batch = Pair(torch.zeros(2), {"label": torch.zeros(2)})
        moved = _learner().transfer_batch(batch, torch.device("meta"))
        assert isinstance(moved, Pair)
        assert moved.x.device.type == "meta" and moved.y["label"].device.type == "meta"

    def test_trainable_state_dict_round_trips(self):
        learner = _learner()
        state = {k: v.clone() for k, v in learner.trainable_state_dict().items()}
        with torch.no_grad():
            for p in learner.parameters():
                p.add_(1.0)
        learner.load_trainable_state_dict(state)
        for k, v in learner.trainable_state_dict().items():
            assert torch.equal(v, state[k])

    def test_loss_callable_replaces_the_head_loss(self):
        calls = []

        def hinge(predictions, targets):
            calls.append(1)
            return predictions.sum() * 0.0 + 3.0

        learner = _learner(loss=hinge)
        loss = learner.training_step(next(iter(_bag_loader())))
        assert calls and loss.item() == pytest.approx(3.0)

    def test_loss_callable_is_also_the_validation_loss(self):
        def constant(predictions, targets):
            return predictions.sum() * 0.0 + 3.0

        learner = _learner(loss=constant)
        learner.eval()
        batches = list(_bag_loader())
        summary = learner.validation_summary([learner.validation_step(b) for b in batches])
        assert summary["loss"] == pytest.approx(3.0)
        # The full-cohort reduction (Cox-style heads) goes through the same callable.
        learner.head.full_cohort_eval_loss = True
        summary = learner.validation_summary([learner.validation_step(b) for b in batches])
        assert summary["loss"] == pytest.approx(3.0)

    def test_bf16_validation_hands_float_logits_to_the_head_metrics(self):
        result = Trainer(max_epochs=1, device="cpu", precision="bf16").fit(
            _learner(), _bag_loader(), _bag_loader()
        )
        row = result.history.iloc[0]
        assert math.isfinite(row["loss"]) and math.isfinite(row["auroc"])

    def test_embedding_model_is_the_set_shape_with_one_row(self):
        torch.manual_seed(0)
        learner = MILLearner(EmbeddingModel(task_head=RegressionHead(input_dim=D)))
        batch = SampleBatch(
            features=torch.randn(4, D),
            targets={"value": torch.randn(4)},
            sample_ids=("a", "b", "c", "d"),
        )
        assert learner.training_step(batch).ndim == 0
        summary = learner.validation_summary([learner.validation_step(batch)])
        assert "loss" in summary and "mae" in summary


class TestAuxiliaryLoss:
    def test_clam_auxiliary_term_composes_on_top_of_the_head_loss(self):
        agg = CLAM_SB(input_dim=D, hidden_dim=8, attn_dim=4, k_sample=2, bag_weight=0.7)
        learner = _learner(agg)
        batch = next(iter(_bag_loader()))
        out = learner.forward_batch(batch)
        head_loss = learner.model.task_head.compute_loss(out.logits, batch.targets)
        total = learner.compute_loss(out, batch)
        assert not torch.isclose(total, head_loss)

    def test_label_aware_auxiliary_loss_refuses_a_head_without_a_label(self):
        from soma.models.aggregators.mil.dtfdmil import DTFDMIL

        torch.manual_seed(0)
        agg = DTFDMIL(input_dim=D, embed_dim=D, hidden_dim=8, n_groups=2)
        head = RegressionHead(input_dim=agg.output_dim)  # targets carry "value", no "label"
        learner = MILLearner(MILModel(aggregator=agg, task_head=head))
        batch = BagBatch(
            features=torch.randn(2, 6, D),
            mask=torch.ones(2, 6, dtype=torch.bool),
            targets={"value": torch.tensor([3.0, 1.0])},
            sample_ids=("a", "b"),
        )
        with pytest.raises(ValueError, match="label"):
            learner.training_step(batch)


class TestCoxWindow:
    """The Cox prediction-accumulation window is a batch like any other."""

    @staticmethod
    def _bags_of(sizes, seed=0):
        g = torch.Generator().manual_seed(seed)
        return [torch.randn(n, D, generator=g) for n in sizes]

    def _learner(self, cox_window: int) -> MILLearner:
        torch.manual_seed(0)
        agg = ABMIL(input_dim=D)
        head = CoxSurvivalHead(input_dim=agg.output_dim, cox_window=cox_window)
        return MILLearner(MILModel(aggregator=agg, task_head=head))

    def test_window_bags_stay_graph_connected_through_one_loss(self):
        learner = self._learner(cox_window=3)
        window = CoxWindowBatch(
            bags=self._bags_of([5, 8, 3]),
            targets={"time": torch.tensor([3.0, 2.0, 1.0]), "event": torch.tensor([1.0, 0.0, 1.0])},
            sample_ids=("a", "b", "c"),
        )
        loss = learner.training_step(window)
        loss.backward()
        grads = [p.grad for p in learner.model.aggregator.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_transfer_batch_keeps_the_window_bags_on_the_host(self):
        # A window of large bags must not sit on the accelerator in full: each bag is
        # moved right before its own forward, the targets up front.
        learner = self._learner(cox_window=3)
        window = CoxWindowBatch(
            bags=self._bags_of([5, 8, 3]),
            targets={"time": torch.tensor([3.0, 2.0, 1.0]), "event": torch.tensor([1.0, 0.0, 1.0])},
            sample_ids=("a", "b", "c"),
        )
        moved = learner.transfer_batch(window, torch.device("meta"))
        assert isinstance(moved, CoxWindowBatch)
        assert all(bag.device.type == "cpu" for bag in moved.bags)
        assert all(t.device.type == "meta" for t in moved.targets.values())
        assert learner.training_step(window).isfinite()

    def test_validation_loss_is_the_full_cohort_loss_whatever_the_batching(self):
        learner = self._learner(cox_window=2)
        learner.eval()
        bags = self._bags_of([5, 8, 3, 6, 4, 7])
        time = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        event = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]  # leading batch all-censored
        collate = functools.partial(
            bag_collate_fn, target_dtypes={"time": torch.float, "event": torch.float}
        )
        items = [(bag, {"time": time[i], "event": event[i]}, f"s{i}") for i, bag in enumerate(bags)]
        whole = [collate(items)]
        pairs = [collate(items[i : i + 2]) for i in range(0, 6, 2)]

        with torch.no_grad():
            risk = torch.cat([learner.model(b.unsqueeze(0)).logits.view(1) for b in bags])
        reference = float(cox_breslow_loss(risk, torch.tensor(time), torch.tensor(event)))

        for loader in (whole, pairs):
            summary = learner.validation_summary([learner.validation_step(b) for b in loader])
            assert summary["loss"] == pytest.approx(reference, abs=1e-5)


class TestDenseLearnerTransfer:
    def test_pixels_go_to_the_encoder_stage_input_device_in_one_copy(self, monkeypatch):
        """Targets move to the training device; pixels stay where the frozen stage takes
        them (its ``input_device``), moved once."""
        from soma.learn import DenseLearner, GridBatch
        from soma.testing import FakeEncoder

        class StagedModel(torch.nn.Module):
            head = None

            def __init__(self):
                super().__init__()
                self.encoder = FakeEncoder(target_size=2, patch_size=1)

        batch = GridBatch(
            features=None,
            pixels=torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
            targets={"mask": torch.zeros(1, 2, 2)},
            sample_ids=("a",),
        )
        receivers: list[Tensor] = []
        original_to = torch.Tensor.to

        def spy(self, *args, **kwargs):
            receivers.append(self)
            return original_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", spy)
        moved = DenseLearner(StagedModel()).transfer_batch(batch, torch.device("meta"))
        assert moved.pixels.device.type == "cpu"
        assert moved.targets["mask"].device.type == "meta"
        assert sum(r is batch.pixels for r in receivers) == 1


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestTrainerBudget:
    def test_fit_returns_a_result_with_one_history_row_per_epoch(self):
        learner = _learner()
        result = Trainer(max_epochs=3, device="cpu").fit(learner, _bag_loader(), _bag_loader())
        assert isinstance(result, FitResult)
        assert result.learner is learner
        assert isinstance(result.history, pd.DataFrame)
        assert result.history["epoch"].tolist() == [0, 1, 2]
        assert {"train_loss", "lr", "elapsed_seconds", "loss", "auroc"} <= set(result.history)
        assert result.optimizer_steps == 12
        assert result.selected_epoch == 2 and result.selection_monitor[0] == "loss"

    def test_step_budget_counts_accumulation_windows_and_allows_a_partial_last_epoch(self):
        learner = _learner()
        # 8 bags / batch 2 = 4 batches; accumulate 2 -> 2 updates per epoch; 5 steps = 3 epochs.
        result = Trainer(max_steps=5, accumulate_grad_batches=2, device="cpu").fit(
            learner, _bag_loader(), _bag_loader()
        )
        assert result.optimizer_steps == 5
        assert result.history["epoch"].tolist() == [0, 1, 2]

    def test_validation_every_n_epochs_always_includes_the_final_epoch(self):
        learner = _learner()
        result = Trainer(max_epochs=5, val_every_n_epochs=2, device="cpu").fit(
            learner, _bag_loader(), _bag_loader()
        )
        evaluated = result.history["loss"].notna().tolist()
        assert evaluated == [False, True, False, True, True]

    def test_no_tune_loader_means_no_validation_and_last_epoch_selection(self):
        learner = _learner()
        result = Trainer(max_epochs=2, device="cpu").fit(learner, _bag_loader())
        assert "loss" not in result.history.columns
        assert result.selected_epoch == 1

    def test_an_empty_tune_loader_is_no_validation_and_early_stopping_is_inert(self, tmp_path):
        """A fold whose tune split holds only placeholder samples trains to the budget and
        keeps the final epoch; the installed EarlyStopping never fires."""
        empty = DataLoader(
            [], batch_size=1, collate_fn=functools.partial(bag_collate_fn, target_dtypes={"label": torch.long})
        )
        result = Trainer(
            max_epochs=3, callbacks=[Checkpoint(dirpath=tmp_path), EarlyStopping(patience=1)], device="cpu"
        ).fit(_learner(), _bag_loader(), empty)
        assert result.selected_epoch == 2 and len(result.history) == 3
        assert torch.load(result.checkpoint_path, weights_only=True)["selection"] == {"strategy": "last"}

    def test_epoch_schedulers_step_per_epoch_and_step_schedulers_per_update(self):
        learner = _learner(
            optimizer={"name": "sgd", "params": {"lr": 1.0}},
            schedulers=[
                {"name": "constant", "params": {"factor": 0.5, "total_iters": 1}},
                "cosine",
            ],
        )
        result = Trainer(max_steps=6, device="cpu").fit(learner, _bag_loader(), _bag_loader())
        # 4 updates per epoch; cosine T_max=6 stepped per update, warm-up lifted after epoch 1.
        lrs = result.history["lr"].tolist()
        cosine = lambda t: 0.5 * (1 + math.cos(math.pi * t / 6))  # noqa: E731
        assert lrs[0] == pytest.approx(0.5 * cosine(4))
        assert lrs[1] == pytest.approx(cosine(6), abs=1e-9)

    def test_set_epoch_is_called_on_a_batch_sampler_that_has_it(self):
        from torch.utils.data import BatchSampler, SequentialSampler

        class Recording(BatchSampler):
            epochs: list[int] = []

            def set_epoch(self, epoch: int) -> None:
                self.epochs.append(epoch)

        items = _bags()
        loader = DataLoader(
            items,
            batch_sampler=Recording(SequentialSampler(items), batch_size=2, drop_last=False),
            collate_fn=functools.partial(bag_collate_fn, target_dtypes={"label": torch.long}),
        )
        Trainer(max_epochs=2, device="cpu").fit(_learner(), loader)
        assert Recording.epochs == [0, 1]

    def test_training_reduces_the_loss(self):
        learner = _learner(optimizer={"name": "adam", "params": {"lr": 1e-2}})
        result = Trainer(max_epochs=10, device="cpu").fit(learner, _bag_loader(), _bag_loader())
        train_loss = result.history["train_loss"]
        assert train_loss.iloc[-1] < 0.5 * train_loss.iloc[0]


class TestCallbacks:
    def test_hooks_fire_in_order_and_should_stop_ends_the_run(self):
        events: list[str] = []

        class Spy:
            def on_fit_start(self, trainer, learner):
                events.append("fit_start")

            def on_epoch_start(self, trainer, learner):
                events.append(f"epoch_start:{trainer.epoch}")

            def on_train_batch_end(self, trainer, learner, loss):
                events.append("batch")

            def on_validation_end(self, trainer, learner, summary):
                events.append("validation")

            def on_epoch_end(self, trainer, learner):
                events.append(f"epoch_end:{trainer.epoch}")
                if trainer.epoch == 1:
                    trainer.should_stop = True

            def on_fit_end(self, trainer, learner):
                events.append("fit_end")

        Trainer(max_epochs=5, callbacks=[Spy()], device="cpu").fit(
            _learner(), _bag_loader(), _bag_loader()
        )
        assert events == [
            "fit_start",
            "epoch_start:0", *["batch"] * 4, "validation", "epoch_end:0",
            "epoch_start:1", *["batch"] * 4, "validation", "epoch_end:1",
            "fit_end",
        ]

    def test_checkpoint_restores_the_best_epoch_in_memory(self, tmp_path: Path):
        class Monotone:
            """Summaries scripted so epoch 1 is the best: 0.9, 0.2, 0.7."""

            values = iter([0.9, 0.2, 0.7])

            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = next(self.values)

        learner = _learner()
        snapshots = {}

        class Snap:
            def on_epoch_end(self, trainer, learner):
                snapshots[trainer.epoch] = {
                    k: v.clone() for k, v in learner.trainable_state_dict().items()
                }

        checkpoint = Checkpoint(monitor="score", mode="min")
        result = Trainer(
            max_epochs=3, callbacks=[Monotone(), Snap(), checkpoint], device="cpu"
        ).fit(learner, _bag_loader(), _bag_loader())
        assert result.selected_epoch == 1
        assert result.selection_monitor == ("score", 0.2)
        assert result.checkpoint_path is None
        for k, v in learner.trainable_state_dict().items():
            assert torch.equal(v, snapshots[1][k])

    def test_checkpoint_refuses_an_optimized_parameter_outside_the_trainable_state(self):
        class WithAux(MILLearner):
            """Owns a trainable module but keeps the default (model-only) trainable state."""

            def __init__(self, model):
                super().__init__(model)
                self.aux = torch.nn.Linear(D, 1)

        with pytest.raises(ValueError, match=r"aux\.weight"):
            Trainer(max_epochs=1, callbacks=[Checkpoint()], device="cpu").fit(
                WithAux(_learner().model), _bag_loader(), _bag_loader()
            )

    def test_checkpoint_restores_learner_owned_state_the_learner_declares(self):
        class WithAux(MILLearner):
            def __init__(self, model):
                super().__init__(model)
                self.aux = torch.nn.Linear(D, 1)

            def training_step(self, batch):
                return super().training_step(batch) + self.aux(batch.features).mean()

            def trainable_state_dict(self):
                return self.state_dict()

            def load_trainable_state_dict(self, state):
                self.load_state_dict(state)

        class Monotone:
            values = iter([0.9, 0.2, 0.7])

            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = next(self.values)

        snapshots = {}

        class Snap:
            def on_epoch_end(self, trainer, learner):
                snapshots[trainer.epoch] = learner.aux.weight.detach().clone()

        learner = WithAux(_learner().model)
        result = Trainer(
            max_epochs=3, callbacks=[Monotone(), Snap(), Checkpoint(monitor="score")], device="cpu"
        ).fit(learner, _bag_loader(), _bag_loader())
        assert result.selected_epoch == 1
        assert torch.equal(learner.aux.weight, snapshots[1])
        assert not torch.equal(snapshots[1], snapshots[2])

    def test_best_model_file_loads_into_the_bare_model_when_the_learner_widens_its_state(
        self, tmp_path: Path
    ):
        class WithAux(MILLearner):
            def __init__(self, model):
                super().__init__(model)
                self.aux = torch.nn.Linear(D, 1)

            def training_step(self, batch):
                return super().training_step(batch) + self.aux(batch.features).mean()

            def trainable_state_dict(self):
                return self.state_dict()

            def load_trainable_state_dict(self, state):
                self.load_state_dict(state)

        learner = WithAux(_learner().model)
        result = Trainer(max_epochs=1, callbacks=[Checkpoint(dirpath=tmp_path)], device="cpu").fit(
            learner, _bag_loader(), _bag_loader()
        )
        state = torch.load(result.checkpoint_path, weights_only=True)["model_state_dict"]
        assert set(state) == set(learner.model.state_dict())
        fresh = _learner().model
        fresh.load_state_dict(state)  # strict: the file is the bare model, not the learner
        for k, v in fresh.state_dict().items():
            assert torch.equal(v, learner.model.state_dict()[k])

    def test_checkpoint_writes_best_model_only_with_a_dirpath(self, tmp_path: Path):
        learner = _learner()
        result = Trainer(
            max_epochs=2, callbacks=[Checkpoint(dirpath=tmp_path)], device="cpu"
        ).fit(learner, _bag_loader(), _bag_loader())
        assert result.checkpoint_path == tmp_path / "best_model.pt"
        state = torch.load(result.checkpoint_path, weights_only=True)
        assert set(state) >= {"model_state_dict", "epoch", "selection", "metrics"}
        assert "optimizer_state_dict" not in state
        assert state["epoch"] == result.selected_epoch
        assert state["selection"] == {
            "strategy": "best", "monitor": "loss", "mode": "min",
            "value": pytest.approx(result.selection_monitor[1]),
        }
        assert state["metrics"]["loss"] == pytest.approx(result.selection_monitor[1])
        assert not list(tmp_path.glob(".*tmp*"))

    def test_checkpoint_last_keeps_the_final_epoch(self):
        class Worsening:
            values = iter([0.1, 0.5, 0.9])

            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = next(self.values)

        result = Trainer(
            max_epochs=3,
            callbacks=[Worsening(), Checkpoint(monitor="score", selection="last")],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        assert result.selected_epoch == 2
        assert result.selection_monitor == ("score", 0.9)

    def test_checkpoint_last_refuses_an_installed_early_stopping(self):
        with pytest.raises(ValueError, match="last"):
            Trainer(
                max_epochs=1,
                callbacks=[Checkpoint(selection="last"), EarlyStopping()],
                device="cpu",
            ).fit(_learner(), _bag_loader(), _bag_loader())

    def test_checkpoint_raises_on_a_non_finite_monitor_at_first_observation(self):
        class Nan:
            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = float("nan")

        with pytest.raises(RuntimeError, match="score"):
            Trainer(
                max_epochs=3, callbacks=[Nan(), Checkpoint(monitor="score")], device="cpu"
            ).fit(_learner(), _bag_loader(), _bag_loader())

    def test_checkpoint_last_tolerates_a_non_finite_diagnostic_monitor(self):
        class Nan:
            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = float("nan")

        # Under ``last`` the monitor is a logged diagnostic, not a selection signal (#282):
        # an undefined AUROC on a one-class tune split must not abort the run.
        result = Trainer(
            max_epochs=2,
            callbacks=[Nan(), Checkpoint(monitor="score", selection="last")],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        assert result.selected_epoch == 1

    def test_checkpoint_refuses_an_unknown_monitor(self):
        with pytest.raises(ValueError, match="nope"):
            Trainer(max_epochs=1, callbacks=[Checkpoint(monitor="nope")], device="cpu").fit(
                _learner(), _bag_loader(), _bag_loader()
            )

    @staticmethod
    def _scripted(values):
        class Scripted:
            it = iter(values)

            def on_validation_end(self, trainer, learner, summary):
                summary["score"] = next(self.it)

        return Scripted()

    def test_early_stopping_stops_after_patience_non_improving_evaluations(self):
        result = Trainer(
            max_epochs=10,
            callbacks=[
                self._scripted([0.5, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]),
                EarlyStopping(monitor="score", mode="min", patience=2),
            ],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        assert len(result.history) == 4

    def test_early_stopping_holds_while_the_monitor_sits_at_its_first_value(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.1, 0.1]
        result = Trainer(
            max_epochs=9,
            callbacks=[self._scripted(values), EarlyStopping(monitor="score", mode="max", patience=2)],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        # Held through the floor at epochs 0-3; moves at epoch 4; stops 2 evaluations later.
        assert len(result.history) == 7

    def test_early_stopping_without_hold_stops_on_an_initial_plateau(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.3]
        result = Trainer(
            max_epochs=5,
            callbacks=[
                self._scripted(values),
                EarlyStopping(monitor="score", mode="max", patience=2, hold_until_moves=False),
            ],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        assert len(result.history) == 3

    def test_early_stopping_counts_evaluations_not_epochs(self):
        result = Trainer(
            max_epochs=10,
            val_every_n_epochs=2,
            callbacks=[
                self._scripted([0.5, 0.6, 0.7, 0.8, 0.9]),
                EarlyStopping(monitor="score", mode="min", patience=2),
            ],
            device="cpu",
        ).fit(_learner(), _bag_loader(), _bag_loader())
        assert len(result.history) == 6

    def test_history_is_installed_by_the_trainer(self):
        result = Trainer(max_epochs=1, device="cpu").fit(_learner(), _bag_loader(), _bag_loader())
        assert isinstance(result.history, pd.DataFrame) and len(result.history) == 1
        assert isinstance(History(), object)


# ---------------------------------------------------------------------------
# soma.fit sugar
# ---------------------------------------------------------------------------


class TestFitSugar:
    def test_a_bare_mil_model_is_wrapped_and_selected_by_the_monitor(self, tmp_path: Path):
        import soma

        torch.manual_seed(0)
        model = MILModel(
            aggregator=MeanPool(input_dim=D),
            task_head=BinaryClassificationHead(input_dim=D, num_classes=2),
        )
        result = soma.fit(
            model, _bag_loader(), _bag_loader(),
            optimizer={"name": "adam", "params": {"lr": 1e-2}},
            monitor="auroc", mode="max", patience=3, max_epochs=4, device="cpu", progress=False,
        )
        assert isinstance(result.learner, MILLearner) and result.learner.model is model
        assert result.selection_monitor[0] == "auroc"
        assert result.history["auroc"].max() == pytest.approx(result.selection_monitor[1])

    def test_patience_none_installs_no_early_stopping(self):
        import soma

        class Worsening:
            values = iter(range(100))

            def on_validation_end(self, trainer, learner, summary):
                summary["loss"] = float(next(self.values))

        result = soma.fit(
            _learner(), _bag_loader(), _bag_loader(),
            patience=None, max_epochs=6, device="cpu", progress=False, callbacks=[Worsening()],
        )
        assert len(result.history) == 6 and result.selected_epoch == 0

    def test_callbacks_passed_explicitly_replace_the_ones_fit_would_build(self):
        """A ``Checkpoint`` / ``EarlyStopping`` in ``callbacks`` is the run's selection and
        stopping policy, not a second one next to the policy ``fit`` would build."""
        import soma
        from soma.learn import Checkpoint, EarlyStopping

        class Worsening(MILLearner):
            def validation_summary(self, outputs):
                self.evaluations = getattr(self, "evaluations", 0) + 1
                return {"loss": float(self.evaluations)}

        def worsening():
            return Worsening(_learner().model)

        # A default Checkpoint(monitor="absent") would refuse the first evaluation.
        checkpoint = Checkpoint(selection="last")
        result = soma.fit(
            worsening(), _bag_loader(), _bag_loader(),
            monitor="absent", patience=None, max_epochs=4, device="cpu", progress=False,
            callbacks=[checkpoint],
        )
        assert result.selected_epoch == 3 and checkpoint.best_epoch == 3

        # The default EarlyStopping(patience=1) would stop this run after two epochs.
        stopper = EarlyStopping(patience=50)
        result = soma.fit(
            worsening(), _bag_loader(), _bag_loader(),
            patience=1, max_epochs=5, device="cpu", progress=False, callbacks=[stopper],
        )
        assert len(result.history) == 5 and stopper.evaluations == 5 and not stopper.stopped

    def test_seed_everything_is_a_top_level_verb(self):
        import soma

        soma.seed_everything(3, fold=1)
        first = torch.rand(2)
        soma.seed_everything(4)
        assert torch.equal(first, torch.rand(2))

    def test_a_learner_with_loss_or_optimizer_is_an_error(self):
        import soma

        with pytest.raises(ValueError, match="Learner"):
            soma.fit(_learner(), _bag_loader(), optimizer="sgd", max_epochs=1, device="cpu")
        with pytest.raises(ValueError, match="Learner"):
            soma.fit(_learner(), _bag_loader(), loss=lambda p, t: p.sum(), max_epochs=1, device="cpu")

    def test_subclass_adds_an_intermediate_loss_by_overriding_training_step_only(self):
        import soma

        calls = []

        class AttentionEntropy(MILLearner):
            def training_step(self, batch):
                output = self.forward_batch(batch)
                attention = output.tile_attention.softmax(-1)
                entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
                calls.append(float(entropy))
                return self.compute_loss(output, batch) + 0.01 * entropy

        torch.manual_seed(0)
        agg = ABMIL(input_dim=D, hidden_dim=8)
        learner = AttentionEntropy(
            MILModel(aggregator=agg, task_head=BinaryClassificationHead(input_dim=agg.output_dim, num_classes=2))
        )
        result = soma.fit(learner, _bag_loader(), _bag_loader(), max_epochs=2, device="cpu", progress=False)
        assert len(calls) == 8 and len(result.history) == 2

    def test_progress_stops_the_live_panel_when_fit_raises(self):
        import io
        import sys

        from rich.console import Console

        from soma.learn import Progress

        class Boom(MILLearner):
            def training_step(self, batch):
                raise RuntimeError("boom")

        stdout = sys.stdout
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        progress = Progress(style="rich", console=console)
        learner = Boom(_learner().model)
        with pytest.raises(RuntimeError, match="boom"):
            Trainer(max_epochs=2, callbacks=[progress], device="cpu").fit(learner, _bag_loader())
        assert progress._live is None
        assert sys.stdout is stdout

    def test_progress_stops_the_live_panel_when_a_later_callback_fails_at_fit_start(self):
        import io
        import sys

        from rich.console import Console

        from soma.learn import Progress

        class Boom:
            def on_fit_start(self, trainer, learner):
                raise RuntimeError("boom at fit start")

        stdout = sys.stdout
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        progress = Progress(style="rich", console=console)
        with pytest.raises(RuntimeError, match="boom at fit start"):
            Trainer(max_epochs=1, callbacks=[progress, Boom()], device="cpu").fit(_learner(), _bag_loader())
        assert progress._live is None
        assert sys.stdout is stdout

    def test_progress_plain_style_writes_one_line_per_epoch(self):
        import io

        from rich.console import Console

        from soma.learn import Progress

        buffer = io.StringIO()
        progress = Progress(style="plain", label="fold 1/2", console=Console(file=buffer, width=200))
        Trainer(
            max_epochs=2, callbacks=[Checkpoint(), EarlyStopping(patience=5), progress], device="cpu"
        ).fit(_learner(), _bag_loader(), _bag_loader())
        lines = buffer.getvalue().splitlines()
        assert lines[0].startswith("training started | fold 1/2 | 2 epochs")
        assert lines[1].startswith("fold 1/2 | epoch 01/02")
        assert "selected loss=" in lines[1]
        assert lines[-1].startswith("training complete | fold 1/2 | ran 2/2 epochs")


class TestSegmentationAugment:
    def test_no_enabled_transform_returns_the_inputs_unchanged(self):
        from soma.learn import SegmentationAugment, augment_registry

        pixels = torch.randint(0, 256, (3, 8, 6), dtype=torch.uint8)
        mask = torch.randint(0, 3, (8, 6), dtype=torch.long)
        for augment in (SegmentationAugment(), augment_registry.get("segmentation")()):
            out_pixels, out_mask = augment(pixels, mask)
            assert out_pixels.dtype == torch.uint8 and out_mask.dtype == torch.long
            assert torch.equal(out_pixels, pixels) and torch.equal(out_mask, mask)


def _marked_canvas(size=64, ignored=None):
    """A blue canvas with one 7x7 marker per point; marker ``i`` has red = blue = ``30 (i + 1)``.

    A bilinear blend of a marker with the background (red 0, blue 255) never reads as
    another marker, so a marker's colour identifies it after any transform.

    ``ignored`` (top, left, bottom, right) paints green into the pixels and 255 into the
    returned ignore raster, so the two can be compared after a transform.
    """
    centres = [(8, 8), (55, 9), (32, 32), (12, 50), (50, 52), (32, 5), (58, 33)]
    pixels = torch.zeros(3, size, size, dtype=torch.uint8)
    pixels[2] = 255
    for i, (x, y) in enumerate(centres):
        pixels[0, y - 3 : y + 4, x - 3 : x + 4] = 30 * (i + 1)
        pixels[2, y - 3 : y + 4, x - 3 : x + 4] = 30 * (i + 1)
    ignore = np.zeros((size, size), dtype=np.uint8)
    if ignored is not None:
        top, left, bottom, right = ignored
        pixels[1, top:bottom, left:right] = 255
        ignore[top:bottom, left:right] = 255
    xy = torch.tensor(centres, dtype=torch.float32)
    return pixels, xy, torch.arange(len(centres)), ignore


_GEOMETRIC = dict(
    crop_scale=(0.3, 1.0),
    target_size=64,
    horizontal_flip=0.5,
    vertical_flip=0.5,
    rotation_degrees=40.0,
    translate=0.3,
    scale=0.3,
)


class TestSegmentationAugmentPoints:
    """``SegmentationAugment`` moves ``Points`` (in the pixel frame) with the pixels."""

    def test_points_move_with_the_pixels_and_leave_with_them(self):
        from soma.data import Points
        from soma.learn import SegmentationAugment

        pixels, xy, classes, _ = _marked_canvas()
        augment = SegmentationAugment(**_GEOMETRIC)
        dropped = kept = 0
        for seed in range(40):
            torch.manual_seed(seed)
            out_pixels, out = augment(pixels, Points(xy=xy, classes=classes))
            assert isinstance(out, Points) and out.xy.dtype == torch.float32
            assert out.classes.dtype == torch.long
            height, width = out_pixels.shape[1:]
            inside = (out.xy[:, 0] >= 0) & (out.xy[:, 0] < width) & (out.xy[:, 1] >= 0) & (out.xy[:, 1] < height)
            assert bool(inside.all()), f"seed {seed}: a returned point left the frame"
            for (x, y), cls in zip(out.xy.tolist(), out.classes.tolist()):
                col, row = min(round(x), width - 1), min(round(y), height - 1)
                red, blue = int(out_pixels[0, row, col]), int(out_pixels[2, row, col])
                assert abs(red - 30 * (cls + 1)) <= 8 and abs(blue - 30 * (cls + 1)) <= 8, (
                    f"seed {seed}: point of class {cls} sits on red {red} / blue {blue}"
                )
            returned = set(out.classes.tolist())
            fill = out_pixels[2] <= 5  # pixels the transform synthesised
            for cls in range(len(classes)):
                region = ((out_pixels[0].int() - 30 * (cls + 1)).abs() <= 4) & (
                    (out_pixels[2].int() - 30 * (cls + 1)).abs() <= 4
                )
                rows, cols = region.nonzero(as_tuple=True)
                near = torch.nn.functional.max_pool2d(region[None, None].float(), 5, 1, 2)[0, 0] > 0
                # A marker clear of the edge and of synthesised fill survived whole: its
                # centre (the point) is still on the canvas.
                whole = (
                    rows.numel() > 0
                    and min(rows.min(), cols.min()) > 0
                    and rows.max() < height - 1
                    and cols.max() < width - 1
                    and not bool((near & fill).any())
                )
                if whole:
                    assert cls in returned, f"seed {seed}: marker {cls} is in the crop but its point was dropped"
            dropped += len(classes) - len(returned)
            kept += len(returned)
        assert dropped > 0 and kept > 0

    def test_the_ignore_mask_moves_with_the_pixels_and_covers_synthesised_borders(self):
        from soma.data import Points
        from soma.learn import SegmentationAugment

        pixels, xy, classes, ignore = _marked_canvas(ignored=(10, 5, 30, 40))
        augment = SegmentationAugment(**_GEOMETRIC)
        for seed in range(20):
            torch.manual_seed(seed)
            out_pixels, out = augment(pixels, Points(xy=xy, classes=classes, ignore_mask=ignore))
            moved = np.asarray(out.ignore_mask)
            assert moved.shape == tuple(out_pixels.shape[1:]) and moved.dtype == np.uint8
            green, blue = out_pixels[1].numpy(), out_pixels[2].numpy()
            for name, where, expected in (
                ("ignored tissue", (blue >= 250) & (green >= 250), 255),
                ("supervised tissue", (blue >= 250) & (green <= 5), 0),
                ("synthesised border", blue <= 5, 255),
            ):
                if where.sum() >= 50:
                    agreement = float((moved[where] == expected).mean())
                    assert agreement >= 0.97, f"seed {seed}: {name} agrees {agreement:.3f}"

    def test_points_without_an_ignore_mask_get_one_over_synthesised_borders(self):
        from soma.data import Points
        from soma.learn import SegmentationAugment

        pixels, xy, classes, _ = _marked_canvas()
        torch.manual_seed(0)
        out_pixels, out = SegmentationAugment(rotation_degrees=30.0, translate=0.4)(pixels, Points(xy=xy, classes=classes))
        border = out_pixels[2].numpy() <= 5
        assert border.sum() > 0
        assert out.ignore_mask is not None
        assert float((np.asarray(out.ignore_mask)[border] == 255).mean()) >= 0.97
        assert float((np.asarray(out.ignore_mask)[~border] == 0).mean()) >= 0.97
