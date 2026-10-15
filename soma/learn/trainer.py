"""The generic training loop. It knows nothing about tasks, heads, shapes or folds."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from soma.learn.callbacks import History
from soma.learn.learner import Budget, Learner, SchedulerSpec

__all__ = ["FitResult", "Trainer"]

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    learner: Learner  # in selected state
    history: pd.DataFrame  # one row per epoch
    selected_epoch: int
    selection_monitor: tuple[str, float]
    checkpoint_path: Path | None
    optimizer_steps: int


class Trainer:
    """Epoch/step-budgeted loop with gradient accumulation and callbacks.

    One step is one completed accumulation window; a step budget may end partway
    through an epoch. ``val_every_n_epochs`` evaluates after epochs N, 2N, ... and
    always after the final epoch. ``precision="bf16"`` is autocast without a grad
    scaler. Callbacks stop the run by setting ``trainer.should_stop``: a stop raised
    before ``on_epoch_end`` makes the current epoch the final one and it is evaluated;
    a stop raised in ``on_epoch_end`` only prevents the next epoch, since the epoch
    is already closed (history row written, selection consulted). When the loop
    raises, ``on_exception(trainer, learner, exc)`` fires before the error propagates.
    A learner with no trainable parameter is returned as built: no optimizer, no
    epoch, no callback, and an empty history.
    """

    def __init__(
        self,
        *,
        max_epochs: int | None = 50,
        max_steps: int | None = None,
        accumulate_grad_batches: int = 1,
        val_every_n_epochs: int = 1,
        device: str | torch.device = "auto",
        precision: str = "fp32",
        callbacks: Sequence[Any] = (),
    ) -> None:
        if max_steps is not None:
            max_epochs = None  # a step budget derives its epoch count
        if max_epochs is None and max_steps is None:
            raise ValueError("set max_epochs or max_steps")
        if precision not in {"fp32", "bf16"}:
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
        if accumulate_grad_batches < 1 or val_every_n_epochs < 1:
            raise ValueError("accumulate_grad_batches and val_every_n_epochs must be >= 1")
        self.max_epochs, self.max_steps = max_epochs, max_steps
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.val_every_n_epochs = int(val_every_n_epochs)
        self.device = _resolve_device(device)
        self.precision = precision
        self.history = History()
        self.callbacks: list[Any] = [self.history, *callbacks]
        # Loop state callbacks read; selection is written by a Checkpoint callback.
        self.should_stop = False
        self.epoch = 0
        self.optimizer_steps = 0
        self.train_loss = float("nan")
        self.summary: dict[str, float] | None = None
        self.selection: tuple[int, tuple[str, float], Path | None] | None = None
        self.started_at = 0.0

    def _call(self, hook: str, *args: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(self, *args)

    @property
    def validates(self) -> bool:
        return self._tune_loader is not None

    def fit(self, learner: Learner, train_loader, tune_loader=None) -> FitResult:
        if tune_loader is not None and len(tune_loader) == 0:
            logger.warning("the tune loader is empty; training without validation.")
            tune_loader = None
        self._tune_loader = tune_loader
        self.learner = learner.to(self.device)
        if not any(p.requires_grad for p in learner.parameters()):
            return FitResult(
                learner=learner,
                history=self.history.frame(),
                selected_epoch=0,
                selection_monitor=("loss", float("nan")),
                checkpoint_path=None,
                optimizer_steps=0,
            )
        updates_per_epoch = max(-(-len(train_loader) // self.accumulate_grad_batches), 1)
        self.budget = Budget(self.max_epochs, self.max_steps, updates_per_epoch)
        self.total_epochs = self.budget.total_epochs
        configured = learner.configure_optimizers(self.budget)
        self.optimizer, specs = (configured, []) if isinstance(configured, torch.optim.Optimizer) else configured
        self.schedulers: list[SchedulerSpec] = list(specs)
        self.started_at = time.perf_counter()
        self.should_stop, self.optimizer_steps, self.selection = False, 0, None
        try:
            self._call("on_fit_start", learner)
            self._run_epochs(learner, train_loader, tune_loader)
            self._call("on_fit_end", learner)
        except BaseException as exc:
            self._call("on_exception", learner, exc)
            raise
        selected_epoch, monitor, path = self.selection or (
            self.epoch,
            ("loss", float((self.summary or {}).get("loss", float("nan")))),
            None,
        )
        return FitResult(
            learner=learner,
            history=self.history.frame(),
            selected_epoch=selected_epoch,
            selection_monitor=monitor,
            checkpoint_path=path,
            optimizer_steps=self.optimizer_steps,
        )

    def _run_epochs(self, learner: Learner, train_loader, tune_loader) -> None:
        for self.epoch in range(self.total_epochs):
            set_epoch = getattr(getattr(train_loader, "batch_sampler", None), "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.epoch)
            self.summary = None
            self._call("on_epoch_start", learner)
            self.train_loss = self._train_epoch(learner, train_loader)
            last = self.epoch + 1 == self.total_epochs or self.should_stop
            if tune_loader is not None and ((self.epoch + 1) % self.val_every_n_epochs == 0 or last):
                self.summary = self._validate(learner, tune_loader)
                self._call("on_validation_end", learner, self.summary)
            self._call("on_epoch_end", learner)
            for spec in self.schedulers:
                if spec.interval == "epoch":
                    spec.scheduler.step()
            if self.should_stop:
                break

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _train_epoch(self, learner: Learner, loader) -> float:
        learner.train()
        self.optimizer.zero_grad()
        total, batches, in_window = 0.0, 0, 0
        for batches, batch in enumerate(loader, 1):
            batch = learner.transfer_batch(batch, self.device)
            with self._autocast():
                out = learner.training_step(batch)
            loss = out["loss"] if isinstance(out, dict) else out
            (loss / self.accumulate_grad_batches).backward()
            total += float(loss.item())
            in_window += 1
            self._call("on_train_batch_end", learner, out)
            if in_window == self.accumulate_grad_batches or batches == len(loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_steps += 1
                in_window = 0
                for spec in self.schedulers:
                    if spec.interval == "step":
                        spec.scheduler.step()
                if self.max_steps is not None and self.optimizer_steps >= self.max_steps:
                    self.should_stop = True
                    break
        return total / max(batches, 1)

    @torch.inference_mode()
    def _validate(self, learner: Learner, loader) -> dict[str, float]:
        learner.eval()
        outputs = []
        for batch in loader:
            batch = learner.transfer_batch(batch, self.device)
            with self._autocast():
                outputs.append(learner.validation_step(batch))
        summary = dict(learner.validation_summary(outputs))
        if "loss" not in summary:
            raise ValueError("validation_summary must contain a 'loss' entry")
        return summary


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
