"""Callbacks: run-management side effects. They never change the loss.

Hooks (all optional): ``on_fit_start(trainer, learner)``, ``on_epoch_start(trainer,
learner)``, ``on_train_batch_end(trainer, learner, step_output)``,
``on_validation_end(trainer, learner, summary)``, ``on_epoch_end(trainer, learner)``,
``on_fit_end(trainer, learner)``. A callback stops the run with
``trainer.should_stop = True``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soma.atomic_io import atomic_torch_save
from soma.learn.learner import uncovered_parameters

__all__ = ["Checkpoint", "EarlyStopping", "History", "Progress"]

logger = logging.getLogger(__name__)


def _is_improvement(current: float, best: float, mode: str) -> bool:
    return current < best if mode == "min" else current > best


def _check_mode(mode: str) -> str:
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    return mode


def _monitor_value(summary: dict[str, float], monitor: str) -> float:
    if monitor not in summary:
        available = ", ".join(sorted(summary)) or "(none)"
        raise ValueError(
            f"monitor '{monitor}' is not a validation_summary key; available: {available}"
        )
    return float(summary[monitor])


class History:
    """One row per epoch: ``epoch``, ``train_loss``, ``lr``, ``elapsed_seconds`` and one
    column per summary key (empty on epochs without an evaluation). Always installed."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def on_fit_start(self, trainer, learner) -> None:
        self.rows = []

    def on_epoch_end(self, trainer, learner) -> None:
        row: dict[str, Any] = {
            "epoch": trainer.epoch,
            "train_loss": trainer.train_loss,
            "lr": trainer.lr,
            "elapsed_seconds": time.perf_counter() - trainer.started_at,
        }
        if trainer.summary is not None:
            row.update({k: float(v) for k, v in trainer.summary.items()})
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class Checkpoint:
    """Owns selection. Snapshots ``trainable_state_dict()`` to CPU memory on improvement
    (every evaluation under ``selection="last"``), writes ``best_model.pt`` only with a
    ``dirpath``, and restores the snapshot at ``on_fit_end``.

    Without a tune loader nothing is evaluated and selection falls back to ``last``.
    """

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "min",
        selection: str = "best",
        dirpath: str | Path | None = None,
    ) -> None:
        if selection not in {"best", "last"}:
            raise ValueError(f"selection must be 'best' or 'last', got {selection!r}")
        self.monitor, self.mode, self.selection = monitor, _check_mode(mode), selection
        self.dirpath = Path(dirpath) if dirpath is not None else None
        self.best_value = float("nan")
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, float] = {}
        self.path: Path | None = None
        self._snapshot: dict[str, torch.Tensor] | None = None

    def on_fit_start(self, trainer, learner) -> None:
        self.best_value, self.best_epoch, self.best_metrics, self._snapshot = float("nan"), None, {}, None
        if self.selection == "last" and any(isinstance(c, EarlyStopping) for c in trainer.callbacks):
            raise ValueError(
                "Checkpoint(selection='last') keeps the final epoch and never early-stops; "
                "remove the EarlyStopping callback."
            )
        missing = uncovered_parameters(learner, trainer.optimizer)
        if missing:
            raise ValueError(
                "Checkpoint restores trainable_state_dict(), but the optimizer updates "
                f"parameters it does not hold: {missing[:5]}. Override trainable_state_dict / "
                "load_trainable_state_dict to cover the learner-owned state, or leave it out "
                "of the optimizer."
            )
        # Without a tune loader nothing is evaluated: selection falls back to the last epoch.
        self._last = self.selection == "last" or not trainer.validates

    def on_epoch_end(self, trainer, learner) -> None:
        summary = trainer.summary
        if summary is None and not self._last:
            return
        value = float("nan")
        if summary is not None:
            value = _monitor_value(summary, self.monitor)
            # Under ``last`` the monitor is a logged diagnostic, not a selection signal, so
            # an undefined metric (AUROC on a one-class tune split) does not abort the run.
            if not self._last and not math.isfinite(value):
                raise RuntimeError(
                    f"monitor '{self.monitor}' is {value} at epoch {trainer.epoch + 1}, so no "
                    "checkpoint could ever be selected. Threshold-free metrics (AUROC, AUPRC, "
                    "C-index, ...) are undefined when the tune split holds a single class or "
                    "no comparable pairs; check the split or monitor 'loss' / a "
                    "threshold-based metric instead."
                )
        if not (self._last or self.best_epoch is None or _is_improvement(value, self.best_value, self.mode)):
            return
        self.best_value, self.best_epoch = value, trainer.epoch
        self.best_metrics = dict(summary or {})
        self._snapshot = {k: v.detach().to("cpu", copy=True) for k, v in learner.trainable_state_dict().items()}
        if self.dirpath is not None:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            self.path = self.dirpath / "best_model.pt"
            atomic_torch_save(
                {
                    "model_state_dict": self._snapshot,
                    "epoch": trainer.epoch,
                    "selection": self.selection_record(),
                    "metrics": {k: float(v) for k, v in self.best_metrics.items()},
                },
                self.path,
            )
        trainer.selection = (self.best_epoch, (self.monitor, self.best_value), self.path)

    def selection_record(self) -> dict[str, str | float]:
        record: dict[str, str | float] = {"strategy": "last" if self._last else "best"}
        if not self._last:
            record.update({"monitor": self.monitor, "mode": self.mode, "value": float(self.best_value)})
        return record

    def on_fit_end(self, trainer, learner) -> None:
        if self._snapshot is not None:
            learner.load_trainable_state_dict(self._snapshot)


class EarlyStopping:
    """Stop after ``patience`` evaluations without improvement. ``hold_until_moves`` holds
    the stop while the monitor still equals its first value (a thresholded metric at its
    floor); patience keeps counting, so a run that moves late stops at the same epoch."""

    def __init__(
        self,
        monitor: str = "loss",
        mode: str = "min",
        patience: int = 10,
        hold_until_moves: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.monitor, self.mode, self.patience = monitor, _check_mode(mode), int(patience)
        self.hold_until_moves = hold_until_moves
        self.wait_count = 0
        self.best_value = float("nan")
        self.first_value: float | None = None
        self.moved = False
        self.evaluations = 0
        self.stopped = False

    def on_fit_start(self, trainer, learner) -> None:
        self.wait_count, self.best_value, self.first_value = 0, float("nan"), None
        self.moved, self.evaluations, self.stopped = False, 0, False
        if not trainer.validates:
            logger.warning("EarlyStopping is inert: no tune loader, so nothing is evaluated.")

    def on_validation_end(self, trainer, learner, summary) -> None:
        value = _monitor_value(summary, self.monitor)
        self.evaluations += 1
        if self.first_value is None:
            self.first_value = value
        elif value != self.first_value:
            self.moved = True
        if self.evaluations == 1 or _is_improvement(value, self.best_value, self.mode):
            self.best_value, self.wait_count = value, 0
            return
        self.wait_count += 1
        if self.wait_count >= self.patience and (self.moved or not self.hold_until_moves):
            self.stopped = True
            trainer.should_stop = True

    def on_fit_end(self, trainer, learner) -> None:
        if self.evaluations > 1 and not self.moved:
            logger.warning(
                "%s never left its first-evaluation value (%s) over %d evaluations.",
                self.monitor,
                self.first_value,
                self.evaluations,
            )


# ---------------------------------------------------------------------------
# Progress: Rich panel on a terminal, one plain line per event elsewhere
# ---------------------------------------------------------------------------

#: On a console that cannot redraw, an in-epoch line is written once this many seconds
#: have passed without any line, so a long epoch is visibly alive.
PROGRESS_HEARTBEAT_SECONDS = 30.0


class Progress:
    """Training progress: a live Rich panel on a terminal, plain lines on a file.

    ``label`` names the run in every line (the pipeline passes ``"fold k/n"``). The
    selected epoch and patience come from the installed :class:`Checkpoint` and
    :class:`EarlyStopping`, when present.
    """

    def __init__(self, style: str = "auto", label: str | None = None, console: Console | None = None) -> None:
        if style not in {"auto", "rich", "plain"}:
            raise ValueError(f"style must be 'auto', 'rich' or 'plain', got {style!r}")
        self.style, self.label = style, label
        self.console = console or Console()
        self._live: Live | None = None
        self._status = "waiting for epoch 1"
        self._batch_text: str | None = None
        self._last_line_at = 0.0
        self._processed = 0

    # -- helpers ----------------------------------------------------------------

    def _rich(self) -> bool:
        if self.style == "auto":
            return bool(self.console.is_terminal or self.console.is_jupyter)
        return self.style == "rich"

    def _checkpoint(self, trainer) -> Checkpoint | None:
        return next((c for c in trainer.callbacks if isinstance(c, Checkpoint)), None)

    def _early_stopping(self, trainer) -> EarlyStopping | None:
        return next((c for c in trainer.callbacks if isinstance(c, EarlyStopping)), None)

    def _elapsed(self, trainer) -> float:
        return time.perf_counter() - trainer.started_at

    def _eta(self, trainer) -> float | None:
        """Remaining time from the average epoch (or step) time so far."""
        elapsed = self._elapsed(trainer)
        done_epochs = trainer.epoch + 1
        if trainer.max_steps is None:
            return elapsed / done_epochs * max(trainer.total_epochs - done_epochs, 0)
        return elapsed / max(trainer.optimizer_steps, 1) * max(trainer.max_steps - trainer.optimizer_steps, 0)

    def _emit(self, *parts: str | None) -> None:
        line = " | ".join(part for part in parts if part)
        self.console.print(line, markup=False, highlight=False, soft_wrap=True)
        self.console.file.flush()
        self._last_line_at = time.perf_counter()

    def _epoch_text(self, trainer) -> str:
        return f"epoch {trainer.epoch + 1:02d}/{trainer.total_epochs:02d}"

    def _steps_text(self, trainer) -> str | None:
        if trainer.max_steps is None:
            return None
        return f"steps {trainer.optimizer_steps}/{trainer.max_steps}"

    def _selected_text(self, trainer) -> str:
        checkpoint = self._checkpoint(trainer)
        if checkpoint is None or checkpoint.best_epoch is None:
            return "n/a"
        if math.isfinite(checkpoint.best_value):
            return f"{checkpoint.monitor}={checkpoint.best_value:.4f} @ {checkpoint.best_epoch + 1:02d}"
        return f"epoch {checkpoint.best_epoch + 1:02d}"

    def _patience_text(self, trainer) -> str:
        stopping = self._early_stopping(trainer)
        if stopping is None:
            return "off"
        return f"{stopping.wait_count}/{stopping.patience}"

    # -- hooks ------------------------------------------------------------------

    def on_fit_start(self, trainer, learner) -> None:
        self._params = sum(p.numel() for p in learner.parameters() if p.requires_grad)
        self._status, self._batch_text = "waiting for epoch 1", None
        if self._rich():
            self._live = Live(self._panel(trainer), console=self.console, refresh_per_second=8, transient=False)
            self._live.start()
            return
        budget = (
            f"{trainer.total_epochs} epochs"
            if trainer.max_steps is None
            else f"{trainer.max_steps} steps ({trainer.total_epochs} epochs)"
        )
        self._emit("training started", self.label, budget, f"{self._params:,} trainable params")

    def on_epoch_start(self, trainer, learner) -> None:
        self._status, self._batch_text, self._processed = "training epoch in progress", None, 0
        self._render(trainer)

    def on_train_batch_end(self, trainer, learner, step_output) -> None:
        self._processed += 1
        self._batch_text = f"train batches {self._processed}"
        if self._live is not None:
            self._render(trainer)
        elif time.perf_counter() - self._last_line_at >= PROGRESS_HEARTBEAT_SECONDS:
            self._emit(
                self.label,
                self._epoch_text(trainer),
                self._steps_text(trainer),
                self._batch_text,
                f"elapsed {_format_seconds(self._elapsed(trainer))}",
            )

    def on_epoch_end(self, trainer, learner) -> None:
        checkpoint = self._checkpoint(trainer)
        if trainer.summary is None:
            self._status = "tune not evaluated this epoch" if trainer.validates else "no tune split"
        elif checkpoint is not None and checkpoint.best_epoch == trainer.epoch:
            self._status = f"new selected checkpoint at epoch {trainer.epoch + 1}"
        else:
            self._status = f"no improvement ({self._patience_text(trainer)})"
            stopping = self._early_stopping(trainer)
            if stopping is not None and stopping.hold_until_moves and not stopping.moved:
                self._status += f"; {stopping.monitor} unchanged since first evaluation, early stopping held"
        self._batch_text = None
        if self._live is not None:
            self._render(trainer)
            return
        tune_parts: tuple[str | None, ...] = ()
        if trainer.summary is not None:
            tune_parts = (
                f"tune {trainer.summary['loss']:.4f}",
                f"selected {self._selected_text(trainer)}",
                f"patience {self._patience_text(trainer)}",
                f"lr {trainer.lr:.2e}",
            )
        self._emit(
            self.label,
            self._epoch_text(trainer),
            self._steps_text(trainer),
            f"train {trainer.train_loss:.4f}",
            *tune_parts,
            f"elapsed {_format_seconds(self._elapsed(trainer))} [ETA {_format_seconds(self._eta(trainer))}]",
            self._status,
        )

    def on_exception(self, trainer, learner, exception: BaseException) -> None:
        # The panel redirects stdout/stderr and runs a refresh thread; a failing fit
        # must release both so the error (and whatever follows) prints normally.
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_fit_end(self, trainer, learner) -> None:
        stopping = self._early_stopping(trainer)
        early = stopping is not None and stopping.stopped
        self._status = "early stopping triggered" if early else "training complete"
        if self._live is not None:
            self._render(trainer)
            self._live.stop()
            self._live = None
            return
        metrics = self._checkpoint(trainer).best_metrics if self._checkpoint(trainer) else {}
        self._emit(
            "training complete",
            self.label,
            f"{'early stopped after' if early else 'ran'} {trainer.epoch + 1}/{trainer.total_epochs} epochs",
            self._steps_text(trainer),
            f"selected {self._selected_text(trainer)}",
            _format_metrics(metrics),
            f"elapsed {_format_seconds(self._elapsed(trainer))}",
        )

    # -- panel ------------------------------------------------------------------

    def _render(self, trainer) -> None:
        if self._live is not None:
            self._live.update(self._panel(trainer), refresh=True)

    def _panel(self, trainer) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="left")
        started = trainer.started_at > 0 and hasattr(trainer, "optimizer")
        table.add_row("epoch", Text(f"{trainer.epoch + 1 if started else 0:02d}/{trainer.total_epochs:02d}", style="bold cyan"))
        table.add_row("train", Text(f"{trainer.train_loss:.4f}" if math.isfinite(trainer.train_loss) else "-"))
        if trainer.summary is not None:
            table.add_row("tune", Text(f"{trainer.summary['loss']:.4f}"))
            for key, value in trainer.summary.items():
                if key != "loss":
                    table.add_row(key[:12], Text(f"{value:.4f}"))
        table.add_row("lr", Text(f"{trainer.lr:.2e}" if started else "-"))
        table.add_row("# params", Text(f"{self._params:,}"))
        if self.label:
            table.add_row("run", Text(self.label))
        if trainer.max_steps is not None:
            table.add_row("steps", Text(f"{trainer.optimizer_steps}/{trainer.max_steps}", style="bold cyan"))
        if self._batch_text is not None:
            table.add_row("batch", Text(self._batch_text))
        table.add_row("selected", Text(self._selected_text(trainer), style="green"))
        table.add_row("patience", Text(self._patience_text(trainer)))
        table.add_row("status", Text(self._status, style="bold green" if "selected" in self._status else "bold yellow"))
        eta = self._eta(trainer) if started else None
        table.add_row("elapsed", Text(f"{_format_seconds(self._elapsed(trainer))} [ETA {_format_seconds(eta)}]"))
        return Panel.fit(table, title="Training progress", subtitle=self._status, border_style="cyan", box=box.ROUNDED)


def _format_seconds(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    total = max(0, int(value))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_metrics(metrics: dict[str, float]) -> str:
    preferred = ("accuracy", "balanced_accuracy", "f1_macro", "auc")
    items = [f"{k}={metrics[k]:.4f}" for k in preferred if k in metrics]
    items += [f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if k not in preferred and k != "loss"]
    return " | ".join(items)
