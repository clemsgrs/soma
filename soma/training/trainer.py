"""Trainer — MIL model training with early stopping and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rich import box
from torch.utils.data import DataLoader
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soma.config import TrainingConfig


@dataclass(frozen=True)
class EpochLog:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    tune_loss: float
    tune_metrics: dict[str, float]
    lr: float


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run."""

    best_epoch: int
    best_tune_loss: float
    best_tune_metrics: dict[str, float]
    history: list[EpochLog]
    checkpoint_path: Path


class Trainer:
    """Model trainer with early stopping and checkpointing.

    Supports both MILModel (tile-level, batches have a `mask` attribute)
    and SlideModel (slide-level, batches have no mask).

    Pure PyTorch training loop — no external frameworks needed.

    Args:
        model: nn.Module with a `task_head` attribute to train.
        train_loader: Training DataLoader.
        tune_loader: Tune DataLoader.
        config: Training configuration.
        fold_dir: Directory for checkpoints.
        device: torch.device for training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        tune_loader: DataLoader,
        config: TrainingConfig,
        fold_dir: Path,
        device: torch.device,
        console: Console | None = None,
    ) -> None:
        self._model = model.to(device)
        self._train_loader = train_loader
        self._tune_loader = tune_loader
        self._config = config
        self._fold_dir = Path(fold_dir)
        self._device = device
        self._console = console

        self._optimizer = _build_optimizer(model, config)
        self._scheduler = _build_scheduler(self._optimizer, config)

    def fit(self) -> TrainResult:
        """Run the full training loop.

        Returns:
            TrainResult with best epoch, metrics, history, and checkpoint path.
        """
        self._fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._fold_dir / "best_model.pt"

        history: list[EpochLog] = []
        best_tune_loss = float("inf")
        best_epoch = 0
        best_tune_metrics: dict[str, float] = {}
        patience_counter = 0

        console = self._console or Console()
        current_log: EpochLog | None = None
        current_subtitle = "starting"
        current_status = "waiting for epoch 1"
        current_batch_progress: str | None = None

        def render_panel() -> None:
            live.update(
                _build_training_panel(
                    title="Training progress",
                    subtitle=current_subtitle,
                    log=current_log,
                    total_epochs=self._config.epochs,
                    best_epoch=best_epoch,
                    best_tune_loss=best_tune_loss,
                    best_tune_metrics=best_tune_metrics,
                    patience_counter=patience_counter,
                    patience_limit=self._config.patience,
                    status=current_status,
                    batch_progress=current_batch_progress,
                ),
                refresh=True,
            )

        with Live(
            _build_training_panel(
                title="Training progress",
                subtitle="starting",
                log=None,
                total_epochs=self._config.epochs,
                best_epoch=best_epoch,
                best_tune_loss=best_tune_loss,
                best_tune_metrics=best_tune_metrics,
                patience_counter=patience_counter,
                patience_limit=self._config.patience,
                status="waiting for epoch 1",
                batch_progress=None,
            ),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            render_panel()

            def on_batch_progress(phase: str, batch_index: int, total_batches: int) -> None:
                nonlocal current_batch_progress, current_status
                current_batch_progress = _format_batch_progress(
                    batch_index,
                    total_batches,
                    phase=phase,
                )
                current_status = f"{phase} batch {batch_index}/{total_batches}"
                render_panel()

            for epoch in range(self._config.epochs):
                current_subtitle = f"epoch {epoch + 1}/{self._config.epochs} | train"
                current_status = "training epoch in progress"
                current_batch_progress = None
                render_panel()
                train_loss = self._train_epoch(on_batch_progress=on_batch_progress)

                current_subtitle = f"epoch {epoch + 1}/{self._config.epochs} | tune"
                current_status = "evaluating tune split"
                render_panel()
                tune_loss, tune_metrics = self._tune(on_batch_progress=on_batch_progress)

                lr = self._optimizer.param_groups[0]["lr"]
                if self._scheduler is not None:
                    self._scheduler.step()

                current_log = log = EpochLog(
                    epoch=epoch,
                    train_loss=train_loss,
                    tune_loss=tune_loss,
                    tune_metrics=tune_metrics,
                    lr=lr,
                )
                history.append(log)

                improved = tune_loss < best_tune_loss
                status: str
                if improved:
                    best_tune_loss = tune_loss
                    best_epoch = epoch
                    best_tune_metrics = tune_metrics
                    patience_counter = 0
                    _save_checkpoint(self._model, self._optimizer, epoch, tune_loss, checkpoint_path)
                    status = f"new best checkpoint saved at epoch {epoch + 1}"
                else:
                    patience_counter += 1
                    status = f"no improvement ({patience_counter}/{self._config.patience})"

                current_status = status
                render_panel()

                if not improved and patience_counter >= self._config.patience:
                    current_status = "early stopping triggered"
                    render_panel()
                    break

            current_subtitle = "complete"
            current_status = "training complete"
            live.update(
                _build_training_panel(
                    title="Training progress",
                    subtitle=current_subtitle,
                    log=history[-1] if history else None,
                    total_epochs=self._config.epochs,
                    best_epoch=best_epoch,
                    best_tune_loss=best_tune_loss,
                    best_tune_metrics=best_tune_metrics,
                    patience_counter=patience_counter,
                    patience_limit=self._config.patience,
                    status=current_status,
                    batch_progress=current_batch_progress,
                ),
                refresh=True,
            )

        return TrainResult(
            best_epoch=best_epoch,
            best_tune_loss=best_tune_loss,
            best_tune_metrics=best_tune_metrics,
            history=history,
            checkpoint_path=checkpoint_path,
        )

    def _train_epoch(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self._model.train()
        total_loss = 0.0
        total_batches = len(self._train_loader)
        accum_steps = self._config.gradient_accumulation

        self._optimizer.zero_grad()
        step = 0

        for step, batch in enumerate(self._train_loader, 1):
            if on_batch_progress is not None:
                on_batch_progress("train", step, total_batches)
            features = batch.features.to(self._device)
            labels = batch.labels.to(self._device)

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, labels)

            if hasattr(self._model, "aggregator"):
                mask_tensor = batch.mask.to(self._device) if hasattr(batch, "mask") else None
                loss = self._model.aggregator.combine_losses(
                    loss,
                    out.auxiliary,
                    labels,
                    mask=mask_tensor,
                )

            (loss / accum_steps).backward()
            total_loss += loss.item()

            if step % accum_steps == 0 or step == total_batches:
                self._optimizer.step()
                self._optimizer.zero_grad()

        return total_loss / max(step, 1)

    @torch.inference_mode()
    def _tune(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Run evaluation on tune set. Returns (average loss, metrics dict)."""
        self._model.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_batches = len(self._tune_loader)

        for batch in self._tune_loader:
            batch_index = num_batches + 1
            if on_batch_progress is not None:
                on_batch_progress("tune", batch_index, total_batches)
            features = batch.features.to(self._device)
            labels = batch.labels.to(self._device)

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, labels)
            total_loss += loss.item()
            num_batches += 1
            all_logits.append(out.logits)
            all_labels.append(labels)

        avg_loss = total_loss / max(num_batches, 1)
        if all_logits:
            logits = torch.cat(all_logits, dim=0)
            targets = torch.cat(all_labels, dim=0)
            metrics = self._model.task_head.compute_metrics(logits, targets)
        else:
            metrics = {}
        return avg_loss, metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        msg = f"Unknown optimizer: {config.optimizer}. Use 'adam', 'adamw', or 'sgd'."
        raise ValueError(msg)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "none":
        return None
    else:
        msg = f"Unknown scheduler: {config.scheduler}. Use 'cosine' or 'none'."
        raise ValueError(msg)


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    tune_loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tune_loss": tune_loss,
        },
        path,
    )


def _build_training_panel(
    *,
    title: str,
    subtitle: str,
    log: EpochLog | None,
    total_epochs: int,
    best_epoch: int,
    best_tune_loss: float,
    best_tune_metrics: dict[str, float],
    patience_counter: int,
    patience_limit: int,
    status: str,
    batch_progress: str | None = None,
) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="left")

    if log is not None:
        table.add_row("epoch", Text(f"{log.epoch + 1:02d}/{total_epochs:02d}", style="bold cyan"))
        table.add_row("train", Text(f"{log.train_loss:.4f}", style="white"))
        table.add_row("tune", Text(f"{log.tune_loss:.4f}", style="white"))
        table.add_row("lr", Text(f"{log.lr:.2e}", style="white"))
        for key, value in log.tune_metrics.items():
            table.add_row(key[:6], Text(f"{value:.4f}", style="white"))
    else:
        table.add_row("epoch", Text(f"0/{total_epochs:02d}", style="bold cyan"))
        table.add_row("train", Text("-", style="dim"))
        table.add_row("tune", Text("-", style="dim"))
        table.add_row("lr", Text("-", style="dim"))

    if batch_progress is not None:
        table.add_row("batch", Text(batch_progress, style="white"))

    best_metrics_text = _format_metrics(best_tune_metrics)
    best_loss_text = _format_finite(best_tune_loss)
    best_epoch_text = f"{best_epoch + 1:02d}" if np.isfinite(best_tune_loss) else "n/a"
    table.add_row(
        "best",
        Text.assemble((best_loss_text, "green"), (" @ ", "dim"), (best_epoch_text, "green")),
    )
    if best_metrics_text:
        table.add_row("best metrics", Text(best_metrics_text, style="green"))
    table.add_row("patience", Text(f"{patience_counter}/{patience_limit}", style="white"))
    table.add_row("status", Text(status, style="bold yellow" if "best" not in status else "bold green"))

    return Panel.fit(
        table,
        title=title,
        subtitle=subtitle,
        border_style="cyan",
        box=box.ROUNDED,
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return ""

    preferred_order = ("accuracy", "balanced_accuracy", "f1_macro", "auc")
    items: list[str] = []
    for key in preferred_order:
        if key in metrics:
            items.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics):
        if key not in preferred_order:
            items.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(items)


def _format_finite(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def _format_batch_progress(current_batch: int, total_batches: int, *, phase: str) -> str:
    frames = "|/-\\"
    spinner = frames[current_batch % len(frames)]
    width = 10
    if total_batches > 0:
        filled = min(width, int(round(width * current_batch / total_batches)))
    else:
        filled = 0
    bar = "#" * filled + "-" * (width - filled)
    total_display = f"{total_batches:02d}" if total_batches > 0 else "--"
    return f"{spinner} {phase} {current_batch:02d}/{total_display} [{bar}]"
