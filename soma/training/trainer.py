"""Trainer — MIL model training with early stopping and checkpointing."""

from __future__ import annotations

import time
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
    elapsed_seconds: float | None = None
    avg_epoch_seconds: float | None = None
    eta_seconds: float | None = None


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
        fold: int | None = None,
        num_folds: int = 1,
    ) -> None:
        self._model = model.to(device)
        self._train_loader = train_loader
        self._tune_loader = tune_loader
        self._config = config
        self._fold_dir = Path(fold_dir)
        self._device = device
        self._console = console
        self._fold = fold
        self._num_folds = num_folds
        self._trainable_param_count = _count_trainable_parameters(self._model)

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
        best_monitor_value = _initial_monitor_value(self._config.monitor_mode)
        best_epoch = 0
        best_tune_metrics: dict[str, float] = {}
        patience_counter = 0
        started_at = time.perf_counter()

        console = self._console or Console()
        current_log: EpochLog | None = None
        current_subtitle = "starting"
        current_status = "waiting for epoch 1"
        current_batch_progress: str | None = None

        current_avg_epoch_seconds: float | None = None
        current_eta_seconds: float | None = None

        def render_panel() -> None:
            elapsed_seconds = time.perf_counter() - started_at
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
                    trainable_param_count=self._trainable_param_count,
                    fold=self._fold,
                    num_folds=self._num_folds,
                    elapsed_seconds=elapsed_seconds,
                    avg_epoch_seconds=current_avg_epoch_seconds,
                    eta_seconds=current_eta_seconds,
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
                trainable_param_count=self._trainable_param_count,
                fold=self._fold,
                num_folds=self._num_folds,
                elapsed_seconds=0.0,
                avg_epoch_seconds=None,
                eta_seconds=None,
                batch_progress=None,
            ),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            render_panel()

            def on_batch_progress(phase: str, processed_items: int, total_items: int) -> None:
                nonlocal current_batch_progress, current_status
                current_batch_progress = _format_batch_progress(
                    processed_items,
                    total_items,
                    phase=phase,
                )
                current_status = f"{phase} items {processed_items}/{total_items}"
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

                elapsed_seconds = time.perf_counter() - started_at
                completed_epochs = len(history) + 1
                avg_epoch_seconds = elapsed_seconds / completed_epochs
                remaining_epochs = max(self._config.epochs - completed_epochs, 0)
                eta_seconds = avg_epoch_seconds * remaining_epochs
                current_avg_epoch_seconds = avg_epoch_seconds
                current_eta_seconds = eta_seconds

                current_log = log = EpochLog(
                    epoch=epoch,
                    train_loss=train_loss,
                    tune_loss=tune_loss,
                    tune_metrics=tune_metrics,
                    lr=lr,
                    elapsed_seconds=elapsed_seconds,
                    avg_epoch_seconds=avg_epoch_seconds,
                    eta_seconds=eta_seconds,
                )
                history.append(log)

                monitor_value = _resolve_monitor_value(
                    self._config,
                    tune_loss=tune_loss,
                    tune_metrics=tune_metrics,
                )
                improved = _is_monitor_improvement(
                    monitor_value,
                    best_monitor_value,
                    self._config.monitor_mode,
                )
                status: str
                if improved:
                    best_tune_loss = tune_loss
                    best_monitor_value = monitor_value
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
            elapsed_seconds = time.perf_counter() - started_at
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
                    trainable_param_count=self._trainable_param_count,
                    fold=self._fold,
                    num_folds=self._num_folds,
                    elapsed_seconds=elapsed_seconds,
                    avg_epoch_seconds=current_avg_epoch_seconds,
                    eta_seconds=current_eta_seconds,
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
        total_items = _resolve_total_items(self._train_loader, fallback=total_batches)
        processed_items = 0
        accum_steps = self._config.gradient_accumulation

        self._optimizer.zero_grad()
        step = 0

        for step, batch in enumerate(self._train_loader, 1):
            processed_items = min(total_items, processed_items + _infer_batch_item_count(batch))
            if on_batch_progress is not None:
                on_batch_progress("train", processed_items, total_items)
            features = batch.features.to(self._device)
            targets = {key: value.to(self._device) for key, value in batch.targets.items()}

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, targets)

            if hasattr(self._model, "aggregator"):
                mask_tensor = batch.mask.to(self._device) if hasattr(batch, "mask") else None
                loss = self._model.aggregator.combine_losses(
                    loss,
                    out.auxiliary,
                    _auxiliary_target(targets),
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
        all_targets: dict[str, list[torch.Tensor]] = {}
        total_batches = len(self._tune_loader)
        total_items = _resolve_total_items(self._tune_loader, fallback=total_batches)
        processed_items = 0

        for batch in self._tune_loader:
            processed_items = min(total_items, processed_items + _infer_batch_item_count(batch))
            if on_batch_progress is not None:
                on_batch_progress("tune", processed_items, total_items)
            features = batch.features.to(self._device)
            targets = {key: value.to(self._device) for key, value in batch.targets.items()}

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, targets)
            total_loss += loss.item()
            num_batches += 1
            all_logits.append(out.logits.detach().cpu())
            for key, value in batch.targets.items():
                all_targets.setdefault(key, []).append(value.detach().cpu())

        avg_loss = total_loss / max(num_batches, 1)
        if all_logits:
            logits = torch.cat(all_logits, dim=0)
            targets = {key: torch.cat(values, dim=0) for key, values in all_targets.items()}
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


def _initial_monitor_value(monitor_mode: str) -> float:
    if monitor_mode == "min":
        return float("inf")
    if monitor_mode == "max":
        return float("-inf")
    msg = f"Unknown monitor_mode: {monitor_mode}. Use 'min' or 'max'."
    raise ValueError(msg)


def _resolve_monitor_value(
    config: TrainingConfig,
    *,
    tune_loss: float,
    tune_metrics: dict[str, float],
) -> float:
    if config.monitor == "tune_loss":
        return tune_loss
    if config.monitor in tune_metrics:
        return tune_metrics[config.monitor]
    available = ", ".join(sorted(tune_metrics)) or "(none)"
    raise ValueError(
        f"Training monitor '{config.monitor}' was not found in tune metrics. "
        f"Available tune metrics: {available}; use 'tune_loss' to monitor loss."
    )


def _is_monitor_improvement(current: float, best: float, monitor_mode: str) -> bool:
    if monitor_mode == "min":
        return current < best
    if monitor_mode == "max":
        return current > best
    msg = f"Unknown monitor_mode: {monitor_mode}. Use 'min' or 'max'."
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


def epoch_log_to_dict(log: EpochLog) -> dict[str, object]:
    data = {
        "epoch": log.epoch,
        "train_loss": log.train_loss,
        "tune_loss": log.tune_loss,
        "tune_metrics": log.tune_metrics,
        "lr": log.lr,
        "elapsed_seconds": log.elapsed_seconds,
        "avg_epoch_seconds": log.avg_epoch_seconds,
    }
    return {key: value for key, value in data.items() if value is not None}


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
    trainable_param_count: int | None = None,
    fold: int | None = None,
    num_folds: int = 1,
    elapsed_seconds: float | None = None,
    avg_epoch_seconds: float | None = None,
    eta_seconds: float | None = None,
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

    if trainable_param_count is not None:
        table.add_row("# params", Text(f"{trainable_param_count:,}", style="white"))

    if fold is not None and num_folds > 1:
        table.add_row("fold", Text(f"{fold + 1}/{num_folds}", style="white"))

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
    table.add_row("epoch avg", Text(_format_optional_duration(avg_epoch_seconds), style="white"))
    table.add_row("elapsed", Text(_format_elapsed_with_eta(elapsed_seconds, eta_seconds), style="white"))

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


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_optional_duration(value: float | None) -> str:
    return _format_elapsed_seconds(value) if value is not None else "n/a"


def _format_elapsed_with_eta(elapsed_seconds: float | None, eta_seconds: float | None) -> str:
    elapsed_text = _format_optional_duration(elapsed_seconds)
    eta_text = _format_optional_duration(eta_seconds)
    return f"{elapsed_text} [ETA {eta_text}]"


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _resolve_total_items(loader: object, *, fallback: int) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return max(0, int(len(dataset)))
        except TypeError:
            pass
    return max(0, int(fallback))


def _auxiliary_target(targets: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Pick the supervised target an aggregator's auxiliary loss should use.

    Aggregators with label-aware auxiliary losses (CLAM, DTFD) expect a single
    target tensor. Single-target heads (classification, regression, ordinal)
    expose exactly one key, so this returns the ``"label"`` target when present
    and otherwise the sole target — reproducing the pre-refactor behavior of
    passing the one label tensor. Aggregators without an auxiliary loss ignore
    the value entirely.
    """
    if "label" in targets:
        return targets["label"]
    if len(targets) == 1:
        return next(iter(targets.values()))
    return None


def _infer_batch_item_count(batch: object) -> int:
    targets = getattr(batch, "targets", None)
    if targets:
        for value in targets.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return max(1, int(value.shape[0]))
    sample_ids = getattr(batch, "sample_ids", None)
    if sample_ids is not None:
        return max(1, int(len(sample_ids)))
    return 1


def _format_batch_progress(current_items: int, total_items: int, *, phase: str) -> str:
    width = 10
    if total_items > 0:
        filled = min(width, int(round(width * current_items / total_items)))
    else:
        filled = 0
    bar = "#" * filled + "-" * (width - filled)
    total_display = str(total_items) if total_items > 0 else "--"
    return f"{phase} {current_items}/{total_display} [{bar}]"
