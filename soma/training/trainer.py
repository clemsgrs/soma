"""Trainer — MIL model training with checkpoint selection and early stopping."""

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


def model_input_device(model: torch.nn.Module, default: torch.device | str) -> torch.device:
    """Resolve where a batch's feature tensor belongs before ``model(features)``.

    Most models consume features on their trainable-module device. Live dense models
    deliberately consume CPU pixels because slide2vec's public kit owns device transfer.
    """
    return torch.device(getattr(model, "input_device", default))


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

    selected_epoch: int
    selected_tune_loss: float
    selected_tune_metrics: dict[str, float]
    history: list[EpochLog]
    checkpoint_path: Path


class Trainer:
    """Model trainer with checkpoint selection, early stopping and checkpointing.

    Supports both MILModel (tile-level, batches have a `mask` attribute)
    and SlideModel (slide-level, batches have no mask).

    ``config.checkpoint_selection`` decides which epoch's weights survive on disk:
    ``best`` keeps the best monitored epoch (early-stopping after
    ``config.patience`` epochs without improvement), ``last`` keeps the final epoch
    and never early-stops, while still evaluating the tune split every epoch so the
    history carries per-epoch diagnostics.

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
        on_checkpoint_improved: Callable[[Path, list[EpochLog], int], None] | None = None,
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
        self._on_checkpoint_improved = on_checkpoint_improved
        self._trainable_param_count = _count_trainable_parameters(self._model)

        self._optimizer = _build_optimizer(model, config)
        self._scheduler = _build_scheduler(self._optimizer, config)

    def fit(self) -> TrainResult:
        """Run the full training loop.

        Returns:
            TrainResult with selected epoch, metrics, history, and checkpoint path.
        """
        self._fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._fold_dir / "best_model.pt"

        history: list[EpochLog] = []
        selected_tune_loss = float("inf")
        selected_monitor_value = _initial_monitor_value(self._config.monitor_mode)
        selected_epoch = 0
        selected_tune_metrics: dict[str, float] = {}
        patience_counter = 0
        started_at = time.perf_counter()
        select_last = self._config.checkpoint_selection == "last"
        patience = self._config.patience
        # Under `last` the configured monitor selects nothing, so labelling the panel's
        # tracked value with it would misreport which quantity is shown.
        monitor_name = "tune_loss" if select_last else self._config.monitor

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
                    selected_epoch=selected_epoch,
                    selected_tune_loss=selected_tune_loss,
                    selected_tune_metrics=selected_tune_metrics,
                    monitor_name=monitor_name,
                    selected_monitor_value=selected_monitor_value,
                    patience_counter=patience_counter,
                    patience_limit=patience,
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
                selected_epoch=selected_epoch,
                selected_tune_loss=selected_tune_loss,
                selected_tune_metrics=selected_tune_metrics,
                monitor_name=monitor_name,
                selected_monitor_value=selected_monitor_value,
                patience_counter=patience_counter,
                patience_limit=patience,
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
                batch_sampler = getattr(self._train_loader, "batch_sampler", None)
                set_epoch = getattr(batch_sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
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

                if select_last:
                    # Final-checkpoint protocol (#282): every epoch supersedes the last,
                    # so the checkpoint left on disk is the final-epoch weights. The tune
                    # metrics above are still computed and logged — as diagnostics only.
                    # Resolve the declared monitor to validate direct Trainer callers, then
                    # deliberately discard it as a selection signal.
                    _resolve_monitor_value(
                        self._config,
                        tune_loss=tune_loss,
                        tune_metrics=tune_metrics,
                    )
                    monitor_value = tune_loss
                    improved = True
                else:
                    monitor_value = _resolve_monitor_value(
                        self._config,
                        tune_loss=tune_loss,
                        tune_metrics=tune_metrics,
                    )
                    improved = _is_monitor_improvement(
                        monitor_value,
                        selected_monitor_value,
                        self._config.monitor_mode,
                    )
                status: str
                if improved:
                    selected_tune_loss = tune_loss
                    selected_monitor_value = monitor_value
                    selected_epoch = epoch
                    selected_tune_metrics = tune_metrics
                    patience_counter = 0
                    _save_checkpoint(self._model, self._optimizer, epoch, tune_loss, checkpoint_path)
                    if self._on_checkpoint_improved is not None:
                        self._on_checkpoint_improved(checkpoint_path, history, epoch)
                    status = f"new selected checkpoint saved at epoch {epoch + 1}"
                else:
                    patience_counter += 1
                    status = f"no improvement ({patience_counter}/{_format_patience(patience)})"

                current_status = status
                render_panel()

                if not improved and patience is not None and patience_counter >= patience:
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
                    selected_epoch=selected_epoch,
                    selected_tune_loss=selected_tune_loss,
                    selected_tune_metrics=selected_tune_metrics,
                    monitor_name=monitor_name,
                    selected_monitor_value=selected_monitor_value,
                    patience_counter=patience_counter,
                    patience_limit=patience,
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
            selected_epoch=selected_epoch,
            selected_tune_loss=selected_tune_loss,
            selected_tune_metrics=selected_tune_metrics,
            history=history,
            checkpoint_path=checkpoint_path,
        )

    def _train_epoch(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        if getattr(self._model.task_head, "accumulates_predictions", False):
            return self._train_epoch_windowed(on_batch_progress=on_batch_progress)

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
            features = batch.features.to(model_input_device(self._model, self._device))
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

    def _train_epoch_windowed(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> float:
        """Cox prediction-accumulation training epoch. Returns average loss.

        Each window holds N un-padded bags (a :class:`CoxWindowBatch`). The bags
        are forwarded one at a time and their risk scalars are kept
        graph-connected — no ``detach``/``item``/``no_grad`` until the loss — so a
        single Cox loss couples all N bags through one risk set, with one
        ``optimizer.step`` per window (Lever-1 accumulation only).
        """
        self._model.train()
        total_loss = 0.0
        total_windows = len(self._train_loader)
        total_items = _resolve_total_items(self._train_loader, fallback=total_windows)
        processed_items = 0
        head = self._model.task_head

        windows = 0
        for window in self._train_loader:
            windows += 1
            targets = {key: value.to(self._device) for key, value in window.targets.items()}

            risks = []
            for bag in window.bags:
                bag = bag.to(self._device).unsqueeze(0)  # (1, n_i, D), un-padded
                mask = torch.ones(
                    bag.shape[:2],
                    dtype=torch.bool,
                    device=self._device,
                )
                out = self._model(bag, mask=mask)
                risks.append(out.logits.view(1))  # graph-connected risk scalar

            risk = torch.cat(risks)  # (N,)
            loss = head.compute_loss(risk.unsqueeze(-1), targets)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()

            processed_items = min(total_items, processed_items + len(window.bags))
            if on_batch_progress is not None:
                on_batch_progress("train", processed_items, total_items)

        return total_loss / max(windows, 1)

    @torch.inference_mode()
    def _tune(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Run evaluation on tune set. Returns (average loss, metrics dict)."""
        if getattr(self._model.task_head, "accumulates_eval_metrics", False):
            return self._tune_streaming_metrics(on_batch_progress=on_batch_progress)

        self._model.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits: list[torch.Tensor] = []
        all_targets: dict[str, list[torch.Tensor]] = {}
        total_batches = len(self._tune_loader)
        total_items = _resolve_total_items(self._tune_loader, fallback=total_batches)
        processed_items = 0

        # Losses that couple samples within a batch (e.g. Cox partial likelihood)
        # must be evaluated once over the whole cohort — the mean of per-batch
        # losses is not the full-cohort loss and would make early stopping depend
        # on how the tune loader happened to partition patients.
        full_cohort_loss = getattr(self._model.task_head, "full_cohort_eval_loss", False)

        for batch in self._tune_loader:
            processed_items = min(total_items, processed_items + _infer_batch_item_count(batch))
            if on_batch_progress is not None:
                on_batch_progress("tune", processed_items, total_items)
            features = batch.features.to(model_input_device(self._model, self._device))
            targets = {key: value.to(self._device) for key, value in batch.targets.items()}

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            if not full_cohort_loss:
                loss = self._model.task_head.compute_loss(out.logits, targets)
                total_loss += loss.item()
            num_batches += 1
            all_logits.append(out.logits.detach().cpu())
            for key, value in batch.targets.items():
                all_targets.setdefault(key, []).append(value.detach().cpu())

        if all_logits:
            logits = torch.cat(all_logits, dim=0)
            targets = {key: torch.cat(values, dim=0) for key, values in all_targets.items()}
            if full_cohort_loss:
                avg_loss = float(self._model.task_head.compute_loss(logits, targets).item())
            else:
                avg_loss = total_loss / max(num_batches, 1)
            metrics = self._model.task_head.compute_metrics(logits, targets)
        else:
            avg_loss = total_loss / max(num_batches, 1)
            metrics = {}
        return avg_loss, metrics

    @torch.inference_mode()
    def _tune_streaming_metrics(
        self,
        on_batch_progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Tune eval for heads that accumulate compact per-image metric stats.

        Loss is a per-batch scalar, so its running mean is safe. The actual
        accumulation is the shared :func:`accumulate_dense_stats` (see its docstring
        for why the per-image axis must survive); here we additionally average loss.
        """
        head = self._model.task_head
        stat_rows, total_loss, num_batches = accumulate_dense_stats(
            self._model,
            self._tune_loader,
            self._device,
            compute_loss=True,
            on_batch_progress=on_batch_progress,
            progress_label="tune",
        )
        avg_loss = total_loss / max(num_batches, 1)
        metrics = head.finalize_eval_metrics(torch.cat(stat_rows, dim=0)) if stat_rows else {}
        return avg_loss, metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def accumulate_dense_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    compute_loss: bool = False,
    on_batch_progress: Callable[[str, int, int], None] | None = None,
    on_batch_output: Callable[[object, torch.Tensor, torch.Tensor], None] | None = None,
    progress_label: str = "tune",
) -> tuple[list[torch.Tensor], float, int]:
    """Stream a dense head's compact per-image confusion counts over ``loader``.

    Shared by the trainer's tune eval and the pipeline's split eval so the two
    cannot drift. Dense segmentation logits ``(N, C, H, W)`` would OOM if
    concatenated across a cohort; instead each batch contributes the head's compact
    ``dense_stats`` rows ``(B, C, 3)``. The rows are returned un-concatenated, with
    the **per-image axis preserved** — the caller concatenates along that axis into
    ``(ΣB, C, 3)`` and reduces once via ``finalize_eval_metrics``. Summing the rows
    to ``(C, 3)`` would silently switch the per-image-macro metric to dataset-global.

    Returns ``(stat_rows, total_loss, num_batches)``; ``total_loss`` is summed only
    when ``compute_loss`` (each term a per-batch scalar, so a running mean is safe).

    ``on_batch_output`` (eval-only; ``None`` in the per-epoch tune pass) receives
    ``(batch, out.logits, stat_row)`` per batch *before* the logits are discarded —
    used to stream prediction rasters/overlays to disk without holding all logits.
    """
    model.eval()
    head = model.task_head
    total_loss = 0.0
    num_batches = 0
    stat_rows: list[torch.Tensor] = []
    total_batches = len(loader)
    total_items = _resolve_total_items(loader, fallback=total_batches)
    processed_items = 0

    for batch in loader:
        processed_items = min(total_items, processed_items + _infer_batch_item_count(batch))
        if on_batch_progress is not None:
            on_batch_progress(progress_label, processed_items, total_items)
        features = batch.features.to(model_input_device(model, device))
        targets = {key: value.to(device) for key, value in batch.targets.items()}
        out = model(features)
        if compute_loss:
            total_loss += head.compute_loss(out.logits, targets).item()
        num_batches += 1
        stat_row = head.dense_stats(out.logits, targets).detach().cpu()
        stat_rows.append(stat_row)
        if on_batch_output is not None:
            on_batch_output(batch, out.logits, stat_row)

    return stat_rows, total_loss, num_batches


def _build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    # Only optimize trainable params: the live segmentation model holds a frozen
    # encoder, and an optimizer raises if handed params with requires_grad=False. A
    # strict no-op for fully-trainable models (every param requires grad).
    params = [p for p in model.parameters() if p.requires_grad]
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


def peak_per_metric(history: list[EpochLog]) -> dict[str, dict[str, float | int]]:
    """Per-metric best value across all epochs, with the epoch it occurred at.

    DIAGNOSTIC ONLY. This describes a model that never existed at any single
    epoch (each metric peaks at its own epoch), so it is positively biased and
    must never feed checkpoint selection, ``summary.json`` fold aggregates, or any
    results table. It is recorded in the per-fold ``training_history.json`` purely
    to judge whether the configured ``monitor`` is a good selection proxy -- e.g.
    if a rare class's Dice peaks far from the selected epoch, the monitor may be
    starving that class.

    Every metric is treated as "higher is better", which holds for the metrics
    soma reports (Dice, AUROC, accuracy, ...). ``tune_loss`` is not in
    ``tune_metrics`` and is therefore excluded -- it is min-better, not a metric.

    Returns ``{metric: {"epoch": <1-based epoch>, "value": <peak value>}}``.
    """
    peaks: dict[str, dict[str, float | int]] = {}
    for log in history:
        for name, raw_value in log.tune_metrics.items():
            value = float(raw_value)
            if not np.isfinite(value):
                continue
            current = peaks.get(name)
            if current is None or value > current["value"]:
                peaks[name] = {"epoch": log.epoch + 1, "value": value}
    return peaks


def _build_training_panel(
    *,
    title: str,
    subtitle: str,
    log: EpochLog | None,
    total_epochs: int,
    selected_epoch: int,
    selected_tune_loss: float,
    selected_tune_metrics: dict[str, float],
    monitor_name: str,
    selected_monitor_value: float,
    patience_counter: int,
    patience_limit: int | None,
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

    selected_metrics_text = _format_metrics(selected_tune_metrics)
    selected_monitor_text = _format_selected_monitor(monitor_name, selected_monitor_value)
    selected_epoch_text = f"{selected_epoch + 1:02d}" if np.isfinite(selected_monitor_value) else "n/a"
    table.add_row(
        "selected",
        Text.assemble(
            (selected_monitor_text, "green"),
            (" @ ", "dim"),
            (selected_epoch_text, "green"),
        ),
    )
    if selected_metrics_text:
        table.add_row("selected @ epoch", Text(selected_metrics_text, style="green"))
    table.add_row(
        "patience",
        Text(f"{patience_counter}/{_format_patience(patience_limit)}", style="white"),
    )
    table.add_row("status", Text(status, style="bold green" if "selected" in status else "bold yellow"))
    table.add_row("epoch avg", Text(_format_optional_duration(avg_epoch_seconds), style="white"))
    table.add_row("elapsed", Text(_format_elapsed_with_eta(elapsed_seconds, eta_seconds), style="white"))

    return Panel.fit(
        table,
        title=title,
        subtitle=subtitle,
        border_style="cyan",
        box=box.ROUNDED,
    )


def _format_patience(patience_limit: int | None) -> str:
    """Render the early-stopping budget; ``None`` means early stopping is off."""
    return "off" if patience_limit is None else str(patience_limit)


def _format_selected_monitor(monitor_name: str, selected_monitor_value: float) -> str:
    if not np.isfinite(selected_monitor_value):
        return "n/a"
    return f"{monitor_name}={selected_monitor_value:.4f}"


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
