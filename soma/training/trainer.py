"""Trainer — MIL model training with early stopping and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from soma.config import TrainingConfig


@dataclass(frozen=True)
class EpochLog:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: dict[str, float]
    lr: float


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run."""

    best_epoch: int
    best_val_loss: float
    best_val_metrics: dict[str, float]
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
        val_loader: Validation DataLoader.
        config: Training configuration.
        output_dir: Directory for checkpoints.
        device: torch.device for training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        output_dir: Path,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._output_dir = Path(output_dir)
        self._device = device

        self._optimizer = _build_optimizer(model, config)
        self._scheduler = _build_scheduler(self._optimizer, config)

    def fit(self) -> TrainResult:
        """Run the full training loop.

        Returns:
            TrainResult with best epoch, metrics, history, and checkpoint path.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._output_dir / "best_model.pt"

        history: list[EpochLog] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_val_metrics: dict[str, float] = {}
        patience_counter = 0

        for epoch in range(self._config.epochs):
            train_loss = self._train_epoch()
            val_loss, val_metrics = self._validate()

            lr = self._optimizer.param_groups[0]["lr"]
            if self._scheduler is not None:
                self._scheduler.step()

            log = EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                lr=lr,
            )
            history.append(log)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_metrics = val_metrics
                patience_counter = 0
                _save_checkpoint(self._model, self._optimizer, epoch, val_loss, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self._config.patience:
                    break

        return TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_metrics=best_val_metrics,
            history=history,
            checkpoint_path=checkpoint_path,
        )

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self._model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self._train_loader:
            features = batch.features.to(self._device)
            labels = batch.labels.to(self._device)

            self._optimizer.zero_grad()
            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, labels)

            # Auxiliary loss from aggregator (CLAM, DSMIL, DTFD-MIL)
            if getattr(out, "auxiliary", None) is not None and hasattr(self._model, "aggregator"):
                mask_tensor = batch.mask.to(self._device) if hasattr(batch, "mask") else None
                aux_loss = self._model.aggregator.compute_auxiliary_loss(
                    out.auxiliary, labels, mask=mask_tensor
                )
                if aux_loss is not None:
                    loss = loss + aux_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.inference_mode()
    def _validate(self) -> tuple[float, dict[str, float]]:
        """Run validation. Returns (average loss, metrics dict)."""
        self._model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self._val_loader:
            features = batch.features.to(self._device)
            labels = batch.labels.to(self._device)

            if hasattr(batch, "mask"):
                out = self._model(features, mask=batch.mask.to(self._device))
            else:
                out = self._model(features)
            loss = self._model.task_head.compute_loss(out.logits, labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, {"val_loss": avg_loss}


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
    val_loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
