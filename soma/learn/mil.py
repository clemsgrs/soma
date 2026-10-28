"""``MILLearner``: the set shape ``(N, D)`` — bags, single embeddings and Cox windows."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from soma.learn.learner import Learner, LearnerParams, PredictOutput, StepOutput, learner_registry
from soma.learn.collate import CoxWindowBatch
from soma.models.model import MILModelOutput

__all__ = ["MILLearner", "MILLearnerParams"]


@dataclasses.dataclass
class MILLearnerParams(LearnerParams):
    """:class:`MILLearner`'s params: :class:`~soma.learn.learner.LearnerParams` plus an
    optional ``(predictions, targets) -> Tensor`` loss replacing the head's, and
    ``checkpoint_cox_window`` (see :class:`MILLearner`)."""

    loss: Callable[[Tensor, dict[str, Tensor]], Tensor] | None = None
    checkpoint_cox_window: bool = False


class MILLearner(Learner):
    """Trains a :class:`~soma.models.model.MILModel` (or an ``EmbeddingModel``).

    ``forward_batch`` forwards one batch (a padded bag batch in one call; a
    :class:`~soma.learn.collate.CoxWindowBatch` bag by bag, keeping the risks
    graph-connected); ``compute_loss`` is the head's loss (or ``loss=``) plus the
    aggregator's auxiliary term. An intermediate loss is one override of
    ``training_step`` that calls both and adds a term.

    Args:
        model: The model; its ``task_head`` owns targets, loss and metrics.
        loss: Optional ``(predictions, targets) -> Tensor`` replacing the head's loss.
        checkpoint_cox_window: Checkpoint each bag of a training ``CoxWindowBatch``:
            the forward keeps only the bag's risk, and the backward recomputes the bags
            one at a time. Only one bag's graph is then in memory, not the whole
            window's; the loss and gradients are the same, and a window takes about
            twice as long.
        optimizer: Optimizer leaf (see :class:`Learner`).
        schedulers: Scheduler leaves (see :class:`Learner`).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        loss: Callable[[Tensor, dict[str, Tensor]], Tensor] | None = None,
        checkpoint_cox_window: bool = False,
        optimizer: Any = None,
        schedulers: Sequence[Any] = (),
    ) -> None:
        super().__init__(model, optimizer=optimizer, schedulers=schedulers)
        self.loss_fn = loss
        self.checkpoint_cox_window = checkpoint_cox_window

    @property
    def head(self):
        return self.model.task_head

    def transfer_batch(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, CoxWindowBatch):
            # The window's bags stay on the host; ``forward_batch`` moves each one right
            # before its forward. Without ``checkpoint_cox_window`` a training window still
            # holds every bag's graph (with its device copy) until the backward.
            return dataclasses.replace(batch, targets=super().transfer_batch(batch.targets, device))
        return super().transfer_batch(batch, device)

    def forward_batch(self, batch: Any):
        if isinstance(batch, CoxWindowBatch):
            device = next(self.model.parameters()).device
            # Without grad (validation, prediction) there is no graph to save memory on.
            checkpointed = self.checkpoint_cox_window and torch.is_grad_enabled()
            risks = []
            for bag in batch.bags:
                # Checkpoint takes its device from its tensor arguments: given only the
                # host-side bag, it would restore neither the CUDA RNG (dropout) nor CUDA
                # autocast in the recompute, so the device-side mask goes in too.
                mask = torch.ones((1, bag.shape[0]), dtype=torch.bool, device=device)
                if checkpointed:
                    risks.append(checkpoint(self._bag_risk, bag, mask, use_reentrant=False))
                else:
                    risks.append(self._bag_risk(bag, mask))
            return MILModelOutput(logits=torch.cat(risks).unsqueeze(-1))
        mask = getattr(batch, "mask", None)
        if mask is not None:
            return self.model(batch.features, mask=mask)
        return self.model(batch.features)

    def _bag_risk(self, bag: Tensor, mask: Tensor) -> Tensor:
        """One un-padded bag's risk, shape ``(1,)``. The bag moves to the device here, so
        a checkpoint keeps the host-side bag as its input, not the device copy."""
        return self.model(bag.to(mask.device).unsqueeze(0), mask=mask).logits.view(1)

    def task_loss(self, logits: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """The head's loss, or the ``loss=`` callable; training and validation share it."""
        if self.loss_fn is not None:
            return self.loss_fn(logits, targets)
        return self.head.compute_loss(logits, targets)

    def compute_loss(self, output, batch: Any) -> Tensor:
        loss = self.task_loss(output.logits, batch.targets)
        aggregator = getattr(self.model, "aggregator", None)
        auxiliary = getattr(output, "auxiliary", None)
        if aggregator is not None and auxiliary is not None:
            loss = aggregator.combine_losses(
                loss, auxiliary, batch.targets, mask=getattr(batch, "mask", None)
            )
        return loss

    def training_step(self, batch: Any) -> Tensor:
        return self.compute_loss(self.forward_batch(batch), batch)

    def validation_step(self, batch: Any) -> StepOutput:
        output = self.forward_batch(batch)
        logits = output.logits
        loss = None
        if not getattr(self.head, "full_cohort_eval_loss", False):
            loss = self.task_loss(logits, batch.targets).detach().cpu()
        return StepOutput(
            loss=loss,
            # ``.float()``: under bf16 autocast the heads' CPU metrics (numpy) need fp32.
            outputs=logits.detach().float().cpu(),
            targets={k: v.detach().cpu() for k, v in batch.targets.items()},
            size=int(logits.shape[0]),
        )

    def predict_step(self, batch: Any) -> PredictOutput:
        """The head's declared columns for the batch; ``raw`` keeps the full model output
        (``tile_attention`` for attention callbacks)."""
        output = self.forward_batch(batch)
        columns = self.head.to_columns(self.head.postprocess(output.logits.float()))
        return PredictOutput(columns=columns, raw=output)

    def validation_summary(self, outputs: list[StepOutput]) -> dict[str, float]:
        if not outputs:
            return {"loss": float("nan")}
        logits = torch.cat([o.outputs for o in outputs], dim=0)
        targets = {
            key: torch.cat([o.targets[key] for o in outputs], dim=0) for key in outputs[0].targets
        }
        if getattr(self.head, "full_cohort_eval_loss", False):
            # Losses that couple samples (Cox partial likelihood) exist only over the
            # whole split; a mean of per-batch values would depend on the partition.
            loss = float(self.task_loss(logits, targets).item())
        else:
            loss = float(torch.stack([o.loss for o in outputs]).mean().item())
        metrics = self.head.compute_metrics(logits, targets)
        return {"loss": loss, **{k: float(v) for k, v in metrics.items()}}


learner_registry.register("mil", MILLearner, params=MILLearnerParams)
