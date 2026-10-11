"""``MILLearner``: the set shape ``(N, D)`` — bags, single embeddings and Cox windows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from soma.learn.learner import Learner, StepOutput, learner_registry
from soma.training.collate import CoxWindowBatch
from soma.training.model import MILModelOutput

__all__ = ["MILLearner"]


class MILLearner(Learner):
    """Trains a :class:`~soma.training.model.MILModel` (or an ``EmbeddingModel``).

    ``forward_batch`` forwards one batch (a padded bag batch in one call; a
    :class:`~soma.training.collate.CoxWindowBatch` bag by bag, keeping the risks
    graph-connected); ``compute_loss`` is the head's loss (or ``loss=``) plus the
    aggregator's auxiliary term. An intermediate loss is one override of
    ``training_step`` that calls both and adds a term.

    Args:
        model: The model; its ``task_head`` owns targets, loss and metrics.
        loss: Optional ``(predictions, targets) -> Tensor`` replacing the head's loss.
        optimizer: Optimizer leaf (see :class:`Learner`).
        schedulers: Scheduler leaves (see :class:`Learner`).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        loss: Callable[[Tensor, dict[str, Tensor]], Tensor] | None = None,
        optimizer: Any = None,
        schedulers: Sequence[Any] = (),
    ) -> None:
        super().__init__(model, optimizer=optimizer, schedulers=schedulers)
        self.loss_fn = loss

    @property
    def head(self):
        return self.model.task_head

    def forward_batch(self, batch: Any):
        if isinstance(batch, CoxWindowBatch):
            risks = []
            for bag in batch.bags:
                bag = bag.unsqueeze(0)  # (1, n_i, D), un-padded
                mask = torch.ones(bag.shape[:2], dtype=torch.bool, device=bag.device)
                risks.append(self.model(bag, mask=mask).logits.view(1))
            return MILModelOutput(logits=torch.cat(risks).unsqueeze(-1))
        mask = getattr(batch, "mask", None)
        if mask is not None:
            return self.model(batch.features, mask=mask)
        return self.model(batch.features)

    def compute_loss(self, output, batch: Any) -> Tensor:
        if self.loss_fn is not None:
            loss = self.loss_fn(output.logits, batch.targets)
        else:
            loss = self.head.compute_loss(output.logits, batch.targets)
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
            loss = self.head.compute_loss(logits, batch.targets).detach().cpu()
        return StepOutput(
            loss=loss,
            outputs=logits.detach().cpu(),
            targets={k: v.detach().cpu() for k, v in batch.targets.items()},
            size=int(logits.shape[0]),
        )

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
            loss = float(self.head.compute_loss(logits, targets).item())
        else:
            loss = float(torch.stack([o.loss for o in outputs]).mean().item())
        metrics = self.head.compute_metrics(logits, targets)
        return {"loss": loss, **{k: float(v) for k, v in metrics.items()}}


learner_registry.register("mil", MILLearner)
