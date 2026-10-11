"""``DenseLearner`` v0: the grid shape ``(H, W, D)`` — segmentation and detection heads.

Slice 4 extends this class with the live encoder stage; it must not grow a sibling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from soma.learn.learner import Learner, StepOutput, learner_registry

__all__ = ["DenseLearner"]


class DenseLearner(Learner):
    """Trains a decoder + dense head over cached (or live re-encoded) feature grids.

    ``validation_step`` returns the head's compact per-image stat rows
    (``dense_stats``), never the ``(N, C, H, W)`` logits; ``validation_summary``
    concatenates them along the image axis and calls ``finalize_eval_metrics`` once.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: Any = None,
        schedulers: Sequence[Any] = (),
    ) -> None:
        super().__init__(model, optimizer=optimizer, schedulers=schedulers)

    @property
    def head(self):
        return self.model.task_head

    def transfer_batch(self, batch: Any, device: torch.device) -> Any:
        """Targets go to ``device``; the features go where the model consumes them.

        The live segmentation model consumes CPU pixels (slide2vec's kit owns the
        encoder's device), which it declares through ``input_device``.
        """
        input_device = getattr(self.model, "input_device", None)
        if input_device is None or not dataclasses.is_dataclass(batch) or not hasattr(batch, "features"):
            return super().transfer_batch(batch, device)
        rest = super().transfer_batch(dataclasses.replace(batch, features=None), device)
        return dataclasses.replace(rest, features=batch.features.to(torch.device(input_device)))

    def forward_batch(self, batch: Any):
        # An image-prior decoder reads the batch pixels that ride in ``targets["image"]``.
        side = {"image": batch.targets["image"]} if "image" in batch.targets else {}
        return self.model(batch.features, **side)

    def compute_loss(self, output, batch: Any) -> Tensor:
        return self.head.compute_loss(output.logits, batch.targets)

    def training_step(self, batch: Any) -> Tensor:
        return self.compute_loss(self.forward_batch(batch), batch)

    def validation_step(self, batch: Any) -> StepOutput:
        output = self.forward_batch(batch)
        loss = self.head.compute_loss(output.logits, batch.targets).detach().cpu()
        rows = self.head.dense_stats(output.logits, batch.targets).detach().cpu()
        return StepOutput(loss=loss, outputs=rows, targets={}, size=int(output.logits.shape[0]))

    def validation_summary(self, outputs: list[StepOutput]) -> dict[str, float]:
        if not outputs:
            return {"loss": float("nan")}
        loss = float(torch.stack([o.loss for o in outputs]).mean().item())
        metrics = self.head.finalize_eval_metrics(torch.cat([o.outputs for o in outputs], dim=0))
        return {"loss": loss, **{k: float(v) for k, v in metrics.items()}}


learner_registry.register("dense", DenseLearner)
