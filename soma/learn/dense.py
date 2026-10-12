"""``DenseLearner``: the grid shape — segmentation and detection heads over token grids.

One learner for both feature paths: cached grids (``GridBatch.features``) and live
re-encoding (``GridBatch.pixels`` through the model's frozen encoder stage).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from soma.learn.grid import GridBatch
from soma.learn.learner import Learner, LearnerParams, PredictOutput, StepOutput, _move, learner_registry

__all__ = ["DenseLearner"]


class DenseLearner(Learner):
    """Trains a :class:`~soma.training.model.DenseModel` (decoder + grid head).

    The head encodes the batch's raw supervision at loss time (a label map, or level-0
    points rendered into a heatmap). ``validation_step`` returns the head's compact
    per-image stat rows (``dense_stats``), never the ``(N, C, H, W)`` logits;
    ``validation_summary`` concatenates them and calls ``finalize_eval_metrics`` once.
    ``predict_step`` runs ``head.postprocess``: a label map per sample, or every peak
    above the head's ``min_score`` in level-0 pixels.
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
        return self.model.head

    def transfer_batch(self, batch: Any, device: torch.device) -> Any:
        """Features and targets go to ``device``; pixels go where the encoder stage takes
        them (its ``input_device``: a frozen encoder owns its own transfer)."""
        if not isinstance(batch, GridBatch):
            return super().transfer_batch(batch, device)
        encoder = getattr(self.model, "encoder", None)
        pixel_device = (
            device if encoder is None else torch.device(getattr(encoder, "input_device", device))
        )
        return GridBatch(
            features=None if batch.features is None else batch.features.to(device),
            pixels=None if batch.pixels is None else batch.pixels.to(pixel_device),
            targets=_move(batch.targets, device),
            sample_ids=batch.sample_ids,
        )

    def forward_batch(self, batch: Any):
        return self.model(features=batch.features, pixels=batch.pixels)

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

    def predict_step(self, batch: Any) -> PredictOutput:
        """The head's per-sample outputs (a label map, or level-0 points); ``raw`` keeps
        the model output (logits) for artefact callbacks."""
        output = self.forward_batch(batch)
        columns = self.head.to_columns(self.head.postprocess(output.logits.float()))
        return PredictOutput(columns=columns, raw=output)


learner_registry.register("dense", DenseLearner, params=LearnerParams)
