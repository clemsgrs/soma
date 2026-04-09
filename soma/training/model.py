"""MILModel — composes an aggregator with a task head."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from soma.aggregators.base import Aggregator
from soma.tasks.base import TaskHead


@dataclass
class MILModelOutput:
    """Structured output from a MIL model.

    Attributes:
        logits: Task predictions, shape (B, num_classes) or (B, 1).
        tile_attention: Per-tile attention weights from the aggregator,
            shape (B, N). None if the aggregator has no attention.
        auxiliary: Optional dict of auxiliary tensors from the aggregator.
    """

    logits: Tensor
    tile_attention: Tensor | None = None
    auxiliary: dict[str, Tensor] | None = None


class MILModel(nn.Module):
    """Composes an aggregator and a task head into a full MIL model.

    The aggregator maps (B, N, D) → AggregatorOutput, and the task head
    maps the bag representation to predictions. Tile attention (if available)
    is passed through for interpretability and heatmap generation.

    Args:
        aggregator: MIL aggregator (e.g. ABMIL, MeanPool).
        task_head: Task head (e.g. ClassificationHead).
    """

    def __init__(self, aggregator: Aggregator, task_head: TaskHead) -> None:
        super().__init__()
        aggregator.configure_for_task(task_head)
        self.aggregator = aggregator
        self.task_head = task_head

    def forward(self, X: Tensor, mask: Tensor | None = None) -> MILModelOutput:
        agg_out = self.aggregator(X, mask=mask)
        if (
            agg_out.bag_representation.ndim == 3
            and not getattr(self.task_head, "supports_branch_representation", False)
        ):
            msg = (
                f"{self.task_head.__class__.__name__} does not support branch-aware "
                "bag representations; use a compatible classification head."
            )
            raise ValueError(msg)
        logits = self.task_head(agg_out.bag_representation)
        return MILModelOutput(
            logits=logits,
            tile_attention=agg_out.tile_attention,
            auxiliary=agg_out.auxiliary,
        )
