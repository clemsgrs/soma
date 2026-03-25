"""Classification task head."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry


class ClassificationHead(TaskHead):
    """Linear classification head for binary or multi-class tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)


task_registry.register("classification", ClassificationHead)
