"""PatientModel — task head wrapper for patient-level (pre-aggregated) features."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from soma.tasks.base import TaskHead


@dataclass
class PatientModelOutput:
    """Output of PatientModel.forward."""

    logits: Tensor  # (B, num_classes)


class PatientModel(nn.Module):
    """Model for patient-level features: a task head with no aggregator.

    Takes pre-aggregated patient embeddings (B, D) directly and maps them to
    logits via a task head. Structurally identical to SlideModel but kept
    distinct to allow future extension with a trainable slide-to-patient
    aggregation head.
    """

    def __init__(self, task_head: TaskHead) -> None:
        super().__init__()
        self.task_head = task_head

    def forward(self, X: Tensor) -> PatientModelOutput:
        """Forward pass.

        Args:
            X: Patient-level features of shape (B, D).

        Returns:
            PatientModelOutput with logits of shape (B, num_classes).
        """
        return PatientModelOutput(logits=self.task_head(X))
