"""DSMIL — Dual-Stream Multiple Instance Learning (Li et al., 2021).

The dual-stream mechanism:
1. An instance classifier finds the critical instance (highest scoring).
2. Query-key attention is computed relative to the critical instance.
3. The bag representation is the attention-weighted sum of value projections.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import masked_softmax
from soma.aggregators.registry import aggregator_registry


class DSMIL(Aggregator):
    """Dual-Stream MIL aggregator.

    Uses an instance classifier to identify a critical instance, then
    computes attention via query-key matching with the critical instance.

    Args:
        input_dim: Feature dimension of input tiles.
        att_dim: Attention/query dimension.
        nonlinear_q: If True, use nonlinear query projection.
        nonlinear_v: If True, use nonlinear value projection.
        dropout: Dropout rate for value projection (when nonlinear_v=True).
    """

    def __init__(
        self,
        input_dim: int,
        att_dim: int = 128,
        nonlinear_q: bool = False,
        nonlinear_v: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim

        if nonlinear_q:
            self.q_nn = nn.Sequential(
                nn.Linear(input_dim, att_dim),
                nn.ReLU(),
                nn.Linear(att_dim, att_dim),
                nn.Tanh(),
            )
        else:
            self.q_nn = nn.Linear(input_dim, att_dim)

        if nonlinear_v:
            self.v_nn = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            )
        else:
            self.v_nn = nn.Identity()

        self.inst_classifier = nn.Linear(input_dim, 1)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        # Instance classification to find critical instance
        y_logits = self.inst_classifier(X)  # (B, N, 1)

        V = self.v_nn(X)  # (B, N, input_dim)
        Q = self.q_nn(X)  # (B, N, att_dim)

        # Find critical instance (highest instance logit)
        inst_scores = y_logits.clone()
        if mask is not None:
            inst_scores = inst_scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        indices_max = torch.sort(inst_scores, 1, descending=True)[1]  # (B, N, 1)
        idx_max = indices_max[:, 0, :]  # (B, 1)
        idx_max = idx_max.unsqueeze(-1).expand(-1, -1, Q.size(-1))  # (B, 1, att_dim)

        # Query of critical instance
        Q_max = torch.gather(Q, 1, idx_max)  # (B, 1, att_dim)

        # Attention via query-key matching with critical instance
        A = torch.bmm(Q, Q_max.transpose(1, 2))  # (B, N, 1)
        scale = np.sqrt(Q_max.size(-1))
        A = A / scale

        # Masked softmax
        mask_3d = mask.unsqueeze(-1) if mask is not None else None
        A = masked_softmax(A, mask_3d)  # (B, N, 1)

        # Bag representation
        z = torch.bmm(A.transpose(1, 2), V).squeeze(1)  # (B, input_dim)

        return AggregatorOutput(
            bag_representation=z,
            tile_attention=A.squeeze(-1),  # (B, N)
            auxiliary={"instance_logits": y_logits.squeeze(-1)},  # (B, N)
        )

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def compute_auxiliary_loss(
        self,
        auxiliary: dict[str, Tensor],
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Max-instance prediction loss (DSMIL auxiliary objective)."""
        instance_logits = auxiliary["instance_logits"]  # (B, N)
        if mask is not None:
            instance_logits = instance_logits.masked_fill(~mask, float("-inf"))
        max_inst, _ = instance_logits.max(dim=1)  # (B,)
        return nn.functional.binary_cross_entropy_with_logits(
            max_inst, labels.float()
        )


aggregator_registry.register("dsmil", DSMIL)
