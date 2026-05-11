"""DTFD-MIL — Double-Tier Feature Distillation MIL (Zhang et al., 2022).

The two-tier mechanism:
1. Randomly partition bag into pseudo-bags.
2. Tier 1: AttentionPool each pseudo-bag, compute Grad-CAM importance.
3. Feature distillation: select important instances based on CAM scores.
4. Tier 2: AttentionPool distilled features → final bag representation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import AttentionPool
from soma.aggregators.registry import aggregator_registry
from soma.tasks.base import TaskHead


class DTFDMIL(Aggregator):
    """Double-Tier Feature Distillation MIL aggregator.

    Args:
        input_dim: Feature dimension of input tiles.
        hidden_dim: Attention bottleneck dimension.
        n_groups: Number of pseudo-bags to partition into.
        distill_mode: Feature distillation mode ('maxmin', 'max', 'afs').
        dropout: Dropout rate applied before tier-1 attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_groups: int = 8,
        distill_mode: str = "maxmin",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if distill_mode not in ("maxmin", "max", "afs"):
            msg = f"distill_mode must be 'maxmin', 'max', or 'afs', got '{distill_mode}'"
            raise ValueError(msg)

        self._input_dim = input_dim
        self.n_groups = n_groups
        self.distill_mode = distill_mode
        self._auxiliary_mode = "binary"
        self._t1_output_dim = 1

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Tier 1: pseudo-bag aggregation + classifier (for Grad-CAM)
        self.t1_pool = AttentionPool(input_dim=input_dim, hidden_dim=hidden_dim)
        self.t1_classifier = nn.Linear(input_dim, 1)

        # Tier 2: distilled feature aggregation
        self.t2_pool = AttentionPool(input_dim=input_dim, hidden_dim=hidden_dim)

    def _set_t1_output_dim(self, output_dim: int) -> None:
        if output_dim == self._t1_output_dim:
            return
        old_classifier = self.t1_classifier
        new_classifier = nn.Linear(self._input_dim, output_dim)
        new_classifier.to(
            device=old_classifier.weight.device,
            dtype=old_classifier.weight.dtype,
        )
        self.t1_classifier = new_classifier
        self._t1_output_dim = output_dim

    def configure_for_task(self, task_head: TaskHead) -> None:
        """Resolve tier-1 auxiliary loss shape from the task head."""
        task_family = task_head.task_family
        if task_family == "binary_classification":
            self._auxiliary_mode = "binary"
            self._set_t1_output_dim(1)
            return
        if task_family == "multiclass_classification":
            num_classes = getattr(task_head, "num_classes", None)
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError(
                    "dtfdmil multiclass auxiliary loss requires task_head.num_classes >= 2"
                )
            self._auxiliary_mode = "multiclass"
            self._set_t1_output_dim(num_classes)
            return
        if task_family in {"ordinal_classification", "regression"}:
            self._auxiliary_mode = "scalar_mse"
            self._set_t1_output_dim(1)
            return
        raise ValueError(f"dtfdmil does not support task family '{task_family}'")

    def _cam_1d(self, features: Tensor) -> Tensor:
        """Compute 1D Grad-CAM using tier-1 classifier weights.

        Args:
            features: Tile features, shape (B, chunk_size, D).

        Returns:
            CAM scores, shape (B, chunk_size).
        """
        # Weight of last linear layer: (1, D) → use as importance projection
        weight = self.t1_classifier.weight  # (C, D)
        cam = torch.einsum("bnd,cd->bcn", features, weight)  # (B, 1, chunk_size)
        if cam.size(1) == 1:
            return cam.squeeze(1)  # (B, chunk_size)
        return cam.max(dim=1).values  # (B, chunk_size)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        B, bag_size, feat_dim = X.shape

        X = self.dropout(X)

        n_groups = min(self.n_groups, bag_size)

        # Random partition into pseudo-bags
        bag_index = np.arange(bag_size)
        np.random.shuffle(bag_index)
        bag_chunks = np.array_split(bag_index, n_groups)

        pseudo_pred_list = []
        pseudo_feat_list = []
        inst_cam_list = []

        for chunk_idx in bag_chunks:
            X_chunk = X[:, chunk_idx, :]  # (B, chunk_size, D)
            mask_chunk = mask[:, chunk_idx] if mask is not None else None

            z, _ = self.t1_pool(X_chunk, mask=mask_chunk)  # (B, D)
            pseudo_pred = self.t1_classifier(z)  # (B, 1)
            pseudo_pred_list.append(pseudo_pred)

            inst_cam = self._cam_1d(X_chunk)  # (B, chunk_size)
            inst_cam_list.append(inst_cam)

            chunk_size = X_chunk.size(1)

            if self.distill_mode == "afs":
                pseudo_feat = z.unsqueeze(1)  # (B, 1, D)
            else:
                cam_for_sort = inst_cam
                if mask_chunk is not None:
                    cam_for_sort = inst_cam.masked_fill(~mask_chunk, float("-inf"))

                sort_idx_max = torch.sort(cam_for_sort, 1, descending=True)[1]
                topk_idx_max = sort_idx_max[:, :chunk_size].long()

                if self.distill_mode == "maxmin":
                    cam_for_min = inst_cam
                    if mask_chunk is not None:
                        cam_for_min = inst_cam.masked_fill(~mask_chunk, float("inf"))
                    sort_idx_min = torch.sort(cam_for_min, 1, descending=False)[1]
                    topk_idx_min = sort_idx_min[:, :chunk_size].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=1)
                else:  # "max"
                    topk_idx = topk_idx_max

                index = topk_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
                pseudo_feat = torch.gather(X_chunk, 1, index)

            pseudo_feat_list.append(pseudo_feat)

        # Combine pseudo-bag predictions
        pseudo_pred = torch.stack(pseudo_pred_list, dim=1)  # (B, n_groups, O)
        if pseudo_pred.size(-1) == 1:
            pseudo_pred = pseudo_pred.squeeze(-1)  # (B, n_groups)

        # Combine distilled features and apply tier-2 aggregation
        pseudo_feat = torch.cat(pseudo_feat_list, dim=1)  # (B, total_distilled, D)
        bag_rep, _ = self.t2_pool(pseudo_feat)  # (B, D)

        # Reorder instance CAM to original tile order
        inst_cam = torch.cat(inst_cam_list, dim=1)  # (B, bag_size)
        inst_cam_reorder = torch.zeros_like(inst_cam)
        reorder_idx = np.concatenate(bag_chunks)
        inst_cam_reorder[:, reorder_idx] = inst_cam

        return AggregatorOutput(
            bag_representation=bag_rep,
            tile_attention=inst_cam_reorder,
            auxiliary={"pseudo_predictions": pseudo_pred},
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
        """Pseudo-bag prediction loss (DTFD-MIL tier-1 auxiliary objective)."""
        pseudo_pred = auxiliary["pseudo_predictions"]
        n_groups = pseudo_pred.size(1)
        if self._auxiliary_mode == "binary":
            targets = labels.float().unsqueeze(1).expand(-1, n_groups)
            return F.binary_cross_entropy_with_logits(pseudo_pred, targets)
        if self._auxiliary_mode == "multiclass":
            targets = labels.long().unsqueeze(1).expand(-1, n_groups).reshape(-1)
            logits = pseudo_pred.reshape(-1, pseudo_pred.size(-1))
            return F.cross_entropy(logits, targets)
        targets = labels.float().unsqueeze(1).expand_as(pseudo_pred)
        return F.mse_loss(pseudo_pred, targets)


aggregator_registry.register("dtfdmil", DTFDMIL)
