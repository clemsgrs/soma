"""DTFD-MIL — Double-Tier Feature Distillation MIL (Zhang et al., 2022).

The two-tier mechanism:
1. Randomly partition bag into pseudo-bags.
2. Tier 1: AttentionPool each pseudo-bag, compute Grad-CAM importance.
3. Feature distillation: select important instances based on CAM scores.
4. Tier 2: AttentionPool distilled features → final bag representation.
"""

from __future__ import annotations

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
        instances_per_group: Instances distilled per pseudo-bag and per direction
            ('max' keeps the top-k CAM instances, 'maxmin' the top-k and bottom-k).
            The reference implementation uses ``total_instance // numGroup`` = 1;
            clamped to the pseudo-bag size.
        dropout: Dropout rate applied before tier-1 attention.

    The pseudo-bag partition is a random permutation drawn from torch's global RNG
    while the module is in training mode (so it follows the run seed and differs
    every step, as in the reference), and a deterministic contiguous split in eval
    mode so scoring a checkpoint is reproducible.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_groups: int = 8,
        distill_mode: str = "maxmin",
        instances_per_group: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if distill_mode not in ("maxmin", "max", "afs"):
            msg = f"distill_mode must be 'maxmin', 'max', or 'afs', got '{distill_mode}'"
            raise ValueError(msg)
        if instances_per_group < 1:
            raise ValueError(
                f"instances_per_group must be >= 1, got {instances_per_group}"
            )

        self._input_dim = input_dim
        self.n_groups = n_groups
        self.distill_mode = distill_mode
        self.instances_per_group = instances_per_group
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

    def _forward_one(
        self,
        X: Tensor,
        original_indices: Tensor,
        original_bag_size: int,
        n_groups: int,
    ) -> AggregatorOutput:
        bag_size, feat_dim = X.shape

        if self.training:
            bag_index = torch.randperm(bag_size, device=X.device)
        else:
            bag_index = torch.arange(bag_size, device=X.device)
        bag_chunks = list(torch.tensor_split(bag_index, n_groups))

        pseudo_pred_list = []
        pseudo_feat_list = []
        inst_cam_list = []

        for chunk_idx in bag_chunks:
            X_chunk = X.index_select(0, chunk_idx).unsqueeze(0)  # (1, chunk_size, D)

            z, _ = self.t1_pool(X_chunk)  # (1, D)
            pseudo_pred = self.t1_classifier(z)  # (1, O)
            pseudo_pred_list.append(pseudo_pred)

            inst_cam = self._cam_1d(X_chunk)  # (1, chunk_size)
            inst_cam_list.append(inst_cam)

            chunk_size = X_chunk.size(1)
            k = min(self.instances_per_group, chunk_size)

            if self.distill_mode == "afs":
                pseudo_feat = z.unsqueeze(1)  # (B, 1, D)
            else:
                cam_for_sort = inst_cam
                sort_idx_max = torch.sort(cam_for_sort, 1, descending=True)[1]
                topk_idx_max = sort_idx_max[:, :k].long()

                if self.distill_mode == "maxmin":
                    cam_for_min = inst_cam
                    sort_idx_min = torch.sort(cam_for_min, 1, descending=False)[1]
                    topk_idx_min = sort_idx_min[:, :k].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=1)
                else:  # "max"
                    topk_idx = topk_idx_max

                index = topk_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
                pseudo_feat = torch.gather(X_chunk, 1, index)

            pseudo_feat_list.append(pseudo_feat)

        # Combine pseudo-bag predictions
        pseudo_pred = torch.stack(pseudo_pred_list, dim=1)  # (1, n_groups, O)
        if pseudo_pred.size(-1) == 1:
            pseudo_pred = pseudo_pred.squeeze(-1)  # (1, n_groups)

        # Combine distilled features and apply tier-2 aggregation
        pseudo_feat = torch.cat(pseudo_feat_list, dim=1)  # (1, total_distilled, D)
        bag_rep, _ = self.t2_pool(pseudo_feat)  # (1, D)

        # Reorder instance CAM to original tile order
        inst_cam = torch.cat(inst_cam_list, dim=1)  # (1, valid_bag_size)
        valid_cam_reorder = torch.zeros_like(inst_cam)
        reorder_idx = torch.cat(bag_chunks)
        valid_cam_reorder[:, reorder_idx] = inst_cam
        full_cam = X.new_zeros((1, original_bag_size))
        full_cam[:, original_indices] = valid_cam_reorder

        return AggregatorOutput(
            bag_representation=bag_rep,
            tile_attention=full_cam,
            auxiliary={"pseudo_predictions": pseudo_pred},
        )

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        B, bag_size, _ = X.shape

        X = self.dropout(X)
        if mask is None:
            valid_mask = torch.ones((B, bag_size), device=X.device, dtype=torch.bool)
        else:
            if mask.shape != (B, bag_size):
                raise ValueError(
                    f"DTFD-MIL mask must have shape {(B, bag_size)}, got {tuple(mask.shape)}"
                )
            valid_mask = mask.bool()

        valid_counts = valid_mask.sum(dim=1)
        if (valid_counts == 0).any():
            raise ValueError("DTFD-MIL received an empty bag after applying mask.")
        n_groups = min(self.n_groups, int(valid_counts.min().item()))

        outputs: list[AggregatorOutput] = []
        all_indices = torch.arange(bag_size, device=X.device)
        for batch_idx in range(B):
            original_indices = all_indices[valid_mask[batch_idx]]
            valid_tiles = X[batch_idx, original_indices, :]
            outputs.append(
                self._forward_one(
                    valid_tiles,
                    original_indices,
                    bag_size,
                    n_groups,
                )
            )

        pseudo_predictions = [
            output.auxiliary["pseudo_predictions"]
            for output in outputs
            if output.auxiliary is not None
        ]
        return AggregatorOutput(
            bag_representation=torch.cat(
                [output.bag_representation for output in outputs], dim=0
            ),
            tile_attention=torch.cat([output.tile_attention for output in outputs], dim=0),
            auxiliary={"pseudo_predictions": torch.cat(pseudo_predictions, dim=0)},
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
