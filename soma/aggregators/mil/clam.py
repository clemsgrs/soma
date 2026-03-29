"""CLAM — Clustering-constrained Attention MIL (Lu et al., 2021).

Adapted from torchmil (Apache 2.0). Unlike torchmil's monolithic CLAM,
this is a pure aggregator (no bag classifier, no loss). The classifier lives
in TaskHead, making CLAM composable with any task.

The aggregation is identical to ABMIL (gated attention pooling). The key
difference is the instance-level clustering loss computed via
`compute_instance_loss()`, which the trainer can call during training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import AttentionPool
from soma.aggregators.mil.losses import SmoothTop1SVM
from soma.aggregators.registry import aggregator_registry


class CLAM(Aggregator):
    """CLAM aggregator with instance-level clustering loss.

    Uses gated attention pooling (same as ABMIL) for aggregation, plus
    two instance classifiers for the CLAM clustering regularization.

    Args:
        input_dim: Feature dimension of input tiles.
        hidden_dim: Attention bottleneck dimension.
        activation: Activation function ('tanh', 'relu', 'gelu').
        gated: If True, use gated attention.
        dropout: Dropout rate applied before attention.
        k_sample: Number of top/bottom instances to sample for clustering loss.
        inst_loss: Instance loss type ('smooth_top1_svm' or 'bce').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        activation: str = "tanh",
        gated: bool = False,
        dropout: float = 0.25,
        k_sample: int = 10,
        inst_loss: str = "smooth_top1_svm",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self.k_sample = k_sample

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = AttentionPool(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            gated=gated,
        )
        self.inst_classifiers = nn.ModuleList(
            [nn.Linear(input_dim, 2) for _ in range(2)]
        )

        if inst_loss == "smooth_top1_svm":
            self.inst_loss_fn = SmoothTop1SVM(n_classes=2)
        elif inst_loss == "bce":
            self.inst_loss_fn = nn.CrossEntropyLoss()
        else:
            msg = f"inst_loss must be 'smooth_top1_svm' or 'bce', got '{inst_loss}'"
            raise ValueError(msg)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        X = self.dropout(X)
        z, attn_logits = self.pool(X, mask=mask)
        return AggregatorOutput(
            bag_representation=z,
            tile_attention=attn_logits,
            auxiliary={"embeddings": X, "attention": attn_logits},
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
        """Instance clustering loss (CLAM auxiliary objective)."""
        return self.compute_instance_loss(
            auxiliary["attention"], auxiliary["embeddings"], labels, mask=mask
        )

    def compute_instance_loss(
        self,
        attention: Tensor,
        embeddings: Tensor,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute CLAM instance-level clustering loss.

        Args:
            attention: Attention logits, shape (B, N).
            embeddings: Tile embeddings, shape (B, N, D).
            labels: Bag labels, shape (B,).
            mask: Boolean mask, shape (B, N). True = valid tile.

        Returns:
            Scalar instance loss.
        """
        total_loss = attention.new_zeros(())
        batch_size = attention.shape[0]

        for i in range(batch_size):
            label = int(labels[i].item())
            in_idx = label
            out_idx = 1 - label

            att_i = attention[i]  # (N,)
            emb_i = embeddings[i]  # (N, D)
            mask_i = mask[i] if mask is not None else None

            bag_size = int(mask_i.sum().item()) if mask_i is not None else att_i.shape[0]
            k = min(self.k_sample, bag_size)

            # In-the-class: top-k (positive) + bottom-k (negative)
            in_loss = self._inst_eval(
                att_i, emb_i, self.inst_classifiers[in_idx], k, mask_i
            )
            # Out-of-the-class: top-k instances labeled as negative
            out_loss = self._inst_eval_out(
                att_i, emb_i, self.inst_classifiers[out_idx], k, mask_i
            )
            total_loss = total_loss + in_loss + out_loss

        return total_loss / max(batch_size, 1)

    def _inst_eval(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        k: int,
        mask: Tensor | None,
    ) -> Tensor:
        """In-the-class instance loss: top-k positive + bottom-k negative."""
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        top_p_ids = torch.topk(att, k)[1]
        top_p = emb[top_p_ids]

        att_for_neg = att.clone()
        if mask is not None:
            att_for_neg = att_for_neg.masked_fill(~mask, float("inf"))
        top_n_ids = torch.topk(-att_for_neg, k)[1]
        top_n = emb[top_n_ids]

        p_targets = torch.ones(k, device=att.device, dtype=torch.long)
        n_targets = torch.zeros(k, device=att.device, dtype=torch.long)

        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        logits = classifier(all_instances)
        return self.inst_loss_fn(logits.float(), all_targets)

    def _inst_eval_out(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        k: int,
        mask: Tensor | None,
    ) -> Tensor:
        """Out-of-the-class instance loss: top-k instances labeled negative."""
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
        top_p_ids = torch.topk(att, k)[1]
        top_p = emb[top_p_ids]
        targets = torch.zeros(k, device=att.device, dtype=torch.long)
        logits = classifier(top_p)
        return self.inst_loss_fn(logits.float(), targets)


aggregator_registry.register("clam", CLAM)
