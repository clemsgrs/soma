"""CLAM aggregators — reference-style CLAM-SB and CLAM-MB.

These implementations follow the original CLAM repository more closely than
the previous torchmil-inspired soma version while preserving soma's
aggregator + task-head architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.losses import SmoothTop1SVM
from soma.aggregators.registry import aggregator_registry


class AttnNet(nn.Module):
    """Attention network without gating."""

    def __init__(self, input_dim: int, attn_dim: int, dropout: float, n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, attn_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(attn_dim, n_classes))
        self.module = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        return self.module(X), X


class AttnNetGated(nn.Module):
    """Attention network with gating."""

    def __init__(self, input_dim: int, attn_dim: int, dropout: float, n_classes: int) -> None:
        super().__init__()
        a_layers: list[nn.Module] = [nn.Linear(input_dim, attn_dim), nn.Tanh()]
        b_layers: list[nn.Module] = [nn.Linear(input_dim, attn_dim), nn.Sigmoid()]
        if dropout > 0:
            a_layers.append(nn.Dropout(dropout))
            b_layers.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(attn_dim, n_classes)

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        A = self.attention_a(X).mul(self.attention_b(X))
        return self.attention_c(A), X


class _CLAMBase(Aggregator):
    """Shared reference-style CLAM behavior."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        gated: bool = True,
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        inst_loss: str = "ce",
        use_negative_class_instance_loss: bool = False,
        bag_weight: float = 0.7,
        multi_branch: bool = False,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.use_negative_class_instance_loss = use_negative_class_instance_loss
        self.bag_weight = bag_weight
        self.multi_branch = multi_branch

        fc: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
        attn_cls = AttnNetGated if gated else AttnNet
        fc.append(
            attn_cls(
                input_dim=hidden_dim,
                attn_dim=attn_dim,
                dropout=dropout,
                n_classes=n_classes if multi_branch else 1,
            )
        )
        self.attention_net = nn.Sequential(*fc)
        self.inst_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
        )

        if inst_loss == "svm":
            self.inst_loss_fn = SmoothTop1SVM(n_classes=2)
        elif inst_loss == "ce":
            self.inst_loss_fn = nn.CrossEntropyLoss()
        else:
            msg = f"inst_loss must be 'ce' or 'svm', got '{inst_loss}'"
            raise ValueError(msg)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        A, H = self.attention_net(X)
        A = torch.transpose(A, 2, 1) if A.ndim == 3 else torch.transpose(A, 1, 0)
        if X.ndim == 3:
            # Batched path: (B, N, C_or_1) -> (B, C_or_1, N)
            A_raw = A
            if mask is not None:
                A = A.masked_fill(~mask.unsqueeze(1), float("-inf"))
            A = F.softmax(A, dim=-1)
            M = torch.bmm(A, H)
        else:
            raise ValueError("CLAM aggregators expect batched input of shape (B, N, D)")
        return AggregatorOutput(
            bag_representation=self._bag_representation(M),
            tile_attention=self._attention_output(A_raw),
            auxiliary={"embeddings": H, "attention": A_raw},
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def compute_auxiliary_loss(
        self,
        auxiliary: dict[str, Tensor],
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute_instance_loss(
            auxiliary["attention"], auxiliary["embeddings"], labels, mask=mask
        )

    def combine_losses(
        self,
        task_loss: Tensor,
        auxiliary: dict[str, Tensor] | None,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if auxiliary is None or self.bag_weight >= 1.0:
            return task_loss
        inst_loss = self.compute_auxiliary_loss(auxiliary, labels, mask=mask)
        return self.bag_weight * task_loss + (1.0 - self.bag_weight) * inst_loss

    def compute_instance_loss(
        self,
        attention: Tensor,
        embeddings: Tensor,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        total_loss = embeddings.new_zeros(())
        batch_size = attention.shape[0]

        for i in range(batch_size):
            att_i = attention[i]
            emb_i = embeddings[i]
            mask_i = mask[i] if mask is not None else None
            inst_labels = F.one_hot(labels[i], num_classes=self.n_classes).reshape(-1)
            bag_loss = embeddings.new_zeros(())
            for class_idx, classifier in enumerate(self.inst_classifiers):
                if inst_labels[class_idx].item() == 1:
                    inst_loss, _, _ = self._inst_eval(att_i, emb_i, classifier, class_idx, mask_i)
                elif self.use_negative_class_instance_loss:
                    inst_loss, _, _ = self._inst_eval_out(att_i, emb_i, classifier, class_idx, mask_i)
                else:
                    continue
                bag_loss = bag_loss + inst_loss
            if self.use_negative_class_instance_loss:
                bag_loss = bag_loss / len(self.inst_classifiers)
            total_loss = total_loss + bag_loss
        return total_loss / max(batch_size, 1)

    def _inst_eval(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        branch_idx: int,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        att_branch = self._attention_branch(att, branch_idx)
        bag_size = int(mask.sum().item()) if mask is not None else att_branch.shape[-1]
        k = min(self.k_sample, bag_size)
        if mask is not None:
            att_branch = att_branch.masked_fill(~mask, float("-inf"))
        top_p_ids = torch.topk(att_branch, k)[1]
        top_p = emb[top_p_ids]
        att_for_neg = att_branch.clone()
        if mask is not None:
            att_for_neg = att_for_neg.masked_fill(~mask, float("inf"))
        top_n_ids = torch.topk(-att_for_neg, k, dim=0)[1]
        top_n = emb[top_n_ids]
        p_targets = torch.ones(k, device=emb.device, dtype=torch.long)
        n_targets = torch.zeros(k, device=emb.device, dtype=torch.long)
        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        logits = classifier(all_instances)
        return self.inst_loss_fn(logits.float(), all_targets), logits, all_targets

    def _inst_eval_out(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        branch_idx: int,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        att_branch = self._attention_branch(att, branch_idx)
        bag_size = int(mask.sum().item()) if mask is not None else att_branch.shape[-1]
        k = min(self.k_sample, bag_size)
        if mask is not None:
            att_branch = att_branch.masked_fill(~mask, float("-inf"))
        top_p_ids = torch.topk(att_branch, k)[1]
        top_p = emb[top_p_ids]
        targets = torch.zeros(k, device=emb.device, dtype=torch.long)
        logits = classifier(top_p)
        return self.inst_loss_fn(logits.float(), targets), logits, targets

    def _attention_branch(self, attention: Tensor, branch_idx: int) -> Tensor:
        raise NotImplementedError

    def _bag_representation(self, M: Tensor) -> Tensor:
        raise NotImplementedError

    def _attention_output(self, A_raw: Tensor) -> Tensor:
        return A_raw


class CLAM_SB(_CLAMBase):
    """Single-branch CLAM aggregator."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        gated: bool = True,
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        inst_loss: str = "ce",
        use_negative_class_instance_loss: bool = False,
        bag_weight: float = 0.7,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            gated=gated,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            inst_loss=inst_loss,
            use_negative_class_instance_loss=use_negative_class_instance_loss,
            bag_weight=bag_weight,
            multi_branch=False,
        )

    def _attention_branch(self, attention: Tensor, branch_idx: int) -> Tensor:
        return attention[0]

    def _bag_representation(self, M: Tensor) -> Tensor:
        return M[:, 0, :]

    def _attention_output(self, A_raw: Tensor) -> Tensor:
        return A_raw[:, 0, :]


class CLAM_MB(_CLAMBase):
    """Multi-branch CLAM aggregator."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        gated: bool = True,
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        inst_loss: str = "ce",
        use_negative_class_instance_loss: bool = False,
        bag_weight: float = 0.7,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            gated=gated,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            inst_loss=inst_loss,
            use_negative_class_instance_loss=use_negative_class_instance_loss,
            bag_weight=bag_weight,
            multi_branch=True,
        )

    def _attention_branch(self, attention: Tensor, branch_idx: int) -> Tensor:
        return attention[branch_idx]

    def _bag_representation(self, M: Tensor) -> Tensor:
        return M


aggregator_registry.register("clam_sb", CLAM_SB)
aggregator_registry.register("clam_mb", CLAM_MB)
