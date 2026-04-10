"""CLAM aggregators — reference-style CLAM-SB and CLAM-MB.

These implementations follow the original CLAM repository more closely than
the previous torchmil-inspired soma version while preserving soma's
aggregator + task-head architecture.
"""

from __future__ import annotations

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
    """Shared CLAM behavior."""

    _CLASSIFICATION_MODE = "classification"
    _ORDINAL_MODE = "ordinal"
    _REGRESSION_MODE = "regression"
    _VALID_INSTANCE_LOSS_MODES = {_CLASSIFICATION_MODE, _ORDINAL_MODE, _REGRESSION_MODE}
    _TASK_TO_MODE = {
        "binary_classification": _CLASSIFICATION_MODE,
        "multiclass_classification": _CLASSIFICATION_MODE,
        "ordinal_classification": _ORDINAL_MODE,
        "regression": _REGRESSION_MODE,
    }

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
        instance_loss_mode: str | None = None,
        low_attention_weight: float = 0.1,
        topk_target_weight: float = 1.0,
        multi_branch: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._attn_dim = attn_dim
        self._gated = gated
        self._dropout = dropout
        self._output_dim = hidden_dim
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.use_negative_class_instance_loss = use_negative_class_instance_loss
        self.bag_weight = bag_weight
        self.multi_branch = multi_branch
        self.instance_loss_mode = instance_loss_mode
        self.low_attention_weight = low_attention_weight
        self.topk_target_weight = topk_target_weight
        self._resolved_instance_loss_mode: str | None = None

        if inst_loss == "svm":
            self.classification_instance_loss_fn = SmoothTop1SVM(n_classes=2)
        elif inst_loss == "ce":
            self.classification_instance_loss_fn = nn.CrossEntropyLoss()
        else:
            msg = f"inst_loss must be 'ce' or 'svm', got '{inst_loss}'"
            raise ValueError(msg)

        if instance_loss_mode is not None and instance_loss_mode not in self._VALID_INSTANCE_LOSS_MODES:
            msg = (
                "instance_loss_mode must be one of "
                f"{sorted(self._VALID_INSTANCE_LOSS_MODES)}, got '{instance_loss_mode}'"
            )
            raise ValueError(msg)

        self.attention_net = self._build_attention_net(
            n_classes=n_classes if multi_branch else 1,
        )
        self.class_instance_classifiers = self._build_class_instance_classifiers(n_classes)
        self.instance_regressor = nn.Linear(hidden_dim, 1)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _build_attention_net(self, n_classes: int) -> nn.Sequential:
        fc: list[nn.Module] = [nn.Linear(self._input_dim, self._hidden_dim), nn.ReLU()]
        if self._dropout > 0:
            fc.append(nn.Dropout(self._dropout))
        attn_cls = AttnNetGated if self._gated else AttnNet
        fc.append(
            attn_cls(
                input_dim=self._hidden_dim,
                attn_dim=self._attn_dim,
                dropout=self._dropout,
                n_classes=n_classes,
            )
        )
        return nn.Sequential(*fc)

    def _build_class_instance_classifiers(self, n_classes: int) -> nn.ModuleList:
        return nn.ModuleList([nn.Linear(self._hidden_dim, 2) for _ in range(n_classes)])

    def configure_for_task(self, task_head) -> None:
        task_family = getattr(task_head, "task_family", "generic")
        task_num_classes = getattr(task_head, "num_classes", None)
        if task_num_classes is None:
            task_num_classes = getattr(getattr(task_head, "fc", None), "out_features", None)
        if self.multi_branch:
            if task_family not in {"binary_classification", "multiclass_classification"}:
                raise ValueError("clam_mb only supports classification tasks.")
            if task_num_classes is None:
                raise ValueError("clam_mb requires a classification head with num_classes.")
            if task_num_classes != self.n_classes:
                self.n_classes = int(task_num_classes)
                self.attention_net = self._build_attention_net(self.n_classes)
                self.class_instance_classifiers = self._build_class_instance_classifiers(
                    self.n_classes
                )
            self._resolved_instance_loss_mode = self._CLASSIFICATION_MODE
            return

        if task_family not in self._TASK_TO_MODE:
            raise ValueError(
                "clam_sb only supports binary_classification, multiclass_classification, "
                f"ordinal_classification, and regression tasks, got '{task_family}'."
            )

        if task_family == "regression" and getattr(task_head, "num_targets", 1) != 1:
            raise ValueError("clam_sb currently supports only single-target regression tasks.")

        inferred_mode = self._TASK_TO_MODE[task_family]
        if self.instance_loss_mode is None:
            self._resolved_instance_loss_mode = inferred_mode
        else:
            self._resolved_instance_loss_mode = self.instance_loss_mode
            if self._resolved_instance_loss_mode != inferred_mode:
                raise ValueError(
                    f"instance_loss_mode='{self.instance_loss_mode}' is incompatible with "
                    f"task family '{task_family}'."
                )

        if (
            self._resolved_instance_loss_mode != self._CLASSIFICATION_MODE
            and self.use_negative_class_instance_loss
        ):
            raise ValueError(
                "use_negative_class_instance_loss is only supported for classification CLAM."
            )

        if inferred_mode == self._CLASSIFICATION_MODE:
            if task_num_classes is None:
                raise ValueError("classification CLAM requires a classification head with num_classes.")
            if task_num_classes != self.n_classes:
                self.n_classes = int(task_num_classes)
                self.class_instance_classifiers = self._build_class_instance_classifiers(
                    self.n_classes
                )

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        if self._resolved_instance_loss_mode is None:
            mode = self._CLASSIFICATION_MODE if self.multi_branch else self.instance_loss_mode
            self._resolved_instance_loss_mode = mode or self._CLASSIFICATION_MODE
        if X.ndim != 3:
            raise ValueError("CLAM aggregators expect batched input of shape (B, N, D)")

        A, H = self.attention_net(X)
        A = torch.transpose(A, 2, 1)
        A_raw = A
        if mask is not None:
            A = A.masked_fill(~mask.unsqueeze(1), float("-inf"))
        A = F.softmax(A, dim=-1)
        M = torch.bmm(A, H)
        return AggregatorOutput(
            bag_representation=self._bag_representation(M),
            tile_attention=self._attention_output(A_raw),
            auxiliary={"embeddings": H, "attention": A_raw},
        )

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
        mode = self._resolved_instance_loss_mode or self._CLASSIFICATION_MODE
        if mode == self._CLASSIFICATION_MODE:
            return self._compute_classification_instance_loss(attention, embeddings, labels, mask=mask)
        if mode in {self._ORDINAL_MODE, self._REGRESSION_MODE}:
            return self._compute_scalar_instance_loss(attention, embeddings, labels, mask=mask)
        raise ValueError(f"Unsupported resolved CLAM instance loss mode '{mode}'")

    def _compute_classification_instance_loss(
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
            for class_idx, classifier in enumerate(self.class_instance_classifiers):
                if inst_labels[class_idx].item() == 1:
                    inst_loss = self._classification_inst_eval(
                        att_i, emb_i, classifier, class_idx, mask_i
                    )
                elif self.use_negative_class_instance_loss:
                    inst_loss = self._classification_inst_eval_out(
                        att_i, emb_i, classifier, class_idx, mask_i
                    )
                else:
                    continue
                bag_loss = bag_loss + inst_loss
            if self.use_negative_class_instance_loss:
                bag_loss = bag_loss / len(self.class_instance_classifiers)
            total_loss = total_loss + bag_loss
        return total_loss / max(batch_size, 1)

    def _compute_scalar_instance_loss(
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
            top_instances, bottom_instances = self._select_top_and_bottom_instances(
                att_i, emb_i, 0, mask_i
            )

            target = labels[i].float()
            top_predictions = self.instance_regressor(top_instances).squeeze(-1)
            target_value = target.expand_as(top_predictions)
            positive_instance_loss = F.mse_loss(top_predictions, target_value)

            if bottom_instances.shape[0] > 1:
                bottom_predictions = self.instance_regressor(bottom_instances).squeeze(-1)
                mean_prediction = bottom_predictions.mean()
                low_attention_regularization = ((bottom_predictions - mean_prediction) ** 2).mean()
            else:
                low_attention_regularization = embeddings.new_zeros(())

            bag_loss = (
                self.topk_target_weight * positive_instance_loss
                + self.low_attention_weight * low_attention_regularization
            )
            total_loss = total_loss + bag_loss

        return total_loss / max(batch_size, 1)

    def _classification_inst_eval(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        branch_idx: int,
        mask: Tensor | None,
    ) -> Tensor:
        top_p, top_n = self._select_top_and_bottom_instances(att, emb, branch_idx, mask)
        k = top_p.shape[0]
        p_targets = torch.ones(k, device=emb.device, dtype=torch.long)
        n_targets = torch.zeros(k, device=emb.device, dtype=torch.long)
        all_instances = torch.cat([top_p, top_n], dim=0)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        logits = classifier(all_instances)
        return self.classification_instance_loss_fn(logits.float(), all_targets)

    def _classification_inst_eval_out(
        self,
        att: Tensor,
        emb: Tensor,
        classifier: nn.Module,
        branch_idx: int,
        mask: Tensor | None,
    ) -> Tensor:
        top_p, _ = self._select_top_and_bottom_instances(att, emb, branch_idx, mask)
        targets = torch.zeros(top_p.shape[0], device=emb.device, dtype=torch.long)
        logits = classifier(top_p)
        return self.classification_instance_loss_fn(logits.float(), targets)

    def _select_top_and_bottom_instances(
        self,
        att: Tensor,
        emb: Tensor,
        branch_idx: int,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
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
        return top_p, top_n

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
        instance_loss_mode: str | None = None,
        low_attention_weight: float = 0.1,
        topk_target_weight: float = 1.0,
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
            instance_loss_mode=instance_loss_mode,
            low_attention_weight=low_attention_weight,
            topk_target_weight=topk_target_weight,
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
            instance_loss_mode=_CLAMBase._CLASSIFICATION_MODE,
            multi_branch=True,
        )

    def _attention_branch(self, attention: Tensor, branch_idx: int) -> Tensor:
        return attention[branch_idx]

    def _bag_representation(self, M: Tensor) -> Tensor:
        return M


aggregator_registry.register("clam_sb", CLAM_SB)
aggregator_registry.register("clam_mb", CLAM_MB)