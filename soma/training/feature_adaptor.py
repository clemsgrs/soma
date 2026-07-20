"""The feature adaptor — a buffer-carrying front module over frozen encoder features.

Frozen encoders span 768→4608 dimensions with very different activation scales, so a
shared aggregator/head and its single externally-calibrated learning rate do not see
comparable inputs across encoders. The adaptor closes that gap: it sits at the head of
the model, ahead of the aggregator/head, and applies a composition of independently
optional stages to the features before anything trainable sees them.

Only the **normalize** stage exists today (issue #283); a projection stage is a planned
second stage, which is why :func:`build_feature_adaptor` is written as "assemble the
active stages, and return ``None`` when none are active" rather than as a switch on the
normalization method.

Two invariants define this module:

* **Fitted state lives in buffers, not parameters.** ``zscore``'s per-feature center and
  scale are registered buffers, so the optimizer never sees them (they are not weights to
  learn — they are an estimate of the Support distribution) while they still ride in the
  checkpoint and are re-applied verbatim by the final-checkpoint test pass.
* **Fitting is leak-free.** :meth:`FeatureAdaptor.fit` is the only thing that moves the
  buffers, and the pipeline calls it with the Support (train) split's features alone.
  Running tune/test features through :meth:`FeatureAdaptor.forward` never updates them.

The adaptor is constructed **only when at least one stage is active**. With
``normalization: none`` the module is absent entirely and the model is structurally
identical to a run that predates this module — soma's "omitted section = absent"
convention, and the anchor for byte-identical legacy runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from soma.config import NormalizationConfig

__all__ = [
    "FEATURE_ADAPTER_FILENAME",
    "FeatureAdaptor",
    "build_feature_adaptor",
    "write_feature_adapter_sidecar",
]

#: QC sidecar filename written next to a fold's checkpoint.
FEATURE_ADAPTER_FILENAME = "feature_adapter.json"

#: Normalization methods whose transform needs statistics estimated from the Support set.
_FITTED_METHODS = frozenset({"zscore"})


class FeatureAdaptor(nn.Module):
    """Applies the active adaptor stages to features before the aggregator/head.

    Accepts any tensor whose **last** axis is the feature dimension — ``(N, D)`` tiles,
    ``(B, N, D)`` MIL bags, ``(B, D)`` single embeddings — and returns the same shape, so
    the same module serves every path.

    Args:
        method: Normalization method (``zscore`` | ``l2`` | ``layernorm``). ``none`` never
            reaches here: :func:`build_feature_adaptor` returns ``None`` instead.
        eps: Floor on the divisor, so a constant or near-constant channel cannot blow up.
        num_features: Feature dimension ``D``, fixing the buffer shapes up front so a
            checkpoint loads into an unfitted module.
    """

    def __init__(self, *, method: str, eps: float, num_features: int) -> None:
        super().__init__()
        if method == "none":
            raise ValueError(
                "FeatureAdaptor is never constructed for method='none' — with no active "
                "stage the module must be absent. Use build_feature_adaptor()."
            )
        if num_features < 1:
            raise ValueError(
                f"FeatureAdaptor requires a positive feature dimension, got {num_features}."
            )
        self.method = str(method)
        self.eps = float(eps)
        self.num_features = int(num_features)
        # Buffers, not parameters: fitted state the optimizer must never touch, but which
        # must ride in the checkpoint. Identity-initialized so an unfitted adaptor is a
        # well-defined no-op and a state_dict loads into it.
        self.register_buffer("center", torch.zeros(self.num_features))
        self.register_buffer("scale", torch.ones(self.num_features))
        self.register_buffer("eps_floored", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @property
    def requires_fit(self) -> bool:
        """Whether this adaptor needs :meth:`fit` before it can transform."""
        return self.method in _FITTED_METHODS

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    @property
    def num_eps_floored(self) -> int:
        """How many channels had their scale clamped up to ``eps`` (QC signal)."""
        return int(self.eps_floored.item())

    def fit(self, batches: Iterable[Tensor]) -> "FeatureAdaptor":
        """Estimate the fitted state from ``batches`` of Support-set features.

        ``batches`` is any iterable of tensors whose last axis is the feature dimension —
        one per Support sample for the MIL path, so a whole cohort's tiles never has to be
        materialized at once. Statistics are accumulated in float64 for numerical stability
        across the millions of tiles a real cohort has.

        A stateless method (``l2``/``layernorm``) has nothing to fit; calling this is a
        harmless no-op so callers need not special-case the method.
        """
        if not self.requires_fit:
            return self

        count = 0
        total = torch.zeros(self.num_features, dtype=torch.float64)
        total_sq = torch.zeros(self.num_features, dtype=torch.float64)
        for batch in batches:
            rows = torch.as_tensor(batch).reshape(-1, self.num_features).to(torch.float64)
            count += rows.shape[0]
            total += rows.sum(dim=0)
            total_sq += (rows * rows).sum(dim=0)
        if count == 0:
            raise ValueError(
                f"Cannot fit a '{self.method}' feature adaptor: the Support set yielded no "
                "feature rows. Check that the train split has samples with features."
            )

        mean = total / count
        # E[x^2] - E[x]^2, clamped at 0 against float round-off on constant channels.
        variance = torch.clamp(total_sq / count - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        floored = std < self.eps
        scale = torch.where(floored, torch.full_like(std, self.eps), std)

        self.center.copy_(mean.to(self.center.dtype))
        self.scale.copy_(scale.to(self.scale.dtype))
        self.eps_floored.fill_(int(floored.sum().item()))
        self.fitted.fill_(True)
        return self

    def forward(self, X: Tensor) -> Tensor:
        if self.method == "zscore":
            if not self.is_fitted:
                raise RuntimeError(
                    "FeatureAdaptor(method='zscore') was used before it was fit. Fit it on "
                    "the Support (train) split, or load a checkpoint that carries its state."
                )
            return (X - self.center) / self.scale
        if self.method == "l2":
            return X / torch.clamp(
                X.norm(p=2, dim=-1, keepdim=True), min=self.eps
            )
        if self.method == "layernorm":
            return torch.nn.functional.layer_norm(
                X, (X.shape[-1],), eps=self.eps
            )
        raise ValueError(f"Unhandled feature-adaptor method {self.method!r}.")

    def qc_summary(self) -> dict[str, Any]:
        """The `feature_adapter` QC sidecar payload."""
        return {
            "normalization": {
                "method": self.method,
                "eps": self.eps,
                "eps_floored_channels": self.num_eps_floored,
            },
            "num_features": self.num_features,
            "fitted": self.is_fitted,
        }

    def extra_repr(self) -> str:
        return f"method={self.method!r}, eps={self.eps}, num_features={self.num_features}"


def build_feature_adaptor(
    normalization: NormalizationConfig | None, *, num_features: int
) -> FeatureAdaptor | None:
    """Build the adaptor for a config, or ``None`` when no stage is active.

    ``None`` is the load-bearing return value: it is what keeps a ``normalization: none``
    model structurally identical to one built before this module existed.
    """
    if normalization is None or normalization.method == "none":
        return None
    return FeatureAdaptor(
        method=normalization.method,
        eps=normalization.eps,
        num_features=num_features,
    )


def write_feature_adapter_sidecar(adaptor: FeatureAdaptor | None, fold_dir: Path) -> Path | None:
    """Write the ``feature_adapter.json`` QC sidecar, or nothing when no adaptor exists.

    Reports the normalization method, the ``eps`` floor, and how many channels that floor
    actually caught — a near-zero-variance channel is the failure mode worth noticing, so
    a non-zero count is the signal an operator wants surfaced next to the checkpoint.
    """
    if adaptor is None:
        return None
    path = Path(fold_dir) / FEATURE_ADAPTER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(adaptor.qc_summary(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
