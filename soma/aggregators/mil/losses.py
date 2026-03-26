"""Loss utilities for MIL aggregators.

Adapted from torchmil (Apache 2.0).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def _log_sum_exp(x: Tensor) -> Tensor:
    """Numerically stable log(sum(exp(x), dim=1))."""
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))


def _delta(y: Tensor, labels: Tensor, alpha: float | None = None) -> Tensor:
    """Zero-one loss matrix: delta[i,j] = alpha if y[i] != labels[j]."""
    d = torch.ne(y[:, None], labels[None, :]).float()
    if alpha is not None:
        d = alpha * d
    return d


def _detect_large(
    x: Tensor, k: int, tau: float, thresh: float
) -> tuple[Tensor, Tensor]:
    """Detect samples where hard top-k loss should be used."""
    top, _ = x.topk(k + 1, 1)
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * np.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard


class SmoothTop1SVM(nn.Module):
    """Smooth Top-1 SVM loss.

    From "Smooth Loss Functions for Deep Top-k Classification"
    (Berrada et al., 2018). Adapted from torchmil (Apache 2.0).

    Args:
        n_classes: Number of classes.
        alpha: Regularization parameter.
        tau: Temperature parameter.
    """

    def __init__(
        self, n_classes: int = 2, alpha: float = 1.0, tau: float = 1.0
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_classes = n_classes
        self.tau = tau
        self.thresh = 1e3
        self.register_buffer("labels", torch.arange(n_classes))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute loss.

        Args:
            x: Logits, shape (batch_size, n_classes).
            y: Targets, shape (batch_size,).

        Returns:
            Scalar loss.
        """
        smooth, hard = _detect_large(x, 1, self.tau, self.thresh)

        loss = x.new_zeros(())
        if smooth.data.sum():
            x_s = x[smooth].view(-1, x.size(1))
            y_s = y[smooth]
            loss = loss + self._smooth_loss(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h = x[hard].view(-1, x.size(1))
            y_h = y[hard]
            loss = loss + self._hard_loss(x_h, y_h).sum() / x.size(0)
        return loss

    def _hard_loss(self, x: Tensor, y: Tensor) -> Tensor:
        y = y.long()
        y_idx = y.unsqueeze(1)  # (B,) → (B, 1) for gather
        max_, _ = (x + _delta(y, self.labels, self.alpha)).max(1)
        return max_ - x.gather(1, y_idx).squeeze(1)

    def _smooth_loss(self, x: Tensor, y: Tensor) -> Tensor:
        y = y.long()
        y_idx = y.unsqueeze(1)  # (B,) → (B, 1) for gather
        x = x + _delta(y, self.labels, self.alpha) - x.gather(1, y_idx)
        return self.tau * _log_sum_exp(x / self.tau)
