"""Discrete-time survival task head.

Implements the Gensheimer / Nnet-survival discrete-time model: the network
emits one logit per time bin, interpreted as an independent per-bin hazard via
sigmoid. Training uses the negative log-likelihood from the HIPT reference
(``NLLSurvLoss``), and ranking uses a risk score derived from the survival
function. See ``survival-design.md`` for the full design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from soma.evaluation.metrics import compute_survival_metrics, resolve_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset, SampleRecord

# Matches the HIPT reference clamp; guards log() against saturated hazards.
_EPS = 1e-7


def _infer_num_bins(dataset: Dataset) -> int:
    """Number of discrete time bins = max bin index + 1 over the dataset."""
    bins = [int(record.metadata["bin"]) for record in dataset.samples.values()]
    return max(bins) + 1


def survival_nll_loss(
    logits: Tensor,
    bins: Tensor,
    events: Tensor,
    *,
    alpha: float = 0.15,
    eps: float = _EPS,
) -> Tensor:
    """Discrete-time negative log-likelihood (HIPT ``NLLSurvLoss``).

    Args:
        logits: Per-bin hazard logits, shape (B, num_bins).
        bins: Discrete event-time bin index per sample, shape (B,). For a
            censored sample this is the last bin the sample was known event-free.
        events: Event indicator, 1 = event observed, 0 = censored, shape (B,).
        alpha: Up-weighting of the uncensored likelihood term (HIPT default).

    Returns:
        Scalar mean loss.
    """
    hazards = torch.sigmoid(logits)  # h_j = P(event in bin j | survived past j-1)
    survival = torch.cumprod(1 - hazards, dim=1)  # S_j = P(survive past bin j)

    batch_size = bins.shape[0]
    Y = bins.view(batch_size, 1)  # ground-truth bin index
    # HIPT convention: c = 1 means censored — the inverse of our event indicator.
    c = (1.0 - events).view(batch_size, 1).float()

    # survival_padded[:, 0] = S(-1) = 1 (all samples alive before bin 0), so
    # survival_padded[:, Y] = S(Y-1) and survival_padded[:, Y+1] = S(Y).
    survival_padded = torch.cat([torch.ones_like(c), survival], dim=1)

    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(
        torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
    )
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def _risk_from_logits(logits: Tensor) -> Tensor:
    """Risk score = -(restricted mean survival time) = -sum_j S_j.

    Higher risk corresponds to shorter expected survival, which is the ordering
    ``concordance_index_censored`` expects from its ``estimate`` argument.
    """
    survival = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
    return -survival.sum(dim=1)


class SurvivalHead(TaskHead):
    """Discrete-time survival head (sigmoid-hazard NLL).

    Args:
        input_dim: Dimension of the input representation.
        num_bins: Number of discrete time bins (model output width).
        alpha: Up-weighting of the uncensored NLL term (HIPT default 0.15).
        metrics: Metrics to compute. Empty list uses the survival default
            (c_index).
    """

    target_dtypes = {"bin": torch.long, "event": torch.float, "time": torch.float}
    task_family = "survival"

    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        alpha: float = 0.15,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_bins < 1:
            raise ValueError(f"SurvivalHead requires num_bins >= 1, got {num_bins}.")
        self.fc = nn.Linear(input_dim, num_bins)
        self.num_bins = num_bins
        self.alpha = alpha
        self.metrics = resolve_metrics("survival", metrics or [])

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_bins": _infer_num_bins(dataset)}

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return {
            "bin": int(record.metadata["bin"]),
            "event": float(record.metadata["event"]),
            "time": float(record.label),
        }

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"SurvivalHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return survival_nll_loss(
            predictions, targets["bin"], targets["event"], alpha=self.alpha
        )

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        risk = _risk_from_logits(raw_output).detach().cpu().numpy()
        return {"risk_scores": risk}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        risk = _risk_from_logits(raw_output).detach().cpu().numpy()
        event = targets["event"].detach().cpu().numpy()
        time = targets["time"].detach().cpu().numpy()
        return compute_survival_metrics(self.metrics, event, time, risk)


def validate_survival_dataset(dataset: Dataset, dataset_type: str) -> None:
    """Fail fast on malformed survival columns before training begins.

    Survival datasets reuse ``label`` for the continuous time-to-event and add
    ``event`` (0/1) and ``bin`` (the discrete bin containing ``label``, for
    every sample including censored ones). Validates presence, ranges, bin
    contiguity from 0, and — for patient pipelines — per-patient agreement on
    ``(label, event, bin)`` (since the patient path extracts targets from one
    representative record).
    """
    records = list(dataset.samples.values())
    if not records:
        raise ValueError("Survival dataset has no samples.")

    sample_meta = records[0].metadata
    for col in ("event", "bin"):
        if col not in sample_meta:
            raise ValueError(
                f"Survival task requires a '{col}' column in the dataset CSV "
                "(alongside 'label', which holds the time-to-event)."
            )

    for record in records:
        sid = record.sample_id
        event = record.metadata["event"]
        bin_value = record.metadata["bin"]
        time = record.label
        if int(event) not in (0, 1):
            raise ValueError(
                f"Survival 'event' must be 0 or 1; sample '{sid}' has {event!r}."
            )
        if int(bin_value) != bin_value or int(bin_value) < 0:
            raise ValueError(
                f"Survival 'bin' must be a non-negative integer; sample '{sid}' "
                f"has {bin_value!r}."
            )
        if float(time) < 0:
            raise ValueError(
                f"Survival 'label' (time) must be >= 0; sample '{sid}' has {time!r}."
            )

    bins = sorted({int(record.metadata["bin"]) for record in records})
    if bins[0] != 0 or bins != list(range(bins[-1] + 1)):
        raise ValueError(
            f"Survival 'bin' values must be contiguous integers starting at 0; "
            f"got {bins}."
        )

    if dataset_type == "patient":
        for patient_id, group in dataset.patient_groups.items():
            triples = {
                (r.label, int(r.metadata["event"]), int(r.metadata["bin"]))
                for r in group
            }
            if len(triples) > 1:
                raise ValueError(
                    f"Patient '{patient_id}' has inconsistent survival targets "
                    "across slides. All slides for a patient must share the same "
                    "(label, event, bin)."
                )


task_registry.register("survival", SurvivalHead)
