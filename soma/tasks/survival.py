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


def cox_breslow_loss(
    risk: Tensor,
    time: Tensor,
    events: Tensor,
    *,
    eps: float = _EPS,
) -> Tensor:
    """Cox proportional-hazards negative partial log-likelihood (Breslow ties).

    Computed over the samples in ``risk`` as the risk set — for batched Cox this
    is the batch; for full-cohort evaluation it is every patient at once. The
    loss couples samples through the risk set (each event's denominator sums over
    everyone still at risk), so it is *not* per-sample separable: the batch must
    contain at least one event, and the mean of per-batch losses is not the
    full-cohort loss (hence ``full_cohort_eval_loss`` on the head).

    Breslow risk sets: sample ``j`` is at risk for event ``i`` iff
    ``time_j >= time_i``, so each event's log denominator is the log-sum-exp of
    the risks over that set and the loss is the mean over events of
    ``-(risk_i - log_denominator_i)``. The risk set is built from an explicit
    ``(B, B)`` comparison mask rather than a sort, so tied times share exactly
    the same denominator (true Breslow) and the result is invariant to the
    sample order. Computed in float32 for numerical stability.

    Args:
        risk: Per-sample risk score (higher = higher hazard = shorter survival),
            shape (B,). Must stay graph-connected for training.
        time: Event or last-follow-up time, shape (B,).
        events: Event indicator, 1 = event observed, 0 = censored, shape (B,).
            Note: soma stores ``event`` directly (1 = event), so — unlike the
            HIPT reference, which carries ``censored`` — no ``1 - censored`` flip
            is applied here.

    Returns:
        Scalar mean loss. Returns a graph-connected zero when the batch has no
        events (degenerate risk set, no gradient signal).
    """
    risk = risk.float().view(-1)
    time = time.view(-1)
    events = events.view(-1).float()

    # at_risk[i, j] is True when sample j is still at risk at event i's time. The
    # diagonal is always True, so no row is empty and logsumexp stays finite.
    at_risk = time.unsqueeze(0) >= time.unsqueeze(1)
    masked_risk = risk.unsqueeze(0).expand(risk.numel(), -1).masked_fill(
        ~at_risk, float("-inf")
    )
    log_risk_set = torch.logsumexp(masked_risk, dim=1)
    per_event = (risk - log_risk_set) * events

    n_events = events.sum()
    if n_events < 1:
        # No event in this risk set: nothing to rank. Return a graph-connected
        # zero so training does not crash on an (avoidable) event-free batch.
        return (risk * 0.0).sum()
    return -per_event.sum() / n_events.clamp(min=eps)


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


class CoxSurvivalHead(TaskHead):
    """Continuous-time CoxPH survival head (Breslow partial likelihood).

    Emits a single risk scalar per sample; the Cox partial likelihood is computed
    over a risk set of multiple samples. Unlike :class:`SurvivalHead`, there is no
    binning and risk is the **raw** model output — higher means higher hazard /
    shorter survival, which is what ``concordance_index_censored`` expects.
    (Reusing the NLL head's ``-sum(surv)`` derivation here would double-invert the
    C-index sign.)

    Selected via ``task.params.loss: cox``. Two training modes, switched by
    ``task.params.cox_window`` (carried here as ``accumulation_window``):

    * **Padded mode** (``cox_window`` unset / 1): the risk set is the batch.
      Single-embedding slide/patient features (no aggregator, ``batch_size >= 2``)
      or padded MIL bags (``batch_size >= 2``, masking handles padding).
    * **Accumulation mode** (``cox_window >= 2``): for large variable-size MIL
      bags. ``batch_size`` is pinned to 1; the trainer forwards ``cox_window``
      bags un-padded, keeps their risk scalars graph-connected, and computes one
      Cox loss over the window. ``accumulates_predictions`` signals this to the
      trainer.

    In both modes the pipeline builds the training loader with an event-balanced
    sampler (``needs_event_balanced_batches``) and the tune loss is computed over
    the whole cohort (``full_cohort_eval_loss``). Config validation enforces the
    mode constraints.

    Args:
        input_dim: Dimension of the input representation.
        ties: Tie-handling method. Only ``"breslow"`` is implemented.
        min_events_per_window: Minimum events the event-balanced sampler
            guarantees per batch / window (>= 1).
        cox_window: Prediction-accumulation window size. ``1`` (default) selects
            padded mode; ``>= 2`` selects accumulation mode with this risk-set size.
        metrics: Metrics to compute. Empty list uses the survival default
            (c_index).
    """

    target_dtypes = {"event": torch.float, "time": torch.float}
    task_family = "survival"
    full_cohort_eval_loss = True
    needs_event_balanced_batches = True

    def __init__(
        self,
        input_dim: int,
        ties: str = "breslow",
        min_events_per_window: int = 1,
        cox_window: int = 1,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if ties != "breslow":
            raise ValueError(
                f"CoxSurvivalHead supports ties='breslow' only (got {ties!r}); "
                "Efron ties are not implemented."
            )
        if min_events_per_window < 1:
            raise ValueError(
                f"min_events_per_window must be >= 1, got {min_events_per_window}."
            )
        if cox_window < 1:
            raise ValueError(f"cox_window must be >= 1, got {cox_window}.")
        self.fc = nn.Linear(input_dim, 1)
        self.ties = ties
        self.min_events_per_window = min_events_per_window
        self.accumulation_window = cox_window
        # Accumulation mode forwards cox_window un-padded bags per Cox loss; the
        # trainer keys its windowed training loop on this flag.
        self.accumulates_predictions = cox_window >= 2
        self.metrics = resolve_metrics("survival", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return {
            "event": float(record.metadata["event"]),
            "time": float(record.label),
        }

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"CoxSurvivalHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        return self.fc(X)  # (B, 1)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return cox_breslow_loss(
            predictions.squeeze(-1), targets["time"], targets["event"]
        )

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        risk = raw_output.squeeze(-1).detach().cpu().numpy()
        return {"risk_scores": risk}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        risk = raw_output.squeeze(-1).detach().cpu().numpy()
        event = targets["event"].detach().cpu().numpy()
        time = targets["time"].detach().cpu().numpy()
        return compute_survival_metrics(self.metrics, event, time, risk)


def resolve_survival_head(loss: str = "nll") -> type[TaskHead]:
    """Map the ``task.params.loss`` selector to the survival head class.

    ``"nll"`` (default) → discrete-time :class:`SurvivalHead`; ``"cox"`` →
    continuous-time :class:`CoxSurvivalHead`. Keeps the registry keyed on the
    single task name ``"survival"`` while routing to the right head.
    """
    if loss == "nll":
        return SurvivalHead
    if loss == "cox":
        return CoxSurvivalHead
    raise ValueError(f"Unknown survival loss {loss!r}; use 'nll' or 'cox'.")


def validate_survival_dataset(
    dataset: Dataset, dataset_type: str, loss: str = "nll"
) -> None:
    """Fail fast on malformed survival columns before training begins.

    Survival datasets reuse ``label`` for the continuous time-to-event and add
    ``event`` (0/1). The discrete-time NLL path (``loss='nll'``) additionally
    requires ``bin`` (the discrete bin containing ``label``, for every sample
    including censored ones); the continuous Cox path (``loss='cox'``) ignores
    ``bin`` and so does not require it. Validates presence, ranges, bin
    contiguity from 0 (NLL only), and — for patient pipelines — per-patient
    agreement on the survival targets (since the patient path extracts targets
    from one representative record).
    """
    needs_bin = loss != "cox"
    records = list(dataset.samples.values())
    if not records:
        raise ValueError("Survival dataset has no samples.")

    sample_meta = records[0].metadata
    required_cols = ("event", "bin") if needs_bin else ("event",)
    for col in required_cols:
        if col not in sample_meta:
            raise ValueError(
                f"Survival task requires a '{col}' column in the dataset CSV "
                "(alongside 'label', which holds the time-to-event)."
            )

    for record in records:
        sid = record.sample_id
        event = record.metadata["event"]
        time = record.label
        if int(event) not in (0, 1):
            raise ValueError(
                f"Survival 'event' must be 0 or 1; sample '{sid}' has {event!r}."
            )
        if needs_bin:
            bin_value = record.metadata["bin"]
            if int(bin_value) != bin_value or int(bin_value) < 0:
                raise ValueError(
                    f"Survival 'bin' must be a non-negative integer; sample '{sid}' "
                    f"has {bin_value!r}."
                )
        if float(time) < 0:
            raise ValueError(
                f"Survival 'label' (time) must be >= 0; sample '{sid}' has {time!r}."
            )

    if needs_bin:
        bins = sorted({int(record.metadata["bin"]) for record in records})
        if bins[0] != 0 or bins != list(range(bins[-1] + 1)):
            raise ValueError(
                f"Survival 'bin' values must be contiguous integers starting at 0; "
                f"got {bins}."
            )

    if dataset_type == "patient":
        for patient_id, group in dataset.patient_groups.items():
            if needs_bin:
                targets = {
                    (r.label, int(r.metadata["event"]), int(r.metadata["bin"]))
                    for r in group
                }
            else:
                targets = {(r.label, int(r.metadata["event"])) for r in group}
            if len(targets) > 1:
                raise ValueError(
                    f"Patient '{patient_id}' has inconsistent survival targets "
                    "across slides. All slides for a patient must share the same "
                    "survival target."
                )


task_registry.register("survival", SurvivalHead)
