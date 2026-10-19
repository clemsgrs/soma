"""Discrete-time survival task head.

Implements the Gensheimer / Nnet-survival discrete-time model: the network
emits one logit per time bin, interpreted as an independent per-bin hazard via
sigmoid. Training uses the negative log-likelihood from the HIPT reference
(``NLLSurvLoss``), and ranking uses a risk score derived from the survival
function. See ``survival-design.md`` for the full design.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from soma.evaluation.metrics import compute_survival_metrics, resolve_metrics
from soma._internal.params import check_choice, check_positive, check_probability
from soma.models.heads.base import TaskHead, build_input_dropout
from soma.models.heads.registry import task_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from soma.data import Cohort, SampleRecord

# Matches the HIPT reference clamp; guards log() against saturated hazards.
_EPS = 1e-7

#: Where a sample's time bin comes from: the cohort's ``bin`` column, or cut points fitted
#: on each fold's training split.
BINNINGS = ("column", "train_quantiles")


@dataclass
class DiscreteSurvivalParams:
    """:class:`DiscreteSurvivalHead`'s params. ``binning`` says where each sample's bin
    comes from: ``column`` reads the cohort's ``bin`` column, and ``num_bins`` left unset
    is inferred from it (:func:`infer_num_bins`); ``train_quantiles`` derives ``num_bins``
    bins from each fold's training split (:func:`fit_bin_edges`), so ``num_bins`` is
    required. The fitted ``bin_edges`` are not a param: a config cannot set them."""

    num_bins: int | None = None
    binning: str = "column"
    alpha: float = 0.15
    dropout: float = 0.0

    def __post_init__(self) -> None:
        check_positive("num_bins", self.num_bins, integer=True, optional=True)
        check_choice("binning", self.binning, BINNINGS)
        if self.binning == "train_quantiles" and self.num_bins is None:
            raise ValueError(
                "binning 'train_quantiles' needs num_bins: the number of bins it derives from each "
                "fold's training split (the head width). It has no default."
            )
        check_probability("dropout", self.dropout, below_one=True)


def infer_num_bins(cohort: "Cohort", params: Mapping[str, Any]) -> dict[str, Any]:
    """``num_bins``: the cohort's largest ``bin`` index plus one. An explicit ``num_bins``
    may be wider than the bins the cohort uses, so nothing is inferred beside it; under
    ``binning: train_quantiles`` it is always explicit."""
    if params.get("num_bins") is not None:
        return {}
    return {"num_bins": max(int(record.targets["bin"]) for record in cohort.records) + 1}


def fit_bin_edges(records: "Sequence[SampleRecord]", params: Mapping[str, Any]) -> dict[str, Any]:
    """``bin_edges`` under ``binning: train_quantiles`` (``{}`` under ``column``): the
    ``num_bins - 1`` inner quantiles of the uncensored times of one fold's training
    ``records``, at ``1/K, ..., (K-1)/K`` (``K = num_bins``, linear interpolation).

    The outer bins are open, so a tune or test time outside the training range falls in
    the first or the last bin, and each bin holds its lower cut point
    (:meth:`DiscreteSurvivalHead.extract_targets`). Fewer than ``num_bins`` uncensored
    times, or tied cut points, cannot form ``num_bins`` bins: that raises rather than
    leaving a bin empty.
    """
    if params.get("binning", "column") != "train_quantiles":
        return {}
    num_bins = int(params["num_bins"])
    times = np.asarray(
        [float(record.targets["time"]) for record in records if int(record.targets["event"]) == 1], dtype=float
    )
    cannot = f"binning 'train_quantiles' cannot form num_bins={num_bins} bins:"
    if times.size < num_bins:
        raise ValueError(
            f"{cannot} the training split has {times.size} uncensored training time(s), fewer "
            "than num_bins; lower num_bins."
        )
    edges = np.quantile(times, np.arange(1, num_bins) / num_bins, method="linear")
    repeated = edges[1:][np.diff(edges) <= 0]
    if repeated.size:
        raise ValueError(
            f"{cannot} the training split's {times.size} uncensored training time(s) repeat the "
            f"cut point {repeated[0]:g} (tied times); lower num_bins."
        )
    return {"bin_edges": [float(edge) for edge in edges]}


@dataclass
class CoxSurvivalParams:
    """:class:`CoxSurvivalHead`'s params."""

    ties: str = "breslow"
    min_events_per_window: int = 1
    cox_window: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        check_choice("ties", self.ties, ("breslow",))
        check_positive("min_events_per_window", self.min_events_per_window, integer=True)
        check_positive("cox_window", self.cox_window, integer=True)
        check_probability("dropout", self.dropout, below_one=True)


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


class DiscreteSurvivalHead(TaskHead):
    """Discrete-time survival head (sigmoid-hazard NLL).

    Args:
        input_dim: Dimension of the input representation.
        num_bins: Number of discrete time bins (model output width).
        alpha: Up-weighting of the uncensored NLL term (HIPT default 0.15).
        metrics: Metrics to compute. Empty list uses the survival default
            (c_index).
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
        binning: ``"column"`` (default) reads each sample's bin from its ``bin``
            target; ``"train_quantiles"`` derives it from ``time`` and ``bin_edges``.
        bin_edges: Under ``"train_quantiles"``, the ``num_bins - 1`` increasing cut
            points :func:`fit_bin_edges` fits on a fold's training split. Bin ``j``
            holds the times in ``[bin_edges[j-1], bin_edges[j])``; the first and last
            bins are open. A run fits them per fold; a config cannot set them.
    """

    target_dtypes = {"bin": torch.long, "event": torch.float, "time": torch.float}
    target_schema = {"time": float, "event": int, "bin": int}
    task_family = "survival"

    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        alpha: float = 0.15,
        metrics: list[str] | None = None,
        dropout: float = 0.0,
        binning: str = "column",
        bin_edges: "Sequence[float] | None" = None,
    ) -> None:
        super().__init__()
        if num_bins < 1:
            raise ValueError(f"DiscreteSurvivalHead requires num_bins >= 1, got {num_bins}.")
        check_choice("binning", binning, BINNINGS)
        if bin_edges is not None:
            if binning != "train_quantiles":
                raise ValueError(
                    "bin_edges are the cut points of binning 'train_quantiles'; binning 'column' "
                    "reads each sample's bin from the cohort."
                )
            bin_edges = [float(edge) for edge in bin_edges]
            if len(bin_edges) != num_bins - 1 or any(b <= a for a, b in zip(bin_edges, bin_edges[1:])):
                raise ValueError(
                    f"bin_edges must be num_bins - 1 = {num_bins - 1} strictly increasing cut "
                    f"points, got {bin_edges}."
                )
        self.fc = nn.Linear(input_dim, num_bins)
        self.dropout = build_input_dropout(dropout)
        self.num_bins = num_bins
        self.alpha = alpha
        self.binning = binning
        self.bin_edges = bin_edges
        self.target_schema = self.resolve_target_schema({"binning": binning})
        self.metrics = resolve_metrics("survival", metrics or [])

    @classmethod
    def resolve_target_schema(cls, params: "Mapping[str, Any] | None" = None) -> dict[str, type]:
        """``time`` and ``event``, plus ``bin`` unless the head derives the bins
        (``binning: train_quantiles``)."""
        schema = dict(cls.target_schema)
        if (params or {}).get("binning") == "train_quantiles":
            del schema["bin"]
        return schema

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        time = float(record.targets["time"])
        if self.binning == "train_quantiles":
            if self.bin_edges is None:
                raise ValueError(
                    "this head derives its bins from the training split (binning 'train_quantiles') "
                    "but holds no bin_edges: build it with the edges fit_bin_edges returns."
                )
            # Left-closed bins: a time equal to a cut point goes to the upper bin.
            bin_index = bisect.bisect_right(self.bin_edges, time)
        else:
            bin_index = int(record.targets["bin"])
        return {"bin": bin_index, "event": float(record.targets["event"]), "time": time}

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"DiscreteSurvivalHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        if self.dropout is not None:
            X = self.dropout(X)
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
    over a risk set of multiple samples. Unlike :class:`DiscreteSurvivalHead`, there is no
    binning and risk is the **raw** model output — higher means higher hazard /
    shorter survival, which is what ``concordance_index_censored`` expects.
    (Reusing the NLL head's ``-sum(surv)`` derivation here would double-invert the
    C-index sign.)

    The ``cox_survival`` head. Two training modes, switched by its ``cox_window``
    param (carried here as ``accumulation_window``):

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
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"event": torch.float, "time": torch.float}
    target_schema = {"time": float, "event": int}
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
        dropout: float = 0.0,
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
        self.dropout = build_input_dropout(dropout)
        self.ties = ties
        self.min_events_per_window = min_events_per_window
        self.accumulation_window = cox_window
        # Accumulation mode forwards cox_window un-padded bags per Cox loss; the
        # trainer keys its windowed training loop on this flag.
        self.accumulates_predictions = cox_window >= 2
        self.metrics = resolve_metrics("survival", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return {
            "event": float(record.targets["event"]),
            "time": float(record.targets["time"]),
        }

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"CoxSurvivalHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        if self.dropout is not None:
            X = self.dropout(X)
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


def resolve_survival_task(task_name: str) -> tuple[type[TaskHead], str]:
    """The head class a survival leaf names and its objective: ``cox`` for a
    :class:`CoxSurvivalHead` (or a subclass), else the discrete-time ``nll``. The leaf's
    name selects the objective (``discrete_survival`` or ``cox_survival``); a registered subclass
    inherits its base's."""
    head_cls = task_registry.resolve(task_name)
    return head_cls, ("cox" if issubclass(head_cls, CoxSurvivalHead) else "nll")


def validate_survival_dataset(
    records: "Sequence[SampleRecord]",
    dataset_type: str,
    loss: str = "nll",
    *,
    num_bins: int | None = None,
    binning: str = "column",
) -> None:
    """Fail fast on malformed survival columns before training begins.

    Survival datasets reuse ``label`` for the continuous time-to-event and add
    ``event`` (0/1). The discrete-time NLL path (``loss='nll'``) additionally
    requires ``bin`` (the discrete bin containing ``label``, for every sample
    including censored ones), unless it derives the bins from the training split
    (``binning='train_quantiles'``); the continuous Cox path (``loss='cox'``) ignores
    ``bin`` and so does not require it. Validates presence, ranges, bin
    indices in [0, num_bins) for an explicit positive integer NLL width, or
    contiguity from 0 when width is inferred, and — for patient pipelines — per-patient
    agreement on the survival targets (since the patient path extracts targets
    from one representative record).
    """
    needs_bin = loss != "cox" and binning == "column"
    if needs_bin and num_bins is not None:
        if isinstance(num_bins, bool) or not isinstance(num_bins, Integral) or num_bins < 1:
            raise ValueError(f"Survival num_bins must be a positive integer, got {num_bins!r}.")
    records = list(records)
    if not records:
        raise ValueError("Survival dataset has no samples.")

    required = ("time", "event", "bin") if needs_bin else ("time", "event")
    for record in records:
        sid = record.sample_id
        for key in required:
            if key not in record.targets:
                raise ValueError(
                    f"Survival task requires a '{key}' target (a '{key}' column in the "
                    f"dataset CSV, or the head's 'targets' mapping it); sample '{sid}' has none."
                )
        event = record.targets["event"]
        time = record.targets["time"]
        if int(event) not in (0, 1):
            raise ValueError(
                f"Survival 'event' must be 0 or 1; sample '{sid}' has {event!r}."
            )
        if needs_bin:
            bin_value = record.targets["bin"]
            if int(bin_value) != bin_value or int(bin_value) < 0:
                raise ValueError(
                    f"Survival 'bin' must be a non-negative integer; sample '{sid}' "
                    f"has {bin_value!r}."
                )
        if float(time) < 0:
            raise ValueError(
                f"Survival 'time' must be >= 0; sample '{sid}' has {time!r}."
            )

    if needs_bin:
        bins = sorted({int(record.targets["bin"]) for record in records})
        if num_bins is not None and bins[-1] >= num_bins:
            raise ValueError(
                f"Survival 'bin' values must lie in [0, {num_bins}); got {bins}."
            )
        if num_bins is None and (bins[0] != 0 or bins != list(range(bins[-1] + 1))):
            raise ValueError(
                f"Survival 'bin' values must be contiguous integers starting at 0; "
                f"got {bins}."
            )

    if dataset_type == "patient":
        groups: dict[str, list] = {}
        for record in records:
            if record.patient_id is None:
                raise ValueError(
                    f"Sample '{record.sample_id}' is missing a patient_id. "
                    "All rows must have a patient_id for patient-level pipelines."
                )
            groups.setdefault(record.patient_id, []).append(record)
        for patient_id, group in groups.items():
            targets = {tuple(float(r.targets[key]) for key in required) for r in group}
            if len(targets) > 1:
                raise ValueError(
                    f"Patient '{patient_id}' has inconsistent survival targets "
                    "across slides. All slides for a patient must share the same "
                    "survival target."
                )


task_registry.register(
    "discrete_survival", DiscreteSurvivalHead, params=DiscreteSurvivalParams, infer=infer_num_bins, fit=fit_bin_edges
)
task_registry.register("cox_survival", CoxSurvivalHead, params=CoxSurvivalParams)
