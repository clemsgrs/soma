"""``detection-benchmark`` — the multi-dataset encoder-ranking harness (issue #246).

This is the *agent-buildable* half of Paper 1 (#228/#234): a registered
:class:`~soma.benchmarks.registry.Benchmark` that ranks a roster of frozen encoders across
three point-detection datasets — OCELOT, MIDOG 2022, and MONKEY — each scored with its own
native metric, and emits a machine-readable ``ranking_report.json``.

It generalizes ``examples/ocelot/campaign.py``'s resumable driver from one dataset to the
three-dataset sweep, but replaces the tune-select -> test-confirm flow with a **full-ranking
protocol**: for every ``(encoder, dataset, replicate)`` the decoder trains on ``train``,
every per-encoder knob (detection threshold, matching radius, val-best checkpoint) is frozen
on ``tune``, then the cell is scored on ``test`` with those frozen knobs. There is **no
winner pick** — all encoders are ranked by their mean test metric, and the tune ranks are
reported alongside as a stability signal.

Everything in this module is pure, GPU-free logic over records the driver produces, so the
whole harness unit-tests cold on synthetic fixtures (no real data, no model downloads). The
GPU orchestration — extract dense grids, train the decoder, run inference — lives in the
thin driver (``examples/detection_benchmark/campaign.py``); this module owns the recipe
registry, the replicate abstraction, the native-metric dispatch, the aggregation + ranking +
rank-consistency + bootstrap-stability math, the frozen subset selections, and the
per-sample-prediction cache that makes the deferred robustness stratification (#248) and the
stability bootstrap pure later re-aggregations.

Design axioms this module enforces *structurally*:

* **No pooled cross-dataset scalar.** Each :class:`Cell` records its own dataset's native
  metric; the only cross-dataset object is the rank-consistency (Spearman/Kendall over the
  per-dataset rank vectors). There is nowhere to write a single "overall detection score".
* **Frozen selections are code.** The subset-selection *rule* (``backbone_subset`` /
  ``efficiency_subset``) is implemented + unit-tested here, so the downstream backbone
  (#235) and efficiency (#237) studies read a field instead of re-deriving "who's top-k".
* **Roster is data, size-agnostic.** :data:`DEFAULT_ROSTER` is a list; the ranking,
  rank-consistency, and selections recompute for any roster size — stretching to the fuller
  ~10 encoders later is "extract the new ones + rerun aggregation", never a re-plumb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    load_reference,
    register_benchmark,
)
from soma.config import PipelineConfig, load_config
from soma.curation.manifest import CuratedManifest
from soma.detection.froc import score_monkey_froc
from soma.detection.matching import detection_counts, reduce_f1
from soma.detection.midog_f1 import midog_f1

__all__ = [
    "BENCHMARK_NAME",
    "DEFAULT_SEEDS",
    "RosterEntry",
    "DEFAULT_ROSTER",
    "DatasetSpec",
    "DATASETS",
    "replicate_plan",
    "SamplePrediction",
    "CellPredictions",
    "write_cell_predictions",
    "read_cell_predictions",
    "score_dataset_points",
    "Cell",
    "aggregate_cell",
    "rank_encoders",
    "aggregate_rank",
    "rank_consistency",
    "bootstrap_rank_stability",
    "select_subsets",
    "build_ranking_report",
    "DetectionBenchmark",
    "DETECTION_BENCHMARK",
]

BENCHMARK_NAME = "detection-benchmark"

# When a dataset ships a single fold, the replicates are seeds; this is the default K.
# (The design bumps headline / close-calls to 5 by passing a longer seed list — the
# harness is agnostic to how many.)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# The ranking backbone: the constant ``proj Conv2d(d, 256, 1)`` gives every encoder the
# same ``d -> D`` capacity, so the ranking compares encoders, not decoder width.
DECODER = "lightweight_conv"

_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "detection"


# --- roster --------------------------------------------------------------------------


@dataclass(frozen=True)
class RosterEntry:
    """One encoder in the ranking roster, with the two flags the selection rule reads.

    ``is_compact`` marks a small / cheap backbone (a "mini" pathology FM or the ViT-B
    control) that the backbone (#235) and efficiency (#237) subsets always keep as the
    low-cost reference point. ``is_control`` marks the natural-image control
    (``dinov2-vitb14``), the pretraining-domain baseline the efficiency subset always keeps.
    """

    name: str
    is_compact: bool = False
    is_control: bool = False


# Core pathology FMs + the natural-image control, read at each dataset's native spacing.
# A list, not a hardcoded set: extraction is per-encoder additive and every downstream
# computation recomputes for whatever roster is passed.
DEFAULT_ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry("virchow2"),
    RosterEntry("genbio-pathfm"),
    RosterEntry("midnight"),
    RosterEntry("conchv15"),
    RosterEntry("h0-mini", is_compact=True),
    RosterEntry("h-optimus-1"),
    RosterEntry("dinov2-vitb14", is_compact=True, is_control=True),
)


def roster_names(roster: Sequence[RosterEntry]) -> list[str]:
    """The encoder names of a roster, in roster order."""
    return [entry.name for entry in roster]


# --- dataset specs -------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset's fixed recipe knobs: native spacing, native metric, and match geometry.

    ``metric_name`` is the *unpooled* metric a :class:`Cell` records for this dataset
    (OCELOT ``mean_f1``, MIDOG ``f1``, MONKEY ``mean_froc``). ``match_um`` holds the
    per-class matching margins in µm; the native scorer converts them to point-frame pixels
    via ``spacing_um`` at score time. ``match_method`` is the matcher the native metric uses
    (Hungarian one-to-one for MIDOG, greedy-by-confidence for OCELOT/MONKEY). ``tolerance``
    is the reader's spacing tolerance surfaced in the committed config (MIDOG relaxes it to
    0.10 for its mixed scanners). ``test_source`` records whether the reported test split is
    a local held-out carve or the official test set.
    """

    name: str
    spacing_um: float
    metric_name: str
    num_classes: int
    match_um: tuple[float, ...]
    match_method: str
    tolerance: float | None
    test_source: str
    config_file: str


DATASETS: dict[str, DatasetSpec] = {
    "ocelot": DatasetSpec(
        name="ocelot",
        spacing_um=0.2,
        metric_name="mean_f1",
        num_classes=2,
        match_um=(3.0,),
        match_method="greedy",
        tolerance=None,
        test_source="local_holdout",
        config_file="ocelot.yaml",
    ),
    "midog": DatasetSpec(
        name="midog",
        spacing_um=0.25,
        metric_name="f1",
        num_classes=1,
        match_um=(7.5,),
        match_method="hungarian",
        tolerance=0.10,
        test_source="local_holdout",
        config_file="midog.yaml",
    ),
    "monkey": DatasetSpec(
        name="monkey",
        spacing_um=0.24199951445730394,
        metric_name="mean_froc",
        num_classes=2,
        match_um=(4.0, 5.0),
        match_method="greedy",
        tolerance=None,
        test_source="local_holdout",
        config_file="monkey.yaml",
    ),
}

DATASET_ORDER: tuple[str, ...] = ("ocelot", "midog", "monkey")


def dataset_spec(dataset: str) -> DatasetSpec:
    """Look up a :class:`DatasetSpec` by name (fail-fast with the known datasets)."""
    try:
        return DATASETS[dataset]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown detection dataset {dataset!r}; known: {known}.") from None


# --- replicate abstraction -----------------------------------------------------------


def replicate_plan(
    num_folds: int, *, seeds: Sequence[int] = DEFAULT_SEEDS
) -> tuple[str, list[int]]:
    """Resolve a dataset's replicate axis from its ``splits.csv`` structure.

    A dataset's curated ``splits.csv`` declares its structure: if it ships **>1 fold** the
    replicates *are* the folds (1 seed × F, ``replicate_axis="folds"``); if it ships a
    **single fold** the replicates are seeds (K, ``replicate_axis="seeds"``). The harness
    aggregates mean±std over whichever axis a dataset uses and never needs to know which —
    ``campaign.py``'s seed loop becomes this replicate loop.

    Returns ``(axis, replicate_ids)`` where ``replicate_ids`` are the fold indices
    ``0..F-1`` for the folds axis or the seed values for the seeds axis.
    """
    if num_folds < 1:
        raise ValueError(f"num_folds must be >= 1, got {num_folds}.")
    if num_folds > 1:
        return "folds", list(range(num_folds))
    if not seeds:
        raise ValueError("a single-fold dataset needs at least one seed replicate.")
    return "seeds", list(seeds)


# --- per-sample prediction cache -----------------------------------------------------


@dataclass
class SamplePrediction:
    """One image's persisted detection prediction (the hook for pure re-aggregation).

    Stores the predicted points (``pred_xy`` with ``pred_score`` / ``pred_class``) and the
    ground-truth points (``gt_xy`` / ``gt_class``) in the sample's point frame, plus a
    per-predicted-point ``matched`` flag (True = matched a GT under the native matcher = a
    true positive; False = a false positive). ``area_mm2`` is the evaluated physical area
    (MONKEY's FROC needs it; ``None`` elsewhere).

    Persisting the raw points (not just aggregate counts) is the **hard requirement** that
    makes the deferred MIDOG robustness stratification (#248) and the ``stability`` bootstrap
    pure later re-aggregations: re-score any subset of samples off this cache, no retrain, no
    re-extract. The ``matched`` flags let a stratified re-count skip a full rematch when the
    stratum is a subset of a sample's points.
    """

    sample_id: str
    pred_xy: list[list[float]]
    pred_score: list[float]
    pred_class: list[int]
    gt_xy: list[list[float]]
    gt_class: list[int]
    matched: list[bool] = field(default_factory=list)
    area_mm2: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pred_xy": [[float(x), float(y)] for x, y in self.pred_xy],
            "pred_score": [float(s) for s in self.pred_score],
            "pred_class": [int(c) for c in self.pred_class],
            "gt_xy": [[float(x), float(y)] for x, y in self.gt_xy],
            "gt_class": [int(c) for c in self.gt_class],
            "matched": [bool(m) for m in self.matched],
            "area_mm2": None if self.area_mm2 is None else float(self.area_mm2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplePrediction":
        return cls(
            sample_id=str(data["sample_id"]),
            pred_xy=[[float(x), float(y)] for x, y in data.get("pred_xy", [])],
            pred_score=[float(s) for s in data.get("pred_score", [])],
            pred_class=[int(c) for c in data.get("pred_class", [])],
            gt_xy=[[float(x), float(y)] for x, y in data.get("gt_xy", [])],
            gt_class=[int(c) for c in data.get("gt_class", [])],
            matched=[bool(m) for m in data.get("matched", [])],
            area_mm2=(None if data.get("area_mm2") is None else float(data["area_mm2"])),
        )


@dataclass
class CellPredictions:
    """The persisted per-sample predictions for one ``(encoder, dataset, replicate)`` cell."""

    encoder: str
    dataset: str
    replicate: int
    metric_name: str
    spacing_um: float
    samples: list[SamplePrediction] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "dataset": self.dataset,
            "replicate": self.replicate,
            "metric_name": self.metric_name,
            "spacing_um": float(self.spacing_um),
            "samples": [s.as_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellPredictions":
        return cls(
            encoder=str(data["encoder"]),
            dataset=str(data["dataset"]),
            replicate=int(data["replicate"]),
            metric_name=str(data["metric_name"]),
            spacing_um=float(data["spacing_um"]),
            samples=[SamplePrediction.from_dict(s) for s in data.get("samples", [])],
        )


def write_cell_predictions(path: str | Path, predictions: CellPredictions) -> Path:
    """Persist a cell's per-sample predictions as JSON (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions.as_dict(), indent=2), encoding="utf-8")
    return path


def read_cell_predictions(path: str | Path) -> CellPredictions:
    """Load a cell's persisted per-sample predictions from JSON."""
    return CellPredictions.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- native-metric dispatch (point-level, off the persisted cache) -------------------


def _per_image_arrays(
    samples: Sequence[SamplePrediction],
) -> tuple[list, list, list, list, list, list]:
    """Explode a list of :class:`SamplePrediction` into the per-image array lists scorers want."""
    pred_xy, pred_score, pred_class = [], [], []
    gt_xy, gt_class, area = [], [], []
    for s in samples:
        pred_xy.append(np.asarray(s.pred_xy, dtype=np.float64).reshape(-1, 2))
        pred_score.append(np.asarray(s.pred_score, dtype=np.float64).reshape(-1))
        pred_class.append(np.asarray(s.pred_class, dtype=np.int64).reshape(-1))
        gt_xy.append(np.asarray(s.gt_xy, dtype=np.float64).reshape(-1, 2))
        gt_class.append(np.asarray(s.gt_class, dtype=np.int64).reshape(-1))
        area.append(0.0 if s.area_mm2 is None else float(s.area_mm2))
    return pred_xy, pred_score, pred_class, gt_xy, gt_class, area


def _score_ocelot_points(samples: Sequence[SamplePrediction], spec: DatasetSpec) -> dict[str, float]:
    """OCELOT class-aware greedy mF1 over persisted points (the leaderboard-comparable metric).

    Reuses the exact TP/FP/FN matcher + F1 reduction the live OCELOT path uses
    (:func:`~soma.detection.matching.detection_counts` / :func:`reduce_f1`), so the off-cache
    re-aggregation reproduces the live headline when fed the same thresholded points.
    """
    import torch

    pred_xy, pred_score, pred_class, gt_xy, gt_class, _ = _per_image_arrays(samples)
    delta = float(spec.match_um[0]) / float(spec.spacing_um)
    rows = [
        detection_counts(
            pxy, pcls, ps, gxy, gcls,
            num_classes=spec.num_classes, delta=delta, method=spec.match_method,
        )
        for pxy, ps, pcls, gxy, gcls in zip(
            pred_xy, pred_score, pred_class, gt_xy, gt_class
        )
    ]
    counts = torch.cat(rows, dim=0) if rows else torch.zeros((0, spec.num_classes, 3), dtype=torch.long)
    metrics = reduce_f1(counts, num_classes=spec.num_classes, aggregation="dataset_global")
    return {"mean_f1": float(metrics["mean_f1"])}


def _score_midog_points(samples: Sequence[SamplePrediction], spec: DatasetSpec) -> dict[str, float]:
    """MIDOG-native single-class F1 over persisted points (Hungarian one-to-one within 7.5 µm)."""
    pred_xy, pred_score, _pred_class, gt_xy, _gt_class, _ = _per_image_arrays(samples)
    delta = float(spec.match_um[0]) / float(spec.spacing_um)
    score = midog_f1(pred_xy, pred_score, gt_xy, delta=delta, method=spec.match_method)
    return {"f1": float(score.f1)}


def _score_monkey_points(samples: Sequence[SamplePrediction], spec: DatasetSpec) -> dict[str, float]:
    """MONKEY-native mean FROC over persisted points (per-class 4/5 µm margins, per-mm² sweep)."""
    pred_xy, pred_score, pred_class, gt_xy, gt_class, area = _per_image_arrays(samples)
    out = score_monkey_froc(
        pred_xy, pred_class, pred_score, gt_xy, gt_class, area,
        spacing_um=float(spec.spacing_um), match_um=spec.match_um,
    )
    return {"mean_froc": float(out["mean_froc"])}


_POINT_SCORERS = {
    "ocelot": _score_ocelot_points,
    "midog": _score_midog_points,
    "monkey": _score_monkey_points,
}


def score_dataset_points(
    dataset: str, samples: Sequence[SamplePrediction], *, spec: DatasetSpec | None = None
) -> dict[str, float]:
    """Dispatch a dataset's native scorer over a set of persisted per-sample predictions.

    The single entry point the stability bootstrap and the deferred robustness
    stratification call: it re-scores any subset of ``samples`` (a bootstrap resample, a
    domain stratum) with the dataset's own metric — pure, off the cache, no training.
    """
    spec = spec or dataset_spec(dataset)
    return _POINT_SCORERS[dataset](samples, spec)


# --- cell aggregation ----------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One ``(encoder, dataset)`` ranking cell: its native metric aggregated over replicates.

    Carries the unpooled native ``metric_name`` and the per-replicate test values plus their
    mean±std, and — for the tune/test rank-agreement signal — the tune-split values too. No
    cross-dataset pooling: this is the only place a dataset's metric lives.
    """

    encoder: str
    dataset: str
    metric_name: str
    per_replicate: tuple[float, ...]
    mean: float
    std: float
    n_replicates: int
    replicate_axis: str
    test_source: str
    tune_per_replicate: tuple[float, ...] = ()
    tune_mean: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "encoder": self.encoder,
            "dataset": self.dataset,
            "metric_name": self.metric_name,
            "per_replicate": list(self.per_replicate),
            "mean": self.mean,
            "std": self.std,
            "n_replicates": self.n_replicates,
            "replicate_axis": self.replicate_axis,
            "test_source": self.test_source,
        }
        if self.tune_per_replicate:
            out["tune_per_replicate"] = list(self.tune_per_replicate)
            out["tune_mean"] = self.tune_mean
        return out


def aggregate_cell(
    encoder: str,
    dataset: str,
    per_replicate: Sequence[float],
    *,
    replicate_axis: str,
    metric_name: str | None = None,
    test_source: str | None = None,
    tune_per_replicate: Sequence[float] | None = None,
) -> Cell:
    """Aggregate a cell's per-replicate test metrics into mean±std over the replicate axis.

    ``metric_name`` / ``test_source`` default to the dataset's spec. ``tune_per_replicate``
    (optional) records the frozen-on-tune values whose ranks are reported alongside test.
    """
    spec = dataset_spec(dataset)
    values = [float(v) for v in per_replicate]
    mean = float(np.mean(values))
    std = float(np.std(values)) if len(values) > 1 else 0.0
    tune_values = tuple(float(v) for v in (tune_per_replicate or ()))
    tune_mean = float(np.mean(tune_values)) if tune_values else None
    return Cell(
        encoder=encoder,
        dataset=dataset,
        metric_name=metric_name or spec.metric_name,
        per_replicate=tuple(values),
        mean=mean,
        std=std,
        n_replicates=len(values),
        replicate_axis=replicate_axis,
        test_source=test_source or spec.test_source,
        tune_per_replicate=tune_values,
        tune_mean=tune_mean,
    )


# --- ranking + rank consistency ------------------------------------------------------


def rank_encoders(encoder_to_score: dict[str, float]) -> list[dict[str, Any]]:
    """Rank encoders by score, best first (all three native metrics are higher-is-better).

    Returns ``[{"encoder", "rank", "score"}]`` sorted best-to-worst. Ranks are integer
    competition ranks (ties share the lower rank, next rank skips), so a downstream reader
    consumes ``rank`` ints directly.
    """
    if not encoder_to_score:
        return []
    from scipy.stats import rankdata

    encoders = list(encoder_to_score)
    scores = np.asarray([encoder_to_score[e] for e in encoders], dtype=np.float64)
    # Competition ("min") ranking on the negated score => higher score gets rank 1.
    ranks = rankdata(-scores, method="min").astype(int)
    order = sorted(range(len(encoders)), key=lambda i: (ranks[i], encoders[i]))
    return [
        {"encoder": encoders[i], "rank": int(ranks[i]), "score": float(scores[i])}
        for i in order
    ]


def _rank_lookup(ranking: Sequence[dict[str, Any]]) -> dict[str, int]:
    """``{encoder -> rank}`` from a :func:`rank_encoders` list."""
    return {row["encoder"]: int(row["rank"]) for row in ranking}


def aggregate_rank(
    per_dataset_ranks: dict[str, dict[str, int]], roster: Sequence[str]
) -> dict[str, float]:
    """Mean rank of each encoder across the datasets it was ranked on (lower is better).

    An encoder missing from a dataset (not yet extracted there) is simply averaged over the
    datasets where it *is* present, so the aggregate is roster-size- and coverage-agnostic.
    """
    out: dict[str, float] = {}
    for encoder in roster:
        ranks = [
            float(ranks[encoder])
            for ranks in per_dataset_ranks.values()
            if encoder in ranks
        ]
        if ranks:
            out[encoder] = float(np.mean(ranks))
    return out


def rank_consistency(
    per_dataset_ranks: dict[str, dict[str, int]], roster: Sequence[str]
) -> dict[str, Any]:
    """Spearman + Kendall agreement of the per-dataset encoder rankings (the only cross-dataset object).

    For each pair of datasets it correlates the two rank vectors over the encoders both
    ranked, then reports the mean Spearman/Kendall across pairs plus the full per-pair
    matrix. ``n_encoders`` is carried so a reader can judge the correlation against the
    roster size (a rank correlation over 6 encoders is noisy — cite it).
    """
    from scipy.stats import kendalltau, spearmanr

    datasets = [d for d in DATASET_ORDER if d in per_dataset_ranks]
    datasets += [d for d in per_dataset_ranks if d not in datasets]
    pairs: dict[str, dict[str, float | None]] = {}
    spearmans: list[float] = []
    kendalls: list[float] = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a, b = datasets[i], datasets[j]
            shared = [e for e in roster if e in per_dataset_ranks[a] and e in per_dataset_ranks[b]]
            key = f"{a}|{b}"
            if len(shared) < 2:
                pairs[key] = {"spearman": None, "kendall": None, "n": len(shared)}
                continue
            ra = [per_dataset_ranks[a][e] for e in shared]
            rb = [per_dataset_ranks[b][e] for e in shared]
            rho = spearmanr(ra, rb).statistic
            tau = kendalltau(ra, rb).statistic
            rho = None if rho is None or np.isnan(rho) else float(rho)
            tau = None if tau is None or np.isnan(tau) else float(tau)
            pairs[key] = {"spearman": rho, "kendall": tau, "n": len(shared)}
            if rho is not None:
                spearmans.append(rho)
            if tau is not None:
                kendalls.append(tau)
    return {
        "n_encoders": len(roster),
        "spearman": float(np.mean(spearmans)) if spearmans else None,
        "kendall": float(np.mean(kendalls)) if kendalls else None,
        "pairs": pairs,
    }


# --- bootstrap stability -------------------------------------------------------------


def bootstrap_rank_stability(
    dataset: str,
    encoder_samples: dict[str, list[SamplePrediction]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    spec: DatasetSpec | None = None,
) -> dict[str, dict[str, Any]]:
    """Bootstrap the encoder ranking over test patients → per-encoder rank CIs (post-hoc, no GPU).

    A **paired** bootstrap: every encoder in ``encoder_samples`` shares the same test
    sample-id universe (the same test patients); each iteration resamples those ids with
    replacement, re-scores every encoder on *that* resample via the dataset's native scorer
    (pure, off the persisted cache), ranks the encoders, and records each encoder's rank.
    Over ``n_boot`` iterations this yields a rank distribution → median rank + a
    percentile CI per encoder. This is exactly the re-aggregation the per-sample cache
    exists to make free.
    """
    spec = spec or dataset_spec(dataset)
    encoders = list(encoder_samples)
    if not encoders:
        return {}
    # Shared sample-id universe (ordered by the first encoder; all encoders must cover it).
    sample_ids = [s.sample_id for s in encoder_samples[encoders[0]]]
    by_encoder = {
        enc: {s.sample_id: s for s in samples} for enc, samples in encoder_samples.items()
    }
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    rank_draws: dict[str, list[int]] = {enc: [] for enc in encoders}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        picked = [sample_ids[k] for k in idx]
        scores = np.asarray(
            [
                score_dataset_points(
                    dataset, [by_encoder[enc][sid] for sid in picked], spec=spec
                )[spec.metric_name]
                for enc in encoders
            ],
            dtype=np.float64,
        )
        ranks = rankdata(-scores, method="min").astype(int)
        for enc, r in zip(encoders, ranks):
            rank_draws[enc].append(int(r))
    lo, hi = ci
    out: dict[str, dict[str, Any]] = {}
    for enc in encoders:
        draws = np.asarray(rank_draws[enc], dtype=np.float64)
        out[enc] = {
            "rank_mean": float(draws.mean()),
            "rank_median": float(np.median(draws)),
            "rank_ci": [float(np.percentile(draws, lo)), float(np.percentile(draws, hi))],
            "n_boot": int(n_boot),
        }
    return out


# --- frozen subset selections --------------------------------------------------------


def _ordered_union(order: Sequence[str], *groups: Sequence[str]) -> list[str]:
    """Union of ``groups`` in the priority of ``order`` (aggregate-rank order), deduped."""
    want = {name for group in groups for name in group}
    return [name for name in order if name in want]


def select_subsets(
    per_dataset_ranks: dict[str, dict[str, int]],
    roster: Sequence[RosterEntry],
    *,
    backbone_top_k: int = 3,
    efficiency_top_n: int = 2,
) -> dict[str, list[str]]:
    """Freeze the downstream subset selections from the ranking (the RULE, not real numbers).

    Computed here so the backbone (#235) and efficiency (#237) studies **read a field** and
    never re-derive "who's top-k" (no drift). The rule, over the aggregate rank (mean rank
    across datasets, lower = better):

    * ``backbone_subset`` = the top-``k`` encoders ∪ the compact encoders — #235 sweeps
      these (the strong backbones plus the cheap reference point).
    * ``efficiency_subset`` = the top-``n`` ∪ the mid-band (the central third by aggregate
      rank) ∪ the compact encoders ∪ the control — #237's annotation-efficiency axis.

    Both are returned in aggregate-rank order (best first) and deduped, and recompute for
    any roster size (the mid-band is a fraction, not fixed indices).
    """
    names = roster_names(roster)
    compact = [e.name for e in roster if e.is_compact]
    control = [e.name for e in roster if e.is_control]

    agg = aggregate_rank(per_dataset_ranks, names)
    ranked = sorted(agg, key=lambda e: (agg[e], e))  # best (lowest mean rank) first
    n = len(ranked)

    top_k = ranked[: max(0, backbone_top_k)]
    top_n = ranked[: max(0, efficiency_top_n)]
    # Central third of the aggregate ordering (roster-size-agnostic mid band).
    mid_lo = n // 3
    mid_hi = n - n // 3
    mid = ranked[mid_lo:mid_hi]

    return {
        "backbone_subset": _ordered_union(ranked, top_k, compact),
        "efficiency_subset": _ordered_union(ranked, top_n, mid, compact, control),
    }


# --- reference bands -----------------------------------------------------------------


def _reference_row_dict(row: ReferenceRow) -> dict[str, Any]:
    return {
        "metric": row.metric,
        "expected": row.expected,
        "tolerance": row.tolerance,
        "kind": row.kind,
        "label": row.label,
        "url": row.url,
        "source": row.source,
    }


def reference_bands(datasets: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """The per-dataset reference rows (gate band + non-gating guidance anchors) for the report.

    Each dataset's ``reference/<dataset>.csv`` ships the scaffold (a placeholder gate band +
    the published-winner external anchor, "different test set"); the human fills the numbers
    at run time. Rendered alongside the ranking, never as a gate on it.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        try:
            rows = load_reference(dataset)
        except (FileNotFoundError, ModuleNotFoundError):
            rows = []
        out[dataset] = [_reference_row_dict(r) for r in rows]
    return out


# --- the ranking report --------------------------------------------------------------


def build_ranking_report(
    cells: Sequence[Cell],
    *,
    roster: Sequence[RosterEntry] = DEFAULT_ROSTER,
    stability_samples: dict[str, dict[str, list[SamplePrediction]]] | None = None,
    git_sha: str | None = None,
    replicate_policy: dict[str, Any] | None = None,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Assemble the full ``ranking_report.json`` (issue #246 schema) from the sweep's cells.

    ``cells`` are the aggregated ``(encoder, dataset)`` cells; ``stability_samples`` (dataset
    -> encoder -> shared test predictions) drives the bootstrap when supplied. The report is
    pure over these inputs — no GPU, no training — so a killed sweep re-aggregates from the
    per-cell metric/prediction cache on re-invocation.

    Schema: ``config`` · unpooled ``cells`` (each carries its own ``metric_name``) ·
    ``ranking`` (``per_dataset`` test **and** tune ranks + ``rank_consistency`` +
    bootstrap ``stability``) · ``robustness: {}`` (deferred to #248) · ``reference_bands`` ·
    frozen ``selections``.
    """
    names = roster_names(roster)
    datasets = [d for d in DATASET_ORDER if any(c.dataset == d for c in cells)]
    datasets += sorted({c.dataset for c in cells} - set(datasets))

    # --- per-dataset ranks (test + tune) --------------------------------------------
    per_dataset: dict[str, dict[str, Any]] = {}
    per_dataset_test_ranks: dict[str, dict[str, int]] = {}
    per_dataset_spacing: dict[str, float] = {}
    for dataset in datasets:
        spec = dataset_spec(dataset)
        per_dataset_spacing[dataset] = spec.spacing_um
        ds_cells = [c for c in cells if c.dataset == dataset]
        test_scores = {c.encoder: c.mean for c in ds_cells}
        test_ranking = rank_encoders(test_scores)
        per_dataset_test_ranks[dataset] = _rank_lookup(test_ranking)
        entry: dict[str, Any] = {"test": test_ranking}
        tune_scores = {c.encoder: c.tune_mean for c in ds_cells if c.tune_mean is not None}
        if tune_scores:
            entry["tune"] = rank_encoders(tune_scores)
        per_dataset[dataset] = entry

    # --- rank consistency + bootstrap stability -------------------------------------
    consistency = rank_consistency(per_dataset_test_ranks, names)
    stability: dict[str, Any] = {}
    if stability_samples:
        for dataset, enc_samples in stability_samples.items():
            stability[dataset] = bootstrap_rank_stability(
                dataset, enc_samples, n_boot=n_boot, seed=bootstrap_seed
            )

    # --- frozen selections ----------------------------------------------------------
    selections = select_subsets(per_dataset_test_ranks, roster)

    return {
        "config": {
            "encoder_list": names,
            "per_dataset_spacing": per_dataset_spacing,
            "replicate_policy": replicate_policy or {"single_fold_seeds": list(DEFAULT_SEEDS)},
            "decoder": DECODER,
            "git_sha": git_sha,
        },
        "cells": [c.as_dict() for c in cells],
        "ranking": {
            "per_dataset": per_dataset,
            "rank_consistency": consistency,
            "stability": stability,
        },
        "robustness": {},  # DEFERRED to #248 (leave the key, populate later).
        "reference_bands": reference_bands(datasets),
        "selections": selections,
    }


# --- build_config helpers ------------------------------------------------------------


def config_path_for(dataset: str) -> Path:
    """The committed base config for ``dataset`` (one per-dataset recipe, encoder varies)."""
    return _CONFIG_DIR / dataset_spec(dataset).config_file


# --- benchmark object ----------------------------------------------------------------

FACET = Facet(
    fixed={"task": "detection", "decoder": DECODER, "protocol": "frozen-probe-full-ranking"},
    varied=("encoder", "dataset"),
)

REFERENCE_ENVIRONMENT: dict[str, str] = {}


class DetectionBenchmark:
    """The multi-dataset encoder-ranking detection benchmark (protocol-as-code, issue #246).

    Unlike a single-dataset benchmark, this ranks a roster across OCELOT / MIDOG / MONKEY and
    emits a ``ranking_report.json`` (no single pooled scalar). ``build_config`` resolves an
    ``(encoder, dataset)`` cell to its committed per-dataset config with the encoder swapped
    in; ``score`` re-aggregates a run's native metric off its persisted per-sample cache.
    """

    name = BENCHMARK_NAME
    facet = FACET
    canonical_seeds = DEFAULT_SEEDS
    primary_metric = "mean_f1"  # nominal; each dataset records its own native metric_name.
    reference_environment = REFERENCE_ENVIRONMENT
    roster = DEFAULT_ROSTER
    datasets = DATASET_ORDER

    def metric_for(self, dataset: str) -> str:
        """The unpooled native metric name for ``dataset``."""
        return dataset_spec(dataset).metric_name

    def curate(
        self, raw_root: str | Path, out_dir: str | Path, *, dataset: str
    ) -> CuratedManifest:
        """Curate one of the three raw datasets into a soma detection Manifest (delegates)."""
        if dataset == "ocelot":
            from soma.curation.ocelot import curate_ocelot_detection

            return curate_ocelot_detection(raw_root, out_dir)
        if dataset == "midog":
            from soma.curation.midog import curate_midog_detection

            return curate_midog_detection(raw_root, out_dir)
        if dataset == "monkey":
            from soma.curation.monkey import curate_monkey_detection

            return curate_monkey_detection(raw_root, out_dir)
        raise KeyError(f"unknown detection dataset {dataset!r}; known: {sorted(DATASETS)}.")

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ROSTER[0].name,
        dataset: str = DATASET_ORDER[0],
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineConfig:
        """Resolve a committed per-dataset config with ``encoder`` swapped in (roster-agnostic).

        The recipe (native spacing, ``lightweight_conv`` decoder, native task params) is
        fixed in the committed ``configs/detection/<dataset>.yaml``; only ``encoder.name``
        varies across the roster, so any roster size reuses one config per dataset. Data /
        output paths and seed are repointed exactly like the OCELOT benchmark.
        """
        config_path = config_path_for(dataset)
        merged: dict[str, Any] = {"encoder": {"name": encoder}}
        data_over: dict[str, Any] = {}
        if dataset_csv is not None:
            data_over["dataset_csv"] = str(dataset_csv)
        if splits_csv is not None:
            data_over["splits_csv"] = str(splits_csv)
        if data_over:
            merged["data"] = data_over
        run_over: dict[str, Any] = {}
        if output_root is not None:
            run_over["output_root"] = str(output_root)
        if seed is not None:
            run_over["seed"] = int(seed)
        if run_over:
            merged["run"] = run_over
        if overrides:
            for section, values in overrides.items():
                merged.setdefault(section, {}).update(values)
        return load_config(config_path, overrides=merged)

    def expected(self, *, dataset: str | None = None, **axes: Any) -> list[ReferenceRow]:
        """Reference rows for one dataset's scaffold matching ``axes`` (banner matches any).

        This benchmark keeps a reference scaffold *per dataset*, so ``dataset`` selects which
        ``reference/<dataset>.csv`` to read; a generic no-``dataset`` call (e.g. a leaderboard
        probing for a broad banner) returns ``[]`` rather than raising, since there is no
        single cross-dataset band.
        """
        if dataset is None:
            return []
        rows = load_reference(dataset)
        return [r for r in rows if r.matches(axes)]

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """Re-aggregate a cell's native metric off its persisted per-sample prediction cache.

        Reads ``predictions.json`` under ``run_dir`` (written by the driver at score time)
        and re-runs the dataset's native scorer — the pure, no-training path that the
        stability bootstrap and the deferred robustness stratification also travel.
        """
        run_dir = Path(run_dir)
        direct = run_dir / "predictions.json"
        if direct.is_file():
            pred_path = direct
        else:
            candidates = sorted(run_dir.glob("**/predictions.json"))
            if not candidates:
                raise FileNotFoundError(f"no predictions.json under {run_dir}")
            pred_path = candidates[-1]
        predictions = read_cell_predictions(pred_path)
        return score_dataset_points(predictions.dataset, predictions.samples)


DETECTION_BENCHMARK = DetectionBenchmark()
register_benchmark(DETECTION_BENCHMARK)
