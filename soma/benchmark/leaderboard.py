"""The Leaderboard: a rendered *faceted view* over self-describing run dirs (ADR 0003).

A Leaderboard is **not** a stored file. It is a pure projection over the run dirs a
sweep already wrote: each run stamps its own ``config.yaml`` + ``summary.json`` +
``run.yaml``, and the experiment dir above it stamps ``experiment.json`` (the canonical
config fingerprint + dataset/splits checksums). :func:`project_leaderboard` scans those
artifacts and renders a table — no pipeline execution, no shared mutable index.

A **facet** fixes a ``(dataset + splits + task)`` triple, holds a set of config axes
fixed, and surfaces one (or more) *varied* axes as comparison columns. Single-metric
tables rank by one metric; multi-metric Benchmark tables use neutral row order and expose
independent per-metric ranks plus Pareto membership. Because a run's identity
(``experiment_id``) fingerprints the *entire* config modulo seed, two runs that differ in
any **unfixed** axis are never pooled. Seed-runs of the *same* config collapse to mean ±
std + n.

Reference rows (a registered :class:`~soma.benchmark.registry.Benchmark`'s
``expected()``) inject two ways: a broad, config-agnostic scalar renders as a **threshold
banner** above the table; keyed rows **join** to the measured rows on the varied axis with
a per-row tolerance PASS/FAIL.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from soma.evaluation.metrics import metric_registry
from soma.evaluation.summary import sample_std

if TYPE_CHECKING:
    from soma.benchmark.registry import Benchmark

# A leaderboard scopes to exactly one of these triples.
TripleKey = tuple[str, str, str]  # (dataset_checksum, splits_checksum, task)

# Sentinel for "this axis does not resolve against a run's canonical spec".
_MISSING = object()

# Friendly axis names -> dotted path into a run's canonical_spec. Any other axis name is
# treated as a raw dotted path, so arbitrary config fields can be fixed/varied too.
_AXIS_ALIASES: dict[str, str] = {
    "encoder": "features.encoder.name",
    "aggregator": "learner.params.aggregator.name",
    "decoder": "learner.params.decoder.name",
    "task": "learner.params.head.name",
    "spacing": "features.extraction.tiling.requested_spacing_um",
    "feature_mode": "features.kind",
}

# Summary keys that are provenance bookkeeping, not a rankable metric (mirror the run
# panel's primary-metric detection in soma/pipeline/console.py).
_BOOKKEEPING_KEYS = {
    "coverage",
    "num_samples",
    "num_real_samples",
    "num_placeholder_samples",
}

# Flat config-diff keys suppressed from a row's annotation (machine-local noise that is
# shared-by-checksum within a triple anyway).
_DIFF_SUPPRESS = {"dataset.path", "splits.path", "dataset.checksum", "splits.checksum"}


# --- axis resolution -----------------------------------------------------------------


def axis_value(spec: dict[str, Any], axis: str) -> Any:
    """Resolve ``axis`` against a run's ``canonical_spec`` (alias or raw dotted path).

    Returns :data:`_MISSING` when the path does not exist (e.g. a benchmark facet fixes a
    non-config axis like ``matcher``). Callers treat :data:`_MISSING` as "cannot constrain / cannot vary here".
    """
    path = _AXIS_ALIASES.get(axis, axis)
    cursor: Any = spec
    for part in path.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            return _MISSING
    return cursor


def _axis_flat_key(axis: str) -> str:
    """The flattened canonical-spec key for ``axis`` (used to strip it from config diffs)."""
    return _AXIS_ALIASES.get(axis, axis)


# --- run records ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """One persisted run attempt, loaded from its self-describing dir artifacts."""

    run_dir: Path
    run_id: str
    experiment_id: str
    seed: int | None
    status: str | None
    finished_at: str | None
    canonical_spec: dict[str, Any]
    dataset_checksum: str
    splits_checksum: str
    task: str
    summary: dict[str, float]

    @property
    def triple(self) -> TripleKey:
        return (self.dataset_checksum, self.splits_checksum, self.task)


def _locate_experiment_json(run_dir: Path) -> Path | None:
    """The ``experiment.json`` for ``run_dir`` (standard layout: two levels up)."""
    direct = run_dir.parent.parent / "experiment.json"
    if direct.is_file():
        return direct
    for parent in run_dir.parents:
        candidate = parent / "experiment.json"
        if candidate.is_file():
            return candidate
    return None


def load_run_record(run_dir: str | Path) -> RunRecord | None:
    """Load a :class:`RunRecord`, including incomplete attempts when identity exists."""
    run_dir = Path(run_dir)
    run_yaml = run_dir / "run.yaml"
    meta: dict[str, Any] = {}
    if run_yaml.is_file():
        meta = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = meta.get("summary_metrics") or {}
    summary = {str(k): float(v) for k, v in summary.items()}
    exp_json = _locate_experiment_json(run_dir)
    if exp_json is None:
        return None
    experiment = json.loads(exp_json.read_text(encoding="utf-8"))
    spec = experiment.get("canonical_spec") or {}
    task = axis_value(spec, "task")
    task = "" if task is _MISSING or task is None else str(task)

    seed = meta.get("seed")
    return RunRecord(
        run_dir=run_dir,
        run_id=str(meta.get("run_id") or run_dir.name),
        experiment_id=str(meta.get("experiment_id") or experiment.get("experiment_id") or ""),
        seed=int(seed) if seed not in (None, "") else None,
        status=meta.get("status"),
        finished_at=meta.get("finished_at") or None,
        canonical_spec=spec,
        dataset_checksum=str(experiment.get("dataset_checksum") or ""),
        splits_checksum=str(experiment.get("splits_checksum") or ""),
        task=task,
        summary=summary,
    )


def discover_triples(
    output_root: str | Path, *, include_incomplete: bool = False
) -> dict[TripleKey, list[Path]]:
    """Scan run attempts and group them by their data/task identity triple.

    By default only completed, non-empty attempts are returned. Multi-metric Benchmark
    projection opts into incomplete attempts so missing logical samples can fail loudly.
    """
    root = Path(output_root)
    result: dict[TripleKey, list[Path]] = {}
    for exp_json in sorted(root.glob("**/experiments/*/experiment.json")):
        runs_dir = exp_json.parent / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            record = load_run_record(run_dir)
            if record is None:
                continue
            if not include_incomplete and (
                record.status != "completed" or not record.summary
            ):
                continue
            result.setdefault(record.triple, []).append(run_dir)
    return result


# --- metric selection ----------------------------------------------------------------


def select_ranking_split(summary: dict[str, float]) -> str:
    """The split a leaderboard ranks on: prefer ``test``, else first non-``tune`` split."""
    splits = sorted({k.split("/", 1)[0] for k in summary if "/" in k})
    if "test" in splits:
        return "test"
    non_tune = [s for s in splits if s != "tune"]
    if non_tune:
        return non_tune[0]
    return splits[0] if splits else "test"


def run_primary_metric(summary: dict[str, float], split: str) -> str | None:
    """The run's stamped primary metric on ``split`` (first non-bookkeeping metric).

    Mirrors the run-summary panel in ``soma/pipeline/console.py``: the first metric key under the
    split prefix that is not a ``_std`` companion nor a provenance bookkeeping field.
    """
    prefix = f"{split}/"
    for key in summary:
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name.endswith("_std"):
            continue
        if name.endswith("_mean"):
            name = name[: -len("_mean")]
        if name in _BOOKKEEPING_KEYS:
            continue
        return name
    return None


def _metric_value(summary: dict[str, float], metric: str, split: str) -> float | None:
    """The value of ``metric`` on ``split`` (single-fold or ``_mean`` seed-aggregate).

    ``metric`` is normally a bare name the split is prepended to (``mean_f1`` ->
    ``test/mean_f1``). A benchmark whose ``primary_metric`` already carries a split prefix —
    HEST's multi-fold headline ``test/mean_pearson_mean`` — is looked up as a full key first
    so it isn't double-prefixed into ``test/test/...``.
    """
    candidates = [metric, f"{metric}_mean"] if "/" in metric else []
    candidates += [f"{split}/{metric}", f"{split}/{metric}_mean"]
    for key in candidates:
        if key in summary:
            return float(summary[key])
    return None


# --- facet + table -------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderboardFacet:
    """Axes held fixed and axes varied (surfaced as comparison columns / sort key)."""

    vary: tuple[str, ...] = ()
    fixed: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceBanner:
    """A broad, config-agnostic reference scalar the whole table is read against."""

    metric: str
    expected: float
    tolerance: float
    source: str


@dataclass(frozen=True)
class GuidanceAnchor:
    """A non-gating external/official reference point rendered as context (issue #226).

    An external anchor (an official-challenge baseline, a best-reported SOTA number, …)
    measures a *different* protocol than the one soma runs, so it never gates a run's
    PASS/FAIL — it is shown alongside the gate band purely as guidance ("what's
    achievable"). ``label`` names it, ``url`` links the snapshotted source, and ``source``
    keeps the protocol note + capture date.
    """

    label: str
    metric: str
    expected: float
    url: str
    source: str


@dataclass(frozen=True)
class LeaderboardMetricDescription:
    metric: str
    higher_is_better: bool


@dataclass(frozen=True)
class LeaderboardReference:
    expected: float
    delta: float
    tolerance: float | None = None
    within_tolerance: bool | None = None
    source: str | None = None
    label: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class LeaderboardMetric:
    metric: str
    rank: int | None
    mean: float
    std: float | None
    n: int
    seeds: tuple[int, ...]
    higher_is_better: bool
    ranking: bool
    reference: LeaderboardReference | None = None


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int | None
    experiment_id: str
    vary_values: dict[str, Any]
    metric: str
    mean: float
    std: float | None
    n: int
    seeds: tuple[int, ...]
    config_diff: dict[str, Any]
    reference_expected: float | None = None
    reference_tolerance: float | None = None
    reference_pass: bool | None = None
    reference_source: str | None = None
    ranking_eligible: bool = True
    metrics: dict[str, LeaderboardMetric] = field(default_factory=dict)
    pareto: bool | None = None


@dataclass(frozen=True)
class LeaderboardTable:
    triple: TripleKey
    metric: str
    higher_is_better: bool
    vary: tuple[str, ...]
    split: str
    rows: list[LeaderboardRow]
    banner: ReferenceBanner | None = None
    guidance: tuple[GuidanceAnchor, ...] = ()
    reported_metrics: tuple[LeaderboardMetricDescription, ...] = ()
    ranking_metrics: tuple[LeaderboardMetricDescription, ...] = ()


def _passes_fixed(spec: dict[str, Any], fixed: dict[str, Any]) -> bool:
    """True if every *resolvable* fixed axis equals its target (unresolvable = skipped)."""
    for axis, target in fixed.items():
        got = axis_value(spec, axis)
        if got is _MISSING:
            continue
        if str(got) != str(target):
            return False
    return True


def project_leaderboard(
    run_dirs: list[str | Path],
    facet: LeaderboardFacet,
    *,
    metric: str | None = None,
    benchmark: "Benchmark | None" = None,
    split: str | None = None,
) -> LeaderboardTable:
    """Project run dirs onto a ranked leaderboard for one ``(dataset, splits, task)`` triple.

    Pure over already-written artifacts. Seed-runs of one config collapse to mean ± std;
    every distinct unfixed config is its own row (never pooled) with its config diff.
    """
    records = [r for r in (load_run_record(d) for d in run_dirs) if r is not None]
    records = [r for r in records if _passes_fixed(r.canonical_spec, facet.fixed)]

    triples = {r.triple for r in records}
    if len(triples) > 1:
        raise ValueError(
            "project_leaderboard requires a single (dataset, splits, task) triple; got "
            f"{len(triples)}: {sorted(triples)}. Scope the run dirs first "
            "(discover_triples / --like)."
        )
    triple: TripleKey = next(iter(triples)) if triples else ("", "", "")

    # Ranking split + primary metric + direction.
    chosen_split = split or (select_ranking_split(records[0].summary) if records else "test")
    if metric is not None:
        metric_name: str | None = metric
    elif benchmark is not None:
        metric_name = benchmark.primary_metric
    elif records:
        metric_name = run_primary_metric(records[0].summary, chosen_split)
    else:
        metric_name = None
    metric_name = metric_name or ""
    higher = metric_registry.higher_is_better(metric_name)
    ranking_enabled = True
    if benchmark is not None:
        from soma.benchmark import get_ranking_metrics

        ranking_enabled = metric_name in get_ranking_metrics(benchmark)

    if benchmark is not None and metric is None:
        from soma.benchmark import get_reported_metrics

        reported = get_reported_metrics(benchmark)
        if len(reported) > 1:
            records = _select_latest_attempts(records, reported, chosen_split)
            return _project_multi_metric(
                records,
                facet,
                benchmark=benchmark,
                triple=triple,
                split=chosen_split,
                reported=reported,
                ranking=get_ranking_metrics(benchmark),
            )

    records = _select_latest_attempts(records, (metric_name,), chosen_split)

    # Collapse seed-runs by experiment_id, preserving first-seen order.
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.experiment_id, []).append(record)

    prelim: list[dict[str, Any]] = []
    for experiment_id, recs in groups.items():
        values = [
            v
            for r in recs
            if (v := _metric_value(r.summary, metric_name, chosen_split)) is not None
        ]
        if not values:
            continue
        spec = recs[0].canonical_spec
        vary_values = {ax: _clean(axis_value(spec, ax)) for ax in facet.vary}
        prelim.append(
            {
                "experiment_id": experiment_id,
                "spec": spec,
                "mean": float(np.mean(values)),
                "std": sample_std(values) if len(values) > 1 else None,
                "n": len(values),
                "seeds": tuple(sorted(r.seed for r in recs if r.seed is not None)),
                "vary_values": vary_values,
                "ranking_eligible": _is_ranking_eligible(
                    benchmark, {**facet.fixed, **vary_values}
                ),
            }
        )

    # Config diff across the shown rows only (never-pool annotation).
    specs = [entry["spec"] for entry in prelim]
    _, per_row_diffs = diff_configs(specs) if specs else ({}, [])
    suppress = _DIFF_SUPPRESS | {_axis_flat_key(ax) for ax in facet.vary}
    for entry, diff in zip(prelim, per_row_diffs):
        entry["config_diff"] = {k: v for k, v in diff.items() if k not in suppress}

    # Rank by the primary metric (direction-aware); deterministic tie-break.
    scalar_ranks = (
        _competition_ranks_for(
            prelim, value=lambda entry: entry["mean"], higher_is_better=higher
        )
        if ranking_enabled
        else {}
    )
    if ranking_enabled:
        prelim.sort(
            key=lambda entry: (
                scalar_ranks.get(entry["experiment_id"], float("inf")),
                [str(value) for value in entry["vary_values"].values()],
                entry["experiment_id"],
            )
        )
    else:
        prelim.sort(
            key=lambda e: (
                [str(v) for v in e["vary_values"].values()],
                e["experiment_id"],
            )
        )

    banner = _reference_banner(benchmark, metric_name)
    guidance = _guidance_anchors(benchmark, metric_name)
    rows: list[LeaderboardRow] = []
    for entry in prelim:
        ref = _keyed_reference(benchmark, metric_name, entry["vary_values"], entry["mean"])
        rows.append(
            LeaderboardRow(
                rank=scalar_ranks.get(entry["experiment_id"]),
                experiment_id=entry["experiment_id"],
                vary_values=entry["vary_values"],
                metric=metric_name,
                mean=entry["mean"],
                std=entry["std"],
                n=entry["n"],
                seeds=entry["seeds"],
                config_diff=entry["config_diff"],
                ranking_eligible=entry["ranking_eligible"],
                **ref,
            )
        )

    return LeaderboardTable(
        triple=triple,
        metric=metric_name,
        higher_is_better=higher,
        vary=tuple(facet.vary),
        split=chosen_split,
        rows=rows,
        banner=banner,
        guidance=guidance,
    )


def _project_multi_metric(
    records: list[RunRecord],
    facet: LeaderboardFacet,
    *,
    benchmark: "Benchmark",
    triple: TripleKey,
    split: str,
    reported: tuple[str, ...],
    ranking: tuple[str, ...],
) -> LeaderboardTable:
    """Build the wide projection used only by explicitly multi-metric Benchmarks."""
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.experiment_id, []).append(record)

    incomplete: dict[str, set[str]] = {}
    for experiment_id, recs in groups.items():
        for record in recs:
            missing = {
                metric_name
                for metric_name in reported
                if record.status != "completed"
                or _metric_value(record.summary, metric_name, split) is None
            }
            incomplete.setdefault(experiment_id, set()).update(missing)
    incomplete = {key: value for key, value in incomplete.items() if value}
    if incomplete:
        details = "; ".join(
            f"{experiment_id}: "
            + ", ".join(metric_name for metric_name in reported if metric_name in missing)
            for experiment_id, missing in sorted(incomplete.items())
        )
        raise ValueError(f"Incomplete Reported metrics: {details}")

    prelim: list[dict[str, Any]] = []
    for experiment_id, recs in groups.items():
        spec = recs[0].canonical_spec
        metrics: dict[str, dict[str, Any]] = {}
        for metric_name in reported:
            observations = [
                (record.seed, value)
                for record in recs
                if (value := _metric_value(record.summary, metric_name, split)) is not None
            ]
            values = [value for _, value in observations]
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": sample_std(values) if len(values) > 1 else None,
                "n": len(values),
                "seeds": tuple(sorted(seed for seed, _ in observations if seed is not None)),
            }
        vary_values = {axis: _clean(axis_value(spec, axis)) for axis in facet.vary}
        eligibility_axes = {**facet.fixed, **vary_values}
        ranking_eligible = _is_ranking_eligible(benchmark, eligibility_axes)
        prelim.append(
            {
                "experiment_id": experiment_id,
                "spec": spec,
                "metrics": metrics,
                "vary_values": vary_values,
                "ranking_eligible": ranking_eligible,
            }
        )

    specs = [entry["spec"] for entry in prelim]
    _, per_row_diffs = diff_configs(specs) if specs else ({}, [])
    suppress = _DIFF_SUPPRESS | {_axis_flat_key(axis) for axis in facet.vary}
    for entry, diff in zip(prelim, per_row_diffs):
        entry["config_diff"] = {key: value for key, value in diff.items() if key not in suppress}

    prelim.sort(
        key=lambda entry: (
            [str(value) for value in entry["vary_values"].values()],
            entry["experiment_id"],
        )
    )
    ranks = {
        metric_name: _competition_ranks(prelim, metric_name)
        for metric_name in ranking
    }
    pareto = _pareto_membership(prelim, ranking)

    rows: list[LeaderboardRow] = []
    for entry in prelim:
        experiment_id = entry["experiment_id"]
        row_metrics: dict[str, LeaderboardMetric] = {}
        for metric_name in reported:
            stats = entry["metrics"][metric_name]
            is_ranking = metric_name in ranking
            reference = _multi_metric_reference(
                benchmark,
                metric_name,
                {**facet.fixed, **entry["vary_values"]},
                stats["mean"],
                primary=metric_name == benchmark.primary_metric,
            )
            row_metrics[metric_name] = LeaderboardMetric(
                metric=metric_name,
                rank=ranks[metric_name].get(experiment_id) if is_ranking else None,
                mean=stats["mean"],
                std=stats["std"],
                n=stats["n"],
                seeds=stats["seeds"],
                higher_is_better=metric_registry.higher_is_better(metric_name),
                ranking=is_ranking,
                reference=reference,
            )
        primary = row_metrics[benchmark.primary_metric]
        rows.append(
            LeaderboardRow(
                rank=None,
                experiment_id=experiment_id,
                vary_values=entry["vary_values"],
                metric=benchmark.primary_metric,
                mean=primary.mean,
                std=primary.std,
                n=primary.n,
                seeds=primary.seeds,
                config_diff=entry["config_diff"],
                ranking_eligible=entry["ranking_eligible"],
                metrics=row_metrics,
                pareto=pareto.get(experiment_id),
            )
        )

    descriptions = tuple(
        LeaderboardMetricDescription(
            metric=metric_name,
            higher_is_better=metric_registry.higher_is_better(metric_name),
        )
        for metric_name in reported
    )
    by_name = {description.metric: description for description in descriptions}
    return LeaderboardTable(
        triple=triple,
        metric=benchmark.primary_metric,
        higher_is_better=metric_registry.higher_is_better(benchmark.primary_metric),
        vary=tuple(facet.vary),
        split=split,
        rows=rows,
        reported_metrics=descriptions,
        ranking_metrics=tuple(by_name[name] for name in ranking),
    )


def _select_latest_attempts(
    records: list[RunRecord], reported: tuple[str, ...], split: str
) -> list[RunRecord]:
    """Select one completed attempt per logical ``(experiment_id, seed)`` sample.

    Completion time is authoritative. A stable run ID (falling back to the normalised run
    path) breaks equal-time ties, so caller discovery order cannot affect the projection.
    Incomplete attempts never supersede a complete attempt. When no attempt is complete,
    the latest attempt remains so multi-metric validation can report the logical sample.
    """
    groups: dict[tuple[str, int | None], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.experiment_id, record.seed), []).append(record)

    selected: list[RunRecord] = []
    for attempts in groups.values():
        complete = [
            record
            for record in attempts
            if record.status == "completed"
            and all(
                _metric_value(record.summary, metric_name, split) is not None
                for metric_name in reported
            )
        ]
        completed = [record for record in attempts if record.status == "completed"]
        candidates = complete or completed or attempts
        selected.append(max(candidates, key=_attempt_recency_key))
    return selected


def _is_ranking_eligible(
    benchmark: "Benchmark | None", axes: dict[str, Any]
) -> bool:
    eligibility = getattr(benchmark, "is_ranking_eligible", None)
    return bool(eligibility(**axes)) if callable(eligibility) else True


def _attempt_recency_key(record: RunRecord) -> tuple[datetime, str, str]:
    try:
        completed_at = datetime.fromisoformat(record.finished_at or "")
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        completed_at = completed_at.astimezone(timezone.utc)
    except ValueError:
        completed_at = datetime.min.replace(tzinfo=timezone.utc)
    return completed_at, record.run_id, str(record.run_dir.resolve())


def _competition_ranks(
    entries: list[dict[str, Any]], metric: str
) -> dict[str, int]:
    return _competition_ranks_for(
        entries,
        value=lambda entry: entry["metrics"][metric]["mean"],
        higher_is_better=metric_registry.higher_is_better(metric),
    )


def _competition_ranks_for(
    entries: list[dict[str, Any]],
    *,
    value: Callable[[dict[str, Any]], float],
    higher_is_better: bool,
) -> dict[str, int]:
    """Return standard competition ranks for eligible entries."""
    eligible = [entry for entry in entries if entry["ranking_eligible"]]
    eligible.sort(
        key=lambda entry: (
            -value(entry) if higher_is_better else value(entry),
            entry["experiment_id"],
        )
    )
    result: dict[str, int] = {}
    previous: float | None = None
    previous_rank = 0
    for position, entry in enumerate(eligible, start=1):
        current = value(entry)
        rank = previous_rank if previous is not None and current == previous else position
        result[entry["experiment_id"]] = rank
        previous = current
        previous_rank = rank
    return result


def _multi_metric_reference(
    benchmark: "Benchmark",
    metric: str,
    axes: dict[str, Any],
    measured: float,
    *,
    primary: bool,
) -> LeaderboardReference | None:
    """Return a metric-local keyed comparison; only a primary gate gets a verdict."""
    candidates = [
        row for row in benchmark.expected(**axes) if row.key and row.metric == metric
    ]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    gates = primary and not row.is_external
    return LeaderboardReference(
        expected=row.expected,
        delta=measured - row.expected,
        tolerance=row.tolerance_band() if gates else None,
        within_tolerance=row.within_tolerance(measured) if gates else None,
        source=row.source,
        label=row.label or None,
        url=row.url or None,
    )


def _pareto_membership(
    entries: list[dict[str, Any]], ranking: tuple[str, ...]
) -> dict[str, bool]:
    eligible = [entry for entry in entries if entry["ranking_eligible"]]

    def dominates(candidate: dict[str, Any], target: dict[str, Any]) -> bool:
        comparisons = []
        for metric_name in ranking:
            candidate_value = candidate["metrics"][metric_name]["mean"]
            target_value = target["metrics"][metric_name]["mean"]
            if metric_registry.higher_is_better(metric_name):
                comparisons.append((candidate_value >= target_value, candidate_value > target_value))
            else:
                comparisons.append((candidate_value <= target_value, candidate_value < target_value))
        return all(no_worse for no_worse, _ in comparisons) and any(
            better for _, better in comparisons
        )

    return {
        entry["experiment_id"]: not any(
            other is not entry and dominates(other, entry) for other in eligible
        )
        for entry in eligible
    }


def _clean(value: Any) -> Any:
    """Normalise an axis value for display (drop the _MISSING sentinel)."""
    return None if value is _MISSING else value


def _reference_banner(benchmark: "Benchmark | None", metric: str) -> ReferenceBanner | None:
    """The broad (empty-key) **gate** reference row for ``metric``, rendered as a banner.

    External guidance anchors also carry an empty key + this metric, so they are filtered
    out here (they render in their own guidance section, never as the gate band).
    """
    if benchmark is None:
        return None
    broad = [
        r
        for r in benchmark.expected()
        if not r.key and r.metric == metric and not r.is_external
    ]
    if not broad:
        return None
    row = broad[0]
    return ReferenceBanner(
        metric=row.metric,
        expected=row.expected,
        tolerance=row.tolerance_band(),
        source=row.source,
    )


def _guidance_anchors(benchmark: "Benchmark | None", metric: str) -> tuple[GuidanceAnchor, ...]:
    """External (non-gating) guidance anchors for ``metric`` (issue #226).

    Collected config-agnostically: an external anchor is typically not keyed to an encoder,
    so ``benchmark.expected()`` (no axes) surfaces it for any facet.
    """
    if benchmark is None:
        return ()
    return tuple(
        GuidanceAnchor(
            label=row.label,
            metric=row.metric,
            expected=row.expected,
            url=row.url,
            source=row.source,
        )
        for row in benchmark.expected()
        if row.is_external and row.metric == metric
    )


def _keyed_reference(
    benchmark: "Benchmark | None",
    metric: str,
    vary_values: dict[str, Any],
    measured: float,
) -> dict[str, Any]:
    """Join a keyed reference row to a measured row on the varied axis (per-row tolerance)."""
    empty = {
        "reference_expected": None,
        "reference_tolerance": None,
        "reference_pass": None,
        "reference_source": None,
    }
    if benchmark is None or not vary_values:
        return empty
    axes = {k: v for k, v in vary_values.items() if v is not None}
    keyed = [
        r
        for r in benchmark.expected(**axes)
        if r.key and r.metric == metric and not r.is_external
    ]
    if len(keyed) != 1:
        return empty
    row = keyed[0]
    gates = metric == benchmark.primary_metric
    return {
        "reference_expected": row.expected,
        "reference_tolerance": row.tolerance_band() if gates else None,
        "reference_pass": row.within_tolerance(measured) if gates else None,
        "reference_source": row.source,
    }


# --- rendering -----------------------------------------------------------------------


def _fmt_metric(row: LeaderboardRow) -> str:
    if row.std is None:
        return f"{row.mean:.4f}"
    return f"{row.mean:.4f} ± {row.std:.4f}"


def _fmt_diff(diff: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(diff.items()))


def _row_dict(row: LeaderboardRow) -> dict[str, Any]:
    return {
        "rank": row.rank,
        "experiment_id": row.experiment_id,
        "vary": dict(row.vary_values),
        "metric": row.metric,
        "mean": row.mean,
        "std": row.std,
        "n": row.n,
        "seeds": list(row.seeds),
        "config_diff": dict(row.config_diff),
        "reference": (
            None
            if row.reference_expected is None
            else {
                "expected": row.reference_expected,
                "tolerance": row.reference_tolerance,
                "pass": row.reference_pass,
                "source": row.reference_source,
            }
        ),
    }


def _metric_dict(metric: LeaderboardMetric) -> dict[str, Any]:
    reference = metric.reference
    return {
        "rank": metric.rank,
        "mean": metric.mean,
        "std": metric.std,
        "n": metric.n,
        "seeds": list(metric.seeds),
        "higher_is_better": metric.higher_is_better,
        "ranking": metric.ranking,
        "reference": (
            None
            if reference is None
            else {
                "expected": reference.expected,
                "delta": reference.delta,
                "tolerance": reference.tolerance,
                "pass": reference.within_tolerance,
                "source": reference.source,
                "label": reference.label,
                "url": reference.url,
            }
        ),
    }


def _description_dict(description: LeaderboardMetricDescription) -> dict[str, Any]:
    return {
        "metric": description.metric,
        "higher_is_better": description.higher_is_better,
    }


def to_dict(table: LeaderboardTable) -> dict[str, Any]:
    """A JSON-ready dict for the whole table."""
    if table.reported_metrics:
        return {
            "triple": {
                "dataset_checksum": table.triple[0],
                "splits_checksum": table.triple[1],
                "task": table.triple[2],
            },
            "split": table.split,
            "vary": list(table.vary),
            "reported_metrics": [
                _description_dict(description) for description in table.reported_metrics
            ],
            "ranking_metrics": [
                _description_dict(description) for description in table.ranking_metrics
            ],
            "rows": [
                {
                    "experiment_id": row.experiment_id,
                    "vary": dict(row.vary_values),
                    "ranking_eligible": row.ranking_eligible,
                    "metrics": {
                        description.metric: _metric_dict(row.metrics[description.metric])
                        for description in table.reported_metrics
                    },
                    "pareto": row.pareto,
                    "config_diff": dict(row.config_diff),
                }
                for row in table.rows
            ],
        }
    return {
        "triple": {
            "dataset_checksum": table.triple[0],
            "splits_checksum": table.triple[1],
            "task": table.triple[2],
        },
        "metric": table.metric,
        "higher_is_better": table.higher_is_better,
        "split": table.split,
        "vary": list(table.vary),
        "banner": (
            None
            if table.banner is None
            else {
                "metric": table.banner.metric,
                "expected": table.banner.expected,
                "tolerance": table.banner.tolerance,
                "source": table.banner.source,
            }
        ),
        "guidance": [
            {
                "label": anchor.label,
                "metric": anchor.metric,
                "expected": anchor.expected,
                "url": anchor.url,
                "source": anchor.source,
            }
            for anchor in table.guidance
        ],
        "rows": [_row_dict(r) for r in table.rows],
    }


def render_json(table: LeaderboardTable) -> str:
    return json.dumps(to_dict(table), indent=2)


def render_csv(table: LeaderboardTable) -> str:
    import csv
    import io

    buffer = io.StringIO()
    if table.reported_metrics:
        columns = list(table.vary)
        referenced_metrics = {
            description.metric
            for description in table.reported_metrics
            if any(
                row.metrics[description.metric].reference is not None
                for row in table.rows
            )
        }
        for description in table.reported_metrics:
            name = description.metric
            columns += [f"{name}_mean", f"{name}_std", f"{name}_n", f"{name}_seeds"]
            if name in referenced_metrics:
                columns += [f"{name}_reference_expected", f"{name}_reference_delta"]
                if name == table.metric:
                    columns += [f"{name}_reference_pass"]
        columns += [f"{description.metric}_rank" for description in table.ranking_metrics]
        columns += ["ranking_eligible", "pareto", "config_diff"]
        writer = csv.writer(buffer)
        writer.writerow(columns)
        for row in table.rows:
            cells: list[Any] = [row.vary_values.get(axis) for axis in table.vary]
            for description in table.reported_metrics:
                result = row.metrics[description.metric]
                cells += [
                    f"{result.mean:.6f}",
                    "" if result.std is None else f"{result.std:.6f}",
                    result.n,
                    "|".join(str(seed) for seed in result.seeds),
                ]
                if description.metric in referenced_metrics:
                    reference = result.reference
                    cells += [
                        "" if reference is None else f"{reference.expected:.6f}",
                        "" if reference is None else f"{reference.delta:.6f}",
                    ]
                    if description.metric == table.metric:
                        cells += [
                            "" if reference is None or reference.within_tolerance is None
                            else ("PASS" if reference.within_tolerance else "FAIL")
                        ]
            cells += [
                "" if row.metrics[description.metric].rank is None
                else row.metrics[description.metric].rank
                for description in table.ranking_metrics
            ]
            cells += [
                str(row.ranking_eligible).lower(),
                "" if row.pareto is None else str(row.pareto).lower(),
                _fmt_diff(row.config_diff),
            ]
            writer.writerow(cells)
        return buffer.getvalue()

    columns = ["rank", *table.vary, "mean", "std", "n", "seeds"]
    has_reference = any(r.reference_expected is not None for r in table.rows)
    if has_reference:
        columns += ["reference_expected", "reference_tolerance", "reference_pass"]
    columns += ["config_diff"]
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in table.rows:
        cells = [row.rank, *[row.vary_values.get(ax) for ax in table.vary]]
        cells += [f"{row.mean:.6f}", "" if row.std is None else f"{row.std:.6f}", row.n]
        cells += ["|".join(str(s) for s in row.seeds)]
        if has_reference:
            cells += [
                "" if row.reference_expected is None else f"{row.reference_expected:.6f}",
                "" if row.reference_tolerance is None else f"{row.reference_tolerance:.6f}",
                "" if row.reference_pass is None else ("PASS" if row.reference_pass else "FAIL"),
            ]
        cells += [_fmt_diff(row.config_diff)]
        writer.writerow(cells)
    return buffer.getvalue()


def format_table(table: LeaderboardTable) -> str:
    """A plain-text ranked table for stdout."""
    if table.reported_metrics:
        return _format_multi_metric_table(table)
    lines: list[str] = []
    direction = "higher is better" if table.higher_is_better else "lower is better"
    lines.append(
        f"Leaderboard — task={table.triple[2]} · metric={table.metric} ({direction}) · "
        f"split={table.split}"
    )
    if table.banner is not None:
        b = table.banner
        lines.append(
            f"reference band: {b.metric} = {b.expected:.4f} ± {b.tolerance:.4f}"
            + (f"  [{b.source}]" if b.source else "")
        )
    if table.guidance:
        lines.append("guidance (external reference — context, not a gated target):")
        for anchor in table.guidance:
            link = f"  <{anchor.url}>" if anchor.url else ""
            lines.append(
                f"  · {anchor.label}: {anchor.metric} = {anchor.expected:.4f}{link}"
            )
    has_reference = any(r.reference_expected is not None for r in table.rows)
    header = ["#", *table.vary, table.metric, "n"]
    if has_reference:
        header += ["ref", "check"]
    header += ["config diff"]
    rows_text: list[list[str]] = [header]
    for row in table.rows:
        cells = [
            "" if row.rank is None else str(row.rank),
            *[str(_clean(row.vary_values.get(ax))) for ax in table.vary],
        ]
        cells += [_fmt_metric(row), str(row.n)]
        if has_reference:
            cells += [
                "" if row.reference_expected is None else f"{row.reference_expected:.4f}",
                "" if row.reference_pass is None else ("PASS" if row.reference_pass else "FAIL"),
            ]
        cells += [_fmt_diff(row.config_diff) or "—"]
        rows_text.append(cells)
    widths = [max(len(r[i]) for r in rows_text) for i in range(len(header))]
    for text_row in rows_text:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(text_row)))
    return "\n".join(lines)


def _fmt_wide_metric(metric: LeaderboardMetric) -> str:
    value = f"{metric.mean:.4f}"
    if metric.std is not None:
        value += f" ± {metric.std:.4f}"
    if metric.reference is not None:
        reference = metric.reference
        value += f" (ref {reference.expected:.4f}, Δ {reference.delta:+.4f})"
        if reference.within_tolerance is not None:
            value += " PASS" if reference.within_tolerance else " FAIL"
    return value


def _format_multi_metric_table(table: LeaderboardTable) -> str:
    names = ", ".join(description.metric for description in table.reported_metrics)
    lines = [f"Leaderboard — task={table.triple[2]} · metrics={names} · split={table.split}"]
    header, projected_rows = _wide_table_grid(table)
    rows_text = [list(header)] + [
        [*row[:-1], row[-1] or "—"] for row in projected_rows
    ]
    widths = [max(len(row[index]) for row in rows_text) for index in range(len(header))]
    lines += [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows_text
    ]
    return "\n".join(lines)


def _wide_table_grid(
    table: LeaderboardTable,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Project the shared wide display columns once for text and HTML encoders."""
    header = tuple(table.vary)
    header += tuple(description.metric for description in table.reported_metrics)
    header += tuple(
        f"{description.metric} rank" for description in table.ranking_metrics
    )
    header += ("eligible", "pareto", "config diff")

    rows: list[tuple[str, ...]] = []
    for row in table.rows:
        cells = tuple(str(_clean(row.vary_values.get(axis))) for axis in table.vary)
        cells += tuple(
            _fmt_wide_metric(row.metrics[description.metric])
            for description in table.reported_metrics
        )
        cells += tuple(
            ""
            if row.metrics[description.metric].rank is None
            else str(row.metrics[description.metric].rank)
            for description in table.ranking_metrics
        )
        cells += (
            "eligible" if row.ranking_eligible else "control (ranking-ineligible)",
            "" if row.pareto is None else ("yes" if row.pareto else "no"),
            _fmt_diff(row.config_diff),
        )
        rows.append(cells)
    return header, tuple(rows)


def render_html(table: LeaderboardTable) -> str:
    """A self-contained HTML page for the leaderboard."""

    def esc(value: Any) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    if table.reported_metrics:
        header_cells, projected_rows = _wide_table_grid(table)
        head = "".join(f"<th>{esc(cell)}</th>" for cell in header_cells)
        body_rows = []
        for cells in projected_rows:
            body_rows.append(
                "<tr>"
                + "".join(f"<td>{esc(cell)}</td>" for cell in cells)
                + "</tr>"
            )
        title = f"SOMA leaderboard — {esc(table.triple[2])}"
        names = ", ".join(description.metric for description in table.reported_metrics)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>{_css()}</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="cfg-item">reported metrics <strong>{esc(names)}</strong> · split
     <strong>{esc(table.split)}</strong></p>
  <table class="results-table ranking-table">
    <thead><tr>{head}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>"""

    has_reference = any(r.reference_expected is not None for r in table.rows)
    header_cells = ["#", *table.vary, esc(table.metric), "n"]
    if has_reference:
        header_cells += ["reference", "check"]
    header_cells += ["config diff"]
    head = "".join(f"<th>{esc(c)}</th>" for c in header_cells)

    body_rows = []
    for row in table.rows:
        cells = [
            "" if row.rank is None else str(row.rank),
            *[esc(_clean(row.vary_values.get(ax))) for ax in table.vary],
        ]
        cells.append(esc(_fmt_metric(row)))
        cells.append(str(row.n))
        if has_reference:
            cells.append("" if row.reference_expected is None else f"{row.reference_expected:.4f}")
            if row.reference_pass is None:
                cells.append("")
            else:
                verdict = "PASS" if row.reference_pass else "FAIL"
                cells.append(verdict)
        cells.append(esc(_fmt_diff(row.config_diff)))
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    banner_html = ""
    if table.banner is not None:
        b = table.banner
        banner_html = (
            f'<p class="cfg-item"><strong>reference band:</strong> {esc(b.metric)} = '
            f"{b.expected:.4f} ± {b.tolerance:.4f} {esc(b.source)}</p>"
        )
    guidance_html = ""
    if table.guidance:
        items = []
        for anchor in table.guidance:
            label = esc(anchor.label)
            linked = (
                f'<a href="{esc(anchor.url)}">{label}</a>' if anchor.url else label
            )
            items.append(
                f"<li>{linked}: {esc(anchor.metric)} = {anchor.expected:.4f}"
                + (f" <span class=\"muted\">{esc(anchor.source)}</span>" if anchor.source else "")
                + "</li>"
            )
        guidance_html = (
            '<div class="cfg-item"><strong>guidance</strong> (external reference — '
            "context, not a gated target):<ul>" + "".join(items) + "</ul></div>"
        )
    direction = "higher is better" if table.higher_is_better else "lower is better"
    title = f"SOMA leaderboard — {esc(table.triple[2])}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>{_css()}</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="cfg-item">metric <strong>{esc(table.metric)}</strong> ({direction}) · split
     <strong>{esc(table.split)}</strong></p>
  {banner_html}
  {guidance_html}
  <table class="results-table ranking-table">
    <thead><tr>{head}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>"""


def _slug_for(table: LeaderboardTable) -> str:
    task = table.triple[2] or "task"
    axes = "-".join(table.vary) if table.vary else "flat"
    dataset = (table.triple[0] or "")[:8]
    return f"{task}__vary-{axes}__{dataset}"


def write_leaderboard(
    table: LeaderboardTable,
    output_root: str | Path,
    *,
    name: str | None = None,
) -> dict[str, Path]:
    """Write CSV + JSON + HTML under ``<output_root>/leaderboards/`` (disposable cache)."""
    leaderboards_dir = Path(output_root) / "leaderboards"
    leaderboards_dir.mkdir(parents=True, exist_ok=True)
    stem = name or _slug_for(table)
    paths = {
        "csv": leaderboards_dir / f"{stem}.csv",
        "json": leaderboards_dir / f"{stem}.json",
        "html": leaderboards_dir / f"{stem}.html",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["csv"].write_text(render_csv(table), encoding="utf-8")
    paths["json"].write_text(render_json(table), encoding="utf-8")
    paths["html"].write_text(render_html(table), encoding="utf-8")
    return paths


def diff_configs(configs: list[dict]) -> tuple[dict, list[dict]]:
    """Partition config fields into shared values and per-run diffs.

    Recursively flattens nested dicts using dot notation (e.g.,
    ``trainer.max_epochs``). Fields whose values are identical across
    all configs go into ``shared``; fields that differ go into per-run
    ``diffs`` dicts.

    Args:
        configs: List of config dicts (one per run).

    Returns:
        ``(shared, diffs)`` where ``shared`` maps flat key → common value and
        ``diffs`` is a list of dicts (one per run) mapping flat key → value
        for every key that varies.
    """
    if not configs:
        return {}, []

    flat_configs = [_flatten(c) for c in configs]
    all_keys: list[str] = []
    for fc in flat_configs:
        for k in fc:
            if k not in all_keys:
                all_keys.append(k)

    shared: dict = {}
    diffs: list[dict] = [{} for _ in configs]

    for key in all_keys:
        values = [fc.get(key) for fc in flat_configs]
        if all(v == values[0] for v in values[1:]):
            shared[key] = values[0]
        else:
            for i, v in enumerate(values):
                diffs[i][key] = v

    return shared, diffs


def _flatten(obj: object, prefix: str = "") -> dict:
    """Recursively flatten a nested dict to dot-separated keys."""
    result: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(_flatten(value, full_key))
            else:
                result[full_key] = value
    else:
        result[prefix] = obj
    return result


def _css() -> str:
    return """
:root {
  --soma-bg:           #FFFFFF;
  --soma-bg-subtle:    #FFF7FA;
  --soma-bg-card:      #FFF0F5;
  --soma-text:         #0F172A;
  --soma-text-muted:   #64748B;
  --soma-accent:       #E2558A;
  --soma-accent-light: #FCE3ED;
  --soma-accent-deep:  #B52B65;
  --soma-border:       #E9D5DF;
  --soma-header-bg:    #171018;
  --soma-header-text:  #F8FAFC;
  --soma-success:      #10B981;
  --soma-danger:       #EF4444;
  --soma-warning:      #F59E0B;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  background:
    radial-gradient(circle at top left, rgba(226, 85, 138, 0.08), transparent 35%),
    linear-gradient(180deg, #fff9fc 0%, #fff6f9 42%, #f8f5f8 100%);
  color: var(--soma-text);
  line-height: 1.5;
}
/* ---- Header ---- */
.site-header {
  background: var(--soma-header-bg);
  color: var(--soma-header-text);
  padding: 0 28px;
  height: 58px;
  display: flex;
  align-items: center;
  gap: 14px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
}
.soma-wordmark {
  font-weight: 800;
  font-size: 15px;
  letter-spacing: 0.08em;
  color: var(--soma-accent);
  flex-shrink: 0;
}
.header-run-id {
  font-family: ui-monospace, 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #94A3B8;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.header-meta-item {
  font-size: 12px;
  color: #94A3B8;
}
/* ---- Hero metrics ---- */
.overview-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}
.hero-card {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,246,250,0.92));
  border: 1px solid rgba(226,85,138,0.18);
  border-left: 4px solid var(--soma-accent);
  border-radius: 20px;
  padding: 14px 16px;
  min-width: 0;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.03);
}
.hero-value {
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--soma-accent-deep);
  line-height: 1.1;
}
.hero-std {
  font-size: 1rem;
  font-weight: 400;
  color: var(--soma-text-muted);
  margin-left: 4px;
}
.hero-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--soma-text-muted);
  margin-top: 4px;
}
/* ---- Badges ---- */
.badge {
  padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
}
.status-ok  { background: #10B981; color: #fff; }
.status-err { background: #EF4444; color: #fff; }
.status-run { background: #F59E0B; color: #000; }
.run-badge {
  display: inline-block;
  padding: 2px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 700; color: #fff;
}
/* ---- Sections ---- */
.section {
  background: var(--soma-bg);
  border-radius: 18px;
  margin: 18px 28px;
  padding: 22px 24px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 14px 40px rgba(15, 23, 42, 0.04);
  border: 1px solid var(--soma-border);
}
.section-compact {
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  background: transparent;
}
.section h2 {
  font-size: 15px; font-weight: 700; margin-bottom: 16px;
  border-bottom: 1px solid var(--soma-border); padding-bottom: 10px;
  color: var(--soma-text);
}
.section h3 {
  font-size: 13px; font-weight: 600; margin: 16px 0 8px;
  color: var(--soma-text-muted);
}
.tab-stack {
  display: flex;
  flex-direction: column;
}
.overview-layout {
  display: grid;
  gap: 16px;
}
.overview-banner {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: end;
  padding: 22px 24px;
  border-radius: 26px;
  background:
    radial-gradient(circle at top left, rgba(226, 85, 138, 0.16), transparent 38%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.99), rgba(252, 227, 237, 0.92));
  border: 1px solid rgba(226, 85, 138, 0.18);
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
}
.overview-banner-copy {
  max-width: 52ch;
}
.overview-banner-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  color: var(--soma-text-muted);
  font-size: 12px;
  font-weight: 700;
}
.overview-banner-meta span {
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(226, 85, 138, 0.14);
  padding: 6px 10px;
  border-radius: 999px;
}
.overview-kicker {
  color: var(--soma-accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 10px;
  font-weight: 800;
  margin-bottom: 8px;
}
.overview-banner h2 {
  font-size: 22px;
  line-height: 1.12;
  margin-bottom: 8px;
  border: 0;
  padding: 0;
}
.overview-banner p {
  max-width: 54ch;
  color: var(--soma-text-muted);
  font-size: 13px;
}
.overview-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
}
.overview-panel {
  background: linear-gradient(180deg, #ffffff, #fffafc);
  border: 1px solid var(--soma-border);
  border-radius: 22px;
  padding: 20px 22px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03), 0 12px 28px rgba(15, 23, 42, 0.03);
}
.overview-panel-wide {
  width: 100%;
}
.overview-panel-head {
  margin-bottom: 16px;
}
.overview-inline-head {
  margin-bottom: 12px;
}
.overview-panel-head h3 {
  font-size: 14px;
  font-weight: 800;
  color: var(--soma-text);
  margin-bottom: 4px;
}
.overview-panel-head p {
  color: var(--soma-text-muted);
  font-size: 12px;
}
.overview-panel-foot {
  margin-top: 16px;
}
.overview-spec-grid,
.training-spec-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.overview-spec,
.training-spec {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(226, 85, 138, 0.05);
  border: 1px solid rgba(226, 85, 138, 0.12);
}
.overview-spec-label,
.training-spec-label {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--soma-text-muted);
  margin-bottom: 6px;
}
.overview-spec-value,
.training-spec-value {
  font-size: 13px;
  font-weight: 700;
  color: var(--soma-text);
  word-break: break-word;
}
.overview-spec-grid > .overview-spec:last-child:nth-child(odd),
.training-spec-grid > .training-spec:last-child:nth-child(odd) {
  grid-column: 1 / -1;
}
.muted { color: var(--soma-text-muted); font-style: italic; }
/* ---- Config collapsible ---- */
.cfg-details summary {
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
  color: var(--soma-text);
  padding-bottom: 0;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cfg-details summary::before {
  content: "▶";
  font-size: 10px;
  color: var(--soma-text-muted);
  transition: transform 0.15s;
}
.cfg-details[open] summary::before { transform: rotate(90deg); }
.cfg-details[open] summary { margin-bottom: 16px; border-bottom: 1px solid var(--soma-border); padding-bottom: 10px; }
.cfg-table {
  border-collapse: collapse;
  width: 100%;
  max-width: none;
  margin-top: 4px;
  table-layout: fixed;
}
.cfg-table tr:nth-child(even) td { background: var(--soma-bg-subtle); }
.cfg-key {
  padding: 6px 12px 6px 0; font-weight: 600; color: var(--soma-text-muted);
  white-space: nowrap; width: 32%; vertical-align: top; font-size: 12px;
}
.cfg-val { padding: 6px 0; color: var(--soma-text); word-break: break-all; font-size: 12px; }
.section-note {
  margin: -6px 0 14px;
  color: var(--soma-text-muted);
  font-size: 12px;
}
/* ---- Results table ---- */
.results-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.results-table th {
  background: var(--soma-bg-subtle); padding: 8px 14px;
  text-align: right; font-weight: 600; border-bottom: 2px solid var(--soma-border);
  color: var(--soma-text-muted); font-size: 12px;
}
.results-table th:first-child { text-align: left; }
.results-table td { padding: 7px 14px; text-align: right; border-bottom: 1px solid var(--soma-bg-subtle); }
.results-table td.metric-name { text-align: left; font-weight: 500; }
.results-table tr:hover td { background: var(--soma-bg-subtle); }
.ranking-table .rank-cell { width: 44px; }
.ranking-table .rank-name { text-align: left; font-weight: 700; }
.ranking-table .rank-metric { font-variant-numeric: tabular-nums; }
.rank-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--soma-text);
  text-decoration: none;
}
.rank-link-label {
  transition: transform 0.16s ease;
}
.rank-link-arrow {
  color: var(--soma-accent);
  font-size: 11px;
  line-height: 1;
  opacity: 1;
  transform: translate(0, 0);
  transition: opacity 0.16s ease, transform 0.16s ease;
}
.rank-link:hover .rank-link-label,
.rank-link:focus-visible .rank-link-label {
  transform: translateX(1px);
}
.rank-link:hover .rank-link-arrow,
.rank-link:focus-visible .rank-link-arrow {
  transform: translate(1px, -1px);
}
.rank-link:focus-visible {
  outline: 2px solid rgba(226, 85, 138, 0.35);
  outline-offset: 3px;
  border-radius: 999px;
}
.ranking-table .rank-row:hover td {
  background: rgba(226, 85, 138, 0.07);
}
.rank-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 30px;
  height: 30px;
  border-radius: 999px;
  padding: 0 10px;
  color: #fff;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.01em;
}
.podium-gold { background: linear-gradient(180deg, #F6E08A, #D6A81F); color: #3B2E06; }
.podium-silver { background: linear-gradient(180deg, #E5E7EB, #A8B0BD); color: #1F2937; }
.podium-bronze { background: linear-gradient(180deg, #F1C08B, #B86A24); color: #40220B; }
.soma-brand-rank { background: linear-gradient(180deg, #F7BDD1, var(--soma-accent)); }
.section-overview .results-table tr.rank-row td {
  background: transparent;
}
.section-overview .results-table tr.rank-row.podium-gold td,
.section-overview .results-table tr.rank-row.podium-silver td,
.section-overview .results-table tr.rank-row.podium-bronze td,
.section-overview .results-table tr.rank-row.soma-brand-rank td {
  background: rgba(226, 85, 138, 0.05);
}
/* ---- Chart grid ---- */
.chart-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.chart-full { flex: 1 1 100%; min-width: 0; }
.chart-full svg { width: 100%; height: auto; display: block; }
.chart-half {
  flex: 1 1 calc(50% - 6px);
  max-width: calc(50% - 6px);
  min-width: 280px;
  aspect-ratio: 7 / 4.5;
  overflow: hidden;
}
.chart-half svg { width: 100%; height: 100%; display: block; }
/* ---- Comparison ---- */
.best-val { background: rgba(16, 185, 129, 0.14); }
.sig-worse { background: rgba(239, 68, 68, 0.12); }
/* ---- Subgroup analysis ---- */
.subgroup-sig { background: #FEE2E2; font-weight: 600; }
.subgroup-flag { background: #FEF3C7; font-weight: 600; }
.subgroup-sig-small { background: #EDE9FE; }
.subgroup-legend { font-size: 12px; color: var(--soma-text-muted); margin-top: 6px; }
.legend-sig { color: #FEE2E2; text-shadow: 0 0 1px #666; }
.legend-flag { color: #FEF3C7; text-shadow: 0 0 1px #666; }
.legend-sig-small { color: #EDE9FE; text-shadow: 0 0 1px #666; }
/* ---- Comparison hero ---- */
.comp-hero-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.comp-hero-card {
  background: var(--soma-bg-subtle);
  border: 1px solid var(--soma-border);
  border-radius: 16px;
  padding: 14px 20px;
  min-width: 180px;
}
.comp-hero-metric {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--soma-text-muted);
  margin-bottom: 10px;
}
.comp-hero-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.comp-hero-row:last-child { margin-bottom: 0; }
.comp-hero-val {
  font-size: 1.4rem;
  line-height: 1.1;
}
/* ---- Tabs ---- */
.tab-group {
  margin: 18px 28px 0;
  border: 1px solid rgba(226, 85, 138, 0.18);
  border-radius: 24px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,247,250,0.88));
  backdrop-filter: blur(10px);
  box-shadow:
    0 12px 30px rgba(15, 23, 42, 0.04),
    0 1px 0 rgba(255,255,255,0.8) inset;
  overflow: hidden;
}
.tab-panels {
  padding: 16px;
}
.tab-panel {
  width: 100%;
}
.tab-bar {
  display: flex;
  gap: 8px;
  border-bottom: 1px solid rgba(226, 85, 138, 0.12);
  margin-bottom: 0;
  padding: 10px 12px 0;
  overflow-x: auto;
  background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,247,250,0.28));
}
.tab-btn {
  padding: 10px 16px 11px;
  border: 1px solid transparent;
  background: rgba(255,255,255,0.45);
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
  color: var(--soma-text-muted);
  border-bottom: 0;
  margin-bottom: 0;
  border-radius: 16px 16px 0 0;
  transition:
    color 0.15s ease,
    background 0.15s ease,
    border-color 0.15s ease,
    box-shadow 0.15s ease,
    transform 0.15s ease;
  white-space: nowrap;
}
.tab-btn:hover {
  color: var(--soma-text);
  background: rgba(255,255,255,0.92);
  transform: translateY(-1px);
}
.tab-btn.active {
  color: var(--soma-accent-deep);
  background: linear-gradient(180deg, #ffffff, #fff5f9);
  border-color: rgba(226, 85, 138, 0.22);
  box-shadow:
    0 -1px 0 rgba(255,255,255,0.85) inset,
    0 8px 22px rgba(226, 85, 138, 0.08);
  transform: translateY(-1px);
}
.tab-panels .section { margin: 0 0 16px; }
.tab-panels .section:last-child { margin-bottom: 0; }
.tab-panels .overview-layout { padding: 0; }
.tab-panels .overview-panel { margin: 0; }
.tab-panels .overview-grid { margin-top: 0; }
.tab-panels .section-compact {
  padding: 0;
}
.chart-square { max-width: 560px; }
.chart-square svg { width: 100%; height: auto; display: block; }
.chart-panel {
  border: 1px solid rgba(226, 85, 138, 0.14);
  border-radius: 20px;
  padding: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,247,250,0.8));
}
code { font-family: ui-monospace, 'JetBrains Mono', monospace; font-size: 12px; }
@media (max-width: 768px) {
  .section { margin: 12px; padding: 16px; border-radius: 16px; }
  .chart-half { flex: 1 1 100%; max-width: 100%; aspect-ratio: unset; }
  .site-header { padding: 0 16px; }
  .tab-bar { padding: 8px 8px 0; }
  .tab-group { margin: 12px; border-radius: 18px; }
  .tab-panels { padding: 12px; }
  .overview-grid,
  .overview-spec-grid,
  .training-spec-grid { grid-template-columns: 1fr; }
  .overview-banner { align-items: start; flex-direction: column; }
}
"""
