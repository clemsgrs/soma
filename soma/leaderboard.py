"""The Leaderboard: a rendered *faceted view* over self-describing run dirs (ADR 0003).

A Leaderboard is **not** a stored file. It is a pure projection over the run dirs a
sweep already wrote: each run stamps its own ``config.yaml`` + ``summary.json`` +
``run.yaml``, and the experiment dir above it stamps ``experiment.json`` (the canonical
config fingerprint + dataset/splits checksums). :func:`project_leaderboard` scans those
artifacts and renders a ranked table — no pipeline execution, no shared mutable index.

A **facet** fixes a ``(dataset + splits + task)`` triple, holds a set of config axes
fixed, and surfaces one (or more) *varied* axes as the comparison columns, ranked by a
primary metric. Because a run's identity (``experiment_id``) is a fingerprint of the
*entire* config modulo seed, two runs that differ in any **unfixed** axis are never
pooled: each distinct config becomes its own row, annotated with the config diff that
distinguishes it. Seed-runs of the *same* config collapse to mean ± std + n.

Reference rows (a registered :class:`~soma.benchmarks.registry.Benchmark`'s
``expected()``) inject two ways: a broad, config-agnostic scalar renders as a **threshold
banner** above the table; keyed rows **join** to the measured rows on the varied axis with
a per-row tolerance PASS/FAIL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from soma.evaluation.metrics import metric_higher_is_better
from soma.reporting.data import diff_configs

if TYPE_CHECKING:
    from soma.benchmarks.registry import Benchmark

# A leaderboard scopes to exactly one of these triples.
TripleKey = tuple[str, str, str]  # (dataset_checksum, splits_checksum, task)

# Sentinel for "this axis does not resolve against a run's canonical spec".
_MISSING = object()

# Friendly axis names -> dotted path into a run's canonical_spec. Any other axis name is
# treated as a raw dotted path, so arbitrary config fields can be fixed/varied too.
_AXIS_ALIASES: dict[str, str] = {
    "encoder": "encoder.name",
    "aggregator": "aggregator.name",
    "decoder": "decoder.name",
    "pixel_classifier": "pixel_classifier.name",
    "task": "task.name",
    "spacing": "preprocessing.requested_spacing_um",
    "dataset_type": "dataset_type",
    "feature_mode": "feature_mode",
}

# Summary keys that are provenance bookkeeping, not a rankable metric (mirror the run
# panel's primary-metric detection in soma/pipeline.py).
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
    non-config axis like ``matcher``, or the encoder is a composite so ``encoder.name`` is
    absent). Callers treat :data:`_MISSING` as "cannot constrain / cannot vary here".
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
    """One completed run, loaded from its self-describing dir artifacts."""

    run_dir: Path
    experiment_id: str
    seed: int | None
    status: str | None
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
    """Load a :class:`RunRecord` from a run dir, or ``None`` if it isn't rankable.

    A run is rankable only if it has a non-empty ``summary`` (so failed/running runs, which
    stamp no metrics, are skipped) and its identity artifacts are present.
    """
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
    if not summary:
        return None

    exp_json = _locate_experiment_json(run_dir)
    if exp_json is None:
        return None
    experiment = json.loads(exp_json.read_text(encoding="utf-8"))
    spec = experiment.get("canonical_spec") or {}
    task = str((spec.get("task") or {}).get("name") or "")

    seed = meta.get("seed")
    return RunRecord(
        run_dir=run_dir,
        experiment_id=str(meta.get("experiment_id") or experiment.get("experiment_id") or ""),
        seed=int(seed) if seed not in (None, "") else None,
        status=meta.get("status"),
        canonical_spec=spec,
        dataset_checksum=str(experiment.get("dataset_checksum") or ""),
        splits_checksum=str(experiment.get("splits_checksum") or ""),
        task=task,
        summary=summary,
    )


def discover_triples(output_root: str | Path) -> dict[TripleKey, list[Path]]:
    """Scan ``<output_root>/experiments/*/runs/*`` and group run dirs by their triple."""
    root = Path(output_root)
    result: dict[TripleKey, list[Path]] = {}
    for exp_json in sorted(root.glob("experiments/*/experiment.json")):
        runs_dir = exp_json.parent / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            record = load_run_record(run_dir)
            if record is None:
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

    Mirrors the run-summary panel in ``soma/pipeline.py``: the first metric key under the
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
class LeaderboardRow:
    rank: int
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
    higher = metric_higher_is_better(metric_name)

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
        prelim.append(
            {
                "experiment_id": experiment_id,
                "spec": spec,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else None,
                "n": len(values),
                "seeds": tuple(sorted(r.seed for r in recs if r.seed is not None)),
                "vary_values": {ax: _clean(axis_value(spec, ax)) for ax in facet.vary},
            }
        )

    # Config diff across the shown rows only (never-pool annotation).
    specs = [entry["spec"] for entry in prelim]
    _, per_row_diffs = diff_configs(specs) if specs else ({}, [])
    suppress = _DIFF_SUPPRESS | {_axis_flat_key(ax) for ax in facet.vary}
    for entry, diff in zip(prelim, per_row_diffs):
        entry["config_diff"] = {k: v for k, v in diff.items() if k not in suppress}

    # Rank by the primary metric (direction-aware); deterministic tie-break.
    prelim.sort(
        key=lambda e: (
            -e["mean"] if higher else e["mean"],
            [str(v) for v in e["vary_values"].values()],
            e["experiment_id"],
        )
    )

    banner = _reference_banner(benchmark, metric_name)
    guidance = _guidance_anchors(benchmark, metric_name)
    rows: list[LeaderboardRow] = []
    for index, entry in enumerate(prelim, start=1):
        ref = _keyed_reference(benchmark, metric_name, entry["vary_values"], entry["mean"])
        rows.append(
            LeaderboardRow(
                rank=index,
                experiment_id=entry["experiment_id"],
                vary_values=entry["vary_values"],
                metric=metric_name,
                mean=entry["mean"],
                std=entry["std"],
                n=entry["n"],
                seeds=entry["seeds"],
                config_diff=entry["config_diff"],
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
        metric=row.metric, expected=row.expected, tolerance=row.tolerance, source=row.source
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
    return {
        "reference_expected": row.expected,
        "reference_tolerance": row.tolerance,
        "reference_pass": row.within_tolerance(measured),
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


def to_dict(table: LeaderboardTable) -> dict[str, Any]:
    """A JSON-ready dict for the whole table."""
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
        cells = [str(row.rank), *[str(_clean(row.vary_values.get(ax))) for ax in table.vary]]
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


def render_html(table: LeaderboardTable) -> str:
    """A self-contained HTML page for the leaderboard (reuses the report stylesheet)."""
    from soma.reporting.html import _css

    def esc(value: Any) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    has_reference = any(r.reference_expected is not None for r in table.rows)
    header_cells = ["#", *table.vary, esc(table.metric), "n"]
    if has_reference:
        header_cells += ["reference", "check"]
    header_cells += ["config diff"]
    head = "".join(f"<th>{esc(c)}</th>" for c in header_cells)

    body_rows = []
    for row in table.rows:
        cells = [str(row.rank), *[esc(_clean(row.vary_values.get(ax))) for ax in table.vary]]
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
    paths["csv"].write_text(render_csv(table), encoding="utf-8")
    paths["json"].write_text(render_json(table), encoding="utf-8")
    paths["html"].write_text(render_html(table), encoding="utf-8")
    return paths
