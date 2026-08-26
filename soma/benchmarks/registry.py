"""Benchmark registry: ``name -> Benchmark`` plus the protocol-as-code interface.

A Built-in **Benchmark** is a named, registered *code* object (ADR 0002). Published
Benchmarks may instead be represented by Project protocols unless maintainers explicitly
promote them (ADR 0010). Registration makes ``soma list benchmarks`` and ``soma
reproduce <name>`` structural: they enumerate and drive the registry directly, so a
benchmark cannot silently drift from a doc someone forgot to write.

A Benchmark is *thin*: it wires an existing curator, a committed config (or a builder), a
reference row, and a scorer together behind a uniform interface. The interface is
**structural** (:class:`Benchmark` is a ``Protocol``): any object exposing the right
attributes/methods conforms — there is no base class to subclass. Where a step is static
a benchmark just delegates (load a committed YAML, read ``summary.json``); where it
computes or scores specially it overrides (OCELOT's greedy matcher).

Expected numbers ship as package data at ``soma/benchmarks/reference/<name>.csv`` with
columns ``key…, metric, expected, tolerance, source`` and a **per-row** tolerance. The
tolerance is absolute by default; an optional ``tolerance_mode=relative`` cell reinterprets
it as a fraction of ``expected`` (so ``tolerance=0.02, tolerance_mode=relative`` is a ±2 %
band that scales with the dataset). The ``key…`` columns are everything left of ``metric``;
an empty key cell means the row is a broad, config-agnostic banner that matches any axes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from soma.config import PipelineConfig
from soma.curation.manifest import CuratedManifest

# The fixed, non-key columns every reference table MUST carry, in canonical order.
_REFERENCE_VALUE_COLUMNS = ("metric", "expected", "tolerance", "source")

# Optional non-key columns that promote a row to a structured external/guidance anchor
# (issue #226) or relabel its tolerance. They are recognised as value columns (never treated
# as axis keys) and default when absent: ``kind`` defaults to ``"gate"``, ``label``/``url``
# to ``""``, and ``tolerance_mode`` to ``"absolute"`` (``"relative"`` = fraction of expected).
_OPTIONAL_VALUE_COLUMNS = ("kind", "label", "url", "tolerance_mode")

# ``tolerance_mode`` values: an absolute band on ``metric`` vs. a fraction of ``expected``.
TOLERANCE_ABSOLUTE = "absolute"
TOLERANCE_RELATIVE = "relative"

# The two row kinds: a ``gate`` row is tolerance-checked by ``soma reproduce``; an
# ``external`` row is a non-gating guidance anchor rendered alongside (never as) the gate.
GATE = "gate"
EXTERNAL = "external"


@dataclass(frozen=True)
class Facet:
    """A benchmark's canonical fixed-vs-varied axis set (leaderboard consumes it later).

    ``fixed`` records the axes held constant across the benchmark's runs (the recipe
    backbone); ``varied`` names the axes it sweeps (the columns a leaderboard would facet
    on). This slice only *records* the facet — nothing renders it yet (ADR 0002).
    """

    fixed: dict[str, Any] = field(default_factory=dict)
    varied: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceRow:
    """One row of a benchmark's reference table.

    ``key`` holds only the *populated* key columns (an empty dict is the broad,
    config-agnostic banner). ``tolerance`` is absolute on ``metric`` and is per-row.

    ``kind`` splits rows into two roles (issue #226): a ``"gate"`` row is the
    tolerance-checkable anchor ``soma reproduce`` verifies; a ``"external"`` row is a
    non-gating guidance anchor (an official/best-reported number captured from an outside
    leaderboard) rendered *alongside* — never *as* — the gate. External rows carry a human
    ``label`` and a linkable ``url``; their ``tolerance`` is ignored (may be blank).

    ``relative`` reinterprets ``tolerance`` as a fraction of ``expected`` (a ±2 % band that
    scales with the dataset) rather than an absolute band; :meth:`tolerance_band` resolves
    the effective absolute band either way, so every renderer shows a real number.
    """

    key: dict[str, str]
    metric: str
    expected: float
    tolerance: float
    source: str
    kind: str = GATE
    label: str = ""
    url: str = ""
    relative: bool = False

    @property
    def is_external(self) -> bool:
        """True for a non-gating guidance anchor (never tolerance-checked)."""
        return self.kind == EXTERNAL

    def tolerance_band(self) -> float:
        """The effective absolute tolerance band — ``tolerance·expected`` when relative."""
        return self.tolerance * self.expected if self.relative else self.tolerance

    def matches(self, axes: dict[str, Any]) -> bool:
        """True if every populated key cell equals the matching axis (banner matches all)."""
        return all(str(axes.get(col)) == val for col, val in self.key.items())

    def within_tolerance(self, measured: float) -> bool:
        return abs(measured - self.expected) <= self.tolerance_band()


@runtime_checkable
class Benchmark(Protocol):
    """Structural interface every registered benchmark satisfies (protocol-as-code).

    Attributes:
        name: Registry key at per-dataset sub-benchmark granularity (e.g. ``"ocelot"``).
        facet: The canonical fixed-vs-varied axis set.
        canonical_seeds: Seeds ``reproduce`` runs by default so the tolerance check
            compares like-for-like against the published number.
        primary_metric: The metric the tolerance band is defined on.
        reference_environment: A small reference environment shown alongside a run
            (may be empty).
    """

    name: str
    facet: Facet
    canonical_seeds: tuple[int, ...]
    primary_metric: str
    reference_environment: dict[str, str]

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest: ...

    def build_config(self, **axes: Any) -> PipelineConfig: ...

    def expected(self, **axes: Any) -> list[ReferenceRow]: ...

    def score(self, run_dir: str | Path) -> dict[str, float]: ...


def _metric_declaration(
    benchmark: Benchmark,
    *,
    attribute: str,
    label: str,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    """Read and validate one optional ordered metric declaration."""
    declared = getattr(benchmark, attribute, fallback)
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise ValueError(
            f"Benchmark {benchmark.name!r}: {label} metrics must be an ordered sequence."
        )
    metrics = tuple(declared)
    if not metrics:
        raise ValueError(f"Benchmark {benchmark.name!r}: {label} metrics must be non-empty.")
    primary_count = metrics.count(benchmark.primary_metric)
    if primary_count != 1:
        uniqueness = "unique and " if primary_count > 1 else ""
        raise ValueError(
            f"Benchmark {benchmark.name!r}: {label} metrics must be {uniqueness}include "
            f"primary_metric {benchmark.primary_metric!r} exactly once."
        )
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"Benchmark {benchmark.name!r}: {label} metrics must be unique.")
    return metrics


def get_reported_metrics(benchmark: Benchmark) -> tuple[str, ...]:
    """Metrics a benchmark requires and renders, in declaration order."""
    return _metric_declaration(
        benchmark,
        attribute="reported_metrics",
        label="Reported",
        fallback=(benchmark.primary_metric,),
    )


def get_ranking_metrics(benchmark: Benchmark) -> tuple[str, ...]:
    """Metrics eligible for ranking, defaulting to every Reported metric."""
    reported = get_reported_metrics(benchmark)
    metrics = _metric_declaration(
        benchmark,
        attribute="ranking_metrics",
        label="Ranking",
        fallback=reported,
    )
    unreported = [metric for metric in metrics if metric not in reported]
    if unreported:
        names = ", ".join(repr(metric) for metric in unreported)
        raise ValueError(
            f"Benchmark {benchmark.name!r}: Ranking metrics are not Reported: {names}."
        )
    return metrics


# --- registry ------------------------------------------------------------------------

_REGISTRY: dict[str, Benchmark] = {}


def register_benchmark(benchmark: Benchmark) -> None:
    """Register a benchmark under its ``name`` (last registration wins on re-import)."""
    _REGISTRY[benchmark.name] = benchmark


def get_benchmark(name: str) -> Benchmark:
    """Look up a registered benchmark by name (fail-fast with the known names)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown benchmark {name!r}; registered: {known}.") from None


def list_benchmarks() -> list[str]:
    """All registered benchmark names, sorted."""
    return sorted(_REGISTRY)


# --- reference tables ----------------------------------------------------------------


def load_reference(name: str) -> list[ReferenceRow]:
    """Load ``reference/<name>.csv`` (package data) into :class:`ReferenceRow` objects.

    Every column left of ``metric`` is a key column; only the non-empty cells of a row
    are kept in :attr:`ReferenceRow.key` (so an all-empty key is the broad banner).
    """
    with resources.files("soma.benchmarks.reference").joinpath(f"{name}.csv").open(
        newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in _REFERENCE_VALUE_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"reference/{name}.csv is missing required column(s) {missing}; "
                f"got {fieldnames}."
            )
        value_columns = set(_REFERENCE_VALUE_COLUMNS) | set(_OPTIONAL_VALUE_COLUMNS)
        key_columns = [c for c in fieldnames if c not in value_columns]
        rows: list[ReferenceRow] = []
        for raw in reader:
            key = {col: raw[col] for col in key_columns if (raw.get(col) or "").strip()}
            kind = (raw.get("kind") or "").strip() or GATE
            tolerance_cell = (raw.get("tolerance") or "").strip()
            # An external guidance anchor never gates, so a blank tolerance is fine.
            tolerance = float(tolerance_cell) if tolerance_cell else 0.0
            mode = (raw.get("tolerance_mode") or "").strip().lower() or TOLERANCE_ABSOLUTE
            if mode not in (TOLERANCE_ABSOLUTE, TOLERANCE_RELATIVE):
                raise ValueError(
                    f"reference/{name}.csv has unknown tolerance_mode {mode!r}; "
                    f"expected {TOLERANCE_ABSOLUTE!r} or {TOLERANCE_RELATIVE!r}."
                )
            rows.append(
                ReferenceRow(
                    key=key,
                    metric=raw["metric"].strip(),
                    expected=float(raw["expected"]),
                    tolerance=tolerance,
                    source=(raw.get("source") or "").strip(),
                    kind=kind,
                    label=(raw.get("label") or "").strip(),
                    url=(raw.get("url") or "").strip(),
                    relative=(mode == TOLERANCE_RELATIVE),
                )
            )
    return rows


def expected_rows(name: str, *, metric: str | None = None, **axes: Any) -> list[ReferenceRow]:
    """Reference rows for ``name`` matching ``axes`` (and ``metric`` if given)."""
    rows = [r for r in load_reference(name) if r.matches(axes)]
    if metric is not None:
        rows = [r for r in rows if r.metric == metric]
    return rows


# --- results tables (reproduced measurements) ----------------------------------------

# The fixed, non-key columns every results table carries, in canonical order. Everything
# LEFT of ``metric`` is a key column — the SAME keys as the benchmark's reference table, so
# a results row joins its reference row at ``key`` + ``metric``. Unlike a reference row a
# results row records no expected/tolerance: it is *what soma got*, not the target.
_RESULT_VALUE_COLUMNS = (
    "metric",
    "measured",
    "std",
    "n_seeds",
    "date",
    "soma_commit",
    "slide2vec_version",
    "croma_version",
    "source",
)
# The minimum a results row must carry to be meaningful (a measurement for a metric).
_RESULT_REQUIRED_COLUMNS = ("metric", "measured")


@dataclass(frozen=True)
class MeasuredRow:
    """One reproduced measurement — soma's OWN produced number, with provenance.

    Mirrors :class:`ReferenceRow`'s key convention (``key`` holds only the *populated* key
    cells) so a measured row joins the reference row at the same ``key`` + ``metric``.
    For a single re-scored run (``--from-run-dir``), ``std`` is ``None`` and ``n_seeds`` is
    ``1``. The provenance fields pin the environment that produced the number — a reproduced
    value is only meaningful with the code and feature-extractor that made it.
    Representation benchmarks additionally record their runtime ``croma_version``.
    ``source`` is a free-text note (mirrors ``reference``'s ``source`` column).
    """

    key: dict[str, str]
    metric: str
    measured: float
    std: float | None = None
    n_seeds: int | None = None
    date: str = ""
    soma_commit: str = ""
    slide2vec_version: str = ""
    croma_version: str = ""
    source: str = ""

    def matches(self, axes: dict[str, Any]) -> bool:
        """True if every populated key cell equals the matching axis (banner matches all)."""
        return all(str(axes.get(col)) == val for col, val in self.key.items())


def _results_file(name: str) -> Path:
    """Filesystem path to ``results/<name>.csv`` (may not exist yet — writable in a checkout)."""
    return Path(str(resources.files("soma.benchmarks.results").joinpath(f"{name}.csv")))


def _resolve_results_file(name: str, results_root: str | Path | None) -> Path:
    """The ledger file for ``name``: ``<results_root>/<name>.csv``, or the in-package table.

    ``results_root`` is the seam that lets an external repository host its own committed
    ledger (issue #370): the soma checkout stops being the only place a reproduced number
    can land, while the schema and append-only discipline stay identical.
    """
    if results_root is None:
        return _results_file(name)
    return Path(results_root) / f"{name}.csv"


def load_results(
    name: str, *, results_root: str | Path | None = None
) -> list[MeasuredRow]:
    """Load ``results/<name>.csv`` into :class:`MeasuredRow` objects (``[]`` if absent).

    Unlike :func:`load_reference`, a missing file is **not** an error: a benchmark may carry
    a reference band with no reproduced measurement recorded yet. Every column left of
    ``metric`` is a key column (same convention as the reference table); blank ``std`` /
    ``n_seeds`` cells become ``None``. ``results_root`` reads a ledger hosted outside the
    soma checkout (``<results_root>/<name>.csv``) instead of the in-package table.
    """
    path = _resolve_results_file(name, results_root)
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in _RESULT_REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"results/{name}.csv is missing required column(s) {missing}; got {fieldnames}."
            )
        value_columns = set(_RESULT_VALUE_COLUMNS)
        key_columns = [c for c in fieldnames if c not in value_columns]
        rows: list[MeasuredRow] = []
        for raw in reader:
            key = {col: raw[col] for col in key_columns if (raw.get(col) or "").strip()}
            std_cell = (raw.get("std") or "").strip()
            n_cell = (raw.get("n_seeds") or "").strip()
            rows.append(
                MeasuredRow(
                    key=key,
                    metric=raw["metric"].strip(),
                    measured=float(raw["measured"]),
                    std=float(std_cell) if std_cell else None,
                    n_seeds=int(n_cell) if n_cell else None,
                    date=(raw.get("date") or "").strip(),
                    soma_commit=(raw.get("soma_commit") or "").strip(),
                    slide2vec_version=(raw.get("slide2vec_version") or "").strip(),
                    croma_version=(raw.get("croma_version") or "").strip(),
                    source=(raw.get("source") or "").strip(),
                )
            )
    return rows


def reproduced_rows(
    name: str,
    *,
    metric: str | None = None,
    results_root: str | Path | None = None,
    **axes: Any,
) -> list[MeasuredRow]:
    """Measured rows for ``name`` matching ``axes`` (and ``metric`` if given), in file order.

    The results table is an append-only ledger, so several rows may share a key (the same
    cell reproduced at different commits). Rows keep file order — callers wanting the latest
    reproduction of a cell take the last match. ``results_root`` selects a ledger hosted
    outside the soma checkout (see :func:`load_results`).
    """
    rows = [r for r in load_results(name, results_root=results_root) if r.matches(axes)]
    if metric is not None:
        rows = [r for r in rows if r.metric == metric]
    return rows


def _format_measure(value: float | None) -> str:
    """Serialise a measurement/std without erasing rank-relevant precision."""
    return "" if value is None else f"{value:.12g}"


def append_result(
    name: str,
    row: MeasuredRow,
    *,
    key_order: list[str] | None = None,
    results_root: str | Path | None = None,
) -> Path:
    """Append ``row`` to ``results/<name>.csv`` (created with a header if absent). Returns path.

    Append-only: existing rows are never rewritten, so a cell reproduced at a new commit
    adds a row rather than overwriting history. On a fresh file the header is the row's key
    columns (``key_order`` if given, else the row's key insertion order) followed by the
    canonical value columns; an existing file reuses its own header verbatim.
    ``results_root`` appends to a ledger hosted outside the soma checkout
    (``<results_root>/<name>.csv``) instead of the in-package table.
    """
    path = _resolve_results_file(name, results_root)
    columns = list(key_order) if key_order is not None else list(row.key)
    header = columns + list(_RESULT_VALUE_COLUMNS)
    exists = path.is_file()
    if exists:
        with path.open(newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing:
            header = existing
    record = {
        **row.key,
        "metric": row.metric,
        "measured": _format_measure(row.measured),
        "std": _format_measure(row.std),
        "n_seeds": "" if row.n_seeds is None else str(row.n_seeds),
        "date": row.date,
        "soma_commit": row.soma_commit,
        "slide2vec_version": row.slide2vec_version,
        "croma_version": row.croma_version,
        "source": row.source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({col: record.get(col, "") for col in header})
    return path


# --- default scorer ------------------------------------------------------------------


def score_from_summary(run_dir: str | Path) -> dict[str, float]:
    """DEFAULT scorer: read a Run's ``summary.json`` into ``{metric: value}``.

    ``run_dir`` may be the run directory itself or an ``output_root`` above it; the newest
    ``summary.json`` on disk is used. Benchmarks that score specially (OCELOT's greedy
    matcher) override ``score`` instead of calling this.
    """
    run_dir = Path(run_dir)
    direct = run_dir / "summary.json"
    if direct.is_file():
        summary_path = direct
    else:
        candidates = sorted(
            run_dir.glob("**/summary.json"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError(f"no summary.json found under {run_dir}")
        summary_path = candidates[-1]
    data = json.loads(summary_path.read_text())
    return {str(k): float(v) for k, v in data.items()}
