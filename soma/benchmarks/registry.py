"""Benchmark registry: ``name -> Benchmark`` plus the protocol-as-code interface.

A published **Benchmark** is a named, registered *code* object (ADR 0002) — not a folder
convention under ``examples/``. Registration makes ``soma list benchmarks`` and
``soma reproduce <name>`` structural: they enumerate and drive the registry directly, so a
benchmark cannot silently drift from a doc someone forgot to write.

A Benchmark is *thin*: it wires an existing curator, a committed config (or a builder), a
reference row, and a scorer together behind a uniform interface. The interface is
**structural** (:class:`Benchmark` is a ``Protocol``): any object exposing the right
attributes/methods conforms — there is no base class to subclass. Where a step is static
a benchmark just delegates (load a committed YAML, read ``summary.json``); where it
computes or scores specially it overrides (OCELOT's greedy matcher).

Expected numbers ship as package data at ``soma/benchmarks/reference/<name>.csv`` with
columns ``key…, metric, expected, tolerance, source`` and a **per-row** tolerance (absolute
on the primary metric). The ``key…`` columns are everything left of ``metric``; an empty
key cell means the row is a broad, config-agnostic banner that matches any axes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from soma.config import PipelineConfig
from soma.curation.manifest import CuratedManifest

# The fixed, non-key columns every reference table carries, in canonical order.
_REFERENCE_VALUE_COLUMNS = ("metric", "expected", "tolerance", "source")


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
    """

    key: dict[str, str]
    metric: str
    expected: float
    tolerance: float
    source: str

    def matches(self, axes: dict[str, Any]) -> bool:
        """True if every populated key cell equals the matching axis (banner matches all)."""
        return all(str(axes.get(col)) == val for col, val in self.key.items())

    def within_tolerance(self, measured: float) -> bool:
        return abs(measured - self.expected) <= self.tolerance


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
        key_columns = [c for c in fieldnames if c not in _REFERENCE_VALUE_COLUMNS]
        rows: list[ReferenceRow] = []
        for raw in reader:
            key = {col: raw[col] for col in key_columns if (raw.get(col) or "").strip()}
            rows.append(
                ReferenceRow(
                    key=key,
                    metric=raw["metric"].strip(),
                    expected=float(raw["expected"]),
                    tolerance=float(raw["tolerance"]),
                    source=(raw.get("source") or "").strip(),
                )
            )
    return rows


def expected_rows(name: str, *, metric: str | None = None, **axes: Any) -> list[ReferenceRow]:
    """Reference rows for ``name`` matching ``axes`` (and ``metric`` if given)."""
    rows = [r for r in load_reference(name) if r.matches(axes)]
    if metric is not None:
        rows = [r for r in rows if r.metric == metric]
    return rows


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
