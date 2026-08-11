"""Reproduction report — how faithfully soma's native pipeline reproduces a benchmark.

Joins soma's recorded measurements (``results/<name>.csv``) to the benchmark's external
reference (``reference/<name>.csv``) and quantifies reproduction **soundness** along three
axes (agreed in the HEST reproduction design grill):

* **A — absolute agreement (what is published).** Per ``(dataset, encoder)`` cell: soma's
  Measured value, the published Reference, and the signed delta. Rendered, never gated —
  soma re-extracts features with its own stack (slide2vec) rather than the benchmark's
  original tooling (HEST uses TRIDENT), so the delta is a cross-stack parity gap and a gate
  against another lab's extractor would fire on that rather than on a real regression
  (ADR 0005). Publish it and let the reader compare.

* **B — rank agreement (a bonus).** A foundation-model benchmark exists to *rank* encoders,
  and a ranking survives an extraction-stack change even when absolute values shift. We
  measure **pooled pairwise concordance**: over every ``(dataset, encoder-pair)``, the
  fraction where soma orders the pair the same way the reference does. A pair is
  **resolvable** when the reference gap exceeds :data:`RESOLVABLE_EPS`; concordance is over
  resolvable pairs only, so soma is not graded on within-noise coin-flips (a pair the
  benchmark itself cannot separate). Per-dataset Spearman ρ is reported alongside (coarse
  at few encoders). Corroborates A rather than replacing it.

* **C — drift guard.** ``results/<name>.csv`` is an append-only, provenance-pinned ledger
  (``soma_commit`` / ``slide2vec_version`` / ``date`` per row), so re-running a cell at a
  new commit adds a row and any drift is an explicit diff rather than a silent overwrite.

Pure over the two packaged CSVs — no runs, no network, no heavy deps — so the generated
docs and the tests can both call it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from soma.benchmarks.registry import (
    MeasuredRow,
    ReferenceRow,
    get_benchmark,
    get_ranking_metrics,
    list_benchmarks,
    load_reference,
    load_results,
)

# An encoder pair is "resolvable" when the published reference separates it by more than this
# (absolute, on the primary metric). Below it, the benchmark itself cannot call the ordering,
# so a soma flip is a coin-flip, not a defect — excluded from the concordance.
RESOLVABLE_EPS = 0.005

# The two key columns every family in scope keys on (dataset × encoder grid).
_DATASET_KEY = "dataset"
_ENCODER_KEY = "encoder"


@dataclass(frozen=True)
class Cell:
    """One reproduced ``(dataset, encoder)`` cell: soma Measured vs published Reference (A)."""

    dataset: str
    encoder: str
    measured: float
    reference: float
    n_seeds: int | None = None
    std: float | None = None
    soma_commit: str = ""
    slide2vec_version: str = ""
    date: str = ""

    @property
    def delta(self) -> float:
        """Signed Measured − Reference (soma minus published)."""
        return self.measured - self.reference


@dataclass(frozen=True)
class PairOutcome:
    """One ordered-by-reference encoder pair within a dataset, for rank concordance (B)."""

    dataset: str
    encoder_high: str  # the encoder the REFERENCE ranks higher
    encoder_low: str
    reference_gap: float  # reference[high] − reference[low]  (> 0 by construction)
    measured_gap: float  # soma measured[high] − measured[low]

    @property
    def resolvable(self) -> bool:
        """True when the reference separates the pair by more than :data:`RESOLVABLE_EPS`."""
        return self.reference_gap > RESOLVABLE_EPS

    @property
    def concordant(self) -> bool:
        """True when soma preserves the reference ordering (measured gap same sign, non-zero)."""
        return self.measured_gap > 0


@dataclass(frozen=True)
class ReproductionReport:
    """The full A/B/C picture for one benchmark family, ready for docs/tests to render."""

    name: str
    metric: str
    cells: list[Cell] = field(default_factory=list)
    pairs: list[PairOutcome] = field(default_factory=list)
    spearman_by_dataset: dict[str, float | None] = field(default_factory=dict)

    # --- B: pooled pairwise concordance --------------------------------------------------
    @property
    def n_resolvable(self) -> int:
        return sum(1 for p in self.pairs if p.resolvable)

    @property
    def n_resolvable_concordant(self) -> int:
        return sum(1 for p in self.pairs if p.resolvable and p.concordant)

    @property
    def concordance_resolvable(self) -> float | None:
        """Headline B: fraction of *resolvable* pairs soma orders like the reference."""
        n = self.n_resolvable
        return self.n_resolvable_concordant / n if n else None

    @property
    def concordance_all(self) -> float | None:
        """Concordance over *all* pairs (resolvable + within-noise), reported alongside."""
        if not self.pairs:
            return None
        return sum(1 for p in self.pairs if p.concordant) / len(self.pairs)

    # --- provenance summary (C) ----------------------------------------------------------
    @property
    def soma_commits(self) -> list[str]:
        return sorted({c.soma_commit for c in self.cells if c.soma_commit})

    @property
    def slide2vec_versions(self) -> list[str]:
        return sorted({c.slide2vec_version for c in self.cells if c.slide2vec_version})

    @property
    def datasets(self) -> list[str]:
        """Datasets with at least one reproduced cell, in first-seen order."""
        seen: list[str] = []
        for cell in self.cells:
            if cell.dataset not in seen:
                seen.append(cell.dataset)
        return seen

    @property
    def encoders(self) -> list[str]:
        """Encoders with at least one reproduced cell, in first-seen order."""
        seen: list[str] = []
        for cell in self.cells:
            if cell.encoder not in seen:
                seen.append(cell.encoder)
        return seen


def _primary_metric_for(name: str) -> str:
    """The primary metric of the family ``name`` (read off any registered member)."""
    for registered in list_benchmarks():
        if registered == name or registered.startswith(f"{name}/"):
            return get_benchmark(registered).primary_metric
    raise KeyError(f"No registered benchmark for family {name!r}.")


def _reference_value(rows: list[ReferenceRow], metric: str) -> dict[tuple[str, str], float]:
    """Map ``(dataset, encoder) -> reference value`` for ``metric`` (gate wins over external).

    A cell may carry a gate row (the tolerance target) and/or an external row (a published
    third-party number). The reproduction reference is the gate expected when present, else
    the external expected — the value soma is being compared against.
    """
    gate: dict[tuple[str, str], float] = {}
    external: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.metric != metric:
            continue
        dataset = row.key.get(_DATASET_KEY)
        encoder = row.key.get(_ENCODER_KEY)
        if dataset is None or encoder is None:
            continue  # a broad banner / differently-keyed row is not a cell reference
        (external if row.is_external else gate)[(dataset, encoder)] = row.expected
    return {**external, **gate}  # gate overrides external for the same cell


def _latest_measurements(rows: list[MeasuredRow], metric: str) -> dict[tuple[str, str], MeasuredRow]:
    """Map ``(dataset, encoder) -> latest MeasuredRow`` for ``metric`` (append-only: last wins)."""
    latest: dict[tuple[str, str], MeasuredRow] = {}
    for row in rows:
        if row.metric != metric:
            continue
        dataset = row.key.get(_DATASET_KEY)
        encoder = row.key.get(_ENCODER_KEY)
        if dataset is None or encoder is None:
            continue
        latest[(dataset, encoder)] = row  # later rows overwrite earlier → newest kept
    return latest


def _ranking_cells(
    name: str, dataset: str, metric: str, cells: list[Cell]
) -> list[Cell]:
    """Cells eligible for rank agreement; controls remain in the absolute table."""
    member_name = f"{name}/{dataset}"
    registered = set(list_benchmarks())
    if member_name in registered:
        benchmark = get_benchmark(member_name)
    elif name in registered:
        benchmark = get_benchmark(name)
    else:
        return cells
    if metric not in get_ranking_metrics(benchmark):
        return []
    eligibility = getattr(benchmark, "is_ranking_eligible", None)
    if not callable(eligibility):
        return cells
    return [
        cell
        for cell in cells
        if eligibility(dataset=dataset, encoder=cell.encoder)
    ]


def _rankdata(values: list[float]) -> list[float]:
    """Average-tie ranks (1-based), matching ``scipy.stats.rankdata`` without the dependency."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # average of the tied positions, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman ρ of aligned ``xs``/``ys`` (Pearson of ranks); ``None`` if undefined."""
    if len(xs) < 2:
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:  # a constant ranking (all tied) has no correlation
        return None
    return sxy / (sxx * syy) ** 0.5


def reproduction_report(name: str, *, metric: str | None = None) -> ReproductionReport:
    """Build the A/B/C :class:`ReproductionReport` for benchmark family ``name``.

    Joins ``reference/<name>.csv`` (published reference) to ``results/<name>.csv`` (soma's
    recorded measurements, latest per cell) on ``(dataset, encoder)`` + ``metric``. ``metric``
    defaults to the family's primary metric. Cells with no recorded measurement are omitted,
    so the report grows as ``soma reproduce --record`` fills the ledger.
    """
    metric = metric or _primary_metric_for(name)
    reference = _reference_value(load_reference(name), metric)
    measurements = _latest_measurements(load_results(name), metric)

    cells: list[Cell] = []
    for (dataset, encoder), row in measurements.items():
        ref = reference.get((dataset, encoder))
        if ref is None:  # no published reference to compare against → not a scored cell
            continue
        cells.append(
            Cell(
                dataset=dataset,
                encoder=encoder,
                measured=row.measured,
                reference=ref,
                n_seeds=row.n_seeds,
                std=row.std,
                soma_commit=row.soma_commit,
                slide2vec_version=row.slide2vec_version,
                date=row.date,
            )
        )

    # Deterministic order: dataset first-seen, then encoder first-seen — stable for docs/tests.
    by_cell = {(c.dataset, c.encoder): c for c in cells}
    datasets_seen: list[str] = []
    for c in cells:
        if c.dataset not in datasets_seen:
            datasets_seen.append(c.dataset)

    pairs: list[PairOutcome] = []
    spearman_by_dataset: dict[str, float | None] = {}
    for dataset in datasets_seen:
        cell_here = _ranking_cells(
            name, dataset, metric, [c for c in cells if c.dataset == dataset]
        )
        if len(cell_here) >= 2:
            spearman_by_dataset[dataset] = _spearman(
                [c.measured for c in cell_here], [c.reference for c in cell_here]
            )
        else:
            spearman_by_dataset[dataset] = None
        for a, b in combinations(cell_here, 2):
            ref_gap = a.reference - b.reference
            if ref_gap == 0:
                continue  # a tied reference has no ordering to reproduce
            high, low = (a, b) if ref_gap > 0 else (b, a)
            pairs.append(
                PairOutcome(
                    dataset=dataset,
                    encoder_high=high.encoder,
                    encoder_low=low.encoder,
                    reference_gap=high.reference - low.reference,
                    measured_gap=high.measured - low.measured,
                )
            )

    # Rebuild cells in the deterministic order for stable rendering.
    ordered_cells = [
        by_cell[(d, e)]
        for d in datasets_seen
        for e in _encoders_in_order(cells, d)
    ]
    return ReproductionReport(
        name=name,
        metric=metric,
        cells=ordered_cells,
        pairs=pairs,
        spearman_by_dataset=spearman_by_dataset,
    )


def _encoders_in_order(cells: list[Cell], dataset: str) -> list[str]:
    """Encoders present for ``dataset``, in first-seen order across the cell list."""
    seen: list[str] = []
    for c in cells:
        if c.dataset == dataset and c.encoder not in seen:
            seen.append(c.encoder)
    return seen
