"""``soma.benchmark``: the registered foundation-model benchmarks (ADR 0002, ADR 0022).

A Built-in :class:`~soma.benchmark.registry.Benchmark` is a named entry in the
``benchmarks`` registry, shipped inside the ``soma`` wheel but outside the core: ``import
soma`` does not import this package. Importing it registers its components, like any
user package: the benchmarks themselves, and the benchmark-private heads, decoders,
aggregators and learners their recipes name. The ``soma`` CLI imports it at startup; from
Python, ``import soma.benchmark`` first or name a component by its dotted path.

The root holds the generic mechanism (registry, reproduction, run, leaderboard, curation
helpers, the dataset-agnostic detection harness); each benchmark protocol is one package
(``eva``, ``hest``, ``ocelot``, ``midog``, ``monkey``, ``croma``) holding its curator,
recipes, scorer, ``reference.csv`` and ``results.csv``. The protocol packages never import
each other.
"""

from __future__ import annotations

from soma.benchmark.registry import (
    Benchmark,
    Facet,
    MeasuredRow,
    ReferenceRow,
    append_result,
    benchmark_registry,
    expected_rows,
    get_benchmark,
    get_ranking_metrics,
    get_reported_metrics,
    load_reference,
    load_results,
    register_benchmark,
    reproduced_rows,
    score_from_summary,
)

# Import for side effect: each protocol registers its benchmarks (and its private
# components) at import. The detection protocols register their datasets with the
# detection harness in this order (ocelot, midog, monkey), which is the order its
# ranking report lists them in.
from soma.benchmark import eva as _eva  # noqa: F401
from soma.benchmark.eva import segmentation as _eva_segmentation  # noqa: F401
from soma.benchmark.eva import slide as _eva_slide  # noqa: F401
from soma.benchmark import hest as _hest  # noqa: F401
from soma.benchmark import ocelot as _ocelot  # noqa: F401
from soma.benchmark import midog as _midog  # noqa: F401
from soma.benchmark import monkey as _monkey  # noqa: F401
from soma.benchmark import croma as _croma  # noqa: F401
# The decoder ladder's rung 0 (``attention_map``), shared by the detection protocols.
from soma.benchmark import attention as _attention  # noqa: F401

# Reproduction report (joins <name>/results.csv to <name>/reference.csv; imported after
# the benchmarks register so it can read their primary metrics).
from soma.benchmark.reproduction import (  # noqa: E402
    RESOLVABLE_EPS,
    Cell,
    PairOutcome,
    ReproductionReport,
    ResolvabilityPolicy,
    reproduction_report,
)

# Importable reproduce orchestration (issue #370): the same seed-loop / reference /
# provenance / record guarantees `soma reproduce` provides, callable as an API — with a
# parameterizable results root so an external repo can host its own committed ledger.
from soma.benchmark.run import (  # noqa: E402
    BenchmarkRunResult,
    MetricResult,
    ReportedScoreError,
    run_benchmark,
    run_benchmark_spec,
)
from soma.benchmark.spec import BenchmarkSpec  # noqa: E402

__all__ = [
    "RESOLVABLE_EPS",
    "Benchmark",
    "BenchmarkRunResult",
    "BenchmarkSpec",
    "Cell",
    "Facet",
    "MeasuredRow",
    "MetricResult",
    "PairOutcome",
    "ReferenceRow",
    "ReportedScoreError",
    "ReproductionReport",
    "ResolvabilityPolicy",
    "append_result",
    "benchmark_registry",
    "expected_rows",
    "get_benchmark",
    "get_ranking_metrics",
    "get_reported_metrics",
    "load_reference",
    "load_results",
    "register_benchmark",
    "reproduced_rows",
    "reproduction_report",
    "run_benchmark",
    "run_benchmark_spec",
    "score_from_summary",
]
