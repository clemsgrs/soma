"""First-class, registered foundation-model benchmarks (ADR 0002).

A published :class:`~soma.benchmarks.registry.Benchmark` is a named entry in a registry
that ships *inside* the ``soma`` wheel — not a folder convention under ``examples/``.
Importing this package registers every bundled benchmark, so ``soma list benchmarks`` and
``soma reproduce <name>`` drive the registry directly.

This package registers ``ocelot`` (OCELOT 2023 cell detection), the ``eva/<dataset>``
family (kaiko-ai/eva patch classification, one sub-benchmark per dataset),
``detection-benchmark`` (the multi-dataset encoder-ranking harness, issue #246), and
the HEST and CRoMa benchmark families.
"""

from __future__ import annotations

from soma.benchmarks.registry import (
    Benchmark,
    Facet,
    MeasuredRow,
    ReferenceRow,
    append_result,
    expected_rows,
    get_benchmark,
    get_ranking_metrics,
    get_reported_metrics,
    list_benchmarks,
    load_reference,
    load_results,
    register_benchmark,
    reproduced_rows,
    score_from_summary,
)

# Import for side effect: registers the bundled benchmarks (ocelot, the eva/<dataset>
# family) into the registry.
from soma.benchmarks import ocelot as _ocelot  # noqa: F401
from soma.benchmarks import eva as _eva  # noqa: F401
from soma.benchmarks import detection_benchmark as _detection_benchmark  # noqa: F401
from soma.benchmarks import hest as _hest  # noqa: F401
from soma.benchmarks import croma as _croma  # noqa: F401

# Reproduction report (joins results/<name>.csv to reference/<name>.csv; imported after the
# benchmarks register so it can read their primary metrics).
from soma.benchmarks.reproduction import (  # noqa: E402
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
from soma.benchmarks.run import (  # noqa: E402
    ReportedScoreError,
    run_benchmark,
)

__all__ = [
    "RESOLVABLE_EPS",
    "Benchmark",
    "Cell",
    "Facet",
    "MeasuredRow",
    "PairOutcome",
    "ReferenceRow",
    "ReportedScoreError",
    "ReproductionReport",
    "ResolvabilityPolicy",
    "append_result",
    "expected_rows",
    "get_benchmark",
    "get_ranking_metrics",
    "get_reported_metrics",
    "list_benchmarks",
    "load_reference",
    "load_results",
    "register_benchmark",
    "reproduced_rows",
    "reproduction_report",
    "run_benchmark",
    "score_from_summary",
]
