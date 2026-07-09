"""First-class, registered foundation-model benchmarks (ADR 0002).

A published :class:`~soma.benchmarks.registry.Benchmark` is a named entry in a registry
that ships *inside* the ``soma`` wheel — not a folder convention under ``examples/``.
Importing this package registers every bundled benchmark, so ``soma list benchmarks`` and
``soma reproduce <name>`` drive the registry directly.

This package registers ``ocelot`` (OCELOT 2023 cell detection), the ``eva/<dataset>``
family (kaiko-ai/eva patch classification, one sub-benchmark per dataset), and
``detection-benchmark`` (the multi-dataset encoder-ranking harness, issue #246).
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

__all__ = [
    "Benchmark",
    "Facet",
    "MeasuredRow",
    "ReferenceRow",
    "append_result",
    "expected_rows",
    "get_benchmark",
    "list_benchmarks",
    "load_reference",
    "load_results",
    "register_benchmark",
    "reproduced_rows",
    "score_from_summary",
]
