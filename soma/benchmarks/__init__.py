"""First-class, registered foundation-model benchmarks (ADR 0002).

A published :class:`~soma.benchmarks.registry.Benchmark` is a named entry in a registry
that ships *inside* the ``soma`` wheel — not a folder convention under ``examples/``.
Importing this package registers every bundled benchmark, so ``soma list benchmarks`` and
``soma reproduce <name>`` drive the registry directly.

This slice registers a single benchmark, ``ocelot`` (OCELOT 2023 cell detection). EVA
promotion is a separate follow-up (#219).
"""

from __future__ import annotations

from soma.benchmarks.registry import (
    Benchmark,
    Facet,
    ReferenceRow,
    expected_rows,
    get_benchmark,
    list_benchmarks,
    load_reference,
    register_benchmark,
    score_from_summary,
)

# Import for side effect: registers the OCELOT benchmark under name "ocelot".
from soma.benchmarks import ocelot as _ocelot  # noqa: F401

__all__ = [
    "Benchmark",
    "Facet",
    "ReferenceRow",
    "expected_rows",
    "get_benchmark",
    "list_benchmarks",
    "load_reference",
    "register_benchmark",
    "score_from_summary",
]
