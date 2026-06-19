"""Reproducible recipes for public computational-pathology benchmarks.

Each submodule encodes the *protocol* of one published benchmark (optimizer,
schedule, metric, per-dataset hyper-parameters, expected leaderboard numbers) as
small, testable building blocks. Thin scripts under ``scripts/`` orchestrate
curation + training on top of them so a soma user can reproduce a benchmark cell
with a single command.

See :mod:`soma.benchmarks.eva` for the kaiko-ai/eva patch-level benchmark.
"""

from soma.benchmarks import eva

__all__ = ["eva"]
