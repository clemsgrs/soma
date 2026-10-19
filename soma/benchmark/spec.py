"""Typed benchmark specifications owned by external projects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from soma.config import ExperimentConfig


class BenchmarkConfigBuilder(Protocol):
    """The exact configuration-builder contract an external spec must provide."""

    def __call__(
        self,
        *,
        dataset_csv: str | Path,
        splits_csv: str | Path,
        output_root: str | Path,
        seed: int,
        overrides: dict[str, Any],
        encoder: str,
    ) -> ExperimentConfig: ...


BenchmarkScorer = Callable[[str | Path], dict[str, float]]


class SeedRunResult(Protocol):
    """What one seed's run returns: the manifest checksums it trained and scored against.

    A :class:`~soma.pipeline.PipelineResult` is one; a custom runner returns the same
    three values, as its run's ``experiment.json`` and ``run.yaml`` record them.
    """

    @property
    def dataset_checksum(self) -> str: ...

    @property
    def splits_checksum(self) -> str: ...

    @property
    def test_checksum(self) -> str: ...


BenchmarkRunner = Callable[[ExperimentConfig], SeedRunResult]


def _validate_seed_sequence(seeds: object, *, message: str) -> tuple[int, ...]:
    """Return one exact ordered seed tuple or fail with the caller's public message."""
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError(message)
    values = tuple(seeds)
    if (
        not values
        or not all(type(seed) is int for seed in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(message)
    return values


@dataclass(frozen=True)
class BenchmarkSpec:
    """An external benchmark's executable protocol, independent of Soma's registry.

    ``run`` is optional. Left ``None``, each seed's config runs as ``Pipeline(config).run()``.
    Set it for a protocol that no ``Pipeline`` trains, such as HEST's closed-form probe:
    :func:`~soma.benchmark.run_benchmark_spec` then calls ``run(config)`` for each seed
    instead, and reads the dataset, splits and test checksums from what it returns
    (:class:`SeedRunResult`). A spec that wraps a registered recipe passes
    ``run=getattr(recipe, "run", None)``.
    """

    name: str
    canonical_seeds: tuple[int, ...]
    primary_metric: str
    build_config: BenchmarkConfigBuilder
    score: BenchmarkScorer
    reported_metrics: tuple[str, ...] | None = None
    ranking_metrics: tuple[str, ...] | None = None
    run: BenchmarkRunner | None = None

    def __post_init__(self) -> None:
        """Resolve optional metric declarations and validate the public protocol."""
        from soma.benchmark.registry import get_ranking_metrics, get_reported_metrics

        if self.run is not None and not callable(self.run):
            raise TypeError(f"run must be callable or None, got {type(self.run).__name__}")
        object.__setattr__(
            self,
            "canonical_seeds",
            _validate_seed_sequence(
                self.canonical_seeds,
                message="canonical_seeds must be non-empty and unique integers",
            ),
        )

        if self.reported_metrics is None:
            object.__setattr__(self, "reported_metrics", (self.primary_metric,))
        if self.ranking_metrics is None:
            object.__setattr__(self, "ranking_metrics", self.reported_metrics)

        object.__setattr__(self, "reported_metrics", get_reported_metrics(self))
        object.__setattr__(self, "ranking_metrics", get_ranking_metrics(self))
