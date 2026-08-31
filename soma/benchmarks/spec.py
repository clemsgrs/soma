"""Typed benchmark specifications owned by external projects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soma.config import PipelineConfig


BenchmarkConfigBuilder = Callable[..., PipelineConfig]
BenchmarkScorer = Callable[[str | Path], dict[str, float]]


@dataclass(frozen=True)
class BenchmarkSpec:
    """An external benchmark's executable protocol, independent of Soma's registry."""

    name: str
    canonical_seeds: tuple[int, ...]
    primary_metric: str
    build_config: BenchmarkConfigBuilder
    score: BenchmarkScorer
    reported_metrics: tuple[str, ...] | None = None
    ranking_metrics: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Resolve optional metric declarations and validate the public protocol."""
        from soma.benchmarks.registry import get_ranking_metrics, get_reported_metrics

        if not self.canonical_seeds or len(set(self.canonical_seeds)) != len(
            self.canonical_seeds
        ):
            raise ValueError("canonical_seeds must be non-empty and unique")

        if self.reported_metrics is None:
            object.__setattr__(self, "reported_metrics", (self.primary_metric,))
        if self.ranking_metrics is None:
            object.__setattr__(self, "ranking_metrics", self.reported_metrics)

        object.__setattr__(self, "reported_metrics", get_reported_metrics(self))
        object.__setattr__(self, "ranking_metrics", get_ranking_metrics(self))
