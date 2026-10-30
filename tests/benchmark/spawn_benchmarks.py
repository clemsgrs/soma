"""Benchmark specs and recipes that a seed's worker process can import.

``devices`` runs each seed after the first in a new ``spawn`` process, which receives the
spec or benchmark pickled. Pickle stores a function or a class by its module and name, so
everything here lives at module level: a worker imports this module to rebuild it, and
sees none of a test's monkeypatches or local functions.

The tile protocol is the real ``Pipeline`` on CPU: soma.testing's weight-free ``fake``
encoder over the synthetic tiles of ``tests.e2e.synthetic.make_tile_dataset``, then a
binary-classification head trained for a few epochs. The trainer is pinned to ``cpu``, so
a sequential run trains where the ``cpu`` workers do, also on a machine with a GPU.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from slide2vec import ExecutionOptions

from soma import ExperimentConfig
from soma.benchmark import BenchmarkSpec, score_from_summary
from soma.benchmark.curation import CuratedManifest
from soma.benchmark.registry import Facet
from soma.config import (
    CacheOptions,
    CohortConfig,
    EvaluationConfig,
    FeaturesConfig,
    LearnerConfig,
    LoadersConfig,
    RunConfig,
    TrainerConfig,
)

__all__ = [
    "CACHE_SNAPSHOT",
    "MEAN_PROBABILITY",
    "PRIMARY_METRIC",
    "QuickBenchmark",
    "TileBenchmark",
    "UnpicklableError",
    "UnrebuildableError",
    "build_quick_config",
    "build_tile_config",
    "cache_snapshot",
    "quick_spec",
    "run_quick",
    "score_and_snapshot_cache",
    "score_quick",
    "score_tile_run",
    "tile_spec",
]

PRIMARY_METRIC = "test/auroc"
#: The mean test probability of the positive class, which moves with every weight.
MEAN_PROBABILITY = "test/mean_prob_1"
#: The file, in a seed's root, that :func:`score_and_snapshot_cache` writes.
CACHE_SNAPSHOT = "cache_snapshot.json"


def build_tile_config(
    *,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    seed: int,
    overrides: dict[str, Any],
    encoder: str,
) -> ExperimentConfig:
    """A tile-classification config: each image is one sample, embedded whole."""
    return ExperimentConfig(
        run=RunConfig(output_dir=Path(output_root), seed=seed),
        cohort=CohortConfig(records=dataset_csv, splits=splits_csv),
        features=FeaturesConfig(
            encoder=encoder,
            cache=replace(CacheOptions(), **overrides.get("cache", {})),
            execution=ExecutionOptions(
                batch_size=16,
                num_gpus=1,
                num_workers_per_gpu=0,
                num_preprocessing_workers=0,
                precision="fp32",
            ),
        ),
        loaders=LoadersConfig(batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False),
        learner=LearnerConfig(
            "mil",
            {
                "head": "binary_classification",
                "optimizer": {"name": "adam", "params": {"lr": 5e-3}},
            },
        ),
        trainer=TrainerConfig(max_epochs=4, device="cpu"),
        evaluation=EvaluationConfig(metrics=["auroc", "balanced_accuracy"]),
    )


def score_tile_run(run_dir: str | Path) -> dict[str, float]:
    """The run's summary, plus the mean test probability of the positive class.

    AUROC saturates on this easy task; the mean probability moves with every weight, so
    two runs agree on it only if they trained the same model.
    """
    import pandas as pd

    scores = score_from_summary(run_dir)
    predictions = sorted(Path(run_dir).glob("**/predictions_test.parquet"))[-1]
    scores[MEAN_PROBABILITY] = float(pd.read_parquet(predictions)["prob_1"].mean())
    return scores


def score_and_snapshot_cache(run_dir: str | Path) -> dict[str, float]:
    """Score a seed, and record the shared feature cache as it is at that moment.

    The default cache root is ``<output_root>/feature_cache``, beside the seed roots.
    """
    seed_root = Path(run_dir)
    snapshot = cache_snapshot(seed_root.parent / "feature_cache")
    (seed_root / CACHE_SNAPSHOT).write_text(json.dumps(snapshot, sort_keys=True))
    return score_tile_run(seed_root)


def cache_snapshot(cache_root: Path) -> dict[str, list[int]]:
    """Every file under ``cache_root``: its inode, size and modification time."""
    return {
        path.relative_to(cache_root).as_posix(): [
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ]
        for path in sorted(cache_root.rglob("*"))
        if path.is_file()
    }


def tile_spec(seeds: tuple[int, ...] = (0, 1, 2)) -> BenchmarkSpec:
    """The tile protocol as an external spec, scored off each seed's summary."""
    return BenchmarkSpec(
        name="parallel/tiles",
        canonical_seeds=seeds,
        primary_metric=PRIMARY_METRIC,
        reported_metrics=(PRIMARY_METRIC, MEAN_PROBABILITY),
        build_config=build_tile_config,
        score=score_and_snapshot_cache,
    )


class TileBenchmark:
    """The tile protocol as a registered benchmark, for ``run_benchmark``."""

    name = "parallel_tiles_fixture"
    facet = Facet(fixed={}, varied=("encoder",))
    canonical_seeds = (0, 1, 2)
    primary_metric = PRIMARY_METRIC
    reported_metrics = (PRIMARY_METRIC, MEAN_PROBABILITY)
    reference_environment: dict[str, str] = {}

    def curate(self, raw_root, out_dir):  # pragma: no cover - the tests pass curated_dir
        raise NotImplementedError

    def build_config(self, *, encoder: str = "fake", **kwargs: Any) -> ExperimentConfig:
        return build_tile_config(encoder=encoder, **kwargs)

    def expected(self, **axes: Any) -> list:
        return []

    def score(self, run_dir: str | Path) -> dict[str, float]:
        return score_tile_run(run_dir)


# --- a quick protocol: no Pipeline, for the failure paths -----------------------------


class UnpicklableError(RuntimeError):
    """An error that cannot be pickled: it holds a lock."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.lock = threading.Lock()


class UnrebuildableError(RuntimeError):
    """An error that pickles but cannot be rebuilt: its constructor takes two arguments."""

    def __init__(self, seed: int, reason: str) -> None:
        super().__init__(f"seed {seed}: {reason}")


def build_quick_config(*, actions: dict[int, str] | None = None, **kwargs: Any) -> dict[str, Any]:
    """The quick protocol's config: its keywords, and what each seed does (``actions``)."""
    return {"actions": dict(actions or {}), **kwargs}


def run_quick(config: dict[str, Any]) -> SimpleNamespace:
    """Run one seed of the quick protocol: record its process and the GPUs it may see
    (``CUDA_VISIBLE_DEVICES``), then act on its action.

    ``config["actions"]`` maps a seed to what it does instead of succeeding: ``raise`` a
    ``ValueError``, ``die`` (kill its own process), raise an ``unpicklable`` or an
    ``unrebuildable`` error, finish ``slow`` (after 8 seconds), or ``hang`` (sleep for a
    minute).
    """
    seed_root = Path(config["output_root"])
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "pid").write_text(str(os.getpid()))
    (seed_root / "visible").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    action = config["actions"].get(config["seed"])
    if action == "raise":
        raise ValueError(f"seed {config['seed']} failed")
    if action == "die":
        os.kill(os.getpid(), signal.SIGKILL)
    if action == "unpicklable":
        raise UnpicklableError(f"seed {config['seed']} failed")
    if action == "unrebuildable":
        raise UnrebuildableError(config["seed"], "failed")
    if action == "slow":
        time.sleep(8)
    if action == "hang":
        (seed_root / "hanging").write_text("")
        time.sleep(60)
    print(f"seed {config['seed']} ran in process {os.getpid()}")
    return SimpleNamespace(dataset_checksum="d", splits_checksum="s", test_checksum="t")


def score_quick(run_dir: str | Path) -> dict[str, float]:
    """Score a quick seed: its seed number, so the mean is known in advance."""
    return {"metric/a": float(Path(run_dir).name.removeprefix("seed_"))}


def quick_spec(seeds: tuple[int, ...] = (0, 1, 2), actions: dict[int, str] | None = None) -> BenchmarkSpec:
    """The quick protocol as an external spec, run by :func:`run_quick` (no ``Pipeline``).

    ``build_config`` is a ``partial`` of a module-level function, which pickles.
    """
    return BenchmarkSpec(
        name="parallel/quick",
        canonical_seeds=seeds,
        primary_metric="metric/a",
        build_config=partial(build_quick_config, actions=actions),
        score=score_quick,
        run=run_quick,
    )


class QuickBenchmark:
    """A registered benchmark whose own ``run`` stands in for the ``Pipeline``.

    Its configs are real tile configs, so ``soma reproduce --encoders`` can check them.
    ``missing_metric_seeds`` names the seeds whose score leaves the Reported metric out;
    ``failing`` maps an encoder to the seed that raises in its cell.
    """

    name = "parallel_quick_fixture"
    facet = Facet(fixed={}, varied=("encoder",))
    canonical_seeds = (0, 1, 2)
    primary_metric = "metric/a"
    reference_environment: dict[str, str] = {}

    def __init__(self, *, missing_metric_seeds=(), failing=None) -> None:
        self.missing_metric_seeds = tuple(missing_metric_seeds)
        self.failing = dict(failing or {})

    def curate(self, raw_root, out_dir) -> CuratedManifest:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return CuratedManifest(dataset_csv=out_dir / "dataset.csv", splits_csv=out_dir / "splits.csv")

    def build_config(self, *, encoder: str = "fake", **kwargs: Any) -> ExperimentConfig:
        return build_tile_config(encoder=encoder, **kwargs)

    def run(self, config: ExperimentConfig) -> None:
        encoder, seed = config.features.encoder.name, config.run.seed
        seed_root = Path(config.run.output_dir)
        seed_root.mkdir(parents=True, exist_ok=True)
        (seed_root / f"ran-{encoder}").write_text(str(os.getpid()))
        if self.failing.get(encoder) == seed:
            raise RuntimeError(f"{encoder} seed {seed} failed")

    def expected(self, **axes: Any) -> list:
        return []

    def score(self, run_dir: str | Path) -> dict[str, float]:
        seed = int(Path(run_dir).name.removeprefix("seed_"))
        if seed in self.missing_metric_seeds:
            return {}
        return {"metric/a": float(seed)}
