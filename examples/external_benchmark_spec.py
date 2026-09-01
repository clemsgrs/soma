"""Run an external ``BenchmarkSpec`` end to end on synthetic data.

The external benchmark API (soma 1.12.0) lets a project outside soma own its
benchmark protocol as a typed object and hand it to soma for execution:

1. build a :class:`soma.benchmarks.BenchmarkSpec` — a config builder, the
   canonical seed set, and a scorer;
2. execute it with :func:`soma.benchmarks.run_benchmark_spec`, which runs every
   canonical seed with a shared feature cache;
3. read the aggregated :class:`soma.benchmarks.BenchmarkRunResult` and the
   per-seed evidence roots.

Self-contained: fabricates a tiny synthetic tile dataset, uses the ungated
``phikon`` tile encoder, and runs on CPU. The metric values are meaningless —
the execution contract is the point.

Run with no arguments:

    python examples/external_benchmark_spec.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

# Force CPU so the example runs anywhere, even on a machine whose GPUs are
# busy (delete this line to use your GPU).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from PIL import Image

from soma import (
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.benchmarks import BenchmarkSpec, run_benchmark_spec, score_from_summary

# --- Scaffolding (not soma API): a toy tile dataset -------------------------------
# One row per tile image; the on-disk contract is the same one every soma tile
# pipeline uses (dataset.csv: sample_id, image_path, label; splits.csv:
# sample_id, split).

WORK = Path(tempfile.mkdtemp(prefix="soma-external-spec-"))
TILES = WORK / "tiles"
TILES.mkdir()
rng = np.random.default_rng(0)


def make_toy_tile(path: Path, label: int, size: int = 224) -> None:
    """A 224x224 RGB tile whose color tint weakly encodes the label."""
    base = np.array([[150, 70, 160], [70, 150, 90]][label % 2], np.int16)
    img = base + rng.integers(-40, 40, (size, size, 3))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


sample_ids = [f"t{i:02d}" for i in range(16)]
labels = [i % 2 for i in range(16)]
split = (["train"] * 8) + (["tune"] * 4) + (["test"] * 4)
for sid, y in zip(sample_ids, labels):
    make_toy_tile(TILES / f"{sid}.png", y)

dataset_csv = WORK / "dataset.csv"
splits_csv = WORK / "splits.csv"
pd.DataFrame(
    {
        "sample_id": sample_ids,
        "image_path": [str(TILES / f"{s}.png") for s in sample_ids],
        "label": labels,
    }
).to_csv(dataset_csv, index=False)
pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)

# --- 1. The external spec ---------------------------------------------------------
# The config builder receives the exact keyword contract soma promises
# (dataset_csv, splits_csv, output_root, seed, overrides, encoder) and returns a
# complete PipelineConfig. The ``overrides`` dict carries the shared feature
# cache soma injects, so every seed reuses one extraction.


def build_config(
    *,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    seed: int,
    overrides: dict[str, Any],
    encoder: str,
) -> PipelineConfig:
    cache = replace(CacheConfig(), **overrides.get("cache", {}))
    return PipelineConfig(
        dataset_csv=str(dataset_csv),
        splits_csv=str(splits_csv),
        output_root=Path(output_root),
        dataset_type="tile",
        encoder=EncoderConfig(name=encoder),
        task=TaskConfig(name="binary_classification"),
        evaluation=EvalConfig(metrics=["balanced_accuracy"]),
        training=TrainingConfig(
            seed=seed,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            pin_memory=False,
            persistent_workers=False,
        ),
        cache=cache,
        # The dataset is tiny: skip dataloader worker processes.
        execution=ExecutionConfig(num_workers_per_gpu=0),
    )


def score(run_dir: str | Path) -> dict[str, float]:
    """Scorer contract: read one seed's evidence root into {metric: value}."""
    return score_from_summary(run_dir)


spec = BenchmarkSpec(
    name="example/synthetic-tiles",
    canonical_seeds=(0, 1),
    primary_metric="test/balanced_accuracy",
    build_config=build_config,
    score=score,
)
print(f"spec:            {spec.name}")
print(f"canonical seeds: {spec.canonical_seeds}")
print(f"reported:        {spec.reported_metrics}")

# --- 2. Execute it ---------------------------------------------------------------
# run_benchmark_spec runs every canonical seed under <output_root>/seed_<n> and
# points every seed at one shared feature cache (<output_root>/feature_cache by
# default), so tile features are extracted once and reused.

output_root = WORK / "benchmark"
result = run_benchmark_spec(
    spec,
    dataset_csv=dataset_csv,
    splits_csv=splits_csv,
    encoder="phikon",
    output_root=output_root,
)

# --- 3. Read the aggregated result ------------------------------------------------
print()
print(f"status: {result.status} ({'success' if result.status == 0 else 'failure'})")
print("per-seed evidence roots:")
for seed, seed_root in zip(spec.canonical_seeds, result.seed_roots):
    print(f"  seed {seed}: {seed_root}")
cache_root = output_root / "feature_cache"
cached_bags = [p for p in cache_root.rglob("*.pt") if p.stem in set(sample_ids)]
print(f"shared feature cache root: {cache_root}")
print(f"  cached feature bags: {len(cached_bags)} (extracted once, reused by every seed)")
print("aggregated metrics (mean over seeds):")
for metric in result.metrics:
    print(
        f"  {metric.metric}: {metric.measured:.3f} "
        f"± {metric.std:.3f} (n_seeds={metric.n_seeds})"
    )
