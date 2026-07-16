"""kaiko-ai/eva patch-level benchmark — registered per-dataset sub-benchmarks (issue #219).

This module promotes the EVA patch-level linear-probe reproduction (PR #87) onto the
landed Benchmark registry (#213), reconciled against the unified Manifest (#212). EVA is
registered as **one sub-benchmark per dataset** — ``eva/bach``, ``eva/breakhis``,
``eva/crc``, ``eva/mhist``, ``eva/gleason_arvaniti``, ``eva/patch_camelyon`` — each
exposing ``encoder`` as a ``build_config`` axis (the dataset × encoder grid). The recipe
encodes the offline linear-probe protocol of the `kaiko-ai/eva
<https://github.com/kaiko-ai/eva>`_ leaderboard so it reproduces with soma's tile path
(``dataset_type="tile"``).

Protocol points that matter for matching the leaderboard:

* Linear head, AdamW ``lr=3e-4`` and ``weight_decay=0.01`` (eva sets only ``lr``, so
  torch's AdamW default ``0.01`` applies), no scheduler, batch size 256.
* Training is step-based: ``max_steps=12500`` with one validation per epoch. We map that to
  soma's epoch loop via :func:`epochs_for_train_size`:
  ``epochs = ceil(12500 / ceil(N_train / 256))``. Early-stopping patience is eva's hand-set
  per-dataset value.
* The selection/report metric is balanced accuracy, which equals eva's
  ``MulticlassAccuracy(average="macro")`` / ``BinaryBalancedAccuracy``.
* Datasets with no eva test split (bach, breakhis, crc, mhist, gleason_arvaniti) report on
  the validation split: curate with ``tune_fraction=0.0`` and run with ``tune_is_test=True``
  so soma's ``test`` split *is* the eva validation set. patch_camelyon has a real val + test
  split (``tune_is_test=False``): the reported (test) number is the headline; the val number
  is recorded alongside it in the reference table.
* virchow2 must use the CLS-only output (1280-d); eva's ``paige_virchow2`` sets
  ``ExtractCLSFeatures(include_patch_tokens=False)``. slide2vec defaults to the 2560-d
  CLS+mean concat, which would *not* match the leaderboard.

The expected balanced-accuracy values live as **keyed rows** in
``reference/eva.csv`` (``dataset, encoder, metric, expected, tolerance, source``) with a
per-row tolerance band. The default ``summary.json`` scorer suffices — EVA's metric is
balanced accuracy; there is no custom scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
    score_from_summary,
)
from soma.config import (
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.eva import curate_eva_patch_dataset
from soma.curation.manifest import CuratedManifest

# --- Protocol constants (eva offline classification configs) -------------------------
MAX_STEPS = 12500
HEAD_BATCH_SIZE = 256
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.01  # eva sets only lr; torch.optim.AdamW default weight_decay
# eva reports on the validation split for tune-is-test datasets, so curation must keep
# every eva-train sample as soma "train" (none diverted to "tune").
CURATION_TUNE_FRACTION = 0.0

# The default (summary.json) scorer returns split-prefixed keys; every eva sub-benchmark
# reports its headline on soma's test split, so this is the primary tolerance metric.
PRIMARY_METRIC = "test/balanced_accuracy"
# eva runs each head over five seeds and averages (see the EVA offline protocol).
CANONICAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

REFERENCE_NAME = "eva"
REFERENCE_ENVIRONMENT: dict[str, str] = {
    "leaderboard": "kaiko-ai.github.io/eva (patch, captured 2026-06)",
}


@dataclass(frozen=True)
class EncoderSpec:
    """How a soma encoder maps onto an EVA leaderboard backbone."""

    eva_key: str  # row name on the EVA leaderboard
    output_variant: str | None  # slide2vec output variant override (None = default)


# soma encoder name -> EVA backbone. The reference rows key on the soma encoder name; the
# output variant is the slide2vec feature eva used (virchow2 is CLS-only, 1280-d).
ENCODERS: dict[str, EncoderSpec] = {
    "uni2": EncoderSpec(eva_key="mahmood_uni2_h", output_variant=None),  # CLS, 1536-d
    "virchow2": EncoderSpec(eva_key="paige_virchow2", output_variant="cls"),  # CLS only, 1280-d
}
DEFAULT_ENCODER = "uni2"


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset EVA protocol parameters."""

    task: str  # soma task head name
    patience: int  # eva's per-dataset EarlyStopping patience
    tune_is_test: bool  # True when eva reports on the validation split


# Curatable EVA patch-classification datasets (see ``soma.curation.eva``). patience values
# are transcribed from eva's offline classification configs.
DATASETS: dict[str, DatasetSpec] = {
    "bach": DatasetSpec("multiclass_classification", 1250, True),
    "breakhis": DatasetSpec("multiclass_classification", 500, True),
    "crc": DatasetSpec("multiclass_classification", 7, True),
    "mhist": DatasetSpec("binary_classification", 278, True),
    "gleason_arvaniti": DatasetSpec("multiclass_classification", 42, True),
    "patch_camelyon": DatasetSpec("binary_classification", 3, False),
}


def epochs_for_train_size(n_train: int) -> int:
    """Map eva's ``max_steps`` to a soma epoch count for ``n_train`` samples."""
    if n_train <= 0:
        raise ValueError(f"n_train must be positive, got {n_train}")
    steps_per_epoch = math.ceil(n_train / HEAD_BATCH_SIZE)
    return max(1, math.ceil(MAX_STEPS / steps_per_epoch))


def count_train_samples(splits_csv: str | Path) -> int:
    """Number of ``train``-split samples in a soma splits manifest."""
    splits = pd.read_csv(splits_csv)
    return int((splits["split"] == "train").sum())


def _require_dataset(dataset: str) -> DatasetSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EVA dataset {dataset!r}. Supported: {', '.join(DATASETS)}"
        ) from exc


def _build_eva_config(
    *,
    dataset: str,
    encoder: str,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    seed: int = 0,
    epochs: int | None = None,
    patience: int | None = None,
    encoder_batch_size: int = 32,
    head_num_workers: int = 0,
    execution: ExecutionConfig | None = None,
    cache: CacheConfig | None = None,
) -> PipelineConfig:
    """Assemble an EVA-faithful :class:`~soma.config.PipelineConfig` for one run.

    ``epochs`` defaults to :func:`epochs_for_train_size` computed from ``splits_csv``;
    ``patience`` defaults to the dataset's eva value. Pass overrides for smoke runs.
    """
    spec = _require_dataset(dataset)
    # Known EVA reference encoders may need a protocol-specific output variant. Any other
    # registered soma encoder uses its own default variant and can extend the benchmark.
    enc = ENCODERS.get(encoder)

    if epochs is None:
        epochs = epochs_for_train_size(count_train_samples(splits_csv))
    if patience is None:
        patience = spec.patience

    return PipelineConfig(
        dataset_csv=str(dataset_csv),
        splits_csv=str(splits_csv),
        output_root=Path(output_root),
        dataset_type="tile",
        execution=execution or ExecutionConfig(),
        cache=cache or CacheConfig(enabled=True),
        encoder=EncoderConfig(
            name=encoder,
            output_variant=enc.output_variant if enc is not None else None,
            batch_size=encoder_batch_size,
        ),
        task=TaskConfig(name=spec.task),
        evaluation=EvalConfig(metrics=["balanced_accuracy", "accuracy"]),
        training=TrainingConfig(
            seed=seed,
            epochs=epochs,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            optimizer="adamw",
            scheduler="none",
            patience=patience,
            monitor="balanced_accuracy",
            monitor_mode="max",
            batch_size=HEAD_BATCH_SIZE,
            tune_is_test=spec.tune_is_test,
            # The head trains on tiny in-memory feature vectors, so dataloader worker spawn
            # would dominate wall time; eva uses 0 workers here too.
            num_workers=head_num_workers,
            pin_memory=False,
            persistent_workers=False,
        ),
        tags=["eva", dataset, encoder],
    )


def _cache_from_overrides(overrides: dict[str, Any] | None) -> CacheConfig | None:
    """Turn a ``{"cache": {...}}`` override block into a :class:`CacheConfig`."""
    if not overrides:
        return None
    cache_over = overrides.get("cache")
    if not cache_over:
        return None
    return replace(CacheConfig(), **cache_over)


class EvaBenchmark:
    """One EVA patch-classification dataset registered as ``eva/<dataset>`` (protocol-as-code).

    The sub-benchmark fixes its dataset (and the EVA linear-probe recipe) and varies the
    ``encoder`` axis. ``build_config`` emits the EVA-faithful config, ``expected`` selects
    the keyed reference row(s) for the dataset × encoder, and the default ``summary.json``
    scorer reads soma's balanced accuracy.
    """

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reference_environment = REFERENCE_ENVIRONMENT

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.name = f"eva/{dataset}"
        self.facet = Facet(
            fixed={
                "dataset": dataset,
                "task": DATASETS[dataset].task,
                "protocol": "eva-linear-probe",
            },
            varied=("encoder",),
        )

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest:
        """Curate this EVA dataset into a soma tile Manifest (delegates to the curator).

        ``tune_fraction=0.0`` keeps every eva-train sample as soma ``train`` so the eva
        validation set lands in soma's reported ``test`` split (tune-is-test protocol).
        """
        return curate_eva_patch_dataset(
            self.dataset, raw_root, out_dir, tune_fraction=CURATION_TUNE_FRACTION
        )

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ENCODER,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        epochs: int | None = None,
        patience: int | None = None,
        encoder_batch_size: int = 32,
        execution: ExecutionConfig | None = None,
    ) -> PipelineConfig:
        """Build the EVA-faithful config for this dataset and the ``encoder`` axis.

        ``dataset_csv`` / ``splits_csv`` / ``output_root`` come from the curated Manifest;
        ``epochs`` defaults to the eva step-budget mapping computed from ``splits_csv``.
        ``overrides`` carries the CLI's shared feature-cache block (``{"cache": {...}}``).
        """
        if splits_csv is None and epochs is None:
            raise ValueError(
                "build_config needs splits_csv (to compute the eva epoch budget) or an "
                "explicit epochs override."
            )
        return _build_eva_config(
            dataset=self.dataset,
            encoder=encoder,
            dataset_csv=dataset_csv if dataset_csv is not None else "dataset.csv",
            splits_csv=splits_csv if splits_csv is not None else "splits.csv",
            output_root=output_root if output_root is not None else "output/eva",
            seed=0 if seed is None else int(seed),
            epochs=epochs,
            patience=patience,
            encoder_batch_size=encoder_batch_size,
            execution=execution,
            cache=_cache_from_overrides(overrides),
        )

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """Keyed reference row(s) for this dataset × the resolved encoder axis.

        Injects the sub-benchmark's own ``dataset`` and defaults the ``encoder`` axis to
        :data:`DEFAULT_ENCODER` so ``soma reproduce eva/<dataset>`` (no ``--encoder``)
        selects a single per-encoder band. patch_camelyon returns two rows (val + test).
        """
        merged: dict[str, Any] = {"dataset": self.dataset, "encoder": DEFAULT_ENCODER}
        merged.update({k: v for k, v in axes.items() if v is not None})
        return expected_rows(REFERENCE_NAME, **merged)

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """DEFAULT scorer: read the run's ``summary.json`` (balanced accuracy per split)."""
        return score_from_summary(run_dir)


# Register one sub-benchmark per curatable EVA dataset (name == "eva/<dataset>").
EVA_BENCHMARKS: dict[str, EvaBenchmark] = {}
for _dataset in DATASETS:
    _bench = EvaBenchmark(_dataset)
    EVA_BENCHMARKS[_bench.name] = _bench
    register_benchmark(_bench)
