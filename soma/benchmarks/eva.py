"""kaiko-ai/eva patch-level benchmark — reproduction recipe.

This module encodes the offline linear-probe protocol used by the
`kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-level leaderboard so it
can be reproduced with soma's tile path (``dataset_type="tile"``). It provides
the per-dataset / per-encoder protocol tables, a :func:`build_config` helper that
returns an EVA-faithful :class:`~soma.PipelineConfig`, and the published expected
balanced-accuracy values for comparison.

The protocol was reconciled against the eva source (offline classification
configs and the dataset/metric/backbone classes). The points that matter for
matching the leaderboard:

* Linear head, AdamW ``lr=3e-4`` and ``weight_decay=0.01`` (eva sets only ``lr``,
  so torch's AdamW default ``0.01`` applies), no scheduler, batch size 256.
* Training is step-based: ``max_steps=12500`` with one validation per epoch. We
  map that to soma's epoch loop via :func:`epochs_for_train_size`:
  ``epochs = ceil(12500 / ceil(N_train / 256))``. Early-stopping patience is
  eva's hand-set per-dataset value (counted in validation checks == epochs here).
* The selection/report metric is balanced accuracy, which equals eva's
  ``MulticlassAccuracy(average="macro")`` / ``BinaryBalancedAccuracy``.
* Datasets with no eva test split (bach, breakhis, crc, mhist, gleason_arvaniti)
  report on the validation split: curate with ``tune_fraction=0.0`` and run with
  ``tune_is_test=True`` so soma's ``test`` split *is* the eva validation set.
  patch_camelyon has a real val + test split: ``tune_is_test=False``, report the
  validation split (``patch_camelyon`` column) and the test split
  (``patch_camelyon/test`` column).
* virchow2 must use the CLS-only output (1280-d); eva's ``paige_virchow2`` sets
  ``ExtractCLSFeatures(include_patch_tokens=False)``. slide2vec defaults to the
  2560-d CLS+mean concat, which would *not* match the leaderboard.

The expected balanced-accuracy values are read from the bundled leaderboard
snapshot ``reference/eva.csv`` (source: https://kaiko-ai.github.io/eva, captured
2026-06). Update that file to refresh against a newer leaderboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import pandas as pd

from soma import (
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)

# --- Protocol constants (eva offline classification configs) -----------------
MAX_STEPS = 12500
HEAD_BATCH_SIZE = 256
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.01  # eva sets only lr; torch.optim.AdamW default weight_decay
# eva reports on the validation split for tune-is-test datasets, so curation must
# keep every eva-train sample as soma "train" (none diverted to "tune").
CURATION_TUNE_FRACTION = 0.0


@dataclass(frozen=True)
class EncoderSpec:
    """How a soma encoder maps onto an EVA leaderboard backbone."""

    eva_key: str  # row name on the EVA leaderboard / in eva.csv
    output_variant: str | None  # slide2vec output variant override (None = default)


# soma encoder name -> EVA backbone. Extend this with the encoder's eva.csv row
# name and the output variant eva used (add matching rows to ``EXPECTED``).
ENCODERS: dict[str, EncoderSpec] = {
    "uni2": EncoderSpec(eva_key="mahmood_uni2_h", output_variant=None),  # CLS, 1536-d
    "virchow2": EncoderSpec(eva_key="paige_virchow2", output_variant="cls"),  # CLS only, 1280-d
}


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset EVA protocol parameters."""

    task: str  # soma task head name
    patience: int  # eva's per-dataset EarlyStopping patience
    tune_is_test: bool  # True when eva reports on the validation split
    val_column: str  # eva.csv column for the reported (validation) split
    test_column: str | None = None  # eva.csv column for the test split, if any


# Curatable EVA patch-classification datasets (see ``soma.curation.eva``).
# patience values are transcribed from eva's offline classification configs.
DATASETS: dict[str, DatasetSpec] = {
    "bach": DatasetSpec("multiclass_classification", 1250, True, "bach"),
    "breakhis": DatasetSpec("multiclass_classification", 500, True, "breakhis"),
    "crc": DatasetSpec("multiclass_classification", 7, True, "crc"),
    "mhist": DatasetSpec("binary_classification", 278, True, "mhist"),
    "gleason_arvaniti": DatasetSpec("multiclass_classification", 42, True, "gleason_arvaniti"),
    "patch_camelyon": DatasetSpec(
        "binary_classification", 3, False, "patch_camelyon", "patch_camelyon/test"
    ),
}

# Bundled EVA leaderboard snapshot (full table, all encoders/columns).
LEADERBOARD_CSV = "eva.csv"


@lru_cache(maxsize=1)
def _leaderboard() -> pd.DataFrame:
    """The bundled EVA leaderboard, indexed by eva backbone key."""
    with resources.files("soma.benchmarks.reference").joinpath(LEADERBOARD_CSV).open() as handle:
        return pd.read_csv(handle).set_index("model")


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


def expected_balanced_accuracy(
    encoder: str, dataset: str, *, split: str = "val"
) -> float | None:
    """Published EVA balanced accuracy for ``(encoder, dataset, split)``.

    ``split`` is ``"val"`` (the reported split) or ``"test"``. Returns ``None``
    when the value is not tabulated (e.g. an unknown encoder, or the test split
    for a dataset that has none).
    """
    spec = _require_dataset(dataset)
    column = spec.val_column if split == "val" else spec.test_column
    if column is None or encoder not in ENCODERS:
        return None
    leaderboard = _leaderboard()
    eva_key = ENCODERS[encoder].eva_key
    if eva_key not in leaderboard.index or column not in leaderboard.columns:
        return None
    return float(leaderboard.loc[eva_key, column])


@dataclass(frozen=True)
class ReportedSplit:
    """One split soma reports for a dataset, and where it lands on the leaderboard.

    ``label`` is the eva-facing split name (``"val"`` / ``"test"``), ``soma_split``
    is the soma split the metric is read from, and ``eva_column`` is the matching
    eva.csv column.
    """

    label: str
    soma_split: str
    eva_column: str


def reported_splits(dataset: str) -> list[ReportedSplit]:
    """Splits soma reports for ``dataset``, mapped to leaderboard columns.

    For tune-is-test datasets the eva validation set *is* soma's ``test`` split;
    patch_camelyon reports its real validation (soma ``tune``) and test splits.
    """
    spec = _require_dataset(dataset)
    if spec.tune_is_test:
        return [ReportedSplit("val", "test", spec.val_column)]
    splits = [ReportedSplit("val", "tune", spec.val_column)]
    if spec.test_column is not None:
        splits.append(ReportedSplit("test", "test", spec.test_column))
    return splits


def result_summary(result) -> dict[str, float]:
    """Flatten a finished run's metrics into ``{"<split>/<metric>": value}``.

    ``result.summary`` only carries test-split metrics, so the tune (validation)
    metrics are merged in from the first fold's tune report.
    """
    summary = {key: float(value) for key, value in result.summary.items()}
    if result.fold_results:
        for key, value in result.fold_results[0].tune_report.metrics.items():
            summary.setdefault(f"tune/{key}", float(value))
    return summary


def balanced_accuracy_from_summary(
    summary: dict[str, float], soma_split: str
) -> float | None:
    """Read balanced accuracy for ``soma_split`` from a flattened summary."""
    for suffix in ("_mean", ""):
        key = f"{soma_split}/balanced_accuracy{suffix}"
        if key in summary:
            return summary[key]
    return None


def build_config(
    *,
    dataset: str,
    encoder: str,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    cache_root: str | Path,
    seed: int = 0,
    epochs: int | None = None,
    patience: int | None = None,
    encoder_batch_size: int = 32,
    head_num_workers: int = 0,
    execution: ExecutionConfig | None = None,
) -> PipelineConfig:
    """Build an EVA-faithful :class:`~soma.PipelineConfig` for one run.

    ``epochs`` defaults to :func:`epochs_for_train_size` computed from
    ``splits_csv``; ``patience`` defaults to the dataset's eva value. Pass
    overrides for smoke runs.
    """
    spec = _require_dataset(dataset)
    enc = _require_encoder(encoder)

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
        cache=CacheConfig(enabled=True, root_dir=str(cache_root)),
        encoder=EncoderConfig(
            name=encoder,
            output_variant=enc.output_variant,
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
            # The head trains on tiny in-memory feature vectors, so dataloader
            # worker spawn would dominate wall time; eva uses 0 workers here too.
            num_workers=head_num_workers,
            pin_memory=False,
            persistent_workers=False,
        ),
        tags=["eva", dataset, encoder],
    )


def _require_dataset(dataset: str) -> DatasetSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EVA dataset '{dataset}'. Supported: {', '.join(DATASETS)}"
        ) from exc


def _require_encoder(encoder: str) -> EncoderSpec:
    try:
        return ENCODERS[encoder]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EVA encoder '{encoder}'. Supported: {', '.join(ENCODERS)}"
        ) from exc
