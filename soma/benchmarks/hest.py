"""HEST-Benchmark gene-expression-from-morphology, registered as ``hest/IDC`` (issue #259).

HEST-Benchmark (Jaume et al., NeurIPS 2024) evaluates a **frozen** patch encoder on
predicting a 50-dimensional highly-variable-gene expression vector from a 112x112 µm tile,
scored by Pearson correlation, k-fold averaged. soma reproduces it **natively** (design §2):
its own slide2vec encoder → soma's tile feature cache → the closed-form Ridge+PCA probe over
the curated ``spatial_expression`` Manifest; it does not depend on the ``hest`` library or
TRIDENT.

All 9 HEST-Benchmark tasks are registered as ``hest/<task>`` (:data:`HEST_TASKS`), mirroring
the ``eva/<dataset>`` family: each shares the closed-form spatial-expression probe recipe and
varies the ``encoder`` axis (``DEFAULT_ENCODER = "uni2"``). ``curate_hest`` is task-generic, so
an unprovisioned task errors cleanly ("provide --raw-root") exactly like an unprovisioned
``eva/<dataset>`` — registering it is not a footgun.

Reference numbers: ``reference/hest.csv`` carries **external** Reference rows only — HEST's
published Pearson per (task, encoder), ``kind=external`` with a ``label`` + ``url``. There is
**no gate row**, so nothing is tolerance-checked: ``soma reproduce hest/<task>`` renders soma's
Measured row next to HEST's Reference, making the slide2vec↔TRIDENT gap an explicit, non-gating
delta rather than hiding it in a loose tolerance (issue #260).

HEST's numbers are ``kind=external``: soma **publishes** its Measured value beside them with
the signed delta and never gates on it — a gate against another lab's extraction stack would
fire on the cross-stack parity gap, not on a real regression (ADR 0005). ``soma reproduce
hest/<task> --record`` logs each Measured value to ``results/hest.csv``; the generated HEST doc
page joins that ledger to the reference and reports the per-cell delta (A, published), pooled
pairwise rank concordance (B, a bonus), and the provenance-pinned ledger as the drift guard
(C, the only axis that gates — soma against soma). See
:func:`soma.benchmarks.reproduction.reproduction_report`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

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
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.hest import curate_hest
from soma.curation.manifest import CuratedManifest
from soma.training.probe import DEFAULT_PCA_COMPONENTS, PROBE_METHOD

REFERENCE_NAME = "hest"

# The 9 HEST-Benchmark tasks (mahmoodlab/HEST leaderboard, results 03.04.26), spanning organ
# types: IDC/LYMPH_IDC breast, PRAD prostate, PAAD pancreas, COAD colon, READ rectum, CCRCC
# kidney, LUNG lung, SKCM skin. Every task is registered as hest/<task>. NOTE: the hest-bench
# HF dataset also ships an HCC/ (liver) data tree, but HCC is NOT one of the benchmark's 9
# scored tasks — it has no published leaderboard number — so it is deliberately not registered
# (a name with no reference row would be the footgun the family is designed to avoid).
HEST_TASKS: tuple[str, ...] = (
    "IDC",
    "PRAD",
    "PAAD",
    "COAD",
    "READ",
    "CCRCC",
    "LUNG",
    "LYMPH_IDC",
    "SKCM",
)
# Default task backing the class default arg (IDC is the vertical slice landed first).
TASK = "IDC"

# `encoder` is the varied axis; uni2 is the headline backbone. slide2vec validates the name,
# so any registered encoder works. The three-model reproduction campaign benchmarks
# {uni2, virchow2, h-optimus-1} — the top HEST cluster (published IDC 0.5898 / 0.5971 / 0.6024),
# a deliberately fine-grained rank test. OUTPUT_VARIANTS only pins the feature variant for
# backbones where the leaderboard used a non-default one:
#   * virchow2 → "cls" (CLS-only 1280-d; slide2vec defaults to the 2560-d CLS+mean concat,
#     which would NOT match TRIDENT). "cls" is the exact slide2vec token (virchow.py
#     output_variants={"cls": 1280, "cls_patch_mean": 2560}, default "cls_patch_mean").
#   * uni2, h-optimus-1 → no override: slide2vec's default for each is the plain CLS token
#     (uni2 1536-d; h-optimus-1's only variant is the 1536-d "default" CLS), matching TRIDENT.
# TODO(#261): confirm slide2vec<->TRIDENT virchow2 parity (transforms + CLS-only variant).
# HEST extracts features via TRIDENT; soma re-extracts natively via slide2vec, so the
# Measured-minus-Reference delta is the accepted, non-gating slide2vec<->TRIDENT parity gap.
DEFAULT_ENCODER = "uni2"
OUTPUT_VARIANTS: dict[str, str] = {"virchow2": "cls"}

# The probe is closed-form and deterministic, so one seed suffices (unlike EVA's SGD head,
# which averages five). random_state=seed only steers PCA's solver.
CANONICAL_SEEDS: tuple[int, ...] = (0,)

# Multi-fold headline written by the shared summary writer: mean over folds of each fold's
# mean-over-genes Pearson (aggregation order per-gene → genes → folds; design §6).
PRIMARY_METRIC = "test/mean_pearson_mean"

# Pearson correlation is the HEST score; the closed-form probe writes it into summary.json.
EVAL_METRIC = "pearson"

# HEST's protocol fixes the tile geometry: a 112x112 µm tile rendered at 224x224 px, i.e.
# 112/224 = 0.5 µm/px. Pin both explicitly rather than letting each encoder's own recommended
# scale decide, because the tile scale is a property of the *benchmark*, not of the encoder:
# a reproduction that fed one encoder 0.5 µm/px and another 1.0 µm/px would not be comparing
# encoders, it would be comparing magnifications. Pinning is also what makes the family
# encoder-agnostic — uni2 and h-optimus-1 declare a scalar supported_spacing_um of 0.5 (so
# they auto-resolve to exactly these values), but virchow2 declares [0.25, 0.5, 1.0, 2.0] and
# soma rightly refuses to guess among them, which used to make hest/<task> unrunnable on it.
TILE_SIZE_PX = 224
SPACING_UM = 0.5

REFERENCE_ENVIRONMENT: dict[str, str] = {
    "leaderboard": "mahmoodlab/HEST-Benchmark (captured for reference; external rows only)",
}


def _cache_from_overrides(overrides: dict[str, Any] | None) -> CacheConfig | None:
    """Turn a ``{"cache": {...}}`` override block into a :class:`CacheConfig`."""
    if not overrides:
        return None
    cache_over = overrides.get("cache")
    if not cache_over:
        return None
    return replace(CacheConfig(), **cache_over)


class HestBenchmark:
    """HEST-Benchmark task registered as ``hest/<task>`` (protocol-as-code; only IDC now).

    Fixes the spatial-expression probe recipe + task and varies the ``encoder`` axis.
    ``build_config`` emits a ``spatial_expression`` config carrying the probe method + PCA
    latent dim 256; ``curate`` delegates to :func:`soma.curation.hest.curate_hest`;
    ``expected`` selects the external reference row(s) for the task × encoder (possibly
    none yet); ``score`` reads the fold-averaged mean-Pearson from ``summary.json``.
    """

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reference_environment = REFERENCE_ENVIRONMENT

    def __init__(self, task: str = TASK) -> None:
        self.task = task
        self.name = f"hest/{task}"
        self.facet = Facet(
            fixed={
                "dataset": task,
                "task": "regression",
                "protocol": "hest-spatial-expression-probe",
            },
            varied=("encoder",),
        )

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest:
        """Curate this HEST task into a soma ``spatial_expression`` Manifest (delegates)."""
        return curate_hest(raw_root, out_dir, task=self.task)

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ENCODER,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        pca_components: int = DEFAULT_PCA_COMPONENTS,
        encoder_batch_size: int = 32,
        execution: ExecutionConfig | None = None,
    ) -> PipelineConfig:
        """Build the HEST-faithful ``spatial_expression`` probe config for the encoder axis.

        Carries ``dataset_type='spatial_expression'``, ``training.method='ridge_pca_probe'``
        (the closed-form probe), PCA latent dim 256 (``task.params.pca_components``), the
        Pearson metric, and ``encoder`` (default ``uni2``). ``dataset_csv`` / ``splits_csv``
        / ``output_root`` come from the curated Manifest; ``overrides`` carries the CLI's
        shared feature-cache block (``{"cache": {...}}``).

        Preprocessing pins HEST's tile geometry (:data:`TILE_SIZE_PX` px @ :data:`SPACING_UM`
        µm/px) for every encoder, so the encoder axis varies the encoder and nothing else.
        """
        return PipelineConfig(
            dataset_csv=str(dataset_csv) if dataset_csv is not None else "dataset.csv",
            splits_csv=str(splits_csv) if splits_csv is not None else "splits.csv",
            output_root=Path(output_root) if output_root is not None else Path("output/hest"),
            dataset_type="spatial_expression",
            execution=execution or ExecutionConfig(),
            cache=_cache_from_overrides(overrides) or CacheConfig(enabled=True),
            preprocessing=PreprocessingConfig(
                requested_tile_size_px=TILE_SIZE_PX,
                requested_spacing_um=SPACING_UM,
            ),
            encoder=EncoderConfig(
                name=encoder,
                output_variant=OUTPUT_VARIANTS.get(encoder),
                batch_size=encoder_batch_size,
            ),
            task=TaskConfig(name="regression", params={"pca_components": int(pca_components)}),
            evaluation=EvalConfig(metrics=[EVAL_METRIC]),
            training=TrainingConfig(
                seed=0 if seed is None else int(seed),
                method=PROBE_METHOD,
                # The probe is closed-form: no epochs / early stopping / tune split, so the
                # gradient-loop knobs keep their (inert) defaults.
            ),
            tags=["hest", self.task, encoder],
        )

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """External reference row(s) for this task × the resolved encoder axis.

        Injects the sub-benchmark's own ``dataset`` (the HEST task) and defaults the
        ``encoder`` axis to :data:`DEFAULT_ENCODER`, so ``expected()`` resolves the uni2
        row and ``expected(encoder="virchow2")`` the virchow2 row. ``reference/hest.csv``
        carries external (``kind=external``, non-gating) rows only — HEST's published IDC
        Pearson per encoder — so these are rendered *beside* the Measured row, never gated.
        Returns ``[]`` for an encoder with no published HEST number.
        """
        merged: dict[str, Any] = {"dataset": self.task, "encoder": DEFAULT_ENCODER}
        merged.update({k: v for k, v in axes.items() if v is not None})
        return expected_rows(REFERENCE_NAME, **merged)

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """DEFAULT scorer: read the run's ``summary.json`` (fold-averaged mean-Pearson)."""
        return score_from_summary(run_dir)


# Register one sub-benchmark per HEST-Benchmark task (name == "hest/<task>"), mirroring the
# eva/<dataset> family. Every task shares the closed-form spatial-expression probe recipe and
# varies only the encoder axis. curate_hest is task-generic (it just needs the <task>/ subtree
# under --raw-root), and reference/hest.csv carries the external HEST numbers per task, so a
# task whose data is not provisioned locally still errors cleanly ("provide --raw-root") the
# same way an unprovisioned eva/<dataset> does — registering it is not a footgun.
HEST_BENCHMARKS: dict[str, HestBenchmark] = {}
for _task in HEST_TASKS:
    _bench = HestBenchmark(_task)
    HEST_BENCHMARKS[_bench.name] = _bench
    register_benchmark(_bench)
