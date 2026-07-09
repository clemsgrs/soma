"""HEST-Benchmark gene-expression-from-morphology, registered as ``hest/IDC`` (issue #259).

HEST-Benchmark (Jaume et al., NeurIPS 2024) evaluates a **frozen** patch encoder on
predicting a 50-dimensional highly-variable-gene expression vector from a 112x112 µm tile,
scored by Pearson correlation, k-fold averaged. soma reproduces it **natively** (design §2):
its own slide2vec encoder → soma's tile feature cache → the closed-form Ridge+PCA probe over
the curated ``spatial_expression`` Manifest; it does not depend on the ``hest`` library or
TRIDENT.

This lands the vertical slice — ``hest/IDC`` only. The eight remaining tasks
(``PRAD … LYMPH_IDC``) follow in fan-out; registering a name whose ``curate()`` cannot run
would be a footgun (design §7). ``encoder`` is the varied ``build_config`` axis
(``DEFAULT_ENCODER = "uni2"``); the spatial-expression probe recipe + task are fixed.

Reference numbers: ``reference/hest.csv`` carries **external** Reference rows only — HEST's
published Pearson per (task, encoder), ``kind=external`` with a ``label`` + ``url`` (issue
#260 populates IDC × the ~18 overlapping encoders from the official HEST leaderboard). There
is **no gate row**, so nothing is tolerance-checked; ``soma reproduce hest/IDC`` renders
soma's Measured row next to HEST's Reference, making the slide2vec↔TRIDENT gap an explicit,
non-gating delta rather than hiding it in a loose tolerance.
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
    TaskConfig,
    TrainingConfig,
)
from soma.curation.hest import curate_hest
from soma.curation.manifest import CuratedManifest
from soma.training.probe import DEFAULT_PCA_COMPONENTS, PROBE_METHOD

REFERENCE_NAME = "hest"

# The only HEST task landed now (design §7 / §10 — vertical slice first).
TASK = "IDC"

# `encoder` is the varied axis; uni2 is the headline backbone. slide2vec validates the name,
# so any registered encoder works — OUTPUT_VARIANTS only pins the feature variant for
# backbones where the leaderboard used a non-default one (virchow2 is CLS-only 1280-d, not
# slide2vec's 2560-d CLS+mean concat; design §11).
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
        """
        return PipelineConfig(
            dataset_csv=str(dataset_csv) if dataset_csv is not None else "dataset.csv",
            splits_csv=str(splits_csv) if splits_csv is not None else "splits.csv",
            output_root=Path(output_root) if output_root is not None else Path("output/hest"),
            dataset_type="spatial_expression",
            execution=execution or ExecutionConfig(),
            cache=_cache_from_overrides(overrides) or CacheConfig(enabled=True),
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


# Register only hest/IDC (the vertical slice). No other HEST name is registered.
HEST_BENCHMARK = HestBenchmark(TASK)
register_benchmark(HEST_BENCHMARK)
