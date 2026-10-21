"""HEST-Benchmark gene-expression-from-morphology, registered as ``hest/<task>`` (issue #259).

HEST-Benchmark (Jaume et al., NeurIPS 2024) evaluates a **frozen** patch encoder on
predicting a 50-dimensional highly-variable-gene expression vector from a 112x112 µm tile,
scored by Pearson correlation, k-fold averaged. soma reproduces it **natively** (design §2):
its own slide2vec encoder → soma's tile feature cache → the closed-form Ridge+PCA probe
(:class:`~soma.benchmark.hest.probe.RidgePCAProbe`) over the curated spatial-expression
Manifest; it does not depend on the ``hest`` library or TRIDENT. The benchmark composes
soma's verbs itself (:meth:`HestBenchmark.run`): extract a rank-1 tile set source, fit the
probe on each fold's train spots, predict its test spots, and ``soma.evaluate`` the
predictions with ``mean_pearson`` plus one Pearson per gene. Its config starts from the
:func:`~soma.presets.tile_regression` preset (a tile encoder, no aggregator) and swaps
the learner for the probe.

All 9 HEST-Benchmark tasks are registered as ``hest/<task>`` (:data:`HEST_TASKS`), mirroring
the ``eva/<dataset>`` family: each shares the closed-form spatial-expression probe recipe and
varies the ``encoder`` axis (``DEFAULT_ENCODER = "uni2"``). ``curate_hest`` is task-generic, so
an unprovisioned task errors cleanly ("provide --raw-root") exactly like an unprovisioned
``eva/<dataset>`` — registering it is not a footgun.

Reference numbers: ``hest/reference.csv`` carries **external** Reference rows only — HEST's
published Pearson per (task, encoder), ``kind=external`` with a ``label`` + ``url``. There is
**no gate row**, so nothing is tolerance-checked: ``soma reproduce hest/<task>`` renders soma's
Measured row next to HEST's Reference, making the slide2vec↔TRIDENT gap an explicit, non-gating
delta rather than hiding it in a loose tolerance (issue #260).

HEST's numbers are ``kind=external``: soma **publishes** its Measured value beside them with
the signed delta and never gates on it — a gate against another lab's extraction stack would
fire on the cross-stack parity gap, not on a real regression (ADR 0005). ``soma reproduce
hest/<task> --record`` logs each Measured value to ``hest/results.csv``; the generated HEST doc
page joins that ledger to the reference and reports the per-cell delta (A, published), pooled
pairwise rank concordance (B, a bonus), and the provenance-pinned ledger as the drift guard
(C, the only axis that gates — soma against soma). See
:func:`soma.benchmark.reproduction.reproduction_report`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from soma.benchmark.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
    score_from_summary,
)
from hs2p import TilingConfig
from slide2vec import ExecutionOptions, PreprocessingConfig

from soma import presets
from soma.config import (
    CacheOptions,
    CohortConfig,
    EncoderLeaf,
    EvaluationConfig,
    ExperimentConfig,
    LearnerConfig,
    LoadersConfig,
    RunConfig,
    TrainerConfig,
)
from soma.benchmark.hest.curate import curate_hest
from soma.benchmark.curation import CuratedManifest
from soma.benchmark.hest.probe import DEFAULT_PCA_COMPONENTS, RidgePCAProbe, gene_pearson
from soma.learn import learner_registry
from soma.pipeline import PipelineResult
from soma.registry import check_positive

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
# a deliberately fine-grained rank test. OUTPUT_VARIANTS pins the feature variant explicitly
# for backbones where more than one variant exists, so the reproduction never depends on a
# slide2vec default silently changing:
#   * virchow2 → "cls_patch_mean" (2560-d CLS + mean-patch-token concat). This is what HEST
#     actually evaluates: TRIDENT's Virchow2InferenceEncoder.forward returns
#     torch.cat([class_token, patch_tokens.mean(1)]) unless the encoder was built with
#     return_cls=True, and HEST's bench builds it with a bare encoder_factory(model_name)
#     (return_cls appears nowhere in the HEST repository). The CLS-only 1280-d "cls"
#     variant would NOT match TRIDENT (#504). "cls_patch_mean" is the exact slide2vec token
#     (virchow.py output_variants={"cls": 1280, "cls_patch_mean": 2560}); it happens to be
#     slide2vec's default too, but it is pinned here on purpose.
#   * uni2, h-optimus-1 → no override: slide2vec's default for each is the plain CLS token
#     (uni2 1536-d; h-optimus-1's only variant is the 1536-d "default" CLS), matching TRIDENT.
# HEST extracts features via TRIDENT; soma re-extracts natively via slide2vec, so the
# Measured-minus-Reference delta is the accepted, non-gating slide2vec<->TRIDENT parity gap.
# The LUNG/virchow2 gap (-2.90% under the old "cls" pin) is resolved by the variant pin
# above: the virchow2 rows were re-recorded with "cls_patch_mean", and LUNG now lands at
# 0.5692 against HEST's 0.5685. COAD/uni2 (+2.99%, 0.3105 vs 0.3015) is an accepted
# residual: an independent extraction reproduces 0.3105 exactly, so it is deterministic, not
# extraction noise; it changes no rank on COAD; and uni2 lands within ~0.1% of HEST on PAAD
# and LUNG. Revisit it only if a uni2-specific tail shows up on a further task or encoder.
# Do not widen a tolerance to hide it (ADR 0005).
DEFAULT_ENCODER = "uni2"
OUTPUT_VARIANTS: dict[str, str] = {"virchow2": "cls_patch_mean"}

# The probe is closed-form and deterministic, so one seed suffices (unlike EVA's SGD head,
# which averages five). random_state=seed only steers PCA's solver.
CANONICAL_SEEDS: tuple[int, ...] = (0,)

# Multi-fold headline written by the shared summary writer: mean over folds of each fold's
# mean-over-genes Pearson (aggregation order per-gene → genes → folds; design §6).
PRIMARY_METRIC = "test/mean_pearson_mean"

# Pearson correlation is the HEST score: mean over genes per fold, written to summary.json.
EVAL_METRIC = "mean_pearson"

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


#: The learner a HEST config names: the closed-form probe, registered benchmark-private.
PROBE_NAME = "hest_probe"


@dataclass
class HestProbeParams:
    """:class:`~soma.benchmark.hest.probe.RidgePCAProbe`'s params (the run's seed seeds
    its PCA solver)."""

    pca_components: int = DEFAULT_PCA_COMPONENTS
    alpha: float | None = None

    def __post_init__(self) -> None:
        check_positive("pca_components", self.pca_components, integer=True)
        check_positive("alpha", self.alpha, optional=True)


if PROBE_NAME not in learner_registry:
    learner_registry.register(
        PROBE_NAME,
        RidgePCAProbe,
        params=HestProbeParams,
        metadata={"description": "HEST's closed-form Ridge+PCA probe (benchmark-private, hest/<task>)"},
    )


def _cache_from_overrides(overrides: dict[str, Any] | None) -> CacheOptions | None:
    """Turn a ``{"cache": {...}}`` override block into ``features.cache``."""
    if not overrides:
        return None
    cache_over = overrides.get("cache")
    if not cache_over:
        return None
    return replace(CacheOptions(), **cache_over)


class HestBenchmark:
    """One HEST-Benchmark task registered as ``hest/<task>`` (protocol-as-code).

    Fixes the spatial-expression probe recipe + task and varies the ``encoder`` axis.
    ``build_config`` emits the tile-extraction config (HEST's tile geometry, the encoder,
    the ``hest_probe`` learner with PCA latent dim 256); ``run`` composes the verbs over it;
    ``curate`` delegates to :func:`soma.benchmark.hest.curate.curate_hest`; ``expected`` selects
    the external reference row(s) for the task × encoder (possibly none yet); ``score``
    reads the fold-averaged mean-Pearson from ``summary.json``.
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
        execution: ExecutionOptions | None = None,
    ) -> ExperimentConfig:
        """Build the HEST-faithful probe config for the encoder axis.

        A tile-encoder config without an aggregator (one spot is one tile, extracted as a
        rank-1 set source) whose learner is the ``hest_probe`` with PCA latent dim 256,
        the ``mean_pearson`` metric, and ``encoder`` (default ``uni2``). :meth:`run` reads
        it; no ``Pipeline`` trains it. ``dataset_csv`` / ``splits_csv`` / ``output_root`` come
        from the curated Manifest; ``overrides`` carries the CLI's shared feature-cache
        block (``{"cache": {...}}``).

        Preprocessing pins HEST's tile geometry (:data:`TILE_SIZE_PX` px @ :data:`SPACING_UM`
        µm/px) for every encoder, so the encoder axis varies the encoder and nothing else.
        """
        cfg = presets.tile_regression(encoder)
        cfg.run = RunConfig(
            output_dir=Path(output_root) if output_root is not None else Path("output/hest"),
            seed=0 if seed is None else int(seed),
        )
        cfg.cohort = CohortConfig(
            records=str(dataset_csv) if dataset_csv is not None else "dataset.csv",
            splits=str(splits_csv) if splits_csv is not None else "splits.csv",
        )
        if OUTPUT_VARIANTS.get(encoder) is not None:
            cfg.features.encoder = EncoderLeaf(name=encoder, params={"output_variant": OUTPUT_VARIANTS[encoder]})
        cfg.features.extraction = PreprocessingConfig(
            tiling=TilingConfig(
                requested_spacing_um=SPACING_UM,
                requested_tile_size_px=TILE_SIZE_PX,
                tolerance=0.05,
                overlap=0.0,
            ),
        )
        cfg.features.execution = replace(execution or ExecutionOptions(), batch_size=encoder_batch_size)
        cfg.features.cache = _cache_from_overrides(overrides) or CacheOptions(enabled=True)
        # The probe is closed-form: no loaders, trainer or tune split take part, so those
        # sections keep their defaults rather than the preset's training recipe.
        cfg.loaders = LoadersConfig()
        cfg.learner = LearnerConfig(PROBE_NAME, {"pca_components": int(pca_components)})
        cfg.trainer = TrainerConfig()
        cfg.evaluation = EvaluationConfig(metrics=[EVAL_METRIC])
        return cfg

    def run(self, config: ExperimentConfig) -> PipelineResult:
        """Extract, fit, predict and evaluate each fold through soma's verbs.

        Extracts the cohort's tiles once into a rank-1 set source (soma's feature cache),
        then per fold: fit :class:`RidgePCAProbe` on the train spots' features and
        expression, predict every test split, and ``soma.evaluate`` the predictions
        against a targets frame with one ``value_<k>`` column per gene (``mean_pearson``
        and one ``pearson_gene<k>_<symbol>`` per gene). Writes each fold's
        ``metrics.json`` and ``predictions_<split>.parquet`` and the run's
        ``summary.json`` into the managed run directory
        (``output_root/experiments/<experiment>/runs/<run_id>``, as a ``Pipeline`` run), so
        runs of different encoders under one ``output_root`` never overwrite each other.
        Like a ``Pipeline`` run it writes ``experiment.json`` and a ``run.yaml`` that
        tracks the run's status, so ``soma leaderboard`` and ``soma reproduce
        --from-run-dir`` can read the run, and returns a :class:`~soma.pipeline.PipelineResult`
        (no fold results) carrying the summary, the run directory and the dataset, splits
        and test checksums those files record, so ``run_benchmark_spec`` can run it as a
        spec's ``run``. ``run.folds`` is refused: the closed-form probe fits every fold in
        one launch. The config resolves to its plan
        against the expression cohort (``plan.yaml`` beside ``config.yaml``), and a resume
        onto an existing run whose saved plan differs is refused before anything is written.
        """
        if config.run.folds is not None:
            raise ValueError("the HEST runner fits every fold in one launch; run.folds is not supported")
        import numpy as np

        from soma.data import Cohort
        from soma.pipeline.layout import (
            RunRecorder,
            guard_resume_plan_drift,
            record_plan,
            resolve_managed_output_paths,
            write_experiment_metadata,
        )
        from soma.plan import resolve

        cohort = Cohort.from_csv(
            config.cohort.records,
            config.cohort.splits,
            targets={"expression": "expression"},
            dtypes={"expression": np.ndarray},
        )
        plan = resolve(config, cohort)
        layout = resolve_managed_output_paths(plan)
        run_dir = layout.run_dir
        # Like a Pipeline run: a resume (resume/run_id) onto a run saved under another plan
        # (e.g. a changed seed, outside experiment identity) is refused before any write,
        # instead of silently replacing that run's plan and fold outputs.
        guard_resume_plan_drift(run_dir, plan)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_experiment_metadata(layout.experiment_dir, layout.experiment)
        record_plan(run_dir, plan, config=config)
        with RunRecorder(layout, plan) as recorder:
            summary = self._run_folds(plan, cohort, run_dir)
            recorder.complete(summary)
            return recorder.with_identity(PipelineResult(fold_results=[], summary=summary, run_dir=run_dir))

    def _run_folds(self, plan, cohort, run_dir: Path) -> dict[str, float]:
        """:meth:`run`'s work inside the managed run directory; returns the summary."""
        import numpy as np
        import pandas as pd

        from soma import extract
        from soma.data import ImageManifest
        from soma.extract.cache import resolve_cache_root
        from soma.evaluation import Evaluation, evaluate
        from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk, save_metrics, save_summary
        from soma.pipeline.extraction import resolve_pipeline_preprocessing

        genes = list(cohort.target_names.get("expression") or [])
        features = plan.features
        cache = replace(
            features.cache,
            root_dir=resolve_cache_root(features.cache, feature_dir=run_dir, output_root=plan.run.output_dir),
        )
        source = extract(
            ImageManifest.from_csv(plan.cohort.records),
            encoder=features.encoder,
            cache=cache,
            unit="tile",
            extraction=resolve_pipeline_preprocessing(plan),
            execution=features.execution,
            output_dir=run_dir,
        )
        probe_params = plan.learner.params
        metrics = ["mean_pearson", *(gene_pearson(k, gene) for k, gene in enumerate(genes))]

        def features(ids) -> np.ndarray:
            return np.stack([source.load(sid).reshape(-1).double().numpy() for sid in ids])

        def targets(ids) -> pd.DataFrame:
            matrix = np.stack([np.asarray(cohort.record(sid).targets["expression"], dtype=np.float64) for sid in ids])
            frame = pd.DataFrame(matrix, columns=[f"value_{k}" for k in range(matrix.shape[1])])
            frame.index = pd.Index(list(ids), name="sample_id", dtype=object)
            return frame

        single = cohort.num_folds == 1
        for index, fold in enumerate(cohort.folds):
            fold_dir = run_dir if single else run_dir / f"fold_{index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_ids = list(fold.train)
            probe = RidgePCAProbe(**probe_params, seed=plan.run.seed).fit(
                features(train_ids), targets(train_ids)
            )
            evaluations = {}
            for split, ids in fold.tests.items():
                predictions = probe.predict(features(ids), ids)
                predictions.to_parquet(fold_dir / f"predictions_{split}.parquet")
                evaluations[split] = evaluate(predictions, targets(ids), metrics=metrics)
            # The probe has no tune split; an empty tune entry is ignored by the summary.
            save_metrics(
                Evaluation(metrics={}, task_family="regression", num_samples=0),
                evaluations,
                fold_dir / "metrics.json",
            )
        summary = aggregate_fold_metrics_from_disk(
            run_dir, single_fold=single, include_tune=False, num_folds=cohort.num_folds
        )
        save_summary(summary, run_dir / "summary.json")
        return summary

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """External reference row(s) for this task × the resolved encoder axis.

        Injects the sub-benchmark's own ``dataset`` (the HEST task) and defaults the
        ``encoder`` axis to :data:`DEFAULT_ENCODER`, so ``expected()`` resolves the uni2
        row and ``expected(encoder="virchow2")`` the virchow2 row. ``hest/reference.csv``
        carries external (``kind=external``, non-gating) rows only — HEST's published
        Pearson per task and encoder — so these are rendered *beside* the Measured row,
        never gated.
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
# under --raw-root), and hest/reference.csv carries the external HEST numbers per task, so a
# task whose data is not provisioned locally still errors cleanly ("provide --raw-root") the
# same way an unprovisioned eva/<dataset> does — registering it is not a footgun.
HEST_BENCHMARKS: dict[str, HestBenchmark] = {}
for _task in HEST_TASKS:
    _bench = HestBenchmark(_task)
    HEST_BENCHMARKS[_bench.name] = _bench
    register_benchmark(_bench)
