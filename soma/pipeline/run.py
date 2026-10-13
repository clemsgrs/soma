"""``Pipeline``: the managed run, composed of the public verbs (ADR 0017).

``Pipeline(config)`` reads the cohort and resolves the config into its
:class:`~soma.plan.Plan` once (ADR 0020); everything after reads the plan. ``run()`` only
orchestrates: the canonical per-fold order is :func:`~soma.pipeline.folds.run_fold`,
plan → objects is :mod:`soma.pipeline.build`, the run layout and resume guards are
:mod:`soma.pipeline.layout`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.console import Console

from soma._logging import ensure_default_logging
from soma.atomic_io import atomic_write_json
from soma.cache import resolve_cache_root
from soma.config import ExperimentConfig
from soma.data import AnnotationManifest, Cohort, ImageManifest
from soma.pipeline._legacy import LegacyFolds, LegacySamples, legacy_inputs
from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk, save_summary
from soma.pipeline.build import RunInputs, attach_source, build_fold, load_pipeline_data
from soma.pipeline.console import build_completed_run_panel, build_run_summary_panel
from soma.pipeline.folds import FoldResult, run_fold
from soma.pipeline.layout import (
    RunRecorder,
    fold_dir,
    fold_is_complete,
    open_run,
    reject_undeclared_fold_dirs,
    register_test_result,
    source_identity,
    test_identity_digest,
)
from soma.pipeline.representation import run_representation
from soma.plan import Plan, check, dotted_packages, resolve
from soma.preprocessing.resolution import resolve_pipeline_dense, resolve_pipeline_preprocessing

__all__ = ["Pipeline", "PipelineResult"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run across all folds."""

    fold_results: list[FoldResult]
    summary: dict[str, float]
    run_dir: Path
    #: Folds still incomplete when this launch ended (``run.folds`` left them to another
    #: launch). Non-empty means no ``summary.json`` was written.
    pending_folds: tuple[int, ...] = ()
    #: The manifests this run trained and scored against, as ``experiment.json`` and
    #: ``run.yaml`` record them: the dataset and splits identity digests and the
    #: test-identity digest (issue #247).
    dataset_checksum: str = ""
    splits_checksum: str = ""
    test_checksum: str = ""


class Pipeline:
    """Runs a config: attach or extract features → train every fold → summarize.

    The constructor reads the cohort and resolves ``config`` into its :attr:`plan`
    (:func:`soma.resolve`), so a bad config fails before anything is extracted.

    Args:
        config: The experiment (:class:`~soma.config.ExperimentConfig`).
        source: Features to train on (a :class:`~soma.data.SetSource`,
            :class:`~soma.data.GridSource` or, for a live grid run, the pixels'
            :class:`~soma.data.ImageSource`). Without one, ``run()`` extracts them as the
            config describes (or builds the live encoder stage).
    """

    def __init__(self, config: ExperimentConfig, *, source: Any = None) -> None:
        ensure_default_logging()
        if not isinstance(config, ExperimentConfig):
            raise TypeError(f"Pipeline takes an ExperimentConfig, got {type(config).__name__}")
        check(config)  # config errors surface before any data is read
        self._config = config
        self._source = source
        self._cohort, self._image_manifest, self._annotation_manifest = load_pipeline_data(config)
        self._plan = resolve(config, self._cohort)
        # Fat-record bridge (slice 7 deletes it): the builders still consume these.
        self._dataset, self._splits = legacy_inputs(
            self._cohort, self._image_manifest, self._annotation_manifest, path=config.cohort.records
        )

    @property
    def config(self) -> ExperimentConfig:
        """The config as given (``config.yaml``)."""
        return self._config

    @property
    def plan(self) -> Plan:
        """The resolved plan (``plan.yaml``) every step of the run reads."""
        return self._plan

    @property
    def source(self) -> Any:
        return self._source

    @property
    def cohort(self) -> Cohort:
        return self._cohort

    @property
    def image_manifest(self) -> ImageManifest:
        return self._image_manifest

    @property
    def annotation_manifest(self) -> AnnotationManifest | None:
        return self._annotation_manifest

    @property
    def dataset(self) -> LegacySamples:
        """Legacy fat records (slice 7 deletes them)."""
        return self._dataset

    @property
    def splits(self) -> LegacyFolds:
        return self._splits

    def run(self) -> PipelineResult:
        """Run every fold not yet complete on disk, then summarize the run from disk."""
        plan, num_folds = self._plan, self._splits.num_folds
        if plan.evaluation.representation is not None:
            return run_representation(self)
        layout, mirror = open_run(plan, config=self._config, num_folds=num_folds)
        with RunRecorder(layout, plan) as recorder:
            inputs = self._prepare(layout)
            results = []
            for fold, fold_split in enumerate(inputs.splits.folds):
                directory = fold_dir(layout.run_dir, fold, num_folds)
                if self._skips(fold, directory):
                    if mirror is not None and fold_is_complete(directory):
                        mirror.fold_completed(fold=fold, fold_dir=directory)
                    continue
                results.append(run_fold(directory, build_fold(inputs, fold, fold_split)))
                if mirror is not None:
                    mirror.fold_completed(fold=fold, fold_dir=directory)
            pending = self._pending_folds(layout.run_dir)
            if pending:  # only with run.folds: the launch completing the run summarizes it
                logger.info("Fold(s) %s still pending; the launch completing them writes summary.json.", list(pending))
                recorder.partial(pending)
                return recorder.with_identity(PipelineResult(results, {}, layout.run_dir, pending_folds=pending))
            # The configured protocol decides the summary's splits, not the clobber guard's
            # hold-out of an already-scored test identity.
            summary = aggregate_fold_metrics_from_disk(
                layout.run_dir,
                single_fold=num_folds == 1,
                include_tune=plan.evaluation.holdout_test,
                num_folds=num_folds,
            )
            save_summary(summary, layout.run_dir / "summary.json")
            if plan.evaluation.attention_heatmaps is not None:
                self._render_heatmaps(layout.run_dir, inputs)
            _write_source_provenance(layout.run_dir, inputs.source)
            Console().print(build_completed_run_panel(summary_metrics=summary))
            recorder.complete(summary)
        return recorder.with_identity(PipelineResult(results, summary, layout.run_dir))

    # -- run() helpers ------------------------------------------------------------

    def _prepare(self, layout) -> RunInputs:
        """Attach the source, apply the test-clobber guard and print the run summary."""
        plan = self._plan
        folds = plan.run.folds
        if self._splits.num_folds > 1:
            reject_undeclared_fold_dirs(layout.run_dir, self._splits.num_folds)
        if folds is not None:
            unknown = sorted(set(folds) - set(range(self._splits.num_folds)))
            if self._splits.num_folds == 1 or unknown:
                raise ValueError(
                    f"run.folds={list(folds)} does not fit the splits: they declare "
                    f"{self._splits.num_folds} fold(s), indexed from 0."
                )
        context = attach_source(
            plan,
            source=self._source,
            dataset=self._dataset,
            splits=self._splits,
            image_manifest=self._image_manifest,
            run_dir=layout.run_dir,
        )
        preprocessing = resolve_pipeline_preprocessing(plan)
        dense = None
        if plan.shape == "grid":
            dense = plan.features.dense
            if dense is None and preprocessing.tiling is not None:
                dense = resolve_pipeline_dense(plan)
        Console().print(build_run_summary_panel(plan=plan, preprocessing=preprocessing, feature_store=context.source))
        return RunInputs(
            plan=plan,
            preprocessing=preprocessing,
            evaluation=self._guard_test_clobber(layout),
            source=context.source,
            cohort=self._cohort,
            dataset=context.dataset,
            splits=context.splits,
            run_id=layout.run_id,
            run_dir=layout.run_dir,
            source_identity=source_identity(context.source),
            encoder_stage=context.encoder_stage,
            dense=dense,
            roi_population_cache_root=(
                resolve_cache_root(
                    plan.features.cache, feature_dir=layout.run_dir / "features", output_root=plan.run.output_dir
                )
                / "segmentation_roi_population"
                if plan.features.cache.enabled
                else layout.run_dir / "segmentation_roi_population"
            ),
            packages=dotted_packages(plan),
        )

    def _pending_folds(self, run_dir: Path) -> tuple[int, ...]:
        num_folds = self._splits.num_folds
        return tuple(f for f in range(num_folds) if not fold_is_complete(fold_dir(run_dir, f, num_folds)))

    def _skips(self, fold: int, directory: Path) -> bool:
        """Resume (issue #244): a complete fold is skipped on a relaunch into its run
        dir (a single fold only under ``resume``/``run_id``); ``run.folds`` leaves the
        other folds to another launch."""
        run = self._plan.run
        resuming = self._splits.num_folds > 1 or bool(run.resume or run.run_id)
        if resuming and fold_is_complete(directory):
            logger.info("Fold %d already complete (%s); skipping (resume).", fold, directory)
            return True
        if run.folds is not None and fold not in run.folds:
            logger.info("Fold %d is not in run.folds; leaving it to another launch.", fold)
            return True
        return False

    def _guard_test_clobber(self, layout):
        """Test-results clobber guard (issue #247): a test identity already scored for
        this run is not re-scored unless ``evaluation.overwrite_test``; its test splits
        are held out instead. A resume scoring still-pending folds is not a re-score."""
        plan, evaluation = self._plan, self._plan.evaluation
        if evaluation.holdout_test:
            return evaluation
        digest = test_identity_digest(plan)
        decision = register_test_result(
            layout.run_dir,
            digest,
            split_names=sorted({name for fold in self._splits.folds for name in fold.test_split_names}),
            run_id=layout.run_id,
            overwrite=evaluation.overwrite_test,
        )
        if not decision.skipped:
            return evaluation
        pending = self._pending_folds(layout.run_dir)
        if pending:
            logger.info(
                "Test identity %s already recorded, but %d fold(s) are unscored; resuming "
                "and scoring them (completed folds are skipped, not re-scored).",
                digest[:12],
                len(pending),
            )
            return evaluation
        logger.warning(
            "Test identity %s was ALREADY scored for this run (recorded %s, splits %s); "
            "skipping test inference so the existing result is not clobbered. Set "
            "evaluation.overwrite_test=True to re-score.",
            digest[:12],
            (decision.prior or {}).get("recorded_at", "?"),
            (decision.prior or {}).get("split_names", []),
        )
        return replace(evaluation, holdout_test=True)

    def _render_heatmaps(self, run_dir: Path, inputs: RunInputs) -> None:
        """Attention heatmaps over the fold bundles (a fresh pass with the attention hook)."""
        from soma.heatmaps import generate_heatmaps

        plan = self._plan
        if plan.shape == "grid":
            logger.info("evaluation.attention_heatmaps: a %s run has no tile attention; skipping.", plan.task_family)
            return
        logger.info("Rendering attention heatmaps...")
        generate_heatmaps(
            run_dir=run_dir,
            dataset=inputs.dataset,
            feature_store=inputs.source,
            tiling_dir=run_dir / "tiling",
            seg_downsample=plan.features.extraction.segmentation.downsample,
            **plan.evaluation.attention_heatmaps,
        )


def _write_source_provenance(run_dir: Path, source: Any) -> None:
    """``dense_source.json``: where a cache-backed grid source came from, when it says."""
    provenance = getattr(source, "provenance", None)
    if provenance is None:
        return
    atomic_write_json(run_dir / "dense_source.json", provenance.to_dict(), sort_keys=True)
