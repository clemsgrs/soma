"""OCELOT 2023 cell-detection benchmark (registered).

This is the first registered :class:`~soma.benchmarks.registry.Benchmark` (ADR 0002). It
absorbs the former ``examples/ocelot/`` harness into first-class package code:

* the greedy (OCELOT-official) matcher — :func:`build_greedy_report` and
  :func:`_greedy_report_for_run`, which re-sweep and re-score the run's persisted
  ``PointPredictions`` with ``matching="greedy"`` — is the :meth:`OcelotBenchmark.score`
  override (greedy is OCELOT's leaderboard-comparable matcher, not soma's headline metric);
* the static per-cell configs become committed package YAML that
  :meth:`OcelotBenchmark.build_config` loads by ``(encoder, spacing)`` axes;
* the expected numbers move to ``reference/ocelot.csv`` (a broad, config-agnostic band
  with a per-row tolerance);
* ``reproduce.py`` / ``eval_greedy.py`` are superseded by the generic ``soma reproduce``.

The recipe backbone (frozen encoder -> dense token grid -> ``lightweight_conv`` decoder ->
per-class peak heatmap -> class-aware F1 @ delta=3 um, greedy-matched) is fixed. Canonical
``soma reproduce`` varies the ``encoder`` and fixes spacing at the Virchow2 @ 0.2 um/px
anchor. ``build_config`` also resolves spacing sweeps, whose recorded results remain
attributable on that second benchmark axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
)
from soma.config import ExperimentConfig
from soma.curation.manifest import CuratedManifest
from soma.curation.ocelot import curate_ocelot_detection

if TYPE_CHECKING:
    from soma.data._legacy import LegacySamples

_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "ocelot"

# (encoder, spacing um/px) -> committed config filename. These are the 2x2 mag-alignment
# ablation plus the native-spacing anchor (issue #152); build_config loads one by axes.
_CONFIG_FILES: dict[tuple[str, float], str] = {
    ("virchow2", 0.2): "ocelot_virchow2_0.20.yaml",
    ("virchow2", 0.25): "ocelot_virchow2_0.25.yaml",
    ("virchow2", 0.5): "ocelot_virchow2_0.50.yaml",
    ("uni2", 0.25): "ocelot_uni2_0.25.yaml",
    ("uni2", 0.5): "ocelot_uni2_0.50.yaml",
}

BENCHMARK_NAME = "ocelot"
PRIMARY_METRIC = "mean_f1"
# The recorded anchor is single-seed (seed 0); the reference band IS that seed-0 value, so
# the default reproduce run is seed 0 only — a like-for-like comparison to the published
# number (examples-era RESULTS.md / expected_metrics.json, canonical_seed = 0).
CANONICAL_SEEDS: tuple[int, ...] = (0,)
ANCHOR_ENCODER = "virchow2"
ANCHOR_SPACING = 0.2

# Canonical reproduction fixes spacing at the anchor. Committed configs still expose the
# historical spacing sweep, but it is not part of this benchmark facet: those protocols are
# migration-validation evidence rather than cells in the canonical encoder comparison.
FACET = Facet(
    fixed={
        "task": "detection",
        "decoder": "lightweight_conv",
        "matcher": "greedy_f1@delta=3um",
    },
    varied=("encoder",),
)

# A small reference environment shown alongside a run (the recorded anchor environment).
REFERENCE_ENVIRONMENT: dict[str, str] = {
    "soma": "1.5.1",
    "slide2vec": "5.1.1",
    "torch": "2.7.1+cu128",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 2080 Ti",
}


# --- greedy (OCELOT-official) scorer -------------------------------------------------


def _locate_run_config(run_dir: Path) -> Path:
    """The saved ``config.yaml`` for the run under ``run_dir`` (output_root or run dir)."""
    direct = run_dir / "config.yaml"
    if direct.is_file():
        return direct
    configs = sorted(
        run_dir.glob("experiments/*/runs/*/config.yaml"), key=lambda p: p.stat().st_mtime
    )
    if configs:
        return configs[-1]
    raise FileNotFoundError(f"no config.yaml found under {run_dir}")


def resolve_dense_cache_dir(
    cfg: ExperimentConfig, manifest: LegacySamples
) -> Path | None:
    """Prepare and return the exact dense-image cache directory described by ``cfg``."""
    from soma import FeatureExtractor
    from soma.preprocessing.resolution import resolve_pipeline_dense, resolve_pipeline_preprocessing

    features = cfg.features
    cache = features.cache
    if cache.root_dir is None:
        cache = replace(cache, root_dir=Path(cfg.run.output_dir) / "feature_cache")
    extraction = FeatureExtractor(
        manifest,
        features.encoder,
        resolve_pipeline_preprocessing(cfg),
        shape="grid",
        unit="tile",
        dense=resolve_pipeline_dense(cfg),
        spacing_policy=features.spacing_policy,
        execution=features.execution,
        cache=cache,
        output_root=Path(cfg.run.output_dir) / "rescore_extraction",
    ).extract()
    return extraction.artifacts.feature_dir.parent


def build_greedy_report(
    *,
    tune,
    tests: dict,
    targets,
    matching: str = "greedy",
    metrics: Sequence[str] = ("mean_f1",),
) -> dict:
    """Greedy (OCELOT-official) report over persisted point predictions.

    Per-class score thresholds are swept on the *tune* predictions with ``matching``,
    frozen, then applied once to each test split — the leakage-free number a real
    submitter reports. ``tune`` and each of ``tests`` are
    :class:`~soma.evaluation.PointPredictions` carrying every peak above the head's
    ``min_score``; ``targets`` is the run's :class:`~soma.data.PointSource`.

    Pass an empty ``tests`` for a tune-only report: the returned dict then carries just
    ``matching``, ``score_threshold_per_class``, and ``tune``.
    """
    from soma.evaluation import evaluate, sweep_threshold

    leaves = [{"name": name, "params": {"matching": matching}} for name in metrics]
    threshold, _ = sweep_threshold(
        tune, targets.subset(tune.sample_ids), per_class=True, matching=matching
    )
    per_class = [float(threshold[c]) for c in range(tune.num_classes)]
    tune_evaluation = evaluate(
        tune.with_threshold(threshold), targets.subset(tune.sample_ids), metrics=leaves
    )
    out: dict = {
        "matching": matching,
        "score_threshold_per_class": per_class,
        "tune": tune_evaluation.metrics,
    }
    for name, predictions in tests.items():
        headline = evaluate(
            predictions.with_threshold(threshold),
            targets.subset(predictions.sample_ids),
            metrics=leaves,
        )
        out[name] = {
            "headline": {
                "note": "reported result — per-class thresholds frozen from the tune split (leakage-free)",
                "score_threshold_per_class": per_class,
                "metrics": headline.metrics,
            },
        }
    return out


def _greedy_report_for_run(run_dir: str | Path, *, matching: str = "greedy") -> dict:
    """Re-score a trained OCELOT run with the greedy matcher (no training, no model).

    Reads the run's saved config and its persisted ``predictions_tune.parquet`` /
    ``predictions_<test>.parquet`` (every peak above ``min_score``, level-0 pixels), then
    re-sweeps and re-scores with ``matching`` against the run's point annotations.
    """
    from soma.class_scheme import resolve_classes
    from soma.data import AnnotationManifest, PointSource
    from soma.evaluation import PointPredictions
    from soma.evaluation.metrics import resolve_metrics

    config_path = _locate_run_config(Path(run_dir))
    folder = config_path.parent
    cfg = ExperimentConfig.from_yaml(config_path)
    head = cfg.learner.params["head"]
    tune_path = folder / "predictions_tune.parquet"
    if not tune_path.is_file():
        raise FileNotFoundError(
            f"no predictions_tune.parquet in {folder}; greedy re-scoring reads the run's "
            "persisted point predictions"
        )
    tests = {
        path.stem.removeprefix("predictions_"): PointPredictions.from_parquet(path)
        for path in sorted(folder.glob("predictions_*.parquet"))
        if path != tune_path
    }
    _, _, class_remap, dropped = resolve_classes(head.params, excluded_key="drop", subject="detection")
    targets = PointSource.from_manifest(
        AnnotationManifest.from_csv(cfg.cohort.records), class_remap=class_remap, drop=dropped
    )
    return build_greedy_report(
        tune=PointPredictions.from_parquet(tune_path),
        tests=tests,
        targets=targets,
        matching=matching,
        metrics=resolve_metrics("detection", cfg.evaluation.metrics),
    )


def extract_test_metrics(report: dict, split: str | None = None) -> dict[str, float]:
    """Flatten the greedy report's leakage-free test headline into ``{metric: value}``.

    ``split`` selects a test split by name; by default the ``test`` split (or the first
    test split present) is used. The tune metrics are merged under ``tune_<metric>``.
    """
    test_splits = [
        k for k, v in report.items() if isinstance(v, dict) and "headline" in v
    ]
    if not test_splits:
        raise ValueError(f"greedy report has no test-split headline block: keys={list(report)}")
    chosen = split or ("test" if "test" in test_splits else test_splits[0])
    metrics = {str(k): float(v) for k, v in report[chosen]["headline"]["metrics"].items()}
    for key, value in report.get("tune", {}).items():
        metrics.setdefault(f"tune_{key}", float(value))
    return metrics


# --- build_config helpers ------------------------------------------------------------


def _config_path_for(encoder: str, spacing: float) -> Path:
    key = (encoder, round(float(spacing), 2))
    filename = _CONFIG_FILES.get(key)
    if filename is None:
        # The committed files define the protocol at each supported spacing, not a closed
        # encoder roster. Reuse the anchor encoder's spacing template for a new encoder;
        # build_config replaces the encoder-specific fields below.
        spacing_key = key[1]
        filename = _CONFIG_FILES.get((ANCHOR_ENCODER, spacing_key))
    if filename is None:
        known = ", ".join(f"{e}@{s}" for e, s in sorted(_CONFIG_FILES))
        raise KeyError(
            f"no committed OCELOT protocol for spacing={spacing!r}; "
            f"available encoder/spacing examples: {known}."
        ) from None
    return _CONFIG_DIR / filename


# --- benchmark object ----------------------------------------------------------------


class OcelotBenchmark:
    """OCELOT 2023 cell-detection benchmark (protocol-as-code)."""

    name = BENCHMARK_NAME
    facet = FACET
    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reference_environment = REFERENCE_ENVIRONMENT

    def curate(
        self,
        raw_root: str | Path,
        out_dir: str | Path,
    ) -> CuratedManifest:
        """Curate raw OCELOT into a soma detection Manifest (delegates to the curator)."""
        return curate_ocelot_detection(raw_root, out_dir)

    def build_config(
        self,
        *,
        encoder: str = ANCHOR_ENCODER,
        spacing: float = ANCHOR_SPACING,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> ExperimentConfig:
        """Resolve a committed OCELOT config for the ``(encoder, spacing)`` axes.

        Loads the static YAML for the axes and repoints it at the caller's data/output
        paths and seed (a committed config carries an author's machine paths). The
        ``overrides`` block (``{"cache": {...}}``) is merged into ``features.cache``.
        """
        config_path = _config_path_for(encoder, spacing)
        extends_encoder_roster = (encoder, round(float(spacing), 2)) not in _CONFIG_FILES
        merged: dict[str, Any] = {}
        cohort_over: dict[str, Any] = {}
        if dataset_csv is not None:
            cohort_over["records"] = str(dataset_csv)
        if splits_csv is not None:
            cohort_over["splits"] = str(splits_csv)
        if cohort_over:
            merged["cohort"] = cohort_over
        run_over: dict[str, Any] = {}
        if output_root is not None:
            run_over["output_dir"] = str(output_root)
        if seed is not None:
            run_over["seed"] = int(seed)
        if run_over:
            merged["run"] = run_over
        if extends_encoder_roster:
            # The benchmark intentionally fixes spacing across encoders, including when
            # that spacing falls outside a new encoder's recommended operating regime.
            merged.setdefault("features", {})["encoder"] = {
                "name": encoder,
                "params": {"allow_non_recommended_settings": True},
            }
        if overrides and overrides.get("cache"):
            merged.setdefault("features", {})["cache"] = dict(overrides["cache"])
        return ExperimentConfig.from_yaml(config_path, overrides=merged)

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """Reference rows for the resolved ``encoder`` axis.

        The gate row is pinned to the anchor encoder (Virchow2; spacing is fixed at
        0.2 µm/px by the facet, so it is not a key column), so an omitted encoder defaults
        to the anchor: ``expected()`` selects the anchor band, while
        ``expected(encoder="uni2")`` returns only external anchors (no gate for uni2).
        """
        merged: dict[str, Any] = {"encoder": ANCHOR_ENCODER}
        merged.update({k: v for k, v in axes.items() if v is not None})
        return expected_rows(self.name, **merged)

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """Greedy (OCELOT-official) re-score of a trained run — the ``score`` override."""
        report = _greedy_report_for_run(run_dir, matching="greedy")
        return extract_test_metrics(report)


OCELOT = OcelotBenchmark()
register_benchmark(OCELOT)
