"""OCELOT 2023 cell-detection benchmark (registered).

This is the first registered :class:`~soma.benchmarks.registry.Benchmark` (ADR 0002). It
absorbs the former ``examples/ocelot/`` harness into first-class package code:

* the greedy (OCELOT-official) matcher — :func:`build_greedy_report` and
  :func:`_greedy_report_for_run` — becomes the :meth:`OcelotBenchmark.score` override
  (greedy is OCELOT's leaderboard-comparable matcher, not soma's headline metric);
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

import functools
import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
)
from soma.config import PipelineConfig, load_config
from soma.curation.manifest import CuratedManifest
from soma.curation.ocelot import curate_ocelot_detection

if TYPE_CHECKING:
    from soma.dataset import DetectionManifest

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


def build_greedy_report(
    *,
    model,
    head,
    device,
    tune_loader,
    test_loaders: dict,
    dataset,
    matching: str,
) -> dict:
    """Greedy (OCELOT-official) report: the leakage-free headline.

    Per-class score thresholds are swept on the *tune* split, frozen, then applied once to
    each test split — the leakage-free number a real submitter reports. No oracle ceiling.
    The greedy matcher is whatever ``head.matching`` selects, so the sweep and the
    evaluations stay greedy-consistent.

    Pass an empty ``test_loaders`` for a tune-only report: the returned dict then carries
    just ``matching``, ``score_threshold_per_class``, and ``tune``.
    """
    from soma.pipeline import _evaluate_detection, _sweep_detection_thresholds

    headline_thresholds = _sweep_detection_thresholds(model, tune_loader, device, head)
    head.score_threshold = headline_thresholds
    tune_report = _evaluate_detection(
        model, tune_loader, "tune", device, head=head, dataset=dataset
    )
    out: dict = {
        "matching": matching,
        "score_threshold_per_class": headline_thresholds,
        "tune": tune_report.metrics,
    }
    for name, loader in test_loaders.items():
        head.score_threshold = headline_thresholds
        headline = _evaluate_detection(model, loader, name, device, head=head, dataset=dataset)
        out[name] = {
            "headline": {
                "note": "reported result — per-class thresholds frozen from the tune split (leakage-free)",
                "score_threshold_per_class": headline_thresholds,
                "metrics": headline.metrics,
            },
        }
    return out


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


def build_detection_model_from_checkpoint(
    *,
    store,
    checkpoint_path: Path,
    decoder,
    task_head,
    geometry,
    normalization=None,
    projection=None,
    encoder_identity: str = "",
):
    """Rebuild a trained dense-detection model from its checkpoint, for re-scoring.

    Reconstructing a trained model from config + checkpoint breaks two ways once a run can
    carry a feature adaptor (issue #286), and only one of them is visible from the keys:

    * the strict ``load_state_dict`` rejects the adaptor's extra buffers unless the adaptor
      is rebuilt here too, and
    * the decoder must be built against the adaptor's **output** width — a decoder built
      against the encoder's native dim loads the checkpoint's ``target_dim``-shaped weights
      by name but then reads the wrong number of channels.

    The adaptor is built **unfitted**: the checkpoint's buffers are the fitted state, and
    loading them is exactly what makes the re-score apply the transform the run trained
    under. Returns the model in eval mode on CPU; the caller moves it.
    """
    import torch

    from soma.decoders.registry import build_decoder_for_grid
    from soma.training.feature_adaptor import (
        build_feature_adaptor,
        feature_adaptor_output_dim,
    )
    from soma.training.model import SegmentationModel

    feature_adaptor = build_feature_adaptor(
        normalization,
        projection,
        num_features=store.feature_dim,
        encoder_identity=encoder_identity,
    )
    decoder_obj = build_decoder_for_grid(
        decoder.name,
        decoder.params,
        geometry=geometry,
        input_dim=feature_adaptor_output_dim(
            feature_adaptor, num_features=store.feature_dim
        ),
        num_classes=task_head.num_classes,
    )
    model = SegmentationModel(
        decoder=decoder_obj, task_head=task_head, feature_adaptor=feature_adaptor
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
    )
    return model.eval()


def _locate_checkpoint(run_dir: Path) -> Path:
    """The newest ``best_model.pt`` under ``run_dir`` (deterministic across re-runs)."""
    direct = run_dir / "best_model.pt"
    if direct.is_file():
        return direct
    candidates = sorted(
        run_dir.glob("experiments/*/runs/*/best_model.pt"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"no best_model.pt under {run_dir}")
    return candidates[-1]


def resolve_dense_cache_dir(
    cfg: PipelineConfig, manifest: DetectionManifest
) -> Path | None:
    """Resolve the exact dense-image cache directory described by ``cfg``."""
    from soma.dense_extraction import DenseTileFeatureExtractor
    from soma.encoders.validation import resolve_preprocessing_config

    preprocessing = resolve_preprocessing_config(cfg.encoder, cfg.preprocessing)
    cache = cfg.cache
    if cache.root_dir is None:
        cache = replace(cache, root_dir=Path(cfg.output_root) / "feature_cache")
    extractor = DenseTileFeatureExtractor(
        manifest,
        cfg.encoder,
        target_size=int(preprocessing.requested_tile_size_px),
        spacing_um=float(preprocessing.requested_spacing_um),
        backend=preprocessing.backend,
        tolerance=float(preprocessing.tolerance),
        window_size=preprocessing.dense_window_size,
        overlap=float(preprocessing.dense_window_overlap),
        execution=cfg.execution,
        cache=cache,
        preprocessing=preprocessing,
    )
    return extractor.cache_dir()


def _greedy_report_for_run(run_dir: str | Path, *, matching: str = "greedy") -> dict:
    """Re-score a trained OCELOT run with the greedy matcher (no training).

    Reloads the run's saved config, the dense grids it trained on (recomputed cache key),
    and its ``best_model.pt``, then re-runs just the evaluation half with greedy matching.
    Requires the cached dense grids + a GPU/CPU torch runtime; this is the live path
    ``soma reproduce ocelot --from-run-dir`` drives. Not exercised in CI (no data/GPU).
    """
    import torch

    from soma.dataset import DetectionManifest, Splits
    from soma.dense import DenseFeatureStore
    from soma.pipeline import (
        _make_loaders,
        _resolve_detection_px,
        _resolve_detection_sample_spacings,
    )
    from soma.tasks.detection import DetectionHead
    from soma.training.detection_dataset import DetectionDataset, detection_collate_fn

    run_dir = Path(run_dir)
    config_path = _locate_run_config(run_dir)
    cfg = load_config(str(config_path))
    manifest = DetectionManifest(cfg.dataset_csv)
    splits = Splits(cfg.splits_csv, manifest)
    fold_split = splits.folds[0]
    train_records = [manifest.samples[s] for s in fold_split.train]
    probe_id = train_records[0].sample_id

    store_dir = resolve_dense_cache_dir(cfg, manifest)
    if store_dir is None:
        raise RuntimeError(
            "caching is disabled in this config; greedy re-scoring needs the cached dense "
            "grids the run trained on."
        )
    store = DenseFeatureStore(store_dir)
    if probe_id not in store.available_samples:
        raise FileNotFoundError(
            f"recomputed dense cache dir {store_dir} does not contain sample '{probe_id}' "
            f"({len(store.available_samples)} samples present); the config likely no longer "
            f"matches the run, or extraction is incomplete."
        )
    ckpt = _locate_checkpoint(run_dir)

    tune_records = [manifest.samples[s] for s in fold_split.tune]
    test_by_split = {n: [manifest.samples[s] for s in ids] for n, ids in fold_split.tests.items()}

    p = dict(cfg.task.params)
    num_classes = int(p["num_classes"])
    all_records = train_records + tune_records + [
        record for records in test_by_split.values() for record in records
    ]
    sample_spacings, effective_spacing_um = _resolve_detection_sample_spacings(
        store, all_records
    )
    delta_px = _resolve_detection_px(
        float(p["match_distance"]), effective_spacing_um, "match_distance"
    )
    sigma_px = (
        _resolve_detection_px(float(p["sigma"]), effective_spacing_um, "sigma")
        if p.get("sigma") is not None
        else delta_px / 3.0
    )
    geometry = store.geometry(train_records[0].sample_id)

    head = DetectionHead(
        num_classes=num_classes,
        geometry=geometry,
        delta_px=delta_px,
        sigma_px=sigma_px,
        nms_distance_px=delta_px,
        matching=matching,
        foreground_weight=float(p.get("foreground_weight", 10.0)),
        sample_spacings=sample_spacings,
        metrics=cfg.evaluation.metrics,
    )

    # Rebuild the trained model — including any feature adaptor the run carried, whose
    # buffers and rewired decoder width the strict load below depends on (issue #286).
    model = build_detection_model_from_checkpoint(
        store=store,
        checkpoint_path=ckpt,
        decoder=cfg.decoder,
        task_head=head,
        geometry=geometry,
        normalization=cfg.normalization,
        projection=cfg.projection,
        encoder_identity=cfg.encoder.name if cfg.encoder is not None else "",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    collate = functools.partial(detection_collate_fn, target_dtypes=head.target_dtypes)
    _, tune_loader, test_loaders = _make_loaders(
        DetectionDataset,
        collate,
        train_records,
        tune_records,
        test_by_split,
        cfg.training,
        store,
        head.extract_targets,
    )
    return build_greedy_report(
        model=model,
        head=head,
        device=device,
        tune_loader=tune_loader,
        test_loaders=test_loaders,
        dataset=manifest,
        matching=matching,
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
    ) -> PipelineConfig:
        """Resolve a committed OCELOT config for the ``(encoder, spacing)`` axes.

        Loads the static YAML for the axes and repoints it at the caller's data/output
        paths and seed (a committed config carries an author's machine paths). Extra
        ``overrides`` are merged onto the user-facing config layout last.
        """
        config_path = _config_path_for(encoder, spacing)
        extends_encoder_roster = (encoder, round(float(spacing), 2)) not in _CONFIG_FILES
        merged: dict[str, Any] = {}
        data_over: dict[str, Any] = {}
        if dataset_csv is not None:
            data_over["dataset_csv"] = str(dataset_csv)
        if splits_csv is not None:
            data_over["splits_csv"] = str(splits_csv)
        if data_over:
            merged["data"] = data_over
        run_over: dict[str, Any] = {}
        if output_root is not None:
            run_over["output_root"] = str(output_root)
        if seed is not None:
            run_over["seed"] = int(seed)
        if run_over:
            merged["run"] = run_over
        if extends_encoder_roster:
            merged.setdefault("run", {})["tags"] = [
                "ocelot",
                "detection",
                encoder,
                "lightweight_conv",
                f"spacing_{float(spacing):.2f}".replace(".", "p"),
            ]
            merged["encoder"] = {
                "name": encoder,
                # The benchmark intentionally fixes spacing across encoders, including when
                # that spacing falls outside a new encoder's recommended operating regime.
                "allow_non_recommended_settings": True,
            }
        if overrides:
            for section, values in overrides.items():
                merged.setdefault(section, {}).update(values)
        return load_config(config_path, overrides=merged or None)

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """Reference rows matching ``axes`` (the broad banner matches any axes)."""
        return expected_rows(self.name, **axes)

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """Greedy (OCELOT-official) re-score of a trained run — the ``score`` override."""
        report = _greedy_report_for_run(run_dir, matching="greedy")
        return extract_test_metrics(report)


OCELOT = OcelotBenchmark()
register_benchmark(OCELOT)
