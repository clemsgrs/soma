"""Drive the detection-benchmark encoder-ranking sweep (#246): extract-all → train+score-all.

This is the resumable driver for the multi-dataset frozen-probe ranking of Paper 1. It
generalizes ``examples/ocelot/campaign.py`` from one dataset to the three-dataset sweep and
replaces its tune-select → test-confirm flow with the **full-ranking protocol**: for every
``(encoder, dataset, replicate)`` it trains the ``lightweight_conv`` decoder on ``train``,
freezes every per-encoder knob (detection threshold, matching radius, val-best checkpoint)
on ``tune``, then scores on ``test`` with those frozen knobs. There is no ``pick_winner`` —
all encoders are ranked by mean test metric, and the tune ranks are reported alongside.

``examples/ocelot/campaign.py``'s **seed loop becomes a replicate loop**: a dataset whose
``splits.csv`` ships >1 fold has its folds as replicates (1 seed × F); a single-fold dataset
uses seeds (K, default 3). The harness aggregates over whichever axis a dataset declares —
``soma.benchmarks.detection_benchmark.replicate_plan`` resolves it from the splits.

Two idempotent phases, each with a per-cell skip guard so an interrupted sweep resumes by
re-invocation:

    # (1) extract-all — the expensive one-time dense grids (roster × 3 datasets), shared
    #     across all replicates of a cell (the cache is keyed by encoder/spacing/geometry).
    python examples/detection_benchmark/campaign.py extract --data-root data

    # (2) train+score-all — cheap per replicate; trains the decoder, freezes knobs on tune,
    #     scores test, persists per-sample predictions + metrics, then aggregates.
    python examples/detection_benchmark/campaign.py rank --data-root data

The ranking + rank-consistency + bootstrap stability + frozen selections + reference bands
are pure post-hoc aggregation over the per-cell metric / prediction cache (no GPU), so
``rank`` re-aggregates from disk on every re-invocation. The per-sample predictions are
persisted (predicted points + matched/unmatched flags), so the deferred MIDOG robustness
stratification (#248) and the stability bootstrap are pure later re-aggregations — no
retrain, no re-extract.

Each cell's config is the committed per-dataset benchmark YAML under
``soma/benchmarks/configs/detection/`` (loaded by the registered ``detection-benchmark``'s
``build_config``, with the roster encoder swapped in); this driver only overrides the
dataset paths (from ``--data-root``), the encoder, the replicate seed/fold, and the output
dir. Needs a GPU + HF token for the gated encoders. Run from the soma repo root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Import the working-tree soma (this repo), not a stale site-packages copy — mirrors
# examples/ocelot/eval_greedy.py's rationale so the dense cache key agrees with training.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soma.benchmarks.detection_benchmark import (  # noqa: E402
    DATASET_ORDER,
    DEFAULT_ROSTER,
    DEFAULT_SEEDS,
    Cell,
    CellPredictions,
    RosterEntry,
    aggregate_cell,
    build_ranking_report,
    dataset_spec,
    read_cell_predictions,
    replicate_plan,
)

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
OUT_DIR = REPO_ROOT / "output" / "detection_benchmark"  # reports land here (git-ignored)


# --- cell planning (pure, unit-tested) -----------------------------------------------


def dataset_replicate_ids(
    splits_csv: str | Path, *, seeds: Sequence[int] = DEFAULT_SEEDS
) -> tuple[str, list[int]]:
    """Resolve a dataset's ``(replicate_axis, ids)`` from its ``splits.csv`` fold structure.

    Reads only the fold count (never loads a dataset), so it is cheap and offline. Multi-fold
    → the folds are the replicates; single-fold → the seeds are. Delegates the policy to
    :func:`~soma.benchmarks.detection_benchmark.replicate_plan`.
    """
    import pandas as pd

    df = pd.read_csv(splits_csv)
    num_folds = int(df["fold"].nunique()) if "fold" in df.columns else 1
    return replicate_plan(num_folds, seeds=seeds)


def splits_csv_for(data_root: str | Path, dataset: str) -> Path:
    """The curated ``splits.csv`` for a dataset under ``--data-root`` (``<root>/<dataset>/curated``)."""
    return Path(data_root) / dataset / "curated" / "splits.csv"


def dataset_csv_for(data_root: str | Path, dataset: str) -> Path:
    """The curated ``dataset.csv`` for a dataset under ``--data-root``."""
    return Path(data_root) / dataset / "curated" / "dataset.csv"


def cell_dir(out_root: str | Path, dataset: str, encoder: str, replicate: int) -> Path:
    """The output dir for one ``(dataset, encoder, replicate)`` cell (checkpoint + metrics live here)."""
    return Path(out_root) / dataset / encoder / f"replicate_{replicate}"


def feature_cache_dir(out_root: str | Path, dataset: str) -> Path:
    """The dense-grid cache shared by every ``(encoder, replicate)`` cell of a dataset.

    ``CacheConfig.root_dir`` defaults to ``None`` ⇒ ``<run.output_root>/feature_cache``, and
    ``run.output_root`` is per-replicate — so without this the identical grids would be
    re-extracted and re-stored once per seed (3x the disk, 3x the extraction). The dense key
    (``build_dense_cache_key``) folds in encoder / geometry / preprocessing / dtype but *not*
    the seed or the output root, so one root per dataset is exactly right: replicates of an
    encoder collide (intended — that is the reuse) and distinct encoders never do.
    """
    return Path(out_root) / dataset / "feature_cache"


def training_done(out_root: str | Path, dataset: str, encoder: str, replicate: int) -> bool:
    """Skip guard for phase 2: this cell already trained *to completion*.

    Two traps here, both of which the old ``best_model.pt``-at-``cell_dir`` probe fell into:

    * **Layout.** soma does not drop artifacts at ``run.output_root``; it writes the run under
      ``<output_root>/experiments/<experiment_key>/runs/<timestamp>/``. Probing the direct path
      meant the guard never fired, so phase 2 retrained every replicate-0 cell phase 1 had
      already trained (~2h x one per encoder).
    * **Completion.** ``best_model.pt`` is rewritten on every epoch improvement, so it exists
      mid-training. Keying on it would let a crashed, half-trained cell be skipped and then
      scored as if it were final. ``summary.json`` is written once, at the end of a run, so it
      is the honest "this finished" marker.

    ``score_cell`` locates the weights with ``_locate_checkpoint``, which globs the same layout.
    """
    directory = cell_dir(out_root, dataset, encoder, replicate)
    if (directory / "summary.json").is_file():
        return True
    return any(directory.glob("experiments/*/runs/*/summary.json"))


def metrics_exists(out_root: str | Path, dataset: str, encoder: str, replicate: int) -> bool:
    """Skip guard: a scored cell already has its ``metrics.json`` (test + tune blocks)."""
    return (cell_dir(out_root, dataset, encoder, replicate) / "metrics.json").is_file()


def extraction_done(out_root: str | Path, dataset: str, encoder: str) -> bool:
    """Skip guard for phase 1: the dense grids for an ``(encoder, dataset)`` pair are cached.

    Extraction is replicate-independent (the dense cache is keyed by encoder/spacing/geometry
    and shared across a cell's replicates), so the marker lives at the ``(dataset, encoder)``
    level, not per replicate.
    """
    return (Path(out_root) / dataset / encoder / "extracted.marker").is_file()


def plan_cells(
    roster: Sequence[RosterEntry],
    datasets: Sequence[str],
    data_root: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    """Enumerate every ``(encoder, dataset, replicate)`` cell of the sweep.

    Roster-size-agnostic: the product of the roster with the datasets, each dataset's
    replicate axis resolved from its ``splits.csv``. Returns plain dicts (``encoder``,
    ``dataset``, ``replicate``, ``replicate_axis``) so the phases and the tests iterate a
    flat plan.
    """
    plan: list[dict[str, Any]] = []
    for dataset in datasets:
        axis, ids = dataset_replicate_ids(splits_csv_for(data_root, dataset), seeds=seeds)
        for entry in roster:
            for rid in ids:
                plan.append(
                    {
                        "encoder": entry.name,
                        "dataset": dataset,
                        "replicate": rid,
                        "replicate_axis": axis,
                    }
                )
    return plan


# --- per-cell metric cache (pure) ----------------------------------------------------


def read_cell_metrics(out_root: str | Path, dataset: str, encoder: str, replicate: int) -> dict:
    """Load one scored cell's ``metrics.json`` (``{"test": {...}, "tune": {...}, ...}``)."""
    path = cell_dir(out_root, dataset, encoder, replicate) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def collect_cells(
    out_root: str | Path, roster: Sequence[RosterEntry], datasets: Sequence[str]
) -> list[Cell]:
    """Aggregate every scored cell on disk into :class:`Cell` objects (native metric, mean±std).

    Skips ``(encoder, dataset)`` pairs with no scored replicate yet, so a partially-complete
    sweep still aggregates its finished cells (the report grows as the sweep progresses).
    """
    cells: list[Cell] = []
    for dataset in datasets:
        spec = dataset_spec(dataset)
        for entry in roster:
            per_replicate: list[float] = []
            tune_values: list[float] = []
            axis = "seeds"
            test_source = spec.test_source
            for path in sorted((Path(out_root) / dataset / entry.name).glob("replicate_*/metrics.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                test_block = data.get("test", {})
                if spec.metric_name not in test_block:
                    continue
                per_replicate.append(float(test_block[spec.metric_name]))
                tune_block = data.get("tune", {})
                if spec.metric_name in tune_block:
                    tune_values.append(float(tune_block[spec.metric_name]))
                axis = data.get("replicate_axis", axis)
                test_source = data.get("test_source", test_source)
            if per_replicate:
                cells.append(
                    aggregate_cell(
                        entry.name,
                        dataset,
                        per_replicate,
                        replicate_axis=axis,
                        test_source=test_source,
                        tune_per_replicate=tune_values or None,
                    )
                )
    return cells


def collect_stability_samples(
    out_root: str | Path, roster: Sequence[RosterEntry], datasets: Sequence[str]
) -> dict[str, dict[str, list]]:
    """Load each dataset's per-encoder test predictions (replicate 0) for the bootstrap.

    Only encoders whose replicate-0 predictions cover the *same* test sample-id set are kept
    per dataset — the paired bootstrap needs a shared universe. Predictions are read from the
    persisted ``predictions.json`` (the per-sample cache), so this is pure and GPU-free.
    """
    out: dict[str, dict[str, list]] = {}
    for dataset in datasets:
        encoder_samples: dict[str, list] = {}
        for entry in roster:
            candidates = sorted(
                (Path(out_root) / dataset / entry.name).glob("replicate_*/predictions.json")
            )
            if not candidates:
                continue
            preds: CellPredictions = read_cell_predictions(candidates[0])
            if preds.samples:
                encoder_samples[entry.name] = preds.samples
        if len(encoder_samples) >= 2:
            id_sets = {frozenset(s.sample_id for s in samples) for samples in encoder_samples.values()}
            if len(id_sets) == 1:
                out[dataset] = encoder_samples
    return out


def aggregate_and_report(
    out_root: str | Path,
    *,
    roster: Sequence[RosterEntry] = DEFAULT_ROSTER,
    datasets: Sequence[str] = DATASET_ORDER,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    git_sha: str | None = None,
    n_boot: int = 1000,
    write: bool = True,
) -> dict:
    """Build ``ranking_report.json`` from the on-disk per-cell metric / prediction cache (no GPU).

    Pure over the sweep's artifacts: collect the scored cells, load the per-encoder test
    predictions for the bootstrap, and assemble the full ranking report. Called at the end
    of the ``rank`` phase and safe to re-run any time (it just re-reads the cache).
    """
    cells = collect_cells(out_root, roster, datasets)
    stability = collect_stability_samples(out_root, roster, datasets)
    report = build_ranking_report(
        cells,
        roster=roster,
        stability_samples=stability or None,
        git_sha=git_sha,
        replicate_policy={"single_fold_seeds": list(seeds)},
        n_boot=n_boot,
    )
    if write:
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "ranking_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"wrote {Path(out_root) / 'ranking_report.json'} ({len(cells)} cells)")
    return report


# --- live orchestration (GPU; not exercised in CI) -----------------------------------


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _data_overrides(data_root: Path, dataset: str) -> list[str]:
    return [
        "--set", f"data.dataset_csv={dataset_csv_for(data_root, dataset)}",
        "--set", f"data.splits_csv={splits_csv_for(data_root, dataset)}",
    ]


def _config_path(dataset: str) -> Path:
    from soma.benchmarks.detection_benchmark import config_path_for

    return config_path_for(dataset)


def train_cell(
    encoder: str, dataset: str, replicate: int, axis: str, data_root: Path, out_root: Path
) -> None:
    """Train (+extract) one cell via ``python -m soma`` — the config supplies the recipe.

    The replicate maps onto ``run.seed`` for the seeds axis and onto a fold selection for the
    folds axis (multi-fold within-run resume is #244's job; here each fold is one cell). The
    committed config's encoder is swapped for the roster encoder.
    """
    cmd = [sys.executable, "-m", "soma", str(_config_path(dataset))]
    cmd += _data_overrides(data_root, dataset)
    cmd += ["--set", f"encoder.name={encoder}"]
    cmd += ["--set", f"run.output_root={cell_dir(out_root, dataset, encoder, replicate)}"]
    # Point every cell at the dataset-wide grid cache, else each replicate builds its own
    # private copy under its run dir. score_cell reloads the run's persisted config, so it
    # inherits this root too and reads the same grids it trained on.
    cmd += ["--set", f"cache.root_dir={feature_cache_dir(out_root, dataset)}"]
    if axis == "seeds":
        cmd += ["--set", f"run.seed={replicate}"]
    _run(cmd, cwd=REPO_ROOT)


def score_cell_isolated(
    encoder: str, dataset: str, replicate: int, axis: str, data_root: Path, out_root: Path
) -> None:
    """Run :func:`score_cell` in a child process, so its GPU memory dies with it.

    Scoring loads the decoder + the dense grids onto the GPU. Done in-process, PyTorch's
    caching allocator keeps those GiB reserved for the lifetime of this driver — and the
    launcher pins the driver and every cell it trains to the *same* GPU, so the next
    ``train_cell`` starts with several GiB already gone and OOMs (training needs nearly the
    whole 12 GB card). A child process hands it all back on exit. Same reason ``train_cell``
    shells out; scoring only got away with it while nothing trained after it.
    """
    cmd = [
        sys.executable, str(SCRIPT), "score",
        "--datasets", dataset,
        "--encoders", encoder,
        "--replicate", str(replicate),
        "--axis", axis,
        "--data-root", str(data_root),
        "--out-root", str(out_root),
    ]
    _run(cmd, cwd=REPO_ROOT)


def run_extract(data_root: Path, out_root: Path, roster, datasets, *, dry_run: bool) -> None:
    """Phase 1: populate the dense cache for every ``(encoder, dataset)`` pair (skip if done)."""
    for dataset in datasets:
        for entry in roster:
            if extraction_done(out_root, dataset, entry.name):
                print(f"[{dataset}/{entry.name}] extraction cached, skip")
                continue
            if dry_run:
                print(f"[{dataset}/{entry.name}] would extract dense grids")
                continue
            axis, ids = dataset_replicate_ids(splits_csv_for(data_root, dataset))
            # One pass over replicate ids[0] extracts the shared grids for all splits.
            train_cell(entry.name, dataset, ids[0], axis, data_root, out_root)
            marker = Path(out_root) / dataset / entry.name / "extracted.marker"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n", encoding="utf-8")


def run_rank(
    data_root: Path, out_root: Path, roster, datasets, seeds, *, dry_run: bool, git_sha: str | None
) -> dict:
    """Phase 2: train + freeze-on-tune + score-on-test every cell, then aggregate the report."""
    for cell in plan_cells(roster, datasets, data_root, seeds=seeds):
        enc, ds, rid, axis = cell["encoder"], cell["dataset"], cell["replicate"], cell["replicate_axis"]
        if metrics_exists(out_root, ds, enc, rid):
            print(f"[{ds}/{enc}/r{rid}] metrics cached, skip")
            continue
        if dry_run:
            print(f"[{ds}/{enc}/r{rid}] would train+score ({axis})")
            continue
        if not training_done(out_root, ds, enc, rid):
            train_cell(enc, ds, rid, axis, data_root, out_root)
        score_cell_isolated(enc, ds, rid, axis, data_root, out_root)
    return aggregate_and_report(
        out_root, roster=roster, datasets=datasets, seeds=seeds, git_sha=git_sha
    )


def score_cell(
    encoder: str, dataset: str, replicate: int, axis: str, data_root: Path, out_root: Path
) -> None:
    """Freeze knobs on tune, score test, and persist per-sample predictions + metrics (live path).

    Loads the trained decoder + the cached dense grids, decodes per-image points at the
    tune-frozen per-class thresholds, records each predicted point's matched/unmatched flag,
    then computes the dataset's native metric off those points. Writes ``predictions.json``
    (the per-sample cache) and ``metrics.json`` (``test`` + ``tune`` native-metric blocks).
    This is GPU/data-bound and not exercised in CI; the pure re-aggregation it feeds is.
    """
    from soma.benchmarks.detection_benchmark import score_dataset_points, write_cell_predictions

    directory = cell_dir(out_root, dataset, encoder, replicate)
    fold_index = replicate if axis == "folds" else 0
    tune_preds, test_preds = _decode_cell_points(dataset, replicate, directory, fold_index=fold_index)
    spec = dataset_spec(dataset)
    test_metrics = score_dataset_points(dataset, test_preds.samples)
    tune_metrics = score_dataset_points(dataset, tune_preds.samples) if tune_preds.samples else {}
    write_cell_predictions(directory / "predictions.json", test_preds)
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "test": test_metrics,
                "tune": tune_metrics,
                "replicate_axis": axis,
                "test_source": spec.test_source,
                "metric_name": spec.metric_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _decode_split_points(model, loader, head, device, manifest) -> list:
    """Decode one loader into per-sample :class:`SamplePrediction` (points + matched flags).

    Mirrors ``soma.pipeline._evaluate_detection``: decode NMS peaks at the frozen per-class
    thresholds, match once per image (the same ``match_assignment`` the headline counts read),
    and record each predicted point's matched/unmatched flag. Points are transformed to the
    level-0 frame, and every sample carries its evaluated ``area_mm2`` for the FROC per-mm²
    axis (MONKEY uses it; the other scorers ignore it). Returned in the frame the native
    point scorers consume.
    """
    import numpy as np
    import torch

    from soma.benchmarks.detection_benchmark import SamplePrediction
    from soma.detection.encode import transform_points_to_level0
    from soma.detection.froc import patch_area_mm2
    from soma.detection.matching import match_assignment

    top, left, crop_w, crop_h = head._crop_box
    samples: list[SamplePrediction] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch.features.to(device))
            gt_points = batch.targets["gt_points"]
            for b, sid in enumerate(batch.sample_ids):
                heatmap = out.logits[b]
                pred_xy, pred_cls, pred_score = head._predict_points(heatmap)
                gt_xy, gt_cls = head._strip_padding(gt_points[b])
                assignment = match_assignment(
                    pred_xy, pred_cls, pred_score, gt_xy, gt_cls,
                    num_classes=head.num_classes, delta=head.delta_px, method=head.matching,
                )
                matched = np.zeros(pred_xy.shape[0], dtype=bool)
                for m in assignment:
                    if m.pairs.shape[0]:
                        matched[m.pairs[:, 0]] = True
                spacing = head.spacing_for_sample(sid)
                xy_l0 = (
                    transform_points_to_level0(
                        pred_xy,
                        source_spacing_um=spacing.source_spacing_um,
                        effective_spacing_um=spacing.effective_spacing_um,
                        crop_top=top,
                        crop_left=left,
                    )
                    if pred_xy.shape[0]
                    else np.zeros((0, 2))
                )
                gt_l0 = (
                    transform_points_to_level0(
                        gt_xy,
                        source_spacing_um=spacing.source_spacing_um,
                        effective_spacing_um=spacing.effective_spacing_um,
                        crop_top=top,
                        crop_left=left,
                    )
                    if gt_xy.shape[0]
                    else np.zeros((0, 2))
                )
                samples.append(
                    SamplePrediction(
                        sample_id=str(sid),
                        pred_xy=xy_l0.tolist(),
                        pred_score=[float(s) for s in pred_score],
                        pred_class=[int(c) for c in pred_cls],
                        gt_xy=gt_l0.tolist(),
                        gt_class=[int(c) for c in gt_cls],
                        matched=matched.tolist(),
                        area_mm2=patch_area_mm2(
                            int(crop_w), int(crop_h), spacing.source_spacing_um
                        ),
                    )
                )
    return samples


def _decode_cell_points(
    dataset: str, replicate: int, run_dir: Path, *, fold_index: int = 0
) -> tuple[CellPredictions, CellPredictions]:
    """Reload a trained cell and decode its tune + test per-sample points (live GPU path).

    Rebuilds the model + dense grids + loaders from the run's saved config (the reload is
    dataset-agnostic — it reads everything from the config), sweeps the per-class detection
    thresholds on ``tune`` and freezes them, then decodes tune + test points at those frozen
    knobs. Requires a GPU/CPU torch runtime + the cached dense grids; not exercised in CI (no
    data/GPU), exactly like ``soma.benchmarks.ocelot._greedy_report_for_run``.
    """
    import functools
    import inspect
    import math

    import torch

    from soma.benchmarks.detection_benchmark import CellPredictions, dataset_spec
    from soma.benchmarks.ocelot import (
        _locate_checkpoint,
        _locate_run_config,
        build_detection_model_from_checkpoint,
    )
    from soma.config import load_config
    from soma.dataset import DetectionManifest, Splits
    from soma.dense import DenseFeatureStore
    from soma.dense_extraction import DenseTileFeatureExtractor
    from soma.encoders.validation import resolve_preprocessing_config
    from soma.pipeline import (
        _make_loaders,
        _resolve_detection_px,
        _resolve_detection_sample_spacings,
        _sweep_detection_thresholds,
    )
    from soma.tasks.detection import DetectionHead
    from soma.training.detection_dataset import DetectionDataset, detection_collate_fn

    spec = dataset_spec(dataset)
    cfg = load_config(str(_locate_run_config(run_dir)))
    manifest = DetectionManifest(cfg.dataset_csv)
    splits = Splits(cfg.splits_csv, manifest)
    fold_split = splits.folds[fold_index]
    train_records = [manifest.samples[s] for s in fold_split.train]
    tune_records = [manifest.samples[s] for s in fold_split.tune]
    test_by_split = {n: [manifest.samples[s] for s in ids] for n, ids in fold_split.tests.items()}

    from dataclasses import replace as _replace

    pre = resolve_preprocessing_config(cfg.encoder, cfg.preprocessing)
    cache_cfg = cfg.cache
    if cache_cfg.root_dir is None:
        cache_cfg = _replace(cache_cfg, root_dir=Path(cfg.output_root) / "feature_cache")
    extractor = DenseTileFeatureExtractor(
        manifest, cfg.encoder,
        target_size=int(pre.requested_tile_size_px), spacing_um=float(pre.requested_spacing_um),
        backend=pre.backend, tolerance=float(pre.tolerance),
        window_size=pre.dense_window_size, overlap=float(pre.dense_window_overlap),
        execution=cfg.execution, cache=cache_cfg, preprocessing=pre,
    )
    store = DenseFeatureStore(extractor.cache_dir())

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
        if p.get("sigma") is not None else delta_px / 3.0
    )
    geometry = store.geometry(train_records[0].sample_id)
    head = DetectionHead(
        num_classes=num_classes, geometry=geometry, delta_px=delta_px, sigma_px=sigma_px,
        nms_distance_px=delta_px, matching=spec.match_method,
        foreground_weight=float(p.get("foreground_weight", 10.0)),
        sample_spacings=sample_spacings,
        metrics=cfg.evaluation.metrics,
    )
    # One reconstruction path for every re-scorer: it rebuilds the run's feature adaptor
    # (issue #286) and sizes the decoder from the adaptor's output width, so a checkpoint
    # trained under a projection loads back into the model that wrote it.
    model = build_detection_model_from_checkpoint(
        store=store,
        checkpoint_path=_locate_checkpoint(run_dir),
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
        DetectionDataset, collate, train_records, tune_records, test_by_split,
        cfg.training, store, head.extract_targets,
    )
    # Freeze per-class detection thresholds on tune, then decode both splits at those knobs.
    head.score_threshold = _sweep_detection_thresholds(model, tune_loader, device, head)
    tune_samples = _decode_split_points(model, tune_loader, head, device, manifest)
    test_samples: list = []
    for loader in test_loaders.values():
        test_samples.extend(_decode_split_points(model, loader, head, device, manifest))

    make = lambda samples: CellPredictions(  # noqa: E731
        encoder=cfg.encoder.name, dataset=dataset, replicate=replicate,
        metric_name=spec.metric_name, spacing_um=spec.spacing_um, samples=samples,
    )
    return make(tune_samples), make(test_samples)


def _resolve_roster(names: Sequence[str] | None) -> tuple[RosterEntry, ...]:
    if not names:
        return DEFAULT_ROSTER
    by_name = {e.name: e for e in DEFAULT_ROSTER}
    return tuple(by_name.get(n, RosterEntry(n)) for n in names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phase", choices=["extract", "rank", "score"])
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                    help="root holding <dataset>/curated/{dataset,splits}.csv per dataset")
    ap.add_argument("--out-root", type=Path, default=OUT_DIR)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER))
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="roster subset by name (default: the full DEFAULT_ROSTER)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="seed replicates for single-fold datasets (folds datasets ignore this)")
    ap.add_argument("--replicate", type=int, default=0,
                    help="score phase only: which replicate of the cell to score")
    ap.add_argument("--axis", choices=["seeds", "folds"], default="seeds",
                    help="score phase only: the cell's replicate axis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    roster = _resolve_roster(args.encoders)
    if args.phase == "score":
        # The out-of-process half of score_cell_isolated: exactly one cell, then exit so the
        # GPU memory is returned before the caller trains the next one.
        if len(args.datasets) != 1 or not args.encoders or len(args.encoders) != 1:
            raise SystemExit(
                "score scores a single cell: pass one --datasets and one --encoders"
            )
        score_cell(
            args.encoders[0], args.datasets[0], args.replicate, args.axis,
            args.data_root, args.out_root,
        )
        return 0
    if args.phase == "extract":
        run_extract(args.data_root, args.out_root, roster, args.datasets, dry_run=args.dry_run)
        return 0
    from soma.output_layout import _git_sha

    run_rank(
        args.data_root, args.out_root, roster, args.datasets, args.seeds,
        dry_run=args.dry_run, git_sha=_git_sha(REPO_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
