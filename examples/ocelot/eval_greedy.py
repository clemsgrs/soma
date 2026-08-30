"""Re-score a trained OCELOT detection fold with greedy (OCELOT-official) matching.

Matching method only affects evaluation, not the trained weights, so we reuse the
cached dense grids + the saved ``best_model.pt`` and re-run just the back half of
``train_one_detection_fold`` with ``matching="greedy"``: re-sweep per-class score
thresholds on the tune split, then score tune + test. Prints both so they can be
compared to the Hungarian headline in the run's ``metrics.json``.

For each test split this emits the leakage-free **headline**: per-class thresholds
frozen from the tune split and applied once to test — exactly what a real submitter
reports. (No oracle ceiling: the test-side threshold sweep that #146 added for the
health gate is gone now the gate has served its purpose — see #152.)

Usage (from the soma repo; slide2vec>=5.0.0 must be importable):
    python examples/ocelot/eval_greedy.py \
        --run-dir /maindisk/clement/runs/ocelot_conch_lightconv \
        --config  examples/ocelot/ocelot.yaml \
        --matching greedy
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

# Import the working-tree soma (this repo), not a stale site-packages copy. soma is not
# editable-installed in this environment, and when this file runs as a script sys.path[0]
# is examples/ocelot/, so `import soma` would otherwise resolve to the installed package.
# `python -m soma` training already uses the working tree; the re-scorer must match it, or
# it could recompute a dense cache key that disagrees with the grids training wrote. Prepend
# the repo root (…/examples/ocelot/eval_greedy.py -> parents[2]) ahead of site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from soma import FeatureExtractor
from soma.config import load_config
from soma.dataset import DetectionManifest, Splits
from soma.encoders.validation import resolve_preprocessing_config
from soma.pipeline import (
    _make_loaders,
    _resolve_detection_px,
    _resolve_detection_sample_spacings,
)

# The greedy matcher is now first-class package code (soma/benchmarks/ocelot.py): it IS the
# OCELOT benchmark's `score` override (ADR 0002). This script stays as the thin CLI that
# examples/ocelot/campaign.py drives per (cell, seed); it re-uses that single definition
# instead of keeping a duplicate. New reproductions should prefer `soma reproduce ocelot`.
from soma.benchmarks.ocelot import (  # noqa: F401
    build_detection_model_from_checkpoint,
    build_greedy_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="output_root of the trained run")
    ap.add_argument("--config", type=Path, required=True, help="the run's config yaml")
    ap.add_argument("--matching", default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument(
        "--run-subdir",
        type=Path,
        default=None,
        help="specific experiments/*/runs/<ts> dir holding best_model.pt; defaults to the "
        "newest run under --run-dir (disambiguates when several runs share an output_root)",
    )
    ap.add_argument(
        "--tune-only",
        action="store_true",
        help="score the tune split only (no test inference). Mirrors evaluation.holdout_test: "
        "use for model-selection re-scoring of runs whose test grids were never cached.",
    )
    args = ap.parse_args()

    cfg = load_config(str(args.config))
    manifest = DetectionManifest(cfg.dataset_csv)
    splits = Splits(cfg.splits_csv, manifest)
    fold_split = splits.folds[0]
    train_records = [manifest.samples[s] for s in fold_split.train]
    probe_id = train_records[0].sample_id

    # Locate the dense grids this run trained on. A run dir can accumulate more than one
    # dense cache-key subdir — e.g. an empty orphan left by a since-changed key (the
    # 1024->410/819 target_size fix) sitting next to the populated one — so a blind
    # next(glob("*")) can grab the wrong dir. Recompute the *exact* key from the config
    # through the same canonical extractor the pipeline uses. On a complete hit this
    # validates and opens the exact cache without loading the encoder or rewriting grids.
    # Resolve encoder-driven preprocessing defaults exactly as the pipeline does before
    # extraction (soma/pipeline.py::_resolve_preprocessing → resolve_preprocessing_config):
    # it fills derived fields (read_tile_size_px / ref_tile_size_px from
    # requested_tile_size_px) that feed the dense cache key. Skipping it recomputes a
    # *different*, wrong key — the raw config leaves those fields None.
    pre = resolve_preprocessing_config(cfg.encoder, cfg.preprocessing)
    cache_cfg = cfg.cache
    if cache_cfg.root_dir is None:
        cache_cfg = replace(cache_cfg, root_dir=args.run_dir / "feature_cache")
    extraction = FeatureExtractor(
        manifest,
        cfg.encoder,
        pre,
        execution=cfg.execution,
        cache=cache_cfg,
        output_root=args.run_dir / "rescore_extraction",
    ).extract()
    store = extraction.source
    store_dir = extraction.artifacts.feature_dir.parent
    # Fail loud if the recomputed key points where the grids aren't — a config that no
    # longer matches what this run trained on, or an incomplete extraction — instead of
    # silently re-scoring against the wrong (or an empty) store.
    if probe_id not in store.available_samples:
        raise FileNotFoundError(
            f"recomputed dense cache dir {store_dir} does not contain sample '{probe_id}' "
            f"({len(store.available_samples)} samples present). The config likely no longer "
            f"matches the one this run was trained with, or extraction is incomplete."
        )
    if args.run_subdir is not None:
        ckpt = args.run_subdir / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"no best_model.pt under --run-subdir {args.run_subdir}")
    else:
        # Pick the most-recently-written checkpoint (not an arbitrary glob hit) so this is
        # deterministic when an output_root accumulates several runs across experiments.
        candidates = sorted(
            args.run_dir.glob("experiments/*/runs/*/best_model.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"no experiments/*/runs/*/best_model.pt under {args.run_dir}")
        ckpt = candidates[-1]
    print(f"feature store: {store_dir}\ncheckpoint:    {ckpt}")

    tune_records = [manifest.samples[s] for s in fold_split.tune]
    # --tune-only drops the test splits before any loader/grid is touched, so a run trained
    # under evaluation.holdout_test (test grids never extracted) can still be re-scored.
    test_by_split = (
        {}
        if args.tune_only
        else {n: [manifest.samples[s] for s in ids] for n, ids in fold_split.tests.items()}
    )

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

    from soma.tasks.detection import DetectionHead

    head = DetectionHead(
        num_classes=num_classes, geometry=geometry, delta_px=delta_px, sigma_px=sigma_px,
        nms_distance_px=delta_px, matching=args.matching,
        foreground_weight=float(p.get("foreground_weight", 10.0)),
        sample_spacings=sample_spacings,
        metrics=cfg.evaluation.metrics,
    )

    # Rebuild the model exactly as the fold did (decoder + any feature adaptor the run
    # carried), then load trained weights.
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

    from soma.training.detection_dataset import DetectionDataset, detection_collate_fn

    collate = functools.partial(detection_collate_fn, target_dtypes=head.target_dtypes)
    _, tune_loader, test_loaders = _make_loaders(
        DetectionDataset, collate, train_records, tune_records, test_by_split,
        cfg.training, store, head.extract_targets,
    )

    out = build_greedy_report(
        model=model, head=head, device=device,
        tune_loader=tune_loader, test_loaders=test_loaders,
        dataset=manifest, matching=args.matching,
    )
    print(f"\nmatching = {args.matching}")
    print(f"tune-frozen per-class thresholds (headline): {out['score_threshold_per_class']}")
    print(f"  [tune] mF1={out['tune'].get('mean_f1')}")
    for name in test_loaders:
        headline_f1 = out[name]["headline"]["metrics"].get("mean_f1")
        print(f"  [{name}] headline mF1={headline_f1}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
