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
from pathlib import Path

import torch

from soma.config import load_config
from soma.dataset import DetectionManifest, Splits
from soma.decoders.registry import decoder_registry
from soma.dense import DenseFeatureStore
from soma.pipeline import (
    _evaluate_detection,
    _make_loaders,
    _resolve_detection_px,
    _sweep_detection_thresholds,
)


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
    each test split — the leakage-free number a real submitter reports. No oracle ceiling:
    the test-side re-sweep #146 added for the health gate is dropped now the gate has done
    its job (#152), so test thresholds are never swept on test here.

    The greedy matcher is whatever ``head.matching`` selects, so the sweep and the
    evaluations stay greedy-consistent.

    Pass an empty ``test_loaders`` for a tune-only report (model-selection sweeps under
    ``evaluation.holdout_test``): the returned dict then carries just ``matching``,
    ``score_threshold_per_class``, and ``tune``.
    """
    headline_thresholds = _sweep_detection_thresholds(model, tune_loader, device, head)
    head.score_threshold = headline_thresholds
    tune_report = _evaluate_detection(model, tune_loader, "tune", device, head=head, dataset=dataset)
    out: dict = {
        "matching": matching,
        "score_threshold_per_class": headline_thresholds,
        "tune": tune_report.metrics,
    }
    for name, loader in test_loaders.items():
        # Headline: the frozen tune thresholds applied once to this test split.
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

    from soma.training.model import SegmentationModel

    cfg = load_config(str(args.config))
    manifest = DetectionManifest(cfg.dataset_csv)
    splits = Splits(cfg.splits_csv, manifest)
    fold_split = splits.folds[0]

    # Locate the cached feature store + best_model.pt for this run.
    store_dir = next((args.run_dir / "feature_cache" / "dense").glob("*"))
    store = DenseFeatureStore(store_dir)
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

    train_records = [manifest.samples[s] for s in fold_split.train]
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
    grid_spacing = store.metadata(train_records[0].sample_id).get("spacing_um")
    delta_px = _resolve_detection_px(float(p["match_distance"]), grid_spacing, "match_distance")
    sigma_px = (
        _resolve_detection_px(float(p["sigma"]), grid_spacing, "sigma")
        if p.get("sigma") is not None else delta_px / 3.0
    )
    geometry = store.geometry(train_records[0].sample_id)

    from soma.tasks.detection import DetectionHead

    head = DetectionHead(
        num_classes=num_classes, geometry=geometry, delta_px=delta_px, sigma_px=sigma_px,
        nms_distance_px=delta_px, matching=args.matching,
        foreground_weight=float(p.get("foreground_weight", 10.0)),
        level0_spacing=float(p.get("level0_spacing", 1.0)),
        run_spacing=float(grid_spacing) if grid_spacing is not None else None,
        metrics=cfg.evaluation.metrics,
    )

    # Rebuild the decoder exactly as the fold did, then load trained weights.
    decoder_cls = decoder_registry.get(cfg.decoder.name)
    dparams = dict(cfg.decoder.params)
    if "num_upsample_blocks" in inspect.signature(decoder_cls.__init__).parameters and "num_upsample_blocks" not in dparams:
        rh = geometry.encoded_size[0] / geometry.grid_shape[0]
        rw = geometry.encoded_size[1] / geometry.grid_shape[1]
        dparams["num_upsample_blocks"] = max(0, math.ceil(math.log2(max(rh, rw))))
    decoder_obj = decoder_cls(input_dim=store.feature_dim, num_classes=num_classes, **dparams)
    model = SegmentationModel(decoder=decoder_obj, task_head=head)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
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
