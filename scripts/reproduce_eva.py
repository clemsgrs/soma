#!/usr/bin/env python
"""Reproduce one kaiko-ai/eva patch-level benchmark cell with soma.

Curates a locally prepared raw EVA dataset, then trains soma's tile path with the
EVA protocol (see :mod:`soma.benchmarks.eva`) for one or more seeds and prints the
soma balanced accuracy next to the published EVA leaderboard value.

Example:
    python scripts/reproduce_eva.py \\
        --dataset bach --encoder uni2 --raw-root /data/raw/eva/bach

The encoders (uni2, virchow2) are gated on HuggingFace; set ``HF_TOKEN`` first.
Feature extraction runs once per (encoder, dataset) and is reused across seeds.
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

from soma import Pipeline
from soma.benchmarks import eva
from soma.curation.eva import CuratedManifest, curate_eva_patch_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce a kaiko-ai/eva patch-level benchmark cell with soma.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(eva.DATASETS))
    parser.add_argument("--encoder", required=True, choices=sorted(eva.ENCODERS))
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Local raw dataset root in the layout expected by EVA curation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eva_reproduction"),
        help="Where curated manifests, feature cache, and run outputs are written.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated training seeds (features are shared across seeds).",
    )
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument(
        "--encoder-workers",
        type=int,
        default=None,
        help="Cap extraction dataloader workers per GPU (default: soma's default).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs (smoke runs)."
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Override early-stopping patience."
    )
    parser.add_argument(
        "--force-curate",
        action="store_true",
        help="Re-curate even if dataset.csv/splits.csv already exist.",
    )
    return parser.parse_args()


def _curate(args) -> "CuratedManifest":
    """Curate the raw dataset, reusing existing manifests unless --force-curate.

    Curation is deterministic here (tune_fraction=0.0), so reusing prior
    manifests is safe and skips re-scanning the raw tree on repeat runs.
    """
    out_dir = args.output_root / "data" / args.dataset
    dataset_csv, splits_csv = out_dir / "dataset.csv", out_dir / "splits.csv"
    if not args.force_curate and dataset_csv.is_file() and splits_csv.is_file():
        print(f"[curate] reusing existing manifests in {out_dir} (--force-curate to rebuild)")
        return CuratedManifest(dataset_csv=dataset_csv, splits_csv=splits_csv)
    # tune_fraction=0.0 so the eva validation split becomes soma's reported "test".
    return curate_eva_patch_dataset(
        args.dataset, args.raw_root, out_dir, tune_fraction=eva.CURATION_TUNE_FRACTION
    )


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if os.environ.get("HF_TOKEN"):
        from slide2vec.utils.config import hf_login

        hf_login()

    # 1) Curate raw data into soma manifests.
    manifest = _curate(args)

    execution = None
    if args.encoder_workers is not None:
        from soma import ExecutionConfig

        execution = ExecutionConfig(num_workers_per_gpu=args.encoder_workers)

    splits = eva.reported_splits(args.dataset)
    collected: dict[str, list[float]] = {s.label: [] for s in splits}

    # 2) Train one head per seed (features are cached and reused).
    for seed in seeds:
        config = eva.build_config(
            dataset=args.dataset,
            encoder=args.encoder,
            dataset_csv=manifest.dataset_csv,
            splits_csv=manifest.splits_csv,
            output_root=args.output_root / "runs" / args.encoder / args.dataset,
            cache_root=args.output_root / "cache" / args.encoder / args.dataset,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            encoder_batch_size=args.encoder_batch_size,
            execution=execution,
        )
        print(
            f"[run] {args.dataset}/{args.encoder} seed={seed} "
            f"epochs={config.training.epochs} patience={config.training.patience}"
        )
        summary = eva.result_summary(Pipeline(config).run())
        for split in splits:
            value = eva.balanced_accuracy_from_summary(summary, split.soma_split)
            if value is not None:
                collected[split.label].append(value)

    # 3) Report soma vs. the published EVA value.
    print(f"\n{args.dataset} / {args.encoder}  ({len(seeds)} seed(s))")
    for split in splits:
        values = collected[split.label]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        expected = eva.expected_balanced_accuracy(args.encoder, args.dataset, split=split.label)
        line = f"  {split.label:<4}: soma {mean:.4f} ± {std:.4f}"
        if expected is not None:
            line += f"   eva {expected:.3f}   Δ {mean - expected:+.3f}"
        print(line)


if __name__ == "__main__":
    main()
