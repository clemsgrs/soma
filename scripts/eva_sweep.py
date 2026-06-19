"""Maintainer sweep: validate soma against the full kaiko-ai/eva leaderboard.

This is the maintainer-facing counterpart to ``scripts/reproduce_eva.py``: it
runs a grid of {encoders} x {datasets} x {seeds}, supports staging EVA tiles off
a slow CIFS mount, and builds a comparison table against ``eva.csv``. The actual
EVA protocol lives in :mod:`soma.benchmarks.eva` (single source of truth); this
script only orchestrates the sweep and reporting.

The end-user path is the single-cell script:

    python scripts/reproduce_eva.py --dataset bach --encoder uni2 --raw-root ...

Commands:
    python scripts/eva_sweep.py stage     # copy tiles to fast local storage
    python scripts/eva_sweep.py run       # run the (sub)grid selected by env
    python scripts/eva_sweep.py compare   # comparison table vs eva.csv

Env knobs (all optional):
    EVA_MODELS      comma list of soma encoder names (default: all in ENCODERS)
    EVA_DATASETS    comma list of dataset keys (default: all in DATASETS)
    EVA_SEEDS       comma list of int seeds (default: 0,1,2,3,4)
    EVA_EPOCHS      override epochs (smoke; e.g. 1)
    EVA_PATIENCE    override patience (smoke)
    EVA_ENC_BATCH   encoder extraction batch size (default 32)
    EVA_ENC_WORKERS cap extraction dataloader workers per GPU (shared-node OOM)
    EVA_NUM_WORKERS head dataloader workers (default 0)
    EVA_DATA_ROOT   curated manifests root (default: main checkout data/eva)
    EVA_STAGE_ROOT  local target for `stage`
    EVA_OUT         output root (default: output/eva_repro)
    HF_TOKEN        HuggingFace token (encoders are gated)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from soma import ExecutionConfig, Pipeline
from soma.benchmarks import eva

# data/eva is gitignored and lives in the main checkout, not this worktree.
DATA_ROOT = Path(
    os.environ.get("EVA_DATA_ROOT", "/data/pathology/projects/clement/code/soma/data/eva")
)
OUT = Path(os.environ.get("EVA_OUT", "output/eva_repro")).resolve()
RESULTS_DIR = OUT / "results"


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _result_path(encoder: str, dataset: str, seed: int) -> Path:
    return RESULTS_DIR / encoder / dataset / f"seed{seed}.json"


def _cell_config(encoder: str, dataset: str, seed: int, epochs: int | None, patience: int | None):
    ds_dir = DATA_ROOT / dataset
    enc_workers = os.environ.get("EVA_ENC_WORKERS")
    execution = ExecutionConfig(num_workers_per_gpu=int(enc_workers)) if enc_workers else None
    return eva.build_config(
        dataset=dataset,
        encoder=encoder,
        dataset_csv=ds_dir / "dataset.csv",
        splits_csv=ds_dir / "splits.csv",
        output_root=OUT / "runs" / encoder / dataset,
        cache_root=OUT / "_cache" / encoder / dataset,
        seed=seed,
        epochs=epochs,
        patience=patience,
        encoder_batch_size=int(os.environ.get("EVA_ENC_BATCH", "32")),
        head_num_workers=int(os.environ.get("EVA_NUM_WORKERS", "0")),
        execution=execution,
    )


def _run_cell(encoder: str, dataset: str, seed: int, epochs: int | None, patience: int | None) -> None:
    rp = _result_path(encoder, dataset, seed)
    if rp.exists():
        print(f"[skip] {encoder}/{dataset}/seed{seed} (cached result)")
        return
    cfg = _cell_config(encoder, dataset, seed, epochs, patience)
    print(
        f"[run]  {encoder}/{dataset}/seed{seed} epochs={cfg.training.epochs} "
        f"patience={cfg.training.patience} variant={eva.ENCODERS[encoder].output_variant}"
    )
    result = Pipeline(cfg).run()
    summary = eva.result_summary(result)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"summary": summary, "run_dir": str(result.run_dir)}, indent=2))


def cmd_stage() -> None:
    """Copy a dataset's tiles to local fast storage and rewrite its manifest.

    The EVA tiles can live on a high-latency CIFS mount (~94 ms/tile), making
    extraction I/O-bound. Staging once (encoders share tiles) to local storage
    with a high-thread bulk copy turns extraction GPU/CPU-bound. Set
    EVA_STAGE_ROOT, then run with EVA_DATA_ROOT=$EVA_STAGE_ROOT.
    """
    datasets = _env_list("EVA_DATASETS", list(eva.DATASETS))
    stage_root = Path(os.environ["EVA_STAGE_ROOT"])
    workers = int(os.environ.get("EVA_STAGE_WORKERS", "128"))
    for dataset in datasets:
        src_dir = DATA_ROOT / dataset
        df = pd.read_csv(src_dir / "dataset.csv")
        files_dir = stage_root / dataset / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        def _copy(row: tuple[str, str]) -> str:
            sample_id, image_path = row
            src = Path(image_path)
            dst = files_dir / f"{sample_id}{src.suffix}"
            if not (dst.exists() and dst.stat().st_size > 0):
                shutil.copyfile(src, dst)
            return str(dst)

        rows = list(zip(df["sample_id"].astype(str), df["image_path"].astype(str)))
        print(f"[stage] {dataset}: copying {len(rows)} tiles -> {files_dir} ({workers} workers)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_paths = list(pool.map(_copy, rows))
        df = df.copy()
        df["image_path"] = new_paths
        df.to_csv(stage_root / dataset / "dataset.csv", index=False)
        shutil.copyfile(src_dir / "splits.csv", stage_root / dataset / "splits.csv")
        print(f"[stage] {dataset}: done")
    print(f"\nStaged. Now run with EVA_DATA_ROOT={stage_root}")


def cmd_run() -> None:
    encoders = _env_list("EVA_MODELS", list(eva.ENCODERS))
    datasets = _env_list("EVA_DATASETS", list(eva.DATASETS))
    seeds = [int(s) for s in _env_list("EVA_SEEDS", ["0", "1", "2", "3", "4"])]
    epochs_override = int(os.environ["EVA_EPOCHS"]) if os.environ.get("EVA_EPOCHS") else None
    patience_override = int(os.environ["EVA_PATIENCE"]) if os.environ.get("EVA_PATIENCE") else None

    for dataset in datasets:
        for encoder in encoders:
            for seed in seeds:
                _run_cell(encoder, dataset, seed, epochs_override, patience_override)
    print("\nDone. Run `python scripts/eva_sweep.py compare` for the comparison table.")


def cmd_compare() -> None:
    rows = []
    for encoder in eva.ENCODERS:
        for dataset in eva.DATASETS:
            res_dir = RESULTS_DIR / encoder / dataset
            if not res_dir.is_dir():
                continue
            for split in eva.reported_splits(dataset):
                vals = [
                    v
                    for f in sorted(res_dir.glob("seed*.json"))
                    if (
                        v := eva.balanced_accuracy_from_summary(
                            json.loads(f.read_text())["summary"], split.soma_split
                        )
                    )
                    is not None
                ]
                if not vals:
                    continue
                s = pd.Series(vals)
                expected = eva.expected_balanced_accuracy(encoder, dataset, split=split.label)
                delta = (s.mean() - expected) if expected is not None else None
                rows.append(
                    {
                        "encoder": encoder,
                        "eva_col": split.eva_column,
                        "soma_mean": round(s.mean(), 4),
                        "soma_std": round(s.std(ddof=0), 4),
                        "n_seeds": len(vals),
                        "eva_expected": expected,
                        "delta": round(delta, 4) if delta is not None else None,
                        "flag": ("SUSPICIOUS" if (delta is not None and abs(delta) > 0.05) else ""),
                    }
                )
    if not rows:
        print("No results found. Run `python scripts/eva_sweep.py run` first.")
        return
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    if os.environ.get("HF_TOKEN"):
        from slide2vec.utils.config import hf_login

        hf_login()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"stage": cmd_stage, "run": cmd_run, "compare": cmd_compare}[cmd]()
