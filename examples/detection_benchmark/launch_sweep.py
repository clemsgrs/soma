#!/usr/bin/env python
"""Fan the detection-benchmark sweep across GPUs: one encoder-shard per GPU.

``campaign.py`` is single-process. Its dense **extraction** is single-GPU by design
(``soma.dense_extraction`` pins ``num_gpus=1``) and cheap (~minutes/encoder); the real cost
is the **rank** phase — training + tune-freeze + test-score of every ``(encoder, replicate)``
cell (~2h each). Those cells are independent and each writes its own ``encoder/replicate_*``
dir, so they parallelise across GPUs with zero code changes. This launcher shards the roster
across the visible GPUs, runs each shard pinned to one GPU via ``CUDA_VISIBLE_DEVICES``, then
does one final single-process aggregation to emit the complete ``ranking_report.json``.

Three barrier-synchronised phases:

  1. extract — G parallel processes (encoder-sharded), each pinned to one GPU, populate the
     dataset-wide dense cache. Note this runs campaign.py's ``extract``, which invokes a full
     ``python -m soma`` run for replicate 0 — so it *extracts and then trains* that replicate
     (~2h/encoder), it is not a minutes-long extraction-only pass. Nothing is wasted (phase 2
     skips the cell it already trained), and an encoder that OOMs (e.g. genbio-pathfm at
     4608-dim) still fails within the first minutes, before any real GPU-hours are spent.
  2. rank    — same sharding; each shard trains + freezes-on-tune + scores its cells. The long
     one. Wall-clock ~= the slowest shard, so speedup is near-linear up to #encoders GPUs.
  3. report  — one process over the *full* roster. Every cell's metrics are cached by now, so
     nothing trains — it only re-aggregates the cross-encoder ranking report.

Per-shard stdout+stderr stream to ``<out-root>/logs/<phase>.gpu<N>.log``; a failing shard
aborts its phase (extract failure never proceeds to the expensive rank phase). Idempotent:
re-invoking resumes via campaign.py's per-cell skip guards (``extracted.marker`` /
``best_model.pt`` / ``metrics.json``), so a crashed shard just picks up where it stopped.

Caveat: training is CPU-bound on dense-grid deserialisation, so G shards contend for CPU —
GPU speedup is real but can be sublinear if the box is CPU-starved. Round-robin over the
roster order interleaves heavy (virchow2/genbio-pathfm/h-optimus-1) and light encoders so the
shards stay roughly balanced by cost, not just by count.

Examples:
    # auto-detect all GPUs, full roster, OCELOT
    python examples/detection_benchmark/launch_sweep.py --datasets ocelot

    # explicit GPUs + a roster subset, skip extraction (cache already warm)
    python examples/detection_benchmark/launch_sweep.py --datasets ocelot \
        --gpus 0,1,2 --encoders virchow2 midnight h0-mini --skip-extract

    # print the GPU->encoder plan and the exact commands without launching anything
    python examples/detection_benchmark/launch_sweep.py --datasets ocelot --gpus 0,1,2 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CAMPAIGN = HERE / "campaign.py"
DEFAULT_OUT_ROOT = Path("/var/tmp/detection_benchmark")

# Import the working-tree soma so the roster never drifts from the committed DEFAULT_ROSTER.
sys.path.insert(0, str(REPO_ROOT))


def detect_gpus() -> list[int]:
    """Physical GPU indices from ``nvidia-smi`` (empty if it is missing or reports none)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    proc = subprocess.run(
        [smi, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    return [int(tok) for tok in proc.stdout.split()]


def default_roster_names() -> list[str]:
    """The canonical Paper-1 roster, imported (not copied) from soma so it stays in sync."""
    from soma.benchmarks.detection_benchmark import DEFAULT_ROSTER

    return [entry.name for entry in DEFAULT_ROSTER]


def shard_roster(names: Sequence[str], n_gpus: int) -> list[list[str]]:
    """Round-robin the roster into <=``n_gpus`` non-empty buckets (balances count; interleaves cost)."""
    buckets: list[list[str]] = [[] for _ in range(n_gpus)]
    for i, name in enumerate(names):
        buckets[i % n_gpus].append(name)
    return [b for b in buckets if b]


def resolve_hf_token() -> str | None:
    """Env ``HF_TOKEN`` if set, else the on-disk token — gated encoders 401 without it."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        return token_file.read_text(encoding="utf-8").strip()
    return None


def _campaign_cmd(
    phase: str, encoders: Sequence[str], datasets: Sequence[str],
    out_root: Path, seeds: Sequence[int],
) -> list[str]:
    return [
        sys.executable, str(CAMPAIGN), phase,
        "--datasets", *datasets,
        "--encoders", *encoders,
        "--out-root", str(out_root),
        "--seeds", *[str(s) for s in seeds],
    ]


def _shard_env(gpu: int | None, token: str | None) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if token:
        env["HF_TOKEN"] = token
    return env


def run_phase(
    phase: str, gpus: Sequence[int], buckets: Sequence[Sequence[str]],
    datasets: Sequence[str], out_root: Path, seeds: Sequence[int],
    token: str | None, log_dir: Path, *, dry_run: bool,
) -> None:
    """Launch one campaign phase as G pinned shards; wait for all; abort if any shard fails."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu, bucket in zip(gpus, buckets):
        cmd = _campaign_cmd(phase, bucket, datasets, out_root, seeds)
        log = log_dir / f"{phase}.gpu{gpu}.log"
        print(f"[gpu {gpu}] {phase}: {' '.join(bucket)}  ->  {log}")
        if dry_run:
            print("    $ CUDA_VISIBLE_DEVICES=%s %s" % (gpu, " ".join(cmd)))
            continue
        handle = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=_shard_env(gpu, token), stdout=handle, stderr=subprocess.STDOUT)
        procs.append((gpu, bucket, proc, handle))
    if dry_run:
        return
    failures = []
    for gpu, bucket, proc, handle in procs:
        rc = proc.wait()
        handle.close()
        print(f"[gpu {gpu}] {phase} {'ok' if rc == 0 else f'FAILED (exit {rc})'}: {' '.join(bucket)}")
        if rc != 0:
            failures.append((gpu, bucket, rc))
    if failures:
        detail = "; ".join(f"gpu{g}:{','.join(b)} (exit {rc})" for g, b, rc in failures)
        raise SystemExit(f"{phase} phase: {len(failures)} shard(s) failed [{detail}]. "
                         f"See logs in {log_dir}. Aborting.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datasets", nargs="+", default=["ocelot"],
                    help="datasets to sweep (default: ocelot)")
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="roster subset by name (default: the full DEFAULT_ROSTER)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seed replicates for single-fold datasets (folds datasets ignore this)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"sweep output + dense cache root (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU indices (default: all visible via nvidia-smi)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip phase 1 (dense cache already warm)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the GPU->encoder plan and commands, launch nothing")
    args = ap.parse_args(argv)

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else detect_gpus()
    if not gpus:
        raise SystemExit("no GPUs detected (nvidia-smi missing or reported none); pass --gpus 0,1,...")

    names = args.encoders or default_roster_names()
    buckets = shard_roster(names, len(gpus))
    gpus = gpus[: len(buckets)]  # fewer encoders than GPUs -> use only as many GPUs as shards

    token = resolve_hf_token()
    if not token:
        print("WARNING: no HF token (env HF_TOKEN or ~/.cache/huggingface/token) — "
              "gated encoders (virchow2/conchv15/h0-mini/h-optimus-1) will 401.")

    print(f"roster ({len(names)}): {' '.join(names)}")
    print(f"GPUs ({len(gpus)}): {gpus}")
    for gpu, bucket in zip(gpus, buckets):
        print(f"  gpu {gpu}  <-  {' '.join(bucket)}")
    print(f"datasets={args.datasets}  seeds={args.seeds}  out-root={args.out_root}")
    log_dir = args.out_root / "logs"

    if not args.skip_extract:
        print("\n=== phase 1/3: extract + train replicate 0  (sharded; OOM fails fast) ===")
        run_phase("extract", gpus, buckets, args.datasets, args.out_root, args.seeds,
                  token, log_dir, dry_run=args.dry_run)

    print("\n=== phase 2/3: rank — train + freeze-on-tune + score  (sharded, the long one) ===")
    run_phase("rank", gpus, buckets, args.datasets, args.out_root, args.seeds,
              token, log_dir, dry_run=args.dry_run)

    print("\n=== phase 3/3: aggregate the complete ranking_report.json  (full roster, GPU-free) ===")
    report_cmd = _campaign_cmd("rank", names, args.datasets, args.out_root, args.seeds)
    if args.dry_run:
        print("    $", " ".join(report_cmd))
        return 0
    subprocess.run(report_cmd, env=_shard_env(None, token), check=True)
    print(f"\ndone — report: {args.out_root}/ranking_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
