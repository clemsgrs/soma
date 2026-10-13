"""Re-score a trained OCELOT detection run with greedy (OCELOT-official) matching.

Matching only affects evaluation, so nothing is retrained or re-encoded: the run's
persisted point predictions (``predictions_tune.parquet`` / ``predictions_<test>.parquet``,
every peak above the head's ``min_score``) are re-swept for per-class score thresholds on
tune and scored once on each test split — the leakage-free headline a real submitter
reports. A thin CLI over :func:`soma.benchmark.ocelot._greedy_report_for_run`, which
``soma reproduce ocelot`` uses too; ``campaign.py`` drives it per (cell, seed).

Usage (from the soma repo, in soma's Python environment):
    python examples/ocelot/eval_greedy.py \\
        --run-dir /maindisk/clement/runs/ocelot_conch_lightconv \\
        --config  examples/ocelot/ocelot.yaml \\
        --matching greedy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the working-tree soma (this repo), not a stale site-packages copy: when this file
# runs as a script sys.path[0] is examples/ocelot/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soma.benchmark.ocelot import _greedy_report_for_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="output_root of the trained run")
    ap.add_argument(
        "--config", type=Path, default=None, help="unused: the run's own saved config.yaml is read"
    )
    ap.add_argument("--matching", default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument(
        "--run-subdir", type=Path, default=None, help="the run directory to score (default: newest)"
    )
    ap.add_argument(
        "--tune-only", action="store_true", help="report the tune split only (a holdout_test run)"
    )
    args = ap.parse_args()

    out = _greedy_report_for_run(args.run_subdir or args.run_dir, matching=args.matching)
    if args.tune_only:
        out = {key: out[key] for key in ("matching", "score_threshold_per_class", "tune")}
    print(f"\nmatching = {args.matching}")
    print(f"tune-frozen per-class thresholds (headline): {out['score_threshold_per_class']}")
    print(f"  [tune] mF1={out['tune'].get('mean_f1')}")
    for name, block in out.items():
        if isinstance(block, dict) and "headline" in block:
            print(f"  [{name}] headline mF1={block['headline']['metrics'].get('mean_f1')}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
