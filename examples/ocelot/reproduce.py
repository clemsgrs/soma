"""Reproduce the OCELOT Virchow2 @ 0.2 anchor and check it against the recorded reference.

This wraps the README workflow (curate → train → greedy-score) end to end and asserts the
greedy **test** mean_F1 lands within tolerance of the recorded reference
(`expected_metrics.json`). Greedy is OCELOT's official matcher, so its headline is the
leaderboard-comparable number. See `RESULTS.md` for the reference values and caveats.

Two modes:

    # full: curate (if needed) → train → score → check   (~3 h on one GPU)
    python examples/ocelot/reproduce.py --data-root <data_root>/ocelot

    # fast: re-score an already-trained run-dir and check (seconds, no training)
    python examples/ocelot/reproduce.py --from-run-dir <output_root>/ocelot_virchow2_0p20_lightconv

Exit code 0 = within tolerance, 1 = outside tolerance (a real environment/plumbing
difference, not noise — the dense grids are cache-identical at extraction batch 8, so
only decoder training is stochastic). Run from the soma repo root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "ocelot_virchow2_0.20.yaml"
REFERENCE = HERE / "expected_metrics.json"


# --- pure helpers (unit-tested) ------------------------------------------------------


def extract_greedy_report(stdout: str) -> dict:
    """Pull the JSON report eval_greedy.py prints after its plain-text summary lines."""
    lines = stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "{"), None)
    if start is None:
        raise ValueError("no JSON object found in eval_greedy output")
    return json.loads("\n".join(lines[start:]))


def greedy_test_mean_f1(report: dict, split: str = "test") -> float:
    """The leakage-free greedy headline mean_F1 for a test split."""
    return float(report[split]["headline"]["metrics"]["mean_f1"])


def build_path_overrides(curated_dir: Path, output_root: Path | None) -> list[str]:
    """`--set key=value` pairs repointing a committed config at this machine's paths."""
    pairs = [
        f"data.dataset_csv={curated_dir / 'dataset.csv'}",
        f"data.splits_csv={curated_dir / 'splits.csv'}",
    ]
    if output_root is not None:
        pairs.append(f"run.output_root={output_root}")
    return pairs


def check_within_tolerance(measured_mean_f1: float, reference: dict) -> tuple[bool, str]:
    """Compare a measured greedy test mean_F1 to the reference within its tolerance band."""
    expected = float(reference["expected"]["test_greedy"]["mean_f1"])
    tol = float(reference["tolerance"]["mean_f1_abs"])
    delta = measured_mean_f1 - expected
    ok = abs(delta) <= tol
    verdict = "PASS" if ok else "FAIL"
    msg = (
        f"[{verdict}] greedy test mean_F1 = {measured_mean_f1:.4f}  "
        f"(reference {expected:.4f}, Δ {delta:+.4f}, tolerance ±{tol:.4f})"
    )
    return ok, msg


# --- orchestration -------------------------------------------------------------------


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _newest_config(run_dir: Path) -> Path:
    """The trained run's own saved config.yaml (has the exact paths it ran with).

    Sorted by mtime so this matches the checkpoint eval_greedy picks (newest run) when an
    output_root holds several runs; falls back to the committed anchor config.
    """
    configs = sorted(run_dir.glob("experiments/*/runs/*/config.yaml"), key=lambda p: p.stat().st_mtime)
    if configs:
        return configs[-1]
    return DEFAULT_CONFIG


def score_and_check(run_dir: Path, config: Path, reference: dict) -> bool:
    proc = _run(
        [sys.executable, str(HERE / "eval_greedy.py"),
         "--run-dir", str(run_dir), "--config", str(config), "--matching", "greedy"],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    report = extract_greedy_report(proc.stdout)
    measured = greedy_test_mean_f1(report)
    ok, msg = check_within_tolerance(measured, reference)
    print("\n" + msg)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, help="raw OCELOT root (full mode)")
    ap.add_argument("--curated-dir", type=Path, help="curated manifest dir (default <data-root>/curated)")
    ap.add_argument("--output-root", type=Path, help="override run.output_root")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="anchor config (full mode)")
    ap.add_argument("--from-run-dir", type=Path, help="fast mode: re-score this output_root, no training")
    ap.add_argument("--skip-curation", action="store_true")
    args = ap.parse_args()

    reference = json.loads(REFERENCE.read_text())

    if args.from_run_dir is not None:
        run_dir = args.from_run_dir
        ok = score_and_check(run_dir, _newest_config(run_dir), reference)
        return 0 if ok else 1

    if args.data_root is None:
        ap.error("need --data-root (full mode) or --from-run-dir (fast mode)")

    curated = args.curated_dir or (args.data_root / "curated")
    if not args.skip_curation and not (curated / "dataset.csv").exists():
        _run([sys.executable, "-m", "soma.curation.ocelot",
              "--raw-root", str(args.data_root), "--output-dir", str(curated)])

    overrides = build_path_overrides(curated, args.output_root)
    train_cmd = [sys.executable, "-m", "soma", str(args.config)]
    for pair in overrides:
        train_cmd += ["--set", pair]
    _run(train_cmd)

    # The config's run.output_root (or the override) is where the run landed.
    from soma.config import load_config

    cfg = load_config(args.config, overrides={
        "data": {"dataset_csv": str(curated / "dataset.csv"), "splits_csv": str(curated / "splits.csv")},
        **({"run": {"output_root": str(args.output_root)}} if args.output_root else {}),
    })
    ok = score_and_check(Path(cfg.output_root), args.config, reference)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
