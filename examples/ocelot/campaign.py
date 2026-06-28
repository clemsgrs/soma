"""Drive the OCELOT encoder×spacing campaign (#152): tune-select → test-confirm.

The campaign ranks five frozen-probe cells — the 2×2 {Virchow2, UNI2} × {0.25, 0.5 µm/px}
plus the Virchow2 @ 0.2 native anchor — on a **tune-only** sweep, picks the winner on tune,
then scores **only** the winner and the anchor on test. Keeping the headline off the test
set avoids the winner's-curse bias of maximising test mF1 over many cells × seeds.

Two phases:

    # selection: train every cell × seed tune-only (holdout_test), greedy-rescore tune,
    # aggregate mean±std + model×magnification interaction + per-class recall, pick winner.
    python examples/ocelot/campaign.py selection --data-root data/ocelot --seeds 0 1 2

    # confirmation: score the tune-selected winner + the anchor on test (3 seeds,
    # tune-frozen thresholds) — the single clean head-to-head.
    python examples/ocelot/campaign.py confirmation --data-root data/ocelot --seeds 0 1 2

Design decisions that make this cheap and resumable:

* Dense extraction is seed-independent (cache keyed by encoder/spacing/geometry), so the
  five extractions are shared across all seeds — running 3 seeds costs ~3× decoder
  training, not 3× extraction.
* Training is idempotent: a (cell, seed) whose checkpoint already exists is skipped, so a
  killed run resumes by re-invocation. The anchor reuses #151's cache + seed-0 decoder;
  only seeds 1–2 are added there.
* Selection runs set ``evaluation.holdout_test`` so no test grids are ever extracted or
  scored. Confirmation re-scores the *same* selection checkpoints on test (after a one-off
  test-grid backfill for a non-anchor winner), so the reported tune and test numbers come
  from identical models.

Each cell's config (``examples/ocelot/ocelot_{encoder}_{spacing}.yaml``) supplies the
output_root, encoder, decoder, and detection protocol; this driver only overrides the
dataset paths (from ``--data-root``), the seed, and the holdout flag. Needs a GPU and an
HF token (Virchow2 and UNI2 are gated). Run from the soma repo root.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_GREEDY = HERE / "eval_greedy.py"

# A throwaway seed for the confirmation test-grid backfill (a 1-epoch holdout_test=false
# pass that only exists to encode the test split into the shared dense cache). Kept distinct
# from the real campaign seeds so its ignored decoder never collides with a scored run.
BACKFILL_SEED = 99


@dataclass(frozen=True)
class Cell:
    """One encoder × spacing point in the campaign grid."""

    key: str
    encoder: str
    spacing: float
    config: str  # filename under examples/ocelot/
    curated: str  # subdir under --data-root holding this spacing's manifest
    is_anchor: bool = False

    @property
    def config_path(self) -> Path:
        return HERE / self.config


# The five cells. The anchor (Virchow2 @ 0.2 native) reuses the curated/ native manifest and
# #151's cache + seed-0 decoder; the other four consume rendered-spacing manifests.
CELLS: list[Cell] = [
    Cell("virchow2_0.20", "virchow2", 0.2, "ocelot_virchow2_0.20.yaml", "curated", is_anchor=True),
    Cell("virchow2_0.25", "virchow2", 0.25, "ocelot_virchow2_0.25.yaml", "curated_0p25"),
    Cell("virchow2_0.50", "virchow2", 0.5, "ocelot_virchow2_0.50.yaml", "curated_0p5"),
    Cell("uni2_0.25", "uni2", 0.25, "ocelot_uni2_0.25.yaml", "curated_0p25"),
    Cell("uni2_0.50", "uni2", 0.5, "ocelot_uni2_0.50.yaml", "curated_0p5"),
]
ANCHOR = next(c for c in CELLS if c.is_anchor)


# --- pure helpers (unit-tested) ------------------------------------------------------


def extract_trailing_json(stdout: str) -> dict:
    """Parse the JSON object eval_greedy.py prints after its plain-text summary lines."""
    lines = stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "{"), None)
    if start is None:
        raise ValueError("no JSON object found in eval_greedy output")
    return json.loads("\n".join(lines[start:]))


def summarize_seed_metrics(per_seed: list[dict]) -> dict:
    """Mean ± std of mean_f1 and the per-class recalls over a cell's seed metric dicts.

    Each input dict is one seed's ``tune`` (selection) or test-split (confirmation) metric
    block: ``mean_f1``, ``recall_class_0`` (BC), ``recall_class_1`` (TC), etc. Returns the
    seed mean/std of mean_f1 plus the seed-mean per-class recall and the raw per-seed
    mean_f1 list (so callers can show the spread).
    """
    if not per_seed:
        raise ValueError("no seed metrics to summarize")
    f1s = [float(m["mean_f1"]) for m in per_seed]
    out = {
        "n_seeds": len(per_seed),
        "mean_f1_mean": statistics.fmean(f1s),
        "mean_f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
        "mean_f1_per_seed": f1s,
    }
    for c, name in ((0, "bc"), (1, "tc")):
        key = f"recall_class_{c}"
        if all(key in m for m in per_seed):
            out[f"recall_{name}_mean"] = statistics.fmean(float(m[key]) for m in per_seed)
    return out


def magnification_interaction(summaries: dict[str, dict]) -> dict:
    """Per-encoder tune-mF1 lift from coarsening 0.5 → 0.25, the alignment signal.

    For each encoder with both spacings present, reports mF1(0.25) − mF1(0.5): positive
    means finer magnification helps. The interaction is the contrast between encoders —
    e.g. finer helping mixed-magnification Virchow2 but not 20×-only UNI2 is the
    pretraining-alignment effect, clean of architecture (each encoder is its own control).
    """
    out: dict = {}
    for encoder in ("virchow2", "uni2"):
        fine = summaries.get(f"{encoder}_0.25")
        coarse = summaries.get(f"{encoder}_0.50")
        if fine and coarse:
            out[encoder] = fine["mean_f1_mean"] - coarse["mean_f1_mean"]
    if "virchow2" in out and "uni2" in out:
        out["interaction"] = out["virchow2"] - out["uni2"]
    return out


def pick_winner(summaries: dict[str, dict]) -> str:
    """The cell key with the highest mean tune mF1 (the selection criterion)."""
    if not summaries:
        raise ValueError("no cell summaries to choose a winner from")
    return max(summaries, key=lambda k: summaries[k]["mean_f1_mean"])


def format_selection_markdown(summaries: dict[str, dict], winner: str, interaction: dict) -> str:
    """A human-readable selection report: per-cell tune mF1 ± std + per-class recall."""
    by_key = {c.key: c for c in CELLS}
    lines = [
        "# OCELOT encoder×spacing — selection (tune only)",
        "",
        "Per-cell tune greedy mF1 (mean ± std over seeds) and per-class recall. No test "
        "inference. Winner is chosen on tune mF1.",
        "",
        "| Encoder | Spacing | Tune mF1 ± std | Recall BC | Recall TC | seeds |",
        "|---|---|---|---|---|---|",
    ]
    for key in by_key:
        s = summaries.get(key)
        if not s:
            continue
        cell = by_key[key]
        anchor = " (anchor)" if cell.is_anchor else ""
        star = " ⭐" if key == winner else ""
        rbc = s.get("recall_bc_mean")
        rtc = s.get("recall_tc_mean")
        lines.append(
            f"| {cell.encoder}{star} | {cell.spacing}{anchor} | "
            f"{s['mean_f1_mean']:.4f} ± {s['mean_f1_std']:.4f} | "
            f"{rbc:.4f} | {rtc:.4f} | {s['n_seeds']} |"
            if rbc is not None and rtc is not None
            else f"| {cell.encoder}{star} | {cell.spacing}{anchor} | "
            f"{s['mean_f1_mean']:.4f} ± {s['mean_f1_std']:.4f} | — | — | {s['n_seeds']} |"
        )
    lines += ["", "## Model × magnification interaction (tune)", ""]
    for enc in ("virchow2", "uni2"):
        if enc in interaction:
            lines.append(f"- {enc}: mF1(0.25) − mF1(0.5) = {interaction[enc]:+.4f}")
    if "interaction" in interaction:
        lines.append(
            f"- **interaction** (virchow2 − uni2 lift) = {interaction['interaction']:+.4f} "
            "(the pretraining-alignment signal, clean of architecture)"
        )
    lines += [
        "",
        "Confounds: the raw Virchow2-vs-UNI2 main effect confounds architecture/scale "
        "(ViT-H/14 vs ViT-g/14); the localization confound rides with spacing (constant "
        "14-px token ⇒ 3.5 µm @ 0.25 vs 7 µm @ 0.5 against δ = 3 µm, so both models are "
        "localization-starved at 0.5) — read per-class recall for coarse-localization "
        "misses.",
        "",
        f"**Winner (tune): {winner}**",
        "",
    ]
    return "\n".join(lines)


# --- orchestration -------------------------------------------------------------------


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def output_root_for(cell: Cell) -> Path:
    """The run output_root a cell lands in (read from its committed config)."""
    from soma.config import load_config

    cfg = load_config(str(cell.config_path))
    root = Path(cfg.output_root)
    return root if root.is_absolute() else REPO_ROOT / root


def find_seed_runs(output_root: Path) -> dict[int, Path]:
    """Map seed → newest run subdir (holding best_model.pt) under an output_root.

    Reads each run's saved config.yaml for its seed; when a seed has several runs (re-runs)
    the most-recently-written checkpoint wins, matching eval_greedy's newest-checkpoint rule.
    """
    import yaml

    by_seed: dict[int, tuple[float, Path]] = {}
    for ckpt in output_root.glob("experiments/*/runs/*/best_model.pt"):
        run_dir = ckpt.parent
        cfg_file = run_dir / "config.yaml"
        if not cfg_file.exists():
            continue
        cfg = yaml.safe_load(cfg_file.read_text())
        seed = int(((cfg or {}).get("run") or {}).get("seed", -1))
        mtime = ckpt.stat().st_mtime
        if seed not in by_seed or mtime > by_seed[seed][0]:
            by_seed[seed] = (mtime, run_dir)
    return {seed: run_dir for seed, (_, run_dir) in by_seed.items()}


def _data_overrides(cell: Cell, data_root: Path) -> list[str]:
    curated = data_root / cell.curated
    return [
        "--set", f"data.dataset_csv={curated / 'dataset.csv'}",
        "--set", f"data.splits_csv={curated / 'splits.csv'}",
    ]


def train_cell_seed(
    cell: Cell, seed: int, data_root: Path, *, holdout_test: bool, extra: list[str] | None = None
) -> None:
    """Run ``python -m soma`` for one (cell, seed); the config supplies the output_root."""
    cmd = [sys.executable, "-m", "soma", str(cell.config_path)]
    cmd += _data_overrides(cell, data_root)
    cmd += ["--set", f"run.seed={seed}", "--set", f"evaluation.holdout_test={str(holdout_test).lower()}"]
    cmd += extra or []
    _run(cmd, cwd=REPO_ROOT)


def score_run(
    cell: Cell, run_subdir: Path, *, tune_only: bool, matching: str = "greedy"
) -> dict:
    """Greedy re-score one checkpoint; returns eval_greedy's parsed JSON report."""
    cmd = [
        sys.executable, str(EVAL_GREEDY),
        "--run-dir", str(output_root_for(cell)),
        "--config", str(cell.config_path),
        "--run-subdir", str(run_subdir),
        "--matching", matching,
    ]
    if tune_only:
        cmd.append("--tune-only")
    proc = _run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout[-2000:])
    return extract_trailing_json(proc.stdout)


def run_selection(data_root: Path, seeds: list[int], *, train: bool, dry_run: bool) -> dict:
    """Train (idempotent) + greedy-rescore tune for every cell × seed; aggregate + pick winner."""
    summaries: dict[str, dict] = {}
    per_cell_seed_metrics: dict[str, dict] = {}
    for cell in CELLS:
        output_root = output_root_for(cell)
        for seed in seeds:
            existing = find_seed_runs(output_root)
            if seed in existing:
                print(f"[{cell.key}] seed {seed}: checkpoint exists ({existing[seed]}), skip train")
            elif train and not dry_run:
                train_cell_seed(cell, seed, data_root, holdout_test=True)
            else:
                print(f"[{cell.key}] seed {seed}: would train (holdout_test=true)")
        if dry_run:
            continue
        runs = find_seed_runs(output_root)
        seed_metrics = []
        for seed in seeds:
            if seed not in runs:
                print(f"[{cell.key}] seed {seed}: no checkpoint to score, skipping")
                continue
            report = score_run(cell, runs[seed], tune_only=True)
            seed_metrics.append(report["tune"])
        if seed_metrics:
            per_cell_seed_metrics[cell.key] = seed_metrics
            summaries[cell.key] = summarize_seed_metrics(seed_metrics)
    if dry_run or not summaries:
        return {"summaries": summaries}
    interaction = magnification_interaction(summaries)
    winner = pick_winner(summaries)
    report = {
        "phase": "selection",
        "seeds": seeds,
        "summaries": summaries,
        "interaction": interaction,
        "winner": winner,
        "per_cell_seed_tune_metrics": per_cell_seed_metrics,
    }
    out_dir = REPO_ROOT / "output" / "campaign152"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "selection_report.md").write_text(
        format_selection_markdown(summaries, winner, interaction)
    )
    print(f"\nwrote {out_dir / 'selection_report.json'} and selection_report.md")
    print(format_selection_markdown(summaries, winner, interaction))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["selection", "confirmation"])
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "ocelot",
                    help="root holding curated/ + curated_0p25/ + curated_0p5/ manifests")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-train", action="store_true",
                    help="score/aggregate existing runs only; do not launch training")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned train/score steps without running anything")
    args = ap.parse_args(argv)

    if args.phase == "selection":
        run_selection(args.data_root, args.seeds, train=not args.no_train, dry_run=args.dry_run)
        return 0
    raise SystemExit("confirmation phase not yet implemented")


if __name__ == "__main__":
    raise SystemExit(main())
