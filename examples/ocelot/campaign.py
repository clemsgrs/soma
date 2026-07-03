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

Each cell's config is the committed benchmark YAML under
``soma/benchmarks/configs/ocelot/`` — the same files the registered ``ocelot`` benchmark's
``build_config`` loads — resolved here by ``(encoder, spacing)``. It supplies the
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
OUT_DIR = REPO_ROOT / "output" / "campaign152"  # campaign reports land here (git-ignored)

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
    curated: str  # subdir under --data-root holding this spacing's manifest
    is_anchor: bool = False

    @property
    def config_path(self) -> Path:
        # Reuse the OCELOT benchmark's own (encoder, spacing) -> path resolver so the
        # campaign and ``soma reproduce ocelot`` load byte-identical committed YAML from
        # ``soma/benchmarks/configs/ocelot/`` — a single source of truth for the configs.
        from soma.benchmarks.ocelot import _config_path_for

        return _config_path_for(self.encoder, self.spacing)


# The five cells. The anchor (Virchow2 @ 0.2 native) reuses the curated/ native manifest and
# #151's cache + seed-0 decoder; the other four consume rendered-spacing manifests.
CELLS: list[Cell] = [
    Cell("virchow2_0.20", "virchow2", 0.2, "curated", is_anchor=True),
    Cell("virchow2_0.25", "virchow2", 0.25, "curated_0p25"),
    Cell("virchow2_0.50", "virchow2", 0.5, "curated_0p5"),
    Cell("uni2_0.25", "uni2", 0.25, "curated_0p25"),
    Cell("uni2_0.50", "uni2", 0.5, "curated_0p5"),
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
    """Train (idempotent) + greedy-rescore tune for every cell × seed; aggregate + pick winner.

    Resilient by design for the long unattended run: a (cell, seed) whose training or scoring
    raises is recorded in ``failures`` and skipped, so one *deterministic* failure (e.g. an OOM
    on a single encoder×spacing) can't sink the whole campaign or block the cells ordered after
    it. Failures are written into the report and re-printed at the end; re-invoking resumes the
    survivors (training is idempotent) and retries the failures.
    """
    summaries: dict[str, dict] = {}
    per_cell_seed_metrics: dict[str, dict] = {}
    failures: list[dict] = []
    for cell in CELLS:
        output_root = output_root_for(cell)
        for seed in seeds:
            existing = find_seed_runs(output_root)
            if seed in existing:
                print(f"[{cell.key}] seed {seed}: checkpoint exists ({existing[seed]}), skip train")
            elif train and not dry_run:
                try:
                    train_cell_seed(cell, seed, data_root, holdout_test=True)
                except subprocess.CalledProcessError as e:
                    failures.append({"cell": cell.key, "seed": seed, "stage": "train", "error": str(e)})
                    print(f"[{cell.key}] seed {seed}: TRAIN FAILED ({e}); continuing", flush=True)
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
            try:
                report = score_run(cell, runs[seed], tune_only=True)
                seed_metrics.append(report["tune"])
            except (subprocess.CalledProcessError, ValueError, KeyError) as e:
                failures.append({"cell": cell.key, "seed": seed, "stage": "score", "error": str(e)})
                print(f"[{cell.key}] seed {seed}: SCORE FAILED ({e}); continuing", flush=True)
        if seed_metrics:
            per_cell_seed_metrics[cell.key] = seed_metrics
            summaries[cell.key] = summarize_seed_metrics(seed_metrics)
    if dry_run:
        return {"summaries": summaries, "failures": failures}
    if not summaries:
        print("\nno cells produced metrics; failures:\n" + json.dumps(failures, indent=2))
        return {"summaries": summaries, "failures": failures}
    interaction = magnification_interaction(summaries)
    winner = pick_winner(summaries)
    report = {
        "phase": "selection",
        "seeds": seeds,
        "summaries": summaries,
        "interaction": interaction,
        "winner": winner,
        "per_cell_seed_tune_metrics": per_cell_seed_metrics,
        "failures": failures,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selection_report.json").write_text(json.dumps(report, indent=2))
    (OUT_DIR / "selection_report.md").write_text(
        format_selection_markdown(summaries, winner, interaction)
    )
    print(f"\nwrote {OUT_DIR / 'selection_report.json'} and selection_report.md")
    if failures:
        print(f"\n{len(failures)} (cell, seed) failure(s) recorded:\n" + json.dumps(failures, indent=2))
    print(format_selection_markdown(summaries, winner, interaction))
    return report


# --- confirmation phase --------------------------------------------------------------


def test_split_names(report: dict) -> list[str]:
    """The test-split keys in an eval_greedy report (everything but the fixed top-level keys)."""
    fixed = {"matching", "score_threshold_per_class", "tune"}
    return [k for k in report if k not in fixed]


def test_headline_metrics(report: dict) -> dict:
    """The single test split's tune-frozen headline metrics (OCELOT has one test split)."""
    names = test_split_names(report)
    if len(names) != 1:
        raise ValueError(f"expected exactly one test split in the report, got {names}")
    return report[names[0]]["headline"]["metrics"]


def dense_embeddings_dir(cell: Cell) -> Path | None:
    """The cached dense-grid dir for a cell, or None if it has not been extracted yet."""
    base = output_root_for(cell) / "feature_cache" / "dense"
    if not base.exists():
        return None
    hashes = [p / "dense_embeddings" for p in base.glob("*") if (p / "dense_embeddings").is_dir()]
    return hashes[0] if hashes else None


def test_sample_ids(cell: Cell, data_root: Path) -> list[str]:
    from soma.dataset import DetectionManifest, Splits

    curated = data_root / cell.curated
    manifest = DetectionManifest(curated / "dataset.csv")
    splits = Splits(curated / "splits.csv", manifest)
    return [sid for ids in splits.folds[0].tests.values() for sid in ids]


def test_grids_present(cell: Cell, data_root: Path) -> bool:
    """True iff every test-split dense grid is already cached for this cell."""
    emb = dense_embeddings_dir(cell)
    if emb is None:
        return False
    return all((emb / f"{sid}.pt").exists() for sid in test_sample_ids(cell, data_root))


def backfill_test_grids(cell: Cell, data_root: Path) -> None:
    """Encode this cell's test split into its (shared) dense cache without a real training.

    Selection ran under holdout_test, so test grids were never extracted. A 1-epoch
    holdout_test=false pass extracts them incrementally (train/tune already cached) into the
    seed-independent cache; its throwaway decoder (BACKFILL_SEED) is never scored.
    """
    train_cell_seed(
        cell, BACKFILL_SEED, data_root, holdout_test=False, extra=["--set", "training.epochs=1"]
    )


def format_confirmation_markdown(results: dict[str, dict], winner: str) -> str:
    by_key = {c.key: c for c in CELLS}
    lines = [
        "# OCELOT encoder×spacing — confirmation (test)",
        "",
        "The tune-selected winner and the Virchow2 @ 0.2 native anchor, scored on test only "
        "(3 seeds, tune-frozen thresholds). Greedy is OCELOT's official, leaderboard-"
        "comparable matcher; Hungarian is the secondary. This is the only test exposure.",
        "",
        "| Encoder | Spacing | Test greedy mF1 ± std | Test Hungarian mF1 | Recall BC | Recall TC | seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        cell = by_key[key]
        g = r["greedy"]
        tag = " ⭐winner" if key == winner else ""
        anchor = " (anchor)" if cell.is_anchor else ""
        rbc = g.get("recall_bc_mean")
        rtc = g.get("recall_tc_mean")
        rbc_s = f"{rbc:.4f}" if rbc is not None else "—"
        rtc_s = f"{rtc:.4f}" if rtc is not None else "—"
        lines.append(
            f"| {cell.encoder}{tag} | {cell.spacing}{anchor} | "
            f"{g['mean_f1_mean']:.4f} ± {g['mean_f1_std']:.4f} | "
            f"{r['hungarian_mean_f1_mean']:.4f} | {rbc_s} | {rtc_s} | {g['n_seeds']} |"
        )
    lines += [
        "",
        "Per-class recall tracks the localization confound (the 14-px token spans more µm at "
        "coarser spacing, so coarse localization shows up as missed cells).",
        "",
    ]
    return "\n".join(lines)


def confirmation_cells(winner: str, summaries: dict[str, dict] | None = None) -> list[Cell]:
    """The winner plus the native anchor (deduped if the winner *is* the anchor).

    When the winner *is* the anchor, ``winner + anchor`` collapses to a single cell, which
    would make confirmation score one config against nothing. To keep confirmation a real
    leakage-free test head-to-head, fall back to the runner-up (the best non-winner cell by
    tune mF1) so the recommended config is still compared against its closest competitor.
    """
    by_key = {c.key: c for c in CELLS}
    cells = [by_key[winner]]
    if ANCHOR.key != winner:
        cells.append(ANCHOR)
    elif summaries:
        contenders = {k: v for k, v in summaries.items() if k != winner and k in by_key}
        if contenders:
            runner_up = max(contenders, key=lambda k: contenders[k]["mean_f1_mean"])
            cells.append(by_key[runner_up])
    return cells


def run_confirmation(
    data_root: Path, seeds: list[int], winner: str | None, *, dry_run: bool
) -> dict:
    """Score the tune-selected winner + anchor on test (greedy headline + Hungarian).

    When the winner is the anchor, the runner-up is pulled from the selection report's
    ``summaries`` so confirmation still yields a real test head-to-head (see
    ``confirmation_cells``); an explicit ``--winner`` with no report keeps the plain
    winner+anchor behaviour.
    """
    sel = OUT_DIR / "selection_report.json"
    sel_data = json.loads(sel.read_text()) if sel.exists() else None
    if winner is None:
        if sel_data is None:
            raise SystemExit(
                f"no {sel}; run the selection phase first, or pass --winner <cell-key>"
            )
        winner = sel_data["winner"]
    summaries = sel_data.get("summaries") if sel_data else None
    print(f"confirmation: winner={winner}, anchor={ANCHOR.key}")

    results: dict[str, dict] = {}
    for cell in confirmation_cells(winner, summaries):
        if not test_grids_present(cell, data_root):
            if dry_run:
                print(f"[{cell.key}] would backfill test grids (1-epoch holdout_test=false)")
            else:
                backfill_test_grids(cell, data_root)
        runs = find_seed_runs(output_root_for(cell))
        greedy_metrics, hungarian_f1s = [], []
        for seed in seeds:
            if seed not in runs:
                print(f"[{cell.key}] seed {seed}: no checkpoint to score, skipping")
                continue
            if dry_run:
                print(f"[{cell.key}] seed {seed}: would score test (greedy + hungarian)")
                continue
            greedy = score_run(cell, runs[seed], tune_only=False, matching="greedy")
            greedy_metrics.append(test_headline_metrics(greedy))
            hung = score_run(cell, runs[seed], tune_only=False, matching="hungarian")
            hungarian_f1s.append(float(test_headline_metrics(hung)["mean_f1"]))
        if greedy_metrics:
            results[cell.key] = {
                "greedy": summarize_seed_metrics(greedy_metrics),
                "hungarian_mean_f1_mean": statistics.fmean(hungarian_f1s),
                "hungarian_mean_f1_per_seed": hungarian_f1s,
                "greedy_test_metrics_per_seed": greedy_metrics,
            }
    if dry_run or not results:
        return {"winner": winner, "results": results}
    report = {
        "phase": "confirmation",
        "winner": winner,
        "anchor": ANCHOR.key,
        "seeds": seeds,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "confirmation_report.json").write_text(json.dumps(report, indent=2))
    md = format_confirmation_markdown(results, winner)
    (OUT_DIR / "confirmation_report.md").write_text(md)
    print(f"\nwrote {OUT_DIR / 'confirmation_report.json'} and confirmation_report.md")
    print(md)
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
    ap.add_argument("--winner", default=None,
                    help="confirmation: cell key to confirm (default: read from selection_report.json)")
    args = ap.parse_args(argv)

    if args.phase == "selection":
        run_selection(args.data_root, args.seeds, train=not args.no_train, dry_run=args.dry_run)
        return 0
    run_confirmation(args.data_root, args.seeds, args.winner, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
