"""Assemble the cross-dataset ranking report using the *recorded* OCELOT results.

WHY THIS EXISTS
---------------
The #234 OCELOT sweep's on-disk artifacts (dense cache, per-replicate ``metrics.json`` /
``predictions.json``, ``ranking_report.json``) were lost when the run node's ``/var/tmp``
was cleaned. What survives is the committed record: the per-encoder ``mean_f1`` mean ± std
over 3 seeds in ``docs/ocelot-detection-benchmark.rst`` (recorded 2026-07-15 at
``54601e4``). Re-running the full OCELOT sweep to regenerate what the docs already record
was ruled out — the recorded table is the citable artifact.

The cross-dataset objects in ``ranking_report.json`` (``rank_consistency``, the aggregate
rank, the frozen subset ``selections``) only need each dataset's encoder *ranking*, which
the recorded means fully determine. This driver therefore rebuilds the report from:

* the fresh on-disk cells of the datasets that were swept locally (``--datasets``), plus
* synthetic OCELOT :class:`~soma.benchmarks.detection_benchmark.Cell` rows carrying the
  recorded means (empty ``per_replicate`` — the per-seed values were not recorded).

Bootstrap ``stability`` and ``robustness`` remain scoped to the locally swept datasets
(they need per-sample predictions, which the recorded table does not carry). The report's
``config`` gains a ``recorded_cells`` block naming the provenance so a reader can tell the
recorded ranking from the freshly measured ones.

Usage (after the MIDOG sweep completes)::

    python examples/detection_benchmark/report_with_recorded_ocelot.py \
        --out-root /maindisk/clement/detection_benchmark/out \
        --data-root /maindisk/clement/detection_benchmark/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import (  # noqa: E402
    collect_cells,
    collect_robustness,
    collect_stability_samples,
)

from soma.benchmarks.detection_benchmark import (  # noqa: E402
    DEFAULT_ROSTER,
    DEFAULT_SEEDS,
    Cell,
    build_ranking_report,
)

# The committed OCELOT record: frozen-probe mean_f1 over seeds {0,1,2}, from
# docs/ocelot-detection-benchmark.rst (recorded 2026-07-15 @ 54601e4). The per-seed
# values were not recorded, only mean ± std — hence the empty per_replicate below.
RECORDED_OCELOT_PROVENANCE = {
    "source": "docs/ocelot-detection-benchmark.rst",
    "recorded": "2026-07-15",
    "git_sha": "54601e4",
    "note": (
        "OCELOT cells are the committed record of the #234 sweep (its on-disk artifacts "
        "were lost); per_replicate values were not recorded. Bootstrap stability and "
        "robustness therefore cover only the locally swept datasets."
    ),
}
RECORDED_OCELOT_MEAN_F1: dict[str, tuple[float, float]] = {
    "genbio-pathfm": (0.733, 0.004),
    "h-optimus-1": (0.720, 0.002),
    "h0-mini": (0.719, 0.003),
    "conchv15": (0.711, 0.001),
    "virchow2": (0.699, 0.004),
    "midnight": (0.658, 0.002),
    "dinov2-vitb14": (0.656, 0.002),
}


def recorded_ocelot_cells() -> list[Cell]:
    """The recorded OCELOT sweep as ranking cells (mean ± std only, no per-seed values)."""
    return [
        Cell(
            encoder=encoder,
            dataset="ocelot",
            metric_name="mean_f1",
            per_replicate=(),
            mean=mean,
            std=std,
            n_replicates=3,
            replicate_axis="seeds",
            test_source="local_holdout:recorded@54601e4",
        )
        for encoder, (mean, std) in RECORDED_OCELOT_MEAN_F1.items()
    ]


def build_full_report(
    out_root: str | Path,
    *,
    data_root: str | Path | None = None,
    datasets: list[str] | None = None,
    git_sha: str | None = None,
    n_boot: int = 1000,
) -> dict:
    """The merged report: fresh cells from ``datasets`` on disk + the recorded OCELOT cells."""
    datasets = datasets or ["midog"]
    if "ocelot" in datasets:
        raise ValueError(
            "'ocelot' comes from the recorded table, not from disk; drop it from --datasets."
        )
    roster = DEFAULT_ROSTER
    cells = list(collect_cells(out_root, roster, datasets)) + recorded_ocelot_cells()
    stability = collect_stability_samples(out_root, roster, datasets)
    robustness = (
        collect_robustness(out_root, data_root, roster, datasets)
        if data_root is not None
        else {}
    )
    report = build_ranking_report(
        cells,
        roster=roster,
        stability_samples=stability or None,
        robustness=robustness or None,
        git_sha=git_sha,
        replicate_policy={"single_fold_seeds": list(DEFAULT_SEEDS)},
        n_boot=n_boot,
    )
    report["config"]["recorded_cells"] = {"ocelot": dict(RECORDED_OCELOT_PROVENANCE)}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-root", type=Path, required=True,
                    help="sweep output root (holds <dataset>/<encoder>/replicate_* dirs)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="curated data root; enables the robustness block when given")
    ap.add_argument("--datasets", nargs="+", default=["midog"],
                    help="locally swept datasets to collect from disk (ocelot is recorded)")
    args = ap.parse_args(argv)

    from soma.output_layout import _git_sha

    report = build_full_report(
        args.out_root,
        data_root=args.data_root,
        datasets=list(args.datasets),
        git_sha=_git_sha(REPO_ROOT),
    )
    out_path = Path(args.out_root) / "ranking_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_recorded = len(RECORDED_OCELOT_MEAN_F1)
    print(f"wrote {out_path} ({len(report['cells'])} cells, {n_recorded} recorded ocelot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
