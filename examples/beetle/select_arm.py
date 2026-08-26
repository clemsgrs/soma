"""Select exactly one BEETLE sampling arm from development OOF evidence only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Sequence

from examples.beetle.protocol import ARM_NAMES, NUM_FOLDS


EXPECTED_FOLDS = tuple(range(NUM_FOLDS))


def select_development_arm(oof_report: str | Path) -> dict:
    """Equal-average five fold scores and choose one arm, with uniform as tie-break."""
    payload = json.loads(Path(oof_report).read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if protocol.get("folds") != len(EXPECTED_FOLDS):
        raise ValueError("BEETLE selection requires exactly five development folds")
    if tuple(protocol.get("arms", ())) != ARM_NAMES or set(payload.get("arms", {})) != set(
        ARM_NAMES
    ):
        raise ValueError(f"BEETLE selection requires both arms {list(ARM_NAMES)}")

    reports: dict[str, dict] = {}
    for arm in ARM_NAMES:
        folds = payload["arms"][arm].get("primary", {}).get("folds", {})
        if set(folds) != {str(fold) for fold in EXPECTED_FOLDS}:
            raise ValueError(f"BEETLE arm {arm!r} requires development folds 0 through 4")
        scores = [folds[str(fold)].get("mean_dice") for fold in EXPECTED_FOLDS]
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            for score in scores
        ):
            raise ValueError(f"BEETLE arm {arm!r} has an invalid fold mean class Dice")
        numeric_scores = [float(score) for score in scores]
        reports[arm] = {
            "fold_scores": numeric_scores,
            "mean": statistics.fmean(numeric_scores),
            "standard_deviation": statistics.stdev(numeric_scores),
        }

    best_score = max(report["mean"] for report in reports.values())
    # A fixed uniform-arm tie-break guarantees one decision without consulting another
    # statistic or the sequestered External evaluation set.
    selected = next(arm for arm in ARM_NAMES if reports[arm]["mean"] == best_score)
    return {
        "schema_version": 1,
        "criterion": "fold_macro_class_dice",
        "development_evidence_only": True,
        "tie_breaker": "uniform",
        "arms": reports,
        "selected_arm": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selection = select_development_arm(args.oof_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["select_development_arm", "main"]
