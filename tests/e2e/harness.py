"""Drive ``soma <config.yaml>`` end to end and turn a run into a repeatable artifact.

The artifact is the scenario's verifiable output: the metrics the run reported, the
checks the test applied to them, and content digests of the prediction files, all
free of timestamps, run ids and absolute paths. Two runs of the same code on the
same machine produce byte-identical artifacts, so ``diff`` between two artifacts is
a regression check in its own right.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from soma.cli import main as soma_main

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.environ.get("SOMA_E2E_ARTIFACT_DIR", REPO_ROOT / "e2e-artifacts"))

#: Settings that keep every scenario in-process, single-threaded-loader and CPU-sized.
CPU_EXECUTION = {
    "num_gpus": 1,
    "num_workers_per_gpu": 0,
    "num_preprocessing_workers": 0,
    "precision": "fp32",
}
CPU_LOADER = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


@dataclass
class Run:
    run_dir: Path
    summary: dict[str, float]
    config_path: Path

    def predictions(self) -> dict[str, Path]:
        return {p.name: p for p in sorted(self.run_dir.glob("predictions_*.csv"))}

    def histories(self) -> dict[str, list[dict[str, Any]]]:
        """Per-fold epoch records, keyed ``""`` (single fold) or ``"fold_N"``."""
        paths = sorted(self.run_dir.glob("training_history.json")) + sorted(
            self.run_dir.glob("fold_*/training_history.json")
        )
        return {
            path.parent.name if path.parent != self.run_dir else "": json.loads(path.read_text())[
                "epochs"
            ]
            for path in paths
        }


def run_soma(config: dict[str, Any], config_path: Path) -> Run:
    """Write ``config`` as YAML and run it through the real CLI entry point."""
    output_root = Path(config["run"]["output_root"])
    before = set(output_root.glob("experiments/*/runs/*"))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    soma_main([str(config_path)])
    created = sorted(set(output_root.glob("experiments/*/runs/*")) - before)
    assert len(created) == 1, f"expected one new run dir, found {created}"
    run_dir = created[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    return Run(run_dir=run_dir, summary=summary, config_path=config_path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Artifact:
    """Accumulates a scenario's evidence; ``write`` persists it as sorted JSON."""

    scenario: str
    #: Paths under this directory are recorded as ``<tmp>/...``.
    workdir: Path
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)

    def record_run(self, name: str, run: Run) -> None:
        config = yaml.safe_load(run.config_path.read_text())
        self.runs[name] = {
            "config": _relativize(config, self.workdir),
            "metrics": {k: _round(v) for k, v in sorted(run.summary.items())},
            "predictions_sha256": {k: sha256(p) for k, p in run.predictions().items()},
        }

    def check_at_least(self, name: str, value: float, threshold: float) -> None:
        self._check(name, value, f">= {threshold}", value >= threshold)

    def check_at_most(self, name: str, value: float, threshold: float) -> None:
        self._check(name, value, f"<= {threshold}", value <= threshold)

    def check_training_reduced_loss(self, name: str, run: Run, factor: float = 0.9) -> None:
        """The last epoch's train loss is below ``factor`` x the first epoch's, per fold.

        Metric thresholds alone cannot tell a trained model from an untrained one when a
        random projection of the features already orders the samples; a falling loss can.
        """
        histories = run.histories()
        self._check(f"{name}/training_history", len(histories), ">= 1", len(histories) >= 1)
        for fold, epochs in histories.items():
            ratio = epochs[-1]["train_loss"] / epochs[0]["train_loss"]
            label = f"{name}/{fold}/train_loss_ratio" if fold else f"{name}/train_loss_ratio"
            self._check(label, ratio, f"<= {factor}", ratio <= factor)

    def check_equal(self, name: str, value: Any, expected: Any) -> None:
        self._check(name, value, f"== {expected!r}", value == expected)

    def _check(self, name: str, value: Any, expectation: str, passed: bool) -> None:
        self.checks[name] = {"value": _round(value), "expect": expectation, "passed": bool(passed)}

    def payload(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "passed": all(check["passed"] for check in self.checks.values()),
            "checks": self.checks,
            "facts": self.facts,
            "runs": self.runs,
        }

    def write(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / f"{self.scenario}.json"
        path.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n")
        return path

    def assert_passed(self) -> None:
        """Write the artifact, then fail with every unmet check at once."""
        path = self.write()
        failed = {k: v for k, v in self.checks.items() if not v["passed"]}
        assert not failed, f"{self.scenario}: failed checks {failed} (artifact: {path})"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _relativize(value: Any, workdir: Path) -> Any:
    if isinstance(value, dict):
        return {k: _relativize(v, workdir) for k, v in value.items()}
    if isinstance(value, list):
        return [_relativize(v, workdir) for v in value]
    if isinstance(value, str) and value.startswith(str(workdir)):
        return "<tmp>" + value[len(str(workdir)) :]
    return value
