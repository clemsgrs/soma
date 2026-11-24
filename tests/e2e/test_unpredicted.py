"""Samples without features: a run never invents a prediction for them.

A slide whose tiling kept no tile has no bag, so the model cannot predict it. The
feature manifest records it as ``empty``. A sample of another status but ``success``
(an extraction ``error``, a ``tbp``) has no bag either. In a tune or test split such a
sample is a target without a prediction, and ``evaluation.unpredicted`` decides what happens:
``error`` (the default) stops the run at fold setup, before any training, and
``exclude`` scores the predicted samples only, lists the others in
``not_predicted_<split>.parquet`` and warns. A train sample without features is left
out of training whatever the policy. ``soma evaluate --unpredicted`` and
``soma.evaluate(unpredicted=)`` apply the same policy to a predictions parquet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

import soma
from soma.cli import main as soma_main
from soma.data import CachedSetSource, Cohort
from soma.evaluation import Predictions
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, run_soma, trainer_section
from tests.e2e.synthetic import write_cached_bags

#: write_cached_bags trains s00..s09, tunes on s10..s12 and tests on s13..s15.
METRICS = ["auroc", "balanced_accuracy"]


def _config(tmp_path: Path, dataset_csv: Path, splits_csv: Path, **evaluation) -> dict:
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {"execution": CPU_EXECUTION},
        "loaders": {"batch_size": 2, **CPU_LOADER},
        "learner": learner_section(
            "mil",
            {"aggregator": {"name": "abmil"}, "head": {"name": "binary_classification"}},
            learning_rate=5e-3,
        ),
        "trainer": trainer_section(6),
        "evaluation": {"metrics": METRICS, **evaluation},
    }


def _counts(metrics: dict) -> tuple:
    return metrics["num_samples"], metrics["num_predicted"], round(metrics["coverage"], 6)


def _not_predicted(path: Path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


class _FitWatcher:
    """A read-only runtime observer that notes each fit start and each epoch end."""

    def __init__(self, seen: list) -> None:
        self.seen = seen

    def on_fit_start(self, trainer, learner) -> None:
        self.seen.append("fit_start")

    def on_epoch_end(self, trainer, learner) -> None:
        self.seen.append(f"epoch_{trainer.epoch}")


@pytest.mark.parametrize(
    ("scenario", "split", "sample", "status", "reason"),
    [
        ("tune", "tune", "s10", "empty", "no tiles"),
        ("test", "test", "s15", "empty", "no tiles"),
        ("test_tbp", "test", "s14", "tbp", "feature status 'tbp'"),
    ],
)
def test_a_tune_or_test_sample_without_features_stops_the_run_before_training(
    tmp_path, new_artifact, scenario, split, sample, status, reason
):
    artifact = new_artifact(f"unpredicted_error_{scenario}")
    dataset_csv, splits_csv, feature_dir = write_cached_bags(
        tmp_path / "data", tileless=(), statuses={sample: status}
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path, dataset_csv, splits_csv), sort_keys=False))
    seen: list = []
    pipeline = soma.Pipeline(
        soma.ExperimentConfig.from_yaml(config_path),
        source=CachedSetSource(feature_dir),
        callback_factory=lambda context: [_FitWatcher(seen)],
    )

    with pytest.raises(ValueError) as refused:
        pipeline.run()

    message = str(refused.value)
    artifact.check_equal("message/names_the_sample", f"{split}: ['{sample}'] ({reason})" in message, True)
    artifact.check_equal("message/remove_them", "remove them from the cohort or the splits" in message, True)
    artifact.check_equal("message/or_exclude", "evaluation.unpredicted: exclude" in message, True)
    artifact.check_equal("observer/no_fit_started", seen, [])
    trained = [p.name for p in (tmp_path / "runs").rglob("*") if p.name in {"training_history.json", "bundle"}]
    artifact.check_equal("nothing_trained", trained, [])
    artifact.assert_passed()


def test_exclude_scores_the_predicted_samples_and_lists_the_others(tmp_path, new_artifact, caplog):
    """s15 has no tiles and s16's extraction failed (feature status ``error``): neither
    has a bag, and both are listed with their reason."""
    artifact = new_artifact("unpredicted_exclude")
    dataset_csv, splits_csv, feature_dir = write_cached_bags(
        tmp_path / "data", n=17, tileless=("s15",), statuses={"s16": "error"}
    )
    config = _config(tmp_path, dataset_csv, splits_csv, unpredicted="exclude")

    with caplog.at_level(logging.WARNING, logger="soma"):
        run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(feature_dir))
    artifact.record_run("exclude", run)

    predictions = Predictions.from_parquet(run.run_dir / "predictions_test.parquet")
    artifact.check_equal("predictions/ids", predictions.sample_ids, ["s13", "s14"])
    artifact.check_equal(
        "not_predicted",
        _not_predicted(run.run_dir / "not_predicted_test.parquet"),
        [{"sample_id": "s15", "reason": "no tiles"}, {"sample_id": "s16", "reason": "feature status 'error'"}],
    )
    metrics = json.loads((run.run_dir / "metrics.json").read_text())
    artifact.check_equal("metrics/test/counts", _counts(metrics["test"]), (4, 2, 0.5))
    artifact.check_equal("metrics/tune/counts", _counts(metrics["tune"]), (3, 3, 1.0))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    artifact.check_equal(
        "warning", any("test scored 2 of 4 samples" in w and "not_predicted_test.parquet" in w for w in warnings), True
    )
    artifact.check_training_reduced_loss("exclude", run)

    # The run's predictions, scored again: the default refuses the target without one, and
    # exclude gives the run's own test metrics, through the CLI and through Python.
    evaluate = ["evaluate", str(run.run_dir / "predictions_test.parquet"), "--cohort", str(dataset_csv)]
    evaluate += ["--splits", str(splits_csv), "--metrics", *METRICS]
    with pytest.raises(SystemExit) as refused:
        soma_main(evaluate)
    artifact.check_equal("cli/error/exit_code", refused.value.code, 2)
    out = tmp_path / "evaluation.json"
    soma_main([*evaluate, "--unpredicted", "exclude", "--out", str(out)])
    artifact.check_equal("cli/exclude/metrics", json.loads(out.read_text())["metrics"], metrics["test"])
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets={"label": "label"}, dtypes={"label": object})
    with pytest.raises(ValueError, match=r"2 target\(s\) have no prediction: \['s15', 's16'\]"):
        soma.evaluate(predictions, cohort.targets("test"), metrics=METRICS)
    python = soma.evaluate(predictions, cohort.targets("test"), metrics=METRICS, unpredicted="exclude")
    artifact.check_equal("python/exclude/metrics", python.metrics, metrics["test"])

    # A resume under the default policy refuses the completed run, although its test
    # split was already scored and is not scored again.
    resume = _config(tmp_path, dataset_csv, splits_csv)
    resume["run"]["resume"] = True
    with pytest.raises(ValueError) as refused:
        run_soma(resume, tmp_path / "resume.yaml", "--feature-dir", str(feature_dir))
    artifact.check_equal(
        "resume_error/names_the_samples",
        "test: ['s15'] (no tiles), ['s16'] (feature status 'error')" in str(refused.value),
        True,
    )
    run_dirs = sorted((tmp_path / "runs").glob("experiments/*/runs/*"))
    artifact.check_equal("resume_error/reused_the_run", run_dirs == [run.run_dir], True)
    artifact.assert_passed()


def test_a_run_that_predicts_every_tune_and_test_sample_writes_an_empty_list(tmp_path, new_artifact, caplog):
    """s00 is a train sample without features: it is left out of training, with a log
    line, and the default policy does not stop the run."""
    artifact = new_artifact("unpredicted_none")
    dataset_csv, splits_csv, feature_dir = write_cached_bags(tmp_path / "data", tileless=("s00",))
    config = _config(tmp_path, dataset_csv, splits_csv)

    with caplog.at_level(logging.INFO, logger="soma"):
        run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(feature_dir))
    artifact.record_run("none", run)

    not_predicted = pd.read_parquet(run.run_dir / "not_predicted_test.parquet")
    artifact.check_equal("not_predicted/columns", list(not_predicted.columns), ["sample_id", "reason"])
    artifact.check_equal("not_predicted/rows", len(not_predicted), 0)
    metrics = json.loads((run.run_dir / "metrics.json").read_text())
    artifact.check_equal("metrics/test/counts", _counts(metrics["test"]), (3, 3, 1.0))
    artifact.check_equal("metrics/tune/counts", _counts(metrics["tune"]), (3, 3, 1.0))
    left_out = "train=9" in caplog.text and "train empty dropped=1" in caplog.text
    artifact.check_equal("log/train_left_out", left_out, True)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    artifact.check_equal("no_coverage_warning", [w for w in warnings if " scored " in w], [])
    artifact.check_training_reduced_loss("none", run)
    artifact.assert_passed()
