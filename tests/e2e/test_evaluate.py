"""Scoring predictions that soma did not produce: no run directory, no model, no config.

A user brings a predictions table (one row per sample, the head's prediction columns)
and the cohort's ``dataset.csv`` / ``splits.csv``. ``soma.evaluate`` joins the two by
``sample_id`` and reports JSON, through Python and through ``soma evaluate``. The
expected numbers are worked by hand from the synthetic values below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

import soma
from soma.aggregators import aggregator_registry
from soma.cli import main as soma_main
from soma.data import Cohort, from_arrays
from soma.evaluation import Predictions
from soma.learn import loaders
from soma.tasks.classification import BinaryClassificationHead
from soma.training.model import MILModel

TRAIN = [f"s{i:02d}" for i in range(4)]
TEST = [f"s{i:02d}" for i in range(4, 12)]
#: Test labels and the external model's positive-class probabilities. Exactly one
#: (positive, negative) pair is mis-ordered (0.4 < 0.6), so AUROC = 15/16; thresholding
#: at 0.5 gets 3 of 4 positives and 3 of 4 negatives right, so balanced accuracy = 0.75.
TEST_LABELS = ["benign"] * 4 + ["tumor"] * 4
TEST_PROB_TUMOR = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
#: Regression values for every sample and an external regressor's predictions: the
#: absolute errors sum to 6.0 over 12 samples, so MAE = 0.5.
VALUES = [float(i) for i in range(12)]
PREDICTED_VALUES = [v + (1.0 if i % 2 else 0.0) for i, v in enumerate(VALUES)]


def _write_cohort(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    labels = ["benign", "tumor", "benign", "tumor", *TEST_LABELS]
    dataset = pd.DataFrame(
        {
            "sample_id": TRAIN + TEST,
            "label": labels,
            "value": VALUES,
            "site": ["A", "B"] * 6,
        }
    )
    splits = pd.DataFrame({"sample_id": TRAIN + TEST, "split": ["train"] * 4 + ["test"] * 8})
    dataset.to_csv(root / "dataset.csv", index=False)
    splits.to_csv(root / "splits.csv", index=False)
    return root / "dataset.csv", root / "splits.csv"


def test_external_predictions_are_scored_without_a_run_directory(tmp_path, new_artifact):
    artifact = new_artifact("evaluate_external_predictions")
    dataset_csv, splits_csv = _write_cohort(tmp_path / "cohort")

    # A binary classifier's output, written through the public noun. Class order is
    # declared, so the cohort's string labels map onto prob_0 / prob_1.
    frame = pd.DataFrame(
        {
            "sample_id": TEST,
            "prob_0": [1.0 - p for p in TEST_PROB_TUMOR],
            "prob_1": TEST_PROB_TUMOR,
            "predicted_label": [int(p > 0.5) for p in TEST_PROB_TUMOR],
        }
    )
    classifier = tmp_path / "external" / "classifier.parquet"
    classifier.parent.mkdir()
    Predictions(frame, task_family="binary_classification", classes=["benign", "tumor"]).to_parquet(
        classifier
    )

    # Python: cohort targets + predictions read back from disk.
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=["label"])
    evaluation = soma.evaluate(
        Predictions.from_parquet(classifier),
        cohort.targets("test"),
        metrics=["auroc", "balanced_accuracy"],
        subgroups=["site"],
    )
    python_json = tmp_path / "python_evaluation.json"
    evaluation.to_json(python_json)
    artifact.check_equal("python/auroc", evaluation.metrics["auroc"], 15 / 16)
    artifact.check_equal("python/balanced_accuracy", evaluation.metrics["balanced_accuracy"], 0.75)
    artifact.check_equal("python/num_samples", evaluation.num_samples, 8)
    artifact.check_equal(
        "python/subgroup_sizes",
        {group: values["n"] for group, values in evaluation.subgroups["metrics"]["site"].items()},
        {"A": 4, "B": 4},
    )

    # CLI: the same file, the same cohort, the same JSON.
    cli_json = tmp_path / "cli_evaluation.json"
    soma_main(
        [
            "evaluate",
            str(classifier),
            "--cohort",
            str(dataset_csv),
            "--splits",
            str(splits_csv),
            "--split",
            "test",
            "--metrics",
            "auroc",
            "balanced_accuracy",
            "--subgroups",
            "site",
            "--out",
            str(cli_json),
        ]
    )
    artifact.check_equal(
        "cli/json_matches_python",
        json.loads(cli_json.read_text()),
        json.loads(python_json.read_text()),
    )

    # A plain pandas parquet (no soma metadata) over the whole, unsplit cohort: the
    # family is read off the columns and every row of dataset.csv is a target.
    regressor = tmp_path / "external" / "regressor.parquet"
    pd.DataFrame({"sample_id": TRAIN + TEST, "predicted_value": PREDICTED_VALUES}).to_parquet(
        regressor, index=False
    )
    unsplit_json = tmp_path / "unsplit_evaluation.json"
    soma_main(
        ["evaluate", str(regressor), "--cohort", str(dataset_csv), "--metrics", "mae", "--out", str(unsplit_json)]
    )
    unsplit = json.loads(unsplit_json.read_text())
    artifact.check_equal("cli_unsplit/task_family", unsplit["task_family"], "regression")
    artifact.check_equal("cli_unsplit/mae", unsplit["metrics"]["mae"], 0.5)
    artifact.check_equal("cli_unsplit/num_samples", unsplit["num_samples"], 12)
    artifact.check_equal("no_run_directory", sorted(p.name for p in tmp_path.iterdir()), sorted(
        ["cohort", "external", "python_evaluation.json", "cli_evaluation.json", "unsplit_evaluation.json"]
    ))
    artifact.assert_passed()


def test_cli_scores_a_multi_output_regression_parquet(tmp_path, new_artifact):
    """A ``RegressionHead(num_outputs=G)`` writes ``value_0..value_{G-1}``; ``soma evaluate``
    reads the same-named cohort columns and reports ``mean_pearson`` by default."""
    artifact = new_artifact("evaluate_cli_multi_output_regression")
    root = tmp_path / "cohort"
    root.mkdir()
    values_0 = [float(i) for i in range(12)]
    values_1 = [float((i * 5) % 12) for i in range(12)]
    pd.DataFrame({"sample_id": TRAIN + TEST, "value_0": values_0, "value_1": values_1}).to_csv(
        root / "dataset.csv", index=False
    )
    pd.DataFrame({"sample_id": TRAIN + TEST, "split": ["train"] * 4 + ["test"] * 8}).to_csv(
        root / "splits.csv", index=False
    )
    # Output 0 is an affine map of its target (Pearson 1), output 1 a negated one
    # (Pearson -1): the mean over the two outputs is 0.
    test_rows = slice(len(TRAIN), None)
    predictions = tmp_path / "multi_output.parquet"
    Predictions(
        pd.DataFrame(
            {
                "sample_id": TEST,
                "value_0": [2.0 * v + 1.0 for v in values_0[test_rows]],
                "value_1": [-v for v in values_1[test_rows]],
            }
        ),
        task_family="regression",
    ).to_parquet(predictions)
    out = tmp_path / "evaluation.json"
    soma_main(
        [
            "evaluate", str(predictions), "--cohort", str(root / "dataset.csv"),
            "--splits", str(root / "splits.csv"), "--out", str(out),
        ]
    )
    result = json.loads(out.read_text())
    artifact.check_equal("metrics", sorted(result["metrics"]), ["mean_pearson"])
    artifact.check_equal("mean_pearson", round(result["metrics"]["mean_pearson"], 9), 0.0)
    artifact.check_equal("num_samples", result["num_samples"], 8)
    artifact.assert_passed()


def test_fit_predict_evaluate_journey_over_user_arrays(tmp_path, new_artifact):
    """The library journey: arrays + a cohort → ``loaders`` → ``fit`` → ``predict`` →
    ``evaluate``, the same verbs ``Pipeline`` composes."""
    artifact = new_artifact("verbs_journey")
    dataset_csv, splits_csv = _write_cohort(tmp_path / "cohort")
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=["label"])
    # Tumour bags carry a shifted marker channel; benign bags do not.
    generator = torch.Generator().manual_seed(0)
    bags = {}
    for record in cohort.records:
        bag = torch.randn(6, 4, generator=generator)
        bag[:, 0] += 3.0 if record.targets["label"] == "tumor" else -3.0
        bags[record.sample_id] = bag

    torch.manual_seed(0)
    aggregator = aggregator_registry.get("abmil")(input_dim=4, hidden_dim=8)
    head = BinaryClassificationHead(
        input_dim=aggregator.output_dim, num_classes=2, label_map={"benign": 0, "tumor": 1}
    )
    fold = loaders(from_arrays(bags), cohort, head=head, batch_size=2)
    result = soma.fit(
        MILModel(aggregator=aggregator, task_head=head),
        fold.train,
        optimizer={"name": "adam", "params": {"lr": 1e-2}},
        max_epochs=20,
        patience=None,
        progress=False,
        device="cpu",
    )
    predictions = soma.predict(result.learner, fold.tests["test"])
    evaluation = soma.evaluate(predictions, cohort.targets("test"), metrics=["auroc", "accuracy"])

    artifact.check_equal("predictions/ids", predictions.sample_ids, TEST)
    artifact.check_equal("predictions/classes", list(predictions.classes), ["benign", "tumor"])
    artifact.check_equal("test/auroc", evaluation.metrics["auroc"], 1.0)
    artifact.check_equal("test/accuracy", evaluation.metrics["accuracy"], 1.0)
    artifact.assert_passed()
