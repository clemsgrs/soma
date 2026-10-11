"""Public-API pins for the evaluation verbs: ``Predictions``, ``evaluate``, ``Baseline``,
``compare``, ``predict`` (with its callbacks) and ``Cohort.targets``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

import soma
from soma.aggregators import aggregator_registry
from soma.data import Cohort, FoldSplit, SampleRecord, from_arrays
from soma.evaluation import Baseline, Evaluation, Predictions, compare, evaluate, predict
from soma.heatmaps import SaveAttention
from soma.tasks.classification import BinaryClassificationHead
from soma.training.bag_dataset import BagDataset
from soma.training.collate import bag_collate_fn
from soma.training.model import MILModel

IDS = ["a", "b", "c", "d"]


def _binary(ids=IDS, probs=(0.2, 0.9, 0.4, 0.7), **kwargs) -> Predictions:
    frame = pd.DataFrame(
        {
            "sample_id": list(ids),
            "prob_0": [1 - p for p in probs],
            "prob_1": list(probs),
            "predicted_label": [int(p > 0.5) for p in probs],
        }
    )
    return Predictions(frame, task_family="binary_classification", **kwargs)


def _labels(labels=(0, 1, 0, 1), ids=IDS) -> pd.DataFrame:
    return pd.DataFrame({"label": list(labels)}, index=pd.Index(list(ids), name="sample_id"))


# --- Predictions ------------------------------------------------------------------- #


def test_parquet_round_trip_keeps_the_frame_and_its_description(tmp_path):
    predictions = Predictions(
        _binary(classes=["benign", "tumor"]).frame.assign(margin=[0.1, 0.2, 0.3, 0.4]),
        task_family="binary_classification",
        prediction_columns=["prob_0", "prob_1", "predicted_label"],
        classes=["benign", "tumor"],
    )

    path = predictions.to_parquet(tmp_path / "p.parquet")
    reread = Predictions.from_parquet(path)

    pd.testing.assert_frame_equal(reread.frame, predictions.frame)
    assert reread.task_family == "binary_classification"
    assert reread.classes == ("benign", "tumor")
    assert reread.prediction_columns == ("prob_0", "prob_1", "predicted_label")
    meta = json.loads(pq.read_schema(path).metadata[b"soma"])
    assert meta["soma_version"] == soma.__version__
    assert meta["columns"] == {
        "sample_id": "id",
        "prob_0": "prediction",
        "prob_1": "prediction",
        "predicted_label": "prediction",
        "margin": "extra",
    }


def test_a_plain_parquet_reads_its_family_off_the_columns(tmp_path):
    pd.DataFrame({"sample_id": IDS, "risk_score": [1.0, 2.0, 3.0, 4.0]}).to_parquet(tmp_path / "p.parquet")

    assert Predictions.from_parquet(tmp_path / "p.parquet").task_family == "survival"


def test_concat_marks_rows_without_the_flag_as_real():
    real = _binary(ids=["a", "b"], probs=(0.2, 0.9))
    filled = Baseline.fit(_labels(), task_family="binary_classification").predict(["c"])

    combined = Predictions.concat([real, filled])

    assert combined.sample_ids == ["a", "b", "c"]
    assert combined.frame["is_placeholder"].tolist() == [False, False, True]


# --- evaluate ---------------------------------------------------------------------- #


def test_evaluate_scores_the_joined_frames():
    evaluation = evaluate(_binary(), _labels(), metrics=["auroc", "accuracy"])

    assert isinstance(evaluation, Evaluation)
    # Positives (b 0.9, d 0.7) outrank both negatives (a 0.2, c 0.4).
    assert evaluation.metrics == {"auroc": 1.0, "accuracy": 1.0}
    assert evaluation.num_samples == 4
    assert evaluation.task_family == "binary_classification"


def test_default_metrics_are_the_family_catalogue_defaults():
    evaluation = evaluate(_binary(), _labels())

    assert list(evaluation.metrics) == ["auroc", "balanced_accuracy", "auprc", "f1"]


def test_a_prediction_without_a_target_is_refused():
    with pytest.raises(ValueError, match="1 prediction id"):
        evaluate(_binary(), _labels(ids=["a", "b", "c", "x"]))


def test_a_target_without_a_prediction_is_refused_and_points_at_baseline():
    targets = pd.concat([_labels(), _labels(labels=(1,), ids=["e"])])
    with pytest.raises(ValueError, match="1 target.*Baseline"):
        evaluate(_binary(), targets)


def test_duplicate_prediction_ids_are_refused():
    with pytest.raises(ValueError, match="more than once"):
        evaluate(_binary(ids=["a", "a", "c", "d"]), _labels())


def test_a_metric_of_another_family_is_refused():
    with pytest.raises(ValueError, match="'mse' does not apply"):
        evaluate(_binary(), _labels(), metrics=["mse"])


def positives_predicted(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    return float(predictions["predicted_label"].sum())


def test_callables_and_dotted_paths_are_metrics():
    evaluation = evaluate(
        _binary(),
        _labels(),
        metrics=[
            lambda p, t: float((p["predicted_label"] == t["label"]).mean()),
            "tests.test_predictions_and_evaluate.positives_predicted",
        ],
    )

    assert evaluation.metrics == {"<lambda>": 1.0, "positives_predicted": 2.0}


def test_raw_labels_map_through_the_declared_classes():
    targets = _labels(labels=("benign", "tumor", "benign", "tumor"))

    assert evaluate(_binary(classes=["benign", "tumor"]), targets, metrics=["accuracy"]).metrics == {
        "accuracy": 1.0
    }
    with pytest.raises(ValueError, match="not class indices"):
        evaluate(_binary(), targets)
    with pytest.raises(ValueError, match="not among"):
        evaluate(_binary(classes=["normal", "tumor"]), targets)


def test_placeholders_add_coverage_and_counts():
    predictions = Predictions.concat(
        [
            _binary(ids=["a", "b", "c"], probs=(0.2, 0.9, 0.4)),
            Baseline.fit(_labels(), task_family="binary_classification").predict(["d"]),
        ]
    )

    metrics = evaluate(predictions, _labels(), metrics=["accuracy"]).metrics

    assert metrics == {
        "accuracy": 0.75,
        "coverage": 0.75,
        "num_samples": 4,
        "num_real_samples": 3,
        "num_placeholder_samples": 1,
    }


def test_to_json_holds_metrics_and_subgroups(tmp_path):
    targets = _labels().assign(site=["A", "A", "B", "B"])
    evaluation = evaluate(_binary(), targets, metrics=["accuracy"], subgroups=["site"])

    payload = json.loads(evaluation.to_json(tmp_path / "e.json"))

    assert payload == json.loads((tmp_path / "e.json").read_text())
    assert payload["metrics"] == {"accuracy": 1.0}
    assert payload["num_samples"] == 4
    assert payload["subgroups"]["metrics"]["site"]["A"] == {"accuracy": 1.0, "n": 2}


# --- Baseline ---------------------------------------------------------------------- #


def test_baseline_predicts_the_training_class_prior():
    baseline = Baseline.fit(
        _labels(labels=("tumor", "tumor", "tumor", "benign")),
        task_family="binary_classification",
        classes=["benign", "tumor"],
    )

    frame = baseline.predict(["x", "y"]).frame

    assert frame["prob_1"].tolist() == [0.75, 0.75]
    assert frame["predicted_label"].tolist() == [1, 1]
    assert frame["is_placeholder"].all()


@pytest.mark.parametrize(
    ("family", "targets", "expected"),
    [
        ("ordinal_classification", {"label": [0, 1, 2, 2]}, {"raw_score": 1.25, "predicted_label": 1}),
        ("regression", {"value": [1.0, 2.0, 6.0, 7.0]}, {"predicted_value": 4.0}),
        ("survival", {"time": [1.0, 2, 3, 4], "event": [1, 0, 1, 1]}, {"risk_score": 0.0}),
    ],
)
def test_baseline_per_family(family, targets, expected):
    frame = Baseline.fit(pd.DataFrame(targets, index=IDS), task_family=family).predict(["x"]).frame

    assert {column: frame[column].iloc[0] for column in expected} == expected


# --- compare ----------------------------------------------------------------------- #


def test_compare_gives_the_best_set_one_and_others_a_p_value():
    labels = [i % 2 for i in range(20)]
    ids = [f"s{i}" for i in range(20)]
    good = _binary(ids=ids, probs=[0.8 if y else 0.2 for y in labels])
    noisy = _binary(ids=ids, probs=list(np.random.default_rng(0).random(20)))

    p_values = compare([noisy, good], _labels(labels=labels, ids=ids), "auroc", n_permutations=50)

    assert p_values[1] == 1.0
    assert 0.0 <= p_values[0] < 0.5
    assert compare([good], _labels(labels=labels, ids=ids), "auroc") == [None]


# --- Cohort.targets ---------------------------------------------------------------- #


def _cohort() -> Cohort:
    records = [
        SampleRecord("s1", {"label": "tumor"}, patient_id="p1", metadata={"site": "A"}),
        SampleRecord("s2", {"label": "tumor"}, patient_id="p1", metadata={"site": "A"}),
        SampleRecord("s3", {"label": "benign"}, patient_id="p2", metadata={"site": "B"}),
        SampleRecord("s4", {"label": "benign"}, patient_id="p3", metadata={"site": "B"}),
    ]
    return Cohort(records, [FoldSplit(train=("s3",), tune=(), tests={"test": ("s2", "s1", "s4")})])


def test_cohort_targets_is_one_row_per_split_sample():
    targets = _cohort().targets("test")

    assert targets.index.name == "sample_id"
    assert targets.index.tolist() == ["s2", "s1", "s4"]
    assert targets.columns.tolist() == ["label", "patient_id", "site"]
    assert targets.loc["s4"].tolist() == ["benign", "p3", "B"]


def test_collapsed_cohort_targets_are_patient_keyed():
    targets = _cohort().collapse("patient_id").targets("test")

    assert targets.index.tolist() == ["p1", "p3"]
    assert targets["label"].tolist() == ["tumor", "benign"]


def test_cohort_targets_refuses_an_unknown_split():
    with pytest.raises(ValueError, match="no split 'external'"):
        _cohort().targets("external")


# --- predict ----------------------------------------------------------------------- #


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_predict_start(self, learner) -> None:
        self.events.append("start")

    def on_predict_batch(self, batch, output) -> None:
        self.events.append(f"batch:{len(batch.sample_ids)}")

    def on_predict_end(self) -> None:
        self.events.append("end")


def _bag_loader(head) -> DataLoader:
    torch.manual_seed(0)
    bags = {sid: torch.randn(3 + i, 4) for i, sid in enumerate(IDS)}
    records = [SampleRecord(sid, {"label": i % 2}) for i, sid in enumerate(IDS)]
    dataset = BagDataset(records, from_arrays(bags), head.extract_targets)
    collate = lambda items: bag_collate_fn(items, target_dtypes=head.target_dtypes)  # noqa: E731
    return DataLoader(dataset, batch_size=3, collate_fn=collate)


def test_predict_wraps_a_bare_model_and_drives_the_callbacks(tmp_path):
    torch.manual_seed(0)
    aggregator = aggregator_registry.get("abmil")(input_dim=4, hidden_dim=4)
    head = BinaryClassificationHead(input_dim=aggregator.output_dim, num_classes=2)
    model = MILModel(aggregator=aggregator, task_head=head)
    recorder = _Recorder()

    predictions = predict(model, _bag_loader(head), callbacks=[recorder, SaveAttention(tmp_path / "attn")])

    assert predictions.sample_ids == IDS
    assert predictions.prediction_columns == ("prob_0", "prob_1", "predicted_label")
    assert predictions.classes == (0, 1)
    assert recorder.events == ["start", "batch:3", "batch:1", "end"]
    # One normalised attention vector per bag, sliced to its own tiles.
    for i, sid in enumerate(IDS):
        attention = np.load(tmp_path / "attn" / f"{sid}.npz")["attention"]
        assert attention.shape == (3 + i,)
        assert attention.sum() == pytest.approx(1.0, abs=1e-5)
