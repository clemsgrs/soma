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


def _gene_panel(values=((0.1, 2.0), (0.4, 1.0), (0.3, 3.5), (0.9, 0.5))) -> Predictions:
    frame = pd.DataFrame(values, columns=["value_0", "value_1"])
    frame.insert(0, "sample_id", IDS)
    return Predictions(frame, task_family="regression")


def test_a_multi_output_regression_defaults_to_mean_pearson():
    targets = pd.DataFrame(
        {"value_0": [0.0, 0.5, 0.2, 1.0], "value_1": [2.5, 1.5, 3.0, 0.0]},
        index=pd.Index(IDS, name="sample_id"),
    )

    evaluation = evaluate(_gene_panel(), targets)

    assert list(evaluation.metrics) == ["mean_pearson"]
    assert evaluation.metrics == evaluate(_gene_panel(), targets, metrics=["mean_pearson"]).metrics


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


def test_a_binary_baseline_keeps_both_probability_columns_when_training_shows_one_class():
    baseline = Baseline.fit(_labels(labels=(0, 0, 0, 0)), task_family="binary_classification")

    frame = baseline.predict(["x"]).frame

    assert baseline.prediction_columns == ("prob_0", "prob_1", "predicted_label")
    assert frame[["prob_0", "prob_1"]].iloc[0].tolist() == [1.0, 0.0]


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


def test_a_multi_output_regression_baseline_predicts_each_output_mean():
    targets = pd.DataFrame({"value_0": [1.0, 2.0, 6.0, 7.0], "value_1": [0.0, 0.0, 1.0, 1.0]}, index=IDS)

    baseline = Baseline.fit(targets, task_family="regression")
    frame = baseline.predict(["x"]).frame

    assert baseline.prediction_columns == ("value_0", "value_1")
    assert frame[["value_0", "value_1"]].iloc[0].tolist() == [4.0, 0.5]


def test_a_scalar_regression_baseline_ignores_value_like_metadata_columns():
    from soma.data import Cohort

    records = pd.DataFrame(
        {
            "sample_id": IDS,
            "value_0": [1.0, 2.0, 6.0, 7.0],
            "value_1": [0.0, 0.0, 1.0, 1.0],  # an unused manifest column, kept as metadata
            "split": ["train", "train", "tune", "test"],
        }
    )
    cohort = Cohort.from_frames(records, targets={"value": "value_0"}, dtypes={"value": float})

    baseline = Baseline.fit(cohort.targets("train"), task_family="regression")

    assert baseline.prediction_columns == ("predicted_value",)
    assert baseline.predict(["x"]).frame["predicted_value"].iloc[0] == 1.5


def test_a_multi_output_regression_baseline_keeps_to_the_heads_target_keys():
    from soma.data import Cohort
    from soma.tasks.regression import RegressionHead

    records = pd.DataFrame(
        {
            "sample_id": IDS,
            "value_0": [1.0, 2.0, 6.0, 7.0],
            "value_1": [0.0, 0.0, 1.0, 1.0],
            "value_2": [5.0, 5.0, 5.0, 5.0],  # an unused manifest column, kept as metadata
            "split": ["train", "train", "tune", "test"],
        }
    )
    head = RegressionHead(input_dim=4, num_outputs=2)
    cohort = Cohort.from_frames(records, targets=list(head.target_keys), dtypes=dict(head.target_schema))

    baseline = Baseline.fit(cohort.targets("train"), task_family="regression", target_keys=head.target_keys)

    assert baseline.prediction_columns == ("value_0", "value_1")
    assert baseline.predict(["x"]).frame[["value_0", "value_1"]].iloc[0].tolist() == [1.5, 0.0]


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


def test_compare_refuses_sets_with_different_class_orders():
    labels = ["a", "b", "a", "b"]
    forward = _binary(classes=("a", "b"))
    reversed_order = _binary(classes=("b", "a"))

    with pytest.raises(ValueError, match="classes"):
        compare([forward, reversed_order], _labels(labels=labels), "auroc", n_permutations=5)


def test_compare_refuses_a_set_without_classes_next_to_one_with_classes():
    # classes=(1, 0) encodes label 1 as index 0; no classes keeps 1 as index 1.
    reversed_order = _binary(classes=(1, 0))
    implicit = _binary()

    with pytest.raises(ValueError, match="same classes"):
        compare([reversed_order, implicit], _labels(), "auroc", n_permutations=5)


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


def test_a_target_wins_over_a_metadata_column_of_the_same_name():
    records = pd.DataFrame(
        {"sample_id": ["s1", "s2"], "diagnosis": ["tumor", "benign"], "label": ["x", "y"]}
    )
    splits = pd.DataFrame({"sample_id": ["s1", "s2"], "split": ["train", "test"]})
    cohort = Cohort.from_frames(records, splits, targets={"label": "diagnosis"})

    targets = cohort.targets("test")

    assert targets.columns.tolist() == ["label", "patient_id"]
    assert targets["label"].tolist() == ["benign"]


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


# --- soma evaluate ----------------------------------------------------------------- #


def test_soma_evaluate_scores_the_tune_split_of_a_cohort_without_a_test_split(tmp_path):
    from soma.cli import main as soma_main

    pd.DataFrame({"sample_id": IDS, "label": [0, 1, 0, 1]}).to_csv(tmp_path / "dataset.csv", index=False)
    pd.DataFrame({"sample_id": IDS, "split": ["train", "train", "tune", "tune"]}).to_csv(
        tmp_path / "splits.csv", index=False
    )
    _binary(ids=["c", "d"], probs=(0.4, 0.7)).to_parquet(tmp_path / "predictions.parquet")
    out = tmp_path / "evaluation.json"

    soma_main(
        [
            "evaluate",
            str(tmp_path / "predictions.parquet"),
            "--cohort",
            str(tmp_path / "dataset.csv"),
            "--splits",
            str(tmp_path / "splits.csv"),
            "--split",
            "tune",
            "--metrics",
            "accuracy",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text())
    assert payload["num_samples"] == 2
    assert payload["metrics"]["accuracy"] == 1.0


# --- grid nouns -------------------------------------------------------------------- #


def test_integer_threshold_keys_are_class_indices_even_with_decimal_class_names():
    from soma.evaluation import PointPredictions

    frame = pd.DataFrame(
        {
            "sample_id": ["s"] * 4,
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "class": [0, 0, 1, 1],
            "score": [0.2, 0.5, 0.6, 0.8],
        }
    )
    predictions = PointPredictions(frame, classes=["1", "2"], sample_ids=["s"], match_distance_px=3.0)

    # An int key is an index (the shape sweep_threshold(per_class=True) returns); a str is a name.
    assert predictions.with_threshold({0: 0.3, 1: 0.7}).frame["score"].tolist() == [0.5, 0.8]
    assert predictions.with_threshold({"1": 0.3, "2": 0.7}).frame["score"].tolist() == [0.5, 0.8]
    with pytest.raises(ValueError, match="unknown class '3'"):
        predictions.with_threshold({"1": 0.3, "3": 0.7})


class _LabelMaps:
    """A minimal label-map target source."""

    def __init__(self, maps):
        self._maps = maps

    @property
    def sample_ids(self):
        return list(self._maps)

    def load(self, sample_id):
        return torch.as_tensor(self._maps[sample_id])


@pytest.mark.parametrize("undeclared_on", ["both", "prediction", "target"])
def test_label_map_scoring_refuses_labels_outside_the_classes(undeclared_on):
    from soma.evaluation import LabelMapPredictions

    declared = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    undeclared = np.array([[0, 2], [1, 2]], dtype=np.uint8)
    # Undeclared on both sides used to score Dice 1 by silently dropping label 2.
    prediction = declared if undeclared_on == "target" else undeclared
    target = declared if undeclared_on == "prediction" else undeclared
    predictions = LabelMapPredictions({"s": prediction}, classes=["a", "b"])
    with pytest.raises(ValueError, match=r"(?s)'s'.*\[2\]"):
        evaluate(predictions, _LabelMaps({"s": target}))


def test_label_map_scoring_skips_ignored_target_pixels_before_the_range_check():
    from soma.evaluation import LabelMapPredictions

    target = np.array([[0, 255], [1, 255]], dtype=np.uint8)
    prediction = np.array([[0, 7], [1, 9]], dtype=np.uint8)
    evaluation = evaluate(LabelMapPredictions({"s": prediction}, classes=["a", "b"]), _LabelMaps({"s": target}))
    assert evaluation.metrics["mean_dice"] == 1.0
