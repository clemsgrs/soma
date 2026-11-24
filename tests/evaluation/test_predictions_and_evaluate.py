"""Public-API pins for the evaluation verbs: ``Predictions``, ``evaluate``, ``compare``,
``predict`` (with its callbacks) and ``Cohort.targets``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

import soma
from soma.models.aggregators import aggregator_registry
from soma.data import Cohort, FoldSplit, SampleRecord, from_arrays
from soma.evaluation import Evaluation, Predictions, compare, evaluate, predict
from soma.evaluation.heatmaps import SaveAttention
from soma.models.heads.classification import BinaryClassificationHead
from soma.learn.bag_dataset import BagDataset
from soma.learn.collate import bag_collate_fn
from soma.data.geometry import compute_dense_geometry
from soma.learn import learner_for
from soma.learn.grid import GridBatch
from soma.models.decoders.base import Decoder
from soma.models.heads.detection import DetectionHead
from soma.models.model import DenseModel, MILModel

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


#: The counts every set-shape evaluation reports, when each of the four targets is predicted.
COUNTS = {"num_samples": 4, "num_predicted": 4, "coverage": 1.0}


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


def test_concat_stacks_the_rows_and_adds_no_column():
    combined = Predictions.concat([_binary(ids=["a", "b"], probs=(0.2, 0.9)), _binary(ids=["c"], probs=(0.4,))])

    assert combined.sample_ids == ["a", "b", "c"]
    assert list(combined.frame.columns) == ["sample_id", "prob_0", "prob_1", "predicted_label"]


# --- evaluate ---------------------------------------------------------------------- #


def test_evaluate_scores_the_joined_frames():
    evaluation = evaluate(_binary(), _labels(), metrics=["auroc", "accuracy"])

    assert isinstance(evaluation, Evaluation)
    # Positives (b 0.9, d 0.7) outrank both negatives (a 0.2, c 0.4).
    assert evaluation.metrics == {"auroc": 1.0, "accuracy": 1.0, **COUNTS}
    assert evaluation.num_samples == 4
    assert evaluation.task_family == "binary_classification"


def test_default_metrics_are_the_family_catalogue_defaults():
    evaluation = evaluate(_binary(), _labels())

    assert list(evaluation.metrics) == ["auroc", "balanced_accuracy", "auprc", "f1", *COUNTS]


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

    assert list(evaluation.metrics) == ["mean_pearson", *COUNTS]
    assert evaluation.metrics == evaluate(_gene_panel(), targets, metrics=["mean_pearson"]).metrics


def test_a_prediction_without_a_target_is_refused():
    with pytest.raises(ValueError, match="1 prediction id"):
        evaluate(_binary(), _labels(ids=["a", "b", "c", "x"]))


def test_a_target_without_a_prediction_is_refused_by_default_with_both_ways_forward():
    targets = pd.concat([_labels(), _labels(labels=(1,), ids=["e"])])

    with pytest.raises(ValueError) as refused:
        evaluate(_binary(), targets)

    message = str(refused.value)
    assert "1 target(s) have no prediction: ['e']" in message
    assert "remove them from the cohort or the splits" in message
    assert "unpredicted='exclude'" in message


def test_exclude_scores_the_predicted_samples_only_and_counts_the_rest():
    # e is a positive the model never saw; a, b, c, d score as before.
    targets = pd.concat([_labels(), _labels(labels=(1,), ids=["e"])])

    evaluation = evaluate(_binary(), targets, metrics=["auroc"], unpredicted="exclude")

    assert evaluation.metrics == {"auroc": 1.0, "num_samples": 5, "num_predicted": 4, "coverage": 0.8}
    assert evaluation.num_samples == 5


def test_an_unknown_unpredicted_policy_is_refused():
    with pytest.raises(ValueError, match=r"unpredicted must be one of \['error', 'exclude'\], got 'fill'"):
        evaluate(_binary(), _labels(), unpredicted="fill")


def test_exclude_still_refuses_a_prediction_without_a_target():
    with pytest.raises(ValueError, match="1 prediction id"):
        evaluate(_binary(), _labels(ids=["a", "b", "c", "x"]), unpredicted="exclude")


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
            "tests.evaluation.test_predictions_and_evaluate.positives_predicted",
        ],
    )

    assert evaluation.metrics == {"<lambda>": 1.0, "positives_predicted": 2.0, **COUNTS}


def test_raw_labels_map_through_the_declared_classes():
    targets = _labels(labels=("benign", "tumor", "benign", "tumor"))

    assert evaluate(_binary(classes=["benign", "tumor"]), targets, metrics=["accuracy"]).metrics == {
        "accuracy": 1.0,
        **COUNTS,
    }
    with pytest.raises(ValueError, match="not class indices"):
        evaluate(_binary(), targets)
    with pytest.raises(ValueError, match="not among"):
        evaluate(_binary(classes=["normal", "tumor"]), targets)


def _filled(flags) -> Predictions:
    """Predictions with the ``is_placeholder`` column an earlier soma version wrote: true
    on the rows it filled instead of the model."""
    frame = _binary().frame.assign(is_placeholder=list(flags))
    columns = ["prob_0", "prob_1", "predicted_label"]
    return Predictions(frame, task_family="binary_classification", prediction_columns=columns)


@pytest.mark.parametrize("unpredicted", ["error", "exclude"])
def test_rows_an_earlier_version_filled_are_refused(tmp_path, unpredicted):
    path = _filled([False, False, False, True]).to_parquet(tmp_path / "p.parquet")
    filled = Predictions.from_parquet(path)

    with pytest.raises(ValueError) as refused:
        evaluate(filled, _labels(), metrics=["auroc"], unpredicted=unpredicted)

    message = str(refused.value)
    assert "1 row(s) of the predictions are flagged is_placeholder: ['d']" in message
    assert "written by an earlier soma version" in message
    assert "drop those rows" in message
    with pytest.raises(ValueError, match="written by an earlier soma version"):
        compare([filled, _binary()], _labels(), "auroc", n_permutations=5)


def test_an_is_placeholder_column_without_a_flagged_row_is_scored():
    evaluation = evaluate(_filled([False] * 4), _labels(), metrics=["auroc"])

    assert evaluation.metrics == {"auroc": 1.0, **COUNTS}


def test_to_json_holds_metrics_and_subgroups(tmp_path):
    targets = _labels().assign(site=["A", "A", "B", "B"])
    evaluation = evaluate(_binary(), targets, metrics=["accuracy"], subgroups=["site"])

    payload = json.loads(evaluation.to_json(tmp_path / "e.json"))

    assert payload == json.loads((tmp_path / "e.json").read_text())
    assert payload["metrics"] == {"accuracy": 1.0, **COUNTS}
    assert payload["num_samples"] == 4
    assert payload["subgroups"]["metrics"]["site"]["A"] == {"accuracy": 1.0, "n": 2}


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


def test_compare_refuses_a_set_that_misses_a_target_and_says_to_filter_to_the_common_ids():
    everything = _binary()
    without_d = _binary(ids=["a", "b", "c"], probs=(0.2, 0.9, 0.4))

    with pytest.raises(ValueError) as refused:
        compare([everything, without_d], _labels(), "auroc", n_permutations=5)

    message = str(refused.value)
    assert "1 target(s) have no prediction: ['d']" in message
    assert "filter the targets and the sets to the ids every set predicts" in message
    assert "unpredicted" not in message


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



class _MeanDecoder(Decoder):
    """A decoder with no parameter: the channel mean as one class's logits."""

    def forward(self, X):
        return X.mean(dim=1, keepdim=True)

    @property
    def num_classes(self) -> int:
        return 1


def _weightless_grid_model() -> tuple[DenseModel, list[GridBatch]]:
    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = DetectionHead(num_classes=1, geometry=geometry, spacing_um=1.0, match_distance=5.0, sigma=2.0)
    features = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 384.0
    batch = GridBatch(features=features, pixels=None, targets={}, sample_ids=("a", "b"))
    return DenseModel(_MeanDecoder(), head), [batch]


@pytest.mark.parametrize(("cuda", "expected"), [(True, "cuda"), (False, "cpu")])
def test_predict_runs_a_model_without_parameters_on_the_gpu_when_there_is_one(monkeypatch, cuda, expected):
    model, loader = _weightless_grid_model()
    learner = learner_for(model)
    moved_to: list[torch.device] = []
    transferred_to: list[torch.device] = []
    # Record where predict sends the learner and its batches, without moving anything.
    learner.to = lambda device: moved_to.append(device) or learner
    learner.transfer_batch = lambda batch, device: transferred_to.append(device) or batch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    predictions = predict(learner, loader)

    assert moved_to == transferred_to == [torch.device(expected)]
    assert sorted(predictions.frame["sample_id"].unique()) == ["a", "b"]


def test_predict_runs_on_the_device_it_is_given():
    model, loader = _weightless_grid_model()
    devices: list[str] = []

    class Spy:
        def on_predict_batch(self, batch, output) -> None:
            devices.append(output.raw.logits.device.type)

    predict(model, loader, callbacks=[Spy()], device="cpu")
    assert devices == ["cpu"]


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


def test_grid_predictions_take_no_unpredicted_policy():
    from soma.evaluation import PointPredictions

    frame = pd.DataFrame({"sample_id": ["s"], "x": [1.0], "y": [2.0], "class": [0], "score": [0.9]})
    predictions = PointPredictions(frame, classes=["cell"], sample_ids=["s"], match_distance_px=3.0)

    with pytest.raises(ValueError, match="unpredicted applies to set-shape Predictions"):
        evaluate(predictions, object(), unpredicted="exclude")


def test_point_predictions_with_numpy_classes_round_trip(tmp_path):
    from soma.evaluation import PointPredictions

    frame = pd.DataFrame({"sample_id": ["s"], "x": [1.0], "y": [2.0], "class": [1], "score": [0.9]})
    # NumPy class names (np.unique(labels)) are written as plain JSON scalars.
    PointPredictions(frame, classes=np.array([0, 1]), sample_ids=["s"], match_distance_px=3.0).to_parquet(
        tmp_path / "points.parquet"
    )
    assert PointPredictions.from_parquet(tmp_path / "points.parquet").classes == (0, 1)


def test_label_map_predictions_with_numpy_classes_round_trip(tmp_path):
    from soma.evaluation import LabelMapPredictions

    maps = {"s": np.array([[0, 1], [1, 0]], dtype=np.uint8)}
    LabelMapPredictions(maps, classes=np.array([0, 1])).save(tmp_path / "maps")
    assert LabelMapPredictions.load(tmp_path / "maps").classes == (0, 1)


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
