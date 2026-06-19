"""Unit tests for the kaiko-ai/eva reproduction recipe (soma.benchmarks.eva)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from soma.benchmarks import eva


def _write_splits(tmp_path, n_train: int, n_test: int = 3):
    rows = [{"sample_id": f"tr{i}", "split": "train"} for i in range(n_train)]
    rows += [{"sample_id": f"te{i}", "split": "test"} for i in range(n_test)]
    path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "n_train, expected_epochs",
    [
        (268, 6250),  # bach
        (1132, 2500),  # breakhis
        (100_000, 32),  # crc
        (262_144, 13),  # patch_camelyon
    ],
)
def test_epochs_for_train_size_matches_eva_step_budget(n_train, expected_epochs):
    assert eva.epochs_for_train_size(n_train) == expected_epochs
    # equivalently: ceil(max_steps / ceil(n / batch))
    spe = math.ceil(n_train / eva.HEAD_BATCH_SIZE)
    assert eva.epochs_for_train_size(n_train) == max(1, math.ceil(eva.MAX_STEPS / spe))


def test_epochs_for_train_size_is_at_least_one_for_huge_datasets():
    assert eva.epochs_for_train_size(10_000_000) == 1


def test_epochs_for_train_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        eva.epochs_for_train_size(0)


def test_expected_balanced_accuracy_lookup():
    assert eva.expected_balanced_accuracy("uni2", "bach") == 0.915
    assert eva.expected_balanced_accuracy("virchow2", "patch_camelyon", split="test") == 0.938
    # bach has no test split -> no test column
    assert eva.expected_balanced_accuracy("uni2", "bach", split="test") is None
    # unknown encoder is not tabulated
    assert eva.expected_balanced_accuracy("not_an_encoder", "bach") is None


def test_build_config_encodes_eva_protocol(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = eva.build_config(
        dataset="bach",
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        seed=3,
    )
    assert config.dataset_type == "tile"
    assert config.encoder.name == "uni2"
    assert config.encoder.output_variant is None  # uni2 uses the slide2vec default
    assert config.task.name == "multiclass_classification"
    assert "balanced_accuracy" in config.evaluation.metrics

    training = config.training
    assert training.learning_rate == eva.LEARNING_RATE
    assert training.weight_decay == eva.WEIGHT_DECAY  # 0.01, not 0.0
    assert training.optimizer == "adamw"
    assert training.scheduler == "none"
    assert training.batch_size == eva.HEAD_BATCH_SIZE
    assert training.monitor == "balanced_accuracy"
    assert training.monitor_mode == "max"
    assert training.seed == 3
    assert training.tune_is_test is True  # bach reports on the validation split
    assert training.epochs == 6250  # computed from n_train=268
    assert training.patience == 1250  # eva's per-dataset value


def test_build_config_pins_virchow2_cls_variant(tmp_path):
    splits = _write_splits(tmp_path, n_train=512)
    config = eva.build_config(
        dataset="patch_camelyon",
        encoder="virchow2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )
    # eva's paige_virchow2 is CLS-only (1280-d); the 2560-d default would not match.
    assert config.encoder.output_variant == "cls"
    assert config.training.tune_is_test is False  # pcam has a real val + test split


def test_build_config_honours_overrides(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = eva.build_config(
        dataset="bach",
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        epochs=1,
        patience=1,
        encoder_batch_size=8,
    )
    assert config.training.epochs == 1
    assert config.training.patience == 1
    assert config.encoder.batch_size == 8


def test_build_config_rejects_unknown_dataset_and_encoder(tmp_path):
    splits = _write_splits(tmp_path, n_train=10)
    common = dict(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="Unknown EVA dataset"):
        eva.build_config(dataset="nope", encoder="uni2", **common)
    with pytest.raises(ValueError, match="Unknown EVA encoder"):
        eva.build_config(dataset="bach", encoder="nope", **common)


def test_expected_values_cover_supported_grid():
    # Every supported (encoder, dataset) should have a published val value so the
    # reproduce script can always print a comparison.
    for encoder in eva.ENCODERS:
        for dataset in eva.DATASETS:
            assert eva.expected_balanced_accuracy(encoder, dataset) is not None, (
                f"missing expected value for {encoder}/{dataset}"
            )


def test_expected_values_read_from_bundled_leaderboard():
    # The leaderboard snapshot ships as package data and is read via the encoder's
    # eva backbone key (not the soma encoder name).
    leaderboard = eva._leaderboard()
    assert "mahmood_uni2_h" in leaderboard.index  # uni2's eva_key
    assert eva.expected_balanced_accuracy("uni2", "crc") == pytest.approx(
        leaderboard.loc["mahmood_uni2_h", "crc"]
    )


def test_reported_splits_tune_is_test_dataset():
    splits = eva.reported_splits("bach")
    assert splits == [eva.ReportedSplit(label="val", soma_split="test", eva_column="bach")]


def test_reported_splits_patch_camelyon_has_val_and_test():
    splits = eva.reported_splits("patch_camelyon")
    assert splits == [
        eva.ReportedSplit(label="val", soma_split="tune", eva_column="patch_camelyon"),
        eva.ReportedSplit(label="test", soma_split="test", eva_column="patch_camelyon/test"),
    ]


def test_result_summary_merges_tune_metrics():
    result = SimpleNamespace(
        summary={"test/balanced_accuracy": 0.9},
        fold_results=[
            SimpleNamespace(tune_report=SimpleNamespace(metrics={"balanced_accuracy": 0.8}))
        ],
    )
    summary = eva.result_summary(result)
    assert summary["test/balanced_accuracy"] == 0.9
    assert summary["tune/balanced_accuracy"] == 0.8


def test_balanced_accuracy_from_summary_prefers_mean_and_handles_missing():
    assert eva.balanced_accuracy_from_summary({"test/balanced_accuracy": 0.9}, "test") == 0.9
    # the aggregated *_mean key wins when both are present
    assert (
        eva.balanced_accuracy_from_summary(
            {"test/balanced_accuracy_mean": 0.7, "test/balanced_accuracy": 0.9}, "test"
        )
        == 0.7
    )
    assert eva.balanced_accuracy_from_summary({"tune/balanced_accuracy": 0.6}, "tune") == 0.6
    assert eva.balanced_accuracy_from_summary({}, "test") is None
