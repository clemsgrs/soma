"""Cohort: records + folds, validated on construction (design §4.1)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from soma.data import Cohort, FoldSplit, SampleRecord


def _records_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "patient_id": [f"p{i // 2}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "site": ["a", "b"] * (n // 2),
        }
    )


def _splits_df(n: int = 6) -> pd.DataFrame:
    roles = ["train", "train", "tune", "tune", "test", "test"]
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "split": roles[:n]})


def test_from_frames_builds_slim_records_with_targets_and_metadata() -> None:
    cohort = Cohort.from_frames(_records_df(), _splits_df(), targets=["label"])

    record = cohort.record("s0")
    assert record == SampleRecord(
        sample_id="s0", targets={"label": 0}, patient_id="p0", metadata={"site": "a"}
    )
    assert cohort.sample_ids == [f"s{i}" for i in range(6)]
    assert cohort.num_folds == 1
    assert cohort.folds[0] == FoldSplit(
        train=("s0", "s1"), tune=("s2", "s3"), tests={"test": ("s4", "s5")}
    )
    assert cohort.unit == "patient_id"
    assert not hasattr(record, "image_path")


def test_targets_mapping_renames_csv_columns_to_target_keys() -> None:
    frame = _records_df().rename(columns={"label": "diagnosis"})
    cohort = Cohort.from_frames(frame, _splits_df(), targets={"label": "diagnosis"})
    assert cohort.record("s1").targets == {"label": 1}
    assert "diagnosis" not in cohort.record("s1").metadata


def test_missing_target_column_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="event"):
        Cohort.from_frames(_records_df(), _splits_df(), targets=["label", "event"])


def test_blank_target_value_fails_at_construction() -> None:
    frame = _records_df()
    frame.loc[2, "label"] = np.nan
    with pytest.raises(ValueError, match="s2"):
        Cohort.from_frames(frame, _splits_df(), targets=["label"])


def test_target_dtype_is_validated_at_construction() -> None:
    frame = _records_df()
    frame["time"] = [1.0, 2.0, "soon", 4.0, 5.0, 6.0]
    with pytest.raises(ValueError, match="time.*s2"):
        Cohort.from_frames(frame, _splits_df(), targets=["label", "time"], dtypes={"time": float})


def test_target_key_insertion_order_does_not_split_the_cohort() -> None:
    """Heads read targets by name, so two records declaring the same keys in a
    different dict order carry the same targets."""
    records = [
        SampleRecord("a", {"time": 2.0, "event": 1}),
        SampleRecord("b", {"event": 0, "time": 3.0}),
        SampleRecord("c", {"time": 1.0, "event": 1}),
    ]
    folds = [FoldSplit(train=("a",), tune=("b",), tests={"test": ("c",)})]
    cohort = Cohort(records, folds)
    assert set(cohort.target_keys) == {"time", "event"}
    assert cohort.record("b").targets == {"event": 0, "time": 3.0}
    with pytest.raises(ValueError, match="declares targets"):
        Cohort([*records, SampleRecord("d", {"time": 1.0})], folds)


def test_single_csv_with_split_column_is_accepted() -> None:
    frame = _records_df()
    frame["split"] = _splits_df()["split"]
    cohort = Cohort.from_frames(frame, targets=["label"])
    assert cohort.folds[0].tests == {"test": ("s4", "s5")}
    assert "split" not in cohort.record("s0").metadata


def test_fold_column_builds_one_fold_split_per_fold() -> None:
    splits = pd.concat(
        [_splits_df().assign(fold=0), _splits_df().assign(fold=1).iloc[::-1]]
    )
    cohort = Cohort.from_frames(_records_df(), splits, targets=["label"])
    assert cohort.num_folds == 2
    assert cohort.folds[1].train == ("s1", "s0")


def test_unknown_split_sample_id_is_rejected() -> None:
    splits = _splits_df()
    splits.loc[0, "sample_id"] = "ghost"
    with pytest.raises(ValueError, match="ghost"):
        Cohort.from_frames(_records_df(), splits, targets=["label"])


def test_invalid_split_name_is_rejected() -> None:
    splits = _splits_df()
    splits.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="validation"):
        Cohort.from_frames(_records_df(), splits, targets=["label"])


def test_fold_without_test_split_is_rejected() -> None:
    splits = _splits_df().replace({"test": "tune"})
    with pytest.raises(ValueError, match="test"):
        Cohort.from_frames(_records_df(), splits, targets=["label"])


def test_unsafe_sample_id_is_rejected() -> None:
    frame = _records_df()
    frame.loc[0, "sample_id"] = "../s0"
    splits = _splits_df()
    splits.loc[0, "sample_id"] = "../s0"
    with pytest.raises(ValueError, match="Unsafe"):
        Cohort.from_frames(frame, splits, targets=["label"])


def test_duplicate_sample_id_is_rejected() -> None:
    frame = _records_df()
    frame.loc[1, "sample_id"] = "s0"
    with pytest.raises(ValueError, match="Duplicate"):
        Cohort.from_frames(frame, _splits_df(), targets=["label"])


def test_unit_crossing_splits_within_a_fold_is_leakage() -> None:
    splits = _splits_df()
    splits.loc[1, "split"] = "test"  # p0 = {s0 train, s1 test}
    with pytest.raises(ValueError, match="p0"):
        Cohort.from_frames(_records_df(), splits, targets=["label"])


def test_sample_unit_skips_patient_leakage_check() -> None:
    splits = _splits_df()
    splits.loc[1, "split"] = "test"
    cohort = Cohort.from_frames(_records_df(), splits, targets=["label"], unit="sample_id")
    assert cohort.unit == "sample_id"


def test_patient_unit_requires_patient_ids() -> None:
    frame = _records_df().drop(columns=["patient_id"])
    with pytest.raises(ValueError, match="patient_id"):
        Cohort.from_frames(frame, _splits_df(), targets=["label"], unit="patient_id")
    assert Cohort.from_frames(frame, _splits_df(), targets=["label"]).unit == "sample_id"


def test_class_coverage_warning_names_fold_and_split(caplog: pytest.LogCaptureFixture) -> None:
    frame = _records_df()
    frame["label"] = [0, 0, 1, 1, 0, 1]
    with caplog.at_level(logging.WARNING, logger="soma.data.cohort"):
        Cohort.from_frames(frame, _splits_df(), targets=["label"], unit="sample_id")
    assert "Fold 0 split 'train'" in caplog.text


def test_with_test_from_tune_mirrors_tune_into_test() -> None:
    splits = _splits_df().replace({"test": "train"})
    cohort = Cohort.from_frames(
        _records_df(), splits, targets=["label"], allow_missing_test=True
    )
    assert cohort.folds[0].tests == {}
    derived = cohort.with_test_from_tune()
    assert derived.folds[0].tests == {"test": ("s2", "s3")}
    assert derived.folds[0].test_from_tune is True
    assert cohort.folds[0].tests == {}  # immutable


def test_with_test_from_tune_refuses_a_fold_that_already_has_a_test_split() -> None:
    cohort = Cohort.from_frames(_records_df(), _splits_df(), targets=["label"])
    with pytest.raises(ValueError, match="both"):
        cohort.with_test_from_tune()


def test_test_from_tune_flag_requires_a_true_mirror_of_tune() -> None:
    records = [SampleRecord(s, {"label": i % 2}) for i, s in enumerate("abc")]
    mirrored = FoldSplit(train=("a",), tune=("b",), tests={"test": ("b",)}, test_from_tune=True)
    assert Cohort(records, [mirrored]).folds[0].tests == {"test": ("b",)}
    for tests in ({"test": ("a",)}, {"test": ("b",), "test_ext": ("c",)}, {"test": ()}):
        with pytest.raises(ValueError, match="test_from_tune=True requires"):
            Cohort(
                records,
                [FoldSplit(train=("a",), tune=("b",), tests=tests, test_from_tune=True)],
            )


def test_collapse_to_patient_unit_keeps_representative_targets() -> None:
    frame = _records_df()
    frame["label"] = [0, 0, 1, 1, 0, 0]
    cohort = Cohort.from_frames(frame, _splits_df(), targets=["label"])
    patients = cohort.collapse("patient_id")
    assert patients.sample_ids == ["p0", "p1", "p2"]
    assert patients.record("p1").targets == {"label": 1}
    assert patients.record("p1").patient_id == "p1"
    assert patients.folds[0] == FoldSplit(train=("p0",), tune=("p1",), tests={"test": ("p2",)})
    assert patients.unit == "sample_id"


def test_collapse_rejects_members_with_disagreeing_targets() -> None:
    frame = _records_df()
    frame["label"] = [0, 1, 0, 0, 1, 1]
    cohort = Cohort.from_frames(frame, _splits_df(), targets=["label"])
    with pytest.raises(ValueError, match="p0"):
        cohort.collapse("patient_id")


def test_array_targets_are_compared_elementwise_on_collapse() -> None:
    frame = _records_df().drop(columns=["label"])
    frame["expression"] = [np.array([float(i // 2), 1.0]) for i in range(6)]
    cohort = Cohort.from_frames(frame, _splits_df(), targets=["expression"])
    collapsed = cohort.collapse("patient_id")
    np.testing.assert_array_equal(collapsed.record("p2").targets["expression"], [2.0, 1.0])


def test_from_csv_reads_expression_sidecar_into_targets(tmp_path) -> None:
    frame = _records_df().drop(columns=["label"])
    frame["target_index"] = [5, 4, 3, 2, 1, 0]
    frame.to_csv(tmp_path / "dataset.csv", index=False)
    _splits_df().to_csv(tmp_path / "splits.csv", index=False)
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    np.save(tmp_path / "targets.npy", matrix)
    (tmp_path / "genes.json").write_text('["g1", "g2"]')

    cohort = Cohort.from_csv(
        tmp_path / "dataset.csv", tmp_path / "splits.csv", targets=["expression"]
    )
    np.testing.assert_array_equal(cohort.record("s0").targets["expression"], matrix[5])
    assert cohort.record("s0").metadata == {"site": "a"}
    assert cohort.target_names["expression"] == ["g1", "g2"]


def test_list_constructor_validates_like_from_frames() -> None:
    records = [SampleRecord(s, {"label": i % 2}) for i, s in enumerate("abc")]
    folds = [FoldSplit(train=("a",), tune=("b",), tests={"test": ("c",)})]
    cohort = Cohort(records, folds)
    assert cohort.sample_ids == ["a", "b", "c"]
    with pytest.raises(ValueError, match="ghost"):
        Cohort(records, [FoldSplit(train=("ghost",), tune=(), tests={"test": ("a",)})])


def test_integer_ids_and_targets_keep_their_types_next_to_float_columns() -> None:
    # A row-wise walk over a mixed table would upcast every int to float, turning
    # sample_id 1 into "1.0" (unknown to the splits) and label 1 into 1.0.
    records_df = pd.DataFrame(
        {"sample_id": [1, 2], "label": [0, 1], "value": [1.25, 2.5], "n_tiles": [10, 20]}
    )
    splits_df = pd.DataFrame({"sample_id": [1, 2], "split": ["train", "test"]})

    cohort = Cohort.from_frames(records_df, splits_df, targets=["label", "value"])

    assert cohort.sample_ids == ["1", "2"]
    record = cohort.record("2")
    assert record.targets == {"label": 1, "value": 2.5}
    assert isinstance(record.targets["label"], int)
    assert record.metadata == {"n_tiles": 20}
    assert isinstance(record.metadata["n_tiles"], int)


def test_folds_cannot_be_mutated_after_validation() -> None:
    records = [SampleRecord(s, {"label": i % 2}, patient_id=f"p{i}") for i, s in enumerate("abc")]
    tests = {"test": ("c",)}
    cohort = Cohort(records, [FoldSplit(train=("a",), tune=("b",), tests=tests)])
    tests["test"] = ("a",)  # the caller's own mapping is not the cohort's
    assert cohort.folds[0].tests["test"] == ("c",)
    with pytest.raises(TypeError):
        cohort.folds[0].tests["test"] = ("a",)  # type: ignore[index]
