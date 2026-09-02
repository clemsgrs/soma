"""Tests for soma.dataset.Splits — loading user-provided splits from CSV."""

from pathlib import Path

import pandas as pd
import pytest

from soma.dataset import Dataset, FoldSplit, Splits


@pytest.fixture()
def dataset(tmp_path: Path) -> Dataset:
    """Dataset with 6 samples."""
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "image_path": [f"/slides/s{i}.svs" for i in range(1, 7)],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return Dataset(path)


@pytest.fixture()
def splits_csv(tmp_path: Path) -> Path:
    """Single fold: 4 train, 1 tune, 1 test."""
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0, 0, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "train", "train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    return path


def test_load_single_fold(splits_csv: Path, dataset: Dataset):
    splits = Splits(splits_csv, dataset)
    assert splits.num_folds == 1
    fold = splits.folds[0]
    assert isinstance(fold, FoldSplit)
    assert set(fold.train) == {"s1", "s2", "s3", "s4"}
    assert fold.tune == ("s5",)
    assert fold.tests == {"test": ("s6",)}


def test_multi_fold(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"] * 2,
            "split": [
                "train", "train", "train", "tune", "tune", "test",
                "train", "train", "tune", "tune", "test", "test",
            ],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)

    splits = Splits(path, dataset)
    assert splits.num_folds == 2
    assert set(splits.folds[0].train) == {"s1", "s2", "s3"}
    assert set(splits.folds[0].tune) == {"s4", "s5"}
    assert splits.folds[0].tests == {"test": ("s6",)}
    assert set(splits.folds[1].tests["test"]) == {"s5", "s6"}


def test_folds_sorted_by_index(tmp_path: Path, dataset: Dataset):
    """Fold indices don't have to be 0-based or contiguous — sorted by value."""
    df = pd.DataFrame(
        {
            "fold": [2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"] * 2,
            "split": ["train", "train", "train", "tune", "test", "test"] * 2,
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)

    splits = Splits(path, dataset)
    assert splits.num_folds == 2


def test_unknown_sample_id_raises(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["s1", "UNKNOWN"],
            "split": ["train", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="UNKNOWN"):
        Splits(path, dataset)


def test_invalid_split_name_raises(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["s1", "s2"],
            "split": ["train", "validation"],  # should be "tune"
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation"):
        Splits(path, dataset)


def test_missing_split_value_raises_clear_validation_error(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["s1", "s2"],
            "split": ["train", None],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid split name"):
        Splits(path, dataset)


def test_fold_without_test_split_raises_clear_validation_error(
    tmp_path: Path, dataset: Dataset
):
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "sample_id": ["s1", "s2", "s3"],
            "split": ["train", "train", "tune"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Fold 0 must contain at least one test split"):
        Splits(path, dataset)


def test_duplicate_sample_in_fold_raises(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "sample_id": ["s1", "s1", "s2"],
            "split": ["train", "test", "train"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="[Dd]uplicate.*s1"):
        Splits(path, dataset)


def test_missing_column_raises(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame({"fold": [0], "sample_id": ["s1"]})  # missing split
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="split"):
        Splits(path, dataset)


def test_fold_column_is_optional(tmp_path: Path, dataset: Dataset):
    """splits.csv without a fold column is treated as a single fold."""
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "train", "train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, dataset)
    assert splits.num_folds == 1
    assert set(splits.folds[0].train) == {"s1", "s2", "s3", "s4"}
    assert splits.folds[0].tests == {"test": ("s6",)}


def test_accepts_string_path(splits_csv: Path, dataset: Dataset):
    splits = Splits(str(splits_csv), dataset)
    assert splits.num_folds == 1


# ---------------------------------------------------------------------------
# tune_is_test (tune-as-test fallback when no test split is provided)
# ---------------------------------------------------------------------------


def test_tune_is_test_synthesizes_test_from_tune(tmp_path: Path, dataset: Dataset):
    """A train/tune-only fold reuses the tune split for test reporting."""
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "sample_id": ["s1", "s2", "s3", "s4"],
            "split": ["train", "train", "tune", "tune"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)

    splits = Splits(path, dataset, tune_is_test=True)
    fold = splits.folds[0]
    assert set(fold.train) == {"s1", "s2"}
    assert set(fold.tune) == {"s3", "s4"}
    assert fold.tests == {"test": ("s3", "s4")}
    assert fold.test_from_tune is True
    assert fold.test_split_names == ["test"]


def test_tune_is_test_with_test_only_split_no_synthesis(tmp_path: Path, dataset: Dataset):
    """A train/test-only fold (no tune) is used as-is; no test is synthesized."""
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "sample_id": ["s1", "s2", "s3"],
            "split": ["train", "train", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)

    splits = Splits(path, dataset, tune_is_test=True)
    fold = splits.folds[0]
    assert fold.tests == {"test": ("s3",)}
    assert fold.test_from_tune is False


def test_missing_test_without_flag_hints_at_flag(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "sample_id": ["s1", "s2", "s3"],
            "split": ["train", "train", "tune"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="tune_is_test"):
        Splits(path, dataset)


def test_tune_is_test_rejects_both_tune_and_test_split(tmp_path: Path, dataset: Dataset):
    """Providing both splits contradicts tune_is_test (a single held-out set)."""
    df = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "sample_id": ["s1", "s2", "s3"],
            "split": ["train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both a tune and a test split"):
        Splits(path, dataset, tune_is_test=True)


def test_tune_is_test_requires_tune_or_test_split(tmp_path: Path, dataset: Dataset):
    """A fold with neither test nor tune errors even with the flag set."""
    df = pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["s1", "s2"],
            "split": ["train", "train"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no test split and no tune split"):
        Splits(path, dataset, tune_is_test=True)


# ---------------------------------------------------------------------------
# validate_no_patient_leakage
# ---------------------------------------------------------------------------


@pytest.fixture()
def patient_dataset(tmp_path: Path) -> Dataset:
    """6 slides, 3 patients (2 slides each), alternating labels."""
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "image_path": [f"/slides/s{i}.svs" for i in range(1, 7)],
            "label": [0, 0, 1, 1, 0, 0],
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return Dataset(path)


def test_validate_no_patient_leakage_passes(tmp_path: Path, patient_dataset: Dataset):
    """Both slides for each patient are in the same split — no leakage."""
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "tune", "tune", "test", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, patient_dataset)
    splits.validate_no_patient_leakage(patient_dataset)  # should not raise


def test_validate_no_patient_leakage_detects_leakage(tmp_path: Path, patient_dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "test", "tune", "tune", "test", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, patient_dataset)
    with pytest.raises(ValueError, match="p1"):
        splits.validate_no_patient_leakage(patient_dataset)


def test_validate_no_patient_leakage_ignores_synthesized_test(
    tmp_path: Path, patient_dataset: Dataset
):
    """A tune-as-test fold must not be flagged as leakage (test mirrors tune)."""
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "tune", "tune", "tune", "tune"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, patient_dataset, tune_is_test=True)
    assert splits.folds[0].test_from_tune is True
    splits.validate_no_patient_leakage(patient_dataset)  # should not raise


def test_validate_no_patient_leakage_multi_fold(tmp_path: Path, patient_dataset: Dataset):
    """Leakage only in fold 1 is detected."""
    rows = []
    # fold 0: clean
    rows += [
        {"fold": 0, "sample_id": "s1", "split": "train"},
        {"fold": 0, "sample_id": "s2", "split": "train"},
        {"fold": 0, "sample_id": "s3", "split": "tune"},
        {"fold": 0, "sample_id": "s4", "split": "tune"},
        {"fold": 0, "sample_id": "s5", "split": "test"},
        {"fold": 0, "sample_id": "s6", "split": "test"},
    ]
    # fold 1: p2 is split across tune and test
    rows += [
        {"fold": 1, "sample_id": "s1", "split": "train"},
        {"fold": 1, "sample_id": "s2", "split": "train"},
        {"fold": 1, "sample_id": "s3", "split": "tune"},
        {"fold": 1, "sample_id": "s4", "split": "test"},
        {"fold": 1, "sample_id": "s5", "split": "test"},
        {"fold": 1, "sample_id": "s6", "split": "test"},
    ]
    path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    splits = Splits(path, patient_dataset)
    with pytest.raises(ValueError, match="p2"):
        splits.validate_no_patient_leakage(patient_dataset)


def test_validate_no_patient_leakage_no_patient_id_raises(tmp_path: Path, dataset: Dataset):
    """Dataset without patient_id column raises."""
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "train", "train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, dataset)
    with pytest.raises(ValueError, match="patient_id"):
        splits.validate_no_patient_leakage(dataset)


# ---------------------------------------------------------------------------
# Multiple named test sets
# ---------------------------------------------------------------------------


def test_multiple_test_splits_accepted(tmp_path: Path, dataset: Dataset):
    """Split names starting with 'test' are all accepted."""
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "train", "tune", "test", "test_external"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, dataset)
    fold = splits.folds[0]
    assert fold.tests == {"test": ("s5",), "test_external": ("s6",)}
    assert fold.test_split_names == ["test", "test_external"]


def test_multiple_test_splits_all_in_tests_dict(tmp_path: Path, dataset: Dataset):
    """Three test splits are all captured in the tests dict."""
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "tune", "test", "test_ext", "test_prosp"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, dataset)
    fold = splits.folds[0]
    assert set(fold.tests.keys()) == {"test", "test_ext", "test_prosp"}
    assert fold.tests["test"] == ("s4",)
    assert fold.tests["test_ext"] == ("s5",)
    assert fold.tests["test_prosp"] == ("s6",)


def test_invalid_non_test_name_still_rejected(tmp_path: Path, dataset: Dataset):
    """Names like 'validation' that don't start with 'test' are still invalid."""
    df = pd.DataFrame(
        {
            "fold": [0, 0],
            "sample_id": ["s1", "s2"],
            "split": ["train", "validation"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation"):
        Splits(path, dataset)


def test_patient_leakage_detected_across_named_test_splits(
    tmp_path: Path, patient_dataset: Dataset
):
    """Patient leakage is detected even when splits have custom names."""
    # p1 has s1 in train and s2 in test_external → leakage
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "test_external", "tune", "tune", "test", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    splits = Splits(path, patient_dataset)
    with pytest.raises(ValueError, match="p1"):
        splits.validate_no_patient_leakage(patient_dataset)


def test_blank_fold_cell_raises_listing_sample_ids(tmp_path: Path, dataset: Dataset):
    df = pd.DataFrame(
        {
            "fold": [0, 0, None, 0, 0, 0],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "train", "train", "train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    # Previously groupby("fold") dropped s3 silently.
    with pytest.raises(ValueError, match=r"Blank 'fold' value for 1 row\(s\).*\['s3'\]"):
        Splits(path, dataset)


def test_single_class_split_warns_naming_fold_and_split(
    tmp_path: Path, dataset: Dataset, caplog
):
    # labels: s1=0 s2=0 s3=1 s4=1 s5=0 s6=1 -> tune holds only class 0.
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "split": ["train", "tune", "train", "train", "tune", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with caplog.at_level("WARNING", logger="soma.dataset"):
        Splits(path, dataset)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Fold 0 split 'tune' has no samples for class(es) ['1']" in m for m in messages)
    assert not any("split 'train'" in m for m in messages)


def test_full_coverage_emits_no_warning(tmp_path: Path, dataset: Dataset, caplog):
    # Every split holds both classes (s1,s2,s5 = 0; s3,s4,s6 = 1).
    df = pd.DataFrame(
        {
            "fold": [0] * 6,
            "sample_id": ["s1", "s3", "s2", "s4", "s5", "s6"],
            "split": ["train", "train", "tune", "tune", "test", "test"],
        }
    )
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    with caplog.at_level("WARNING", logger="soma.dataset"):
        Splits(path, dataset)
    assert not [r for r in caplog.records if "no samples for class" in r.getMessage()]
