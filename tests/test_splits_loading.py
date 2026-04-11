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
    assert fold.test == ("s6",)


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
    assert splits.folds[0].test == ("s6",)
    assert set(splits.folds[1].test) == {"s5", "s6"}


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


def test_accepts_string_path(splits_csv: Path, dataset: Dataset):
    splits = Splits(str(splits_csv), dataset)
    assert splits.num_folds == 1


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
