"""Tests for soma.dataset — Dataset loading from dataset.csv."""

from pathlib import Path

import pandas as pd
import pytest

from soma.dataset import Dataset, SampleRecord


@pytest.fixture()
def dataset_csv(tmp_path: Path) -> Path:
    """Create a minimal dataset.csv."""
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "image_path": ["/slides/s1.svs", "/slides/s2.svs", "/slides/s3.svs", "/slides/s4.svs"],
            "label": ["tumor", "normal", "tumor", "normal"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return path


def test_load_dataset(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert len(ds.samples) == 4
    assert isinstance(ds.samples["s1"], SampleRecord)


def test_sample_record_fields(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    rec = ds.samples["s1"]
    assert rec.sample_id == "s1"
    assert rec.image_path == Path("/slides/s1.svs")
    assert rec.label == "tumor"
    assert rec.tissue_mask_path is None
    assert rec.metadata == {}


def test_sample_ids(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert sorted(ds.sample_ids) == ["s1", "s2", "s3", "s4"]


def test_label_map_string_labels(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    # Sorted unique: "normal" < "tumor"
    assert ds.label_map == {"normal": 0, "tumor": 1}


def test_label_map_integer_labels(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "image_path": ["/a.svs", "/b.svs", "/c.svs"],
            "label": [2, 0, 1],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    ds = Dataset(path)
    assert ds.label_map == {0: 0, 1: 1, 2: 2}


def test_num_classes(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert ds.num_classes == 2


def test_optional_tissue_mask_path(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": ["/slides/s1.svs"],
            "label": [0],
            "tissue_mask_path": ["/masks/s1.tif"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    ds = Dataset(path)
    assert ds.samples["s1"].tissue_mask_path == Path("/masks/s1.tif")


def test_extra_columns_go_to_metadata(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": ["/slides/s1.svs"],
            "label": [0],
            "site": ["hospital_A"],
            "stain": ["H&E"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    ds = Dataset(path)
    assert ds.samples["s1"].metadata == {"site": "hospital_A", "stain": "H&E"}


def test_missing_sample_id_column_raises(tmp_path: Path):
    df = pd.DataFrame({"image_path": ["/a.svs"], "label": [0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sample_id"):
        Dataset(path)


def test_missing_image_path_column_raises(tmp_path: Path):
    df = pd.DataFrame({"sample_id": ["s1"], "label": [0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="image_path"):
        Dataset(path)


def test_missing_label_column_raises(tmp_path: Path):
    df = pd.DataFrame({"sample_id": ["s1"], "image_path": ["/a.svs"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label"):
        Dataset(path)


def test_duplicate_sample_ids_raises(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "image_path": ["/a.svs", "/b.svs"],
            "label": [0, 1],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        Dataset(path)


def test_accepts_path_or_string(dataset_csv: Path):
    ds = Dataset(str(dataset_csv))
    assert len(ds.samples) == 4
