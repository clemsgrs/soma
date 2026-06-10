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
    assert rec.mask_path is None
    assert rec.metadata == {}


def test_sample_ids(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert sorted(ds.sample_ids) == ["s1", "s2", "s3", "s4"]


def test_categorical_label_map_string_labels(dataset_csv: Path):
    from soma.tasks.classification import categorical_label_map

    ds = Dataset(dataset_csv)
    # Sorted unique: "normal" < "tumor"
    assert categorical_label_map(ds) == {"normal": 0, "tumor": 1}


def test_categorical_label_map_integer_labels(tmp_path: Path):
    from soma.tasks.classification import categorical_label_map

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
    assert categorical_label_map(ds) == {0: 0, 1: 1, 2: 2}


def test_categorical_label_map_count(dataset_csv: Path):
    from soma.tasks.classification import categorical_label_map

    ds = Dataset(dataset_csv)
    assert len(categorical_label_map(ds)) == 2


def test_optional_mask_path(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": ["/slides/s1.svs"],
            "label": [0],
            "mask_path": ["/masks/s1.tif"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    ds = Dataset(path)
    assert ds.samples["s1"].mask_path == Path("/masks/s1.tif")


def test_legacy_tissue_mask_path_is_rejected(tmp_path: Path):
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
    with pytest.raises(ValueError, match="mask_path"):
        Dataset(path)


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


# ---------------------------------------------------------------------------
# patient_id support
# ---------------------------------------------------------------------------


@pytest.fixture()
def patient_dataset_csv(tmp_path: Path) -> Path:
    """Dataset CSV with patient_id column: 4 slides, 2 patients."""
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "image_path": [f"/slides/s{i}.svs" for i in range(1, 5)],
            "label": ["tumor", "tumor", "normal", "normal"],
            "patient_id": ["p1", "p1", "p2", "p2"],
        }
    )
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return path


def test_patient_id_loaded(patient_dataset_csv: Path):
    ds = Dataset(patient_dataset_csv)
    assert ds.samples["s1"].patient_id == "p1"
    assert ds.samples["s3"].patient_id == "p2"


def test_patient_id_none_when_column_absent(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert all(r.patient_id is None for r in ds.samples.values())


def test_has_patient_ids_true(patient_dataset_csv: Path):
    ds = Dataset(patient_dataset_csv)
    assert ds.has_patient_ids is True


def test_has_patient_ids_false(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    assert ds.has_patient_ids is False


def test_patient_id_not_in_metadata(patient_dataset_csv: Path):
    ds = Dataset(patient_dataset_csv)
    assert "patient_id" not in ds.samples["s1"].metadata


def test_patient_groups(patient_dataset_csv: Path):
    ds = Dataset(patient_dataset_csv)
    groups = ds.patient_groups
    assert set(groups.keys()) == {"p1", "p2"}
    assert {r.sample_id for r in groups["p1"]} == {"s1", "s2"}
    assert {r.sample_id for r in groups["p2"]} == {"s3", "s4"}


def test_patient_groups_raises_when_no_patient_id(dataset_csv: Path):
    ds = Dataset(dataset_csv)
    with pytest.raises(ValueError, match="patient_id"):
        _ = ds.patient_groups


def test_patient_record_map_returns_representative_records(patient_dataset_csv: Path):
    ds = Dataset(patient_dataset_csv)
    record_map = ds.patient_record_map
    assert set(record_map) == {"p1", "p2"}
    assert record_map["p1"].label == "tumor"
    assert record_map["p2"].label == "normal"


def test_patient_record_map_inconsistent_raises(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "image_path": ["/a.svs", "/b.svs"],
            "label": ["tumor", "normal"],
            "patient_id": ["p1", "p1"],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    ds = Dataset(path)
    with pytest.raises(ValueError, match="inconsistent"):
        _ = ds.patient_record_map


@pytest.mark.parametrize("bad_id", ["../escaped", "a/b", "/abs", "..", ".", r"a\b"])
def test_unsafe_sample_id_rejected(tmp_path: Path, bad_id: str):
    # sample_id becomes a cache filename (<id>.pt) across every cache kind, so a
    # path-traversal id must be rejected at ingestion (the single chokepoint).
    df = pd.DataFrame(
        {
            "sample_id": ["s1", bad_id],
            "image_path": ["/a.svs", "/b.svs"],
            "label": ["tumor", "normal"],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unsafe sample_id"):
        Dataset(path)


def test_unsafe_patient_id_rejected(tmp_path: Path):
    # patient_id is also written as <patient_id>.pt by the patient cache.
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "image_path": ["/a.svs", "/b.svs"],
            "label": ["tumor", "normal"],
            "patient_id": ["p1", "../p2"],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unsafe patient_id"):
        Dataset(path)


def test_safe_sample_ids_with_dots_allowed(tmp_path: Path):
    # Dots inside the name (not '.'/'..') are legitimate and must pass.
    df = pd.DataFrame(
        {
            "sample_id": ["case.1.2", "TCGA-AB-1234"],
            "image_path": ["/a.svs", "/b.svs"],
            "label": ["tumor", "normal"],
        }
    )
    path = tmp_path / "ok.csv"
    df.to_csv(path, index=False)
    assert sorted(Dataset(path).sample_ids) == ["TCGA-AB-1234", "case.1.2"]
