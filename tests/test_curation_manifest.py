"""Tests for the unified Manifest schema, shared writer, and Protocol-typed curators."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from soma.curation import (
    CuratedManifest,
    Curator,
    curate_beetle_slide_manifest,
    curate_eva_patch_dataset,
    curate_ocelot_detection,
    write_manifest,
)
from soma.dataset import Dataset, DetectionManifest, SegmentationManifest, load_manifest


# --------------------------------------------------------------------------- helpers


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_mhist_raw(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    annotations = pd.DataFrame(
        {
            "Image Name": [
                "ssa_train_0.png",
                "ssa_train_1.png",
                "hp_train_0.png",
                "hp_train_1.png",
                "ssa_test.png",
                "hp_test.png",
            ],
            "Majority Vote Label": ["SSA", "SSA", "HP", "HP", "SSA", "HP"],
            "Partition": ["train", "train", "train", "train", "test", "test"],
        }
    )
    annotations.to_csv(root / "annotations.csv", index=False)
    for image_name in annotations["Image Name"]:
        _touch(root / "images" / image_name)
    return root


def _make_ocelot_raw(root: Path) -> Path:
    for split, stem, points in [
        ("train", "001", [(42, 516, 2), (10, 20, 1)]),
        ("val", "010", [(100, 100, 1)]),
        ("test", "020", [(5, 5, 2)]),
    ]:
        img_dir = root / "images" / split / "cell"
        ann_dir = root / "annotations" / split / "cell"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{stem}.jpg").write_bytes(b"")
        with (ann_dir / f"{stem}.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows(points)
    return root


def _make_beetle_raw(root: Path, *, n_folds: int = 3) -> tuple[Path, Path]:
    """A synthetic BEETLE data_overview.csv + touched WSI/mask files."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_folds):
        wsi_rel = f"images/slide_{i}.tif"
        mask_rel = f"annotations/slide_{i}_mask.tif"
        _touch(root / wsi_rel)
        _touch(root / mask_rel)
        rows.append(
            {
                "name": f"slide_{i}",
                "split": "development",
                "wsi_path": wsi_rel,
                "annotation_mask_path": mask_rel,
                "patient_id": f"p{i}",
                "source": "srcA",
                "specimen_type": "biopsy",
                "validation_fold": f"fold{i}",
            }
        )
    overview = root / "data_overview.csv"
    pd.DataFrame(rows).to_csv(overview, index=False)
    return overview, root


# ------------------------------------------------------------------- write_manifest


def test_write_manifest_orders_columns_and_selects_supervision(tmp_path: Path):
    manifest = write_manifest(
        tmp_path / "out",
        dataset_type="tile",
        dataset_rows=[
            {"label": 1, "image_path": "/a.png", "sample_id": "s0", "class_name": "x"},
            {"label": 0, "image_path": "/b.png", "sample_id": "s1", "class_name": "y"},
        ],
        split_rows=[
            {"sample_id": "s0", "split": "train"},
            {"sample_id": "s1", "split": "test"},
        ],
        summary={"note": "hi"},
    )
    assert isinstance(manifest, CuratedManifest)
    df = pd.read_csv(manifest.dataset_csv)
    # sample_id, image_path, then the supervision column (label), then extras.
    assert list(df.columns) == ["sample_id", "image_path", "label", "class_name"]
    splits = pd.read_csv(manifest.splits_csv)
    # fold is always present; defaulted to 0 when the curator omits it.
    assert list(splits.columns) == ["sample_id", "split", "fold"]
    assert set(splits["fold"]) == {0}
    assert manifest.summary_json.exists()


def test_write_manifest_supervision_column_by_dataset_type(tmp_path: Path):
    seg = write_manifest(
        tmp_path / "seg",
        dataset_type="segmentation",
        dataset_rows=[{"sample_id": "s0", "image_path": "/a.tif", "mask_path": "/m.tif"}],
        split_rows=[{"sample_id": "s0", "split": "test"}],
        summary={},
    )
    det = write_manifest(
        tmp_path / "det",
        dataset_type="detection",
        dataset_rows=[{"sample_id": "s0", "image_path": "/a.jpg", "points_path": "/p.csv"}],
        split_rows=[{"sample_id": "s0", "split": "test"}],
        summary={},
    )
    assert list(pd.read_csv(seg.dataset_csv).columns)[:3] == ["sample_id", "image_path", "mask_path"]
    assert list(pd.read_csv(det.dataset_csv).columns)[:3] == ["sample_id", "image_path", "points_path"]


def test_write_manifest_rejects_missing_supervision_column(tmp_path: Path):
    with pytest.raises(ValueError, match="missing required column"):
        write_manifest(
            tmp_path / "out",
            dataset_type="segmentation",
            dataset_rows=[{"sample_id": "s0", "image_path": "/a.tif"}],  # no mask_path
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


def test_write_manifest_rejects_two_supervision_columns(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one supervision column"):
        write_manifest(
            tmp_path / "out",
            dataset_type="tile",
            dataset_rows=[
                {"sample_id": "s0", "image_path": "/a.png", "label": 1, "mask_path": "/m.tif"}
            ],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


def test_write_manifest_rejects_unknown_dataset_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        write_manifest(
            tmp_path / "out",
            dataset_type="bogus",
            dataset_rows=[{"sample_id": "s0", "image_path": "/a.png", "label": 1}],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


# ------------------------------------------------------------- load-time validation


def test_load_manifest_selects_loader_by_dataset_type(tmp_path: Path):
    ds = tmp_path / "cls.csv"
    ds.write_text("sample_id,image_path,label\ns0,/a.png,1\n")
    assert isinstance(load_manifest(ds, "tile"), Dataset)
    assert isinstance(load_manifest(ds, "slide"), Dataset)

    seg = tmp_path / "seg.csv"
    seg.write_text("sample_id,image_path,mask_path\ns0,/a.tif,/m.tif\n")
    assert isinstance(load_manifest(seg, "segmentation"), SegmentationManifest)

    det = tmp_path / "det.csv"
    det.write_text("sample_id,image_path,points_path\ns0,/a.jpg,/p.csv\n")
    assert isinstance(load_manifest(det, "detection"), DetectionManifest)


def test_load_manifest_rejects_malformed_supervision_column(tmp_path: Path):
    # A segmentation manifest missing mask_path is rejected fail-fast with a clear message.
    bad = tmp_path / "bad.csv"
    bad.write_text("sample_id,image_path,label\ns0,/a.tif,1\n")
    with pytest.raises(ValueError, match="mask_path"):
        load_manifest(bad, "segmentation")


# ----------------------------------------------------------------- Curator Protocol


def test_curators_satisfy_structural_protocol_without_base_class():
    # ADR 0004: no base class — curators are just deterministic functions that return a
    # CuratedManifest, so they satisfy the structural Curator Protocol via isinstance.
    for curator in (
        curate_eva_patch_dataset,
        curate_ocelot_detection,
        curate_beetle_slide_manifest,
    ):
        assert isinstance(curator, Curator)
    # None of them inherit from Curator (it is a Protocol, never a base class).
    assert Curator not in type(curate_eva_patch_dataset).__mro__


def test_curated_manifest_lives_in_neutral_module():
    from soma.curation import manifest as neutral

    assert CuratedManifest is neutral.CuratedManifest
    # It is NOT re-exported from the EVA curator anymore.
    import soma.curation.eva as eva

    assert getattr(eva, "CuratedManifest", None) is neutral.CuratedManifest


# --------------------------------------------------- every curator emits the schema


def test_all_curators_emit_identical_core_schema(tmp_path: Path):
    eva = curate_eva_patch_dataset(
        "mhist", _make_mhist_raw(tmp_path / "mhist_raw"), tmp_path / "eva", tune_fraction=0.5
    )
    ocelot = curate_ocelot_detection(_make_ocelot_raw(tmp_path / "oce_raw"), tmp_path / "oce")
    overview, beetle_root = _make_beetle_raw(tmp_path / "beetle_raw")
    beetle = curate_beetle_slide_manifest(overview, beetle_root, tmp_path / "beetle")

    for manifest, supervision in [
        (eva, "label"),
        (ocelot, "points_path"),
        (beetle, "mask_path"),
    ]:
        ds = pd.read_csv(manifest.dataset_csv)
        assert list(ds.columns)[:3] == ["sample_id", "image_path", supervision]
        splits = pd.read_csv(manifest.splits_csv)
        assert list(splits.columns)[:3] == ["sample_id", "split", "fold"]
        assert manifest.summary_json is not None and manifest.summary_json.exists()


def test_beetle_curator_emits_dataset_csv_not_manifest_csv(tmp_path: Path):
    overview, beetle_root = _make_beetle_raw(tmp_path / "beetle_raw")
    out = tmp_path / "beetle"
    manifest = curate_beetle_slide_manifest(overview, beetle_root, out)

    assert manifest.dataset_csv == out / "dataset.csv"
    assert (out / "dataset.csv").exists()
    assert not (out / "manifest.csv").exists()  # the old shape is retired
    # Loads through the segmentation loader (mask_path supervision).
    seg = SegmentationManifest(manifest.dataset_csv)
    assert set(seg.sample_ids) == {"slide_0", "slide_1", "slide_2"}


# ------------------------------------------------------------- determinism contract


def test_recuration_is_byte_identical(tmp_path: Path):
    raw = _make_mhist_raw(tmp_path / "mhist_raw")
    a = curate_eva_patch_dataset("mhist", raw, tmp_path / "a", tune_fraction=0.5)
    b = curate_eva_patch_dataset("mhist", raw, tmp_path / "b", tune_fraction=0.5)

    for name in ("dataset.csv", "splits.csv", "summary.json"):
        assert (a.dataset_csv.parent / name).read_bytes() == (
            b.dataset_csv.parent / name
        ).read_bytes(), f"{name} is not byte-identical across re-curation"
