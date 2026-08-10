"""Tests for OCELOT 2023 cell-detection dataset curation."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pandas as pd

from soma.curation.ocelot import (
    OCELOT_NATIVE_SPACING_UM,
    OCELOT_SPLIT_ROLE,
    curate_ocelot_detection,
)
from soma.dataset import DetectionManifest, Splits


def _write_raw_sample(
    raw_root: Path,
    ocelot_split: str,
    stem: str,
    points: list[tuple],
) -> None:
    """Create one synthetic OCELOT cell patch + headerless x,y,label annotation."""
    img_dir = raw_root / "images" / ocelot_split / "cell"
    ann_dir = raw_root / "annotations" / ocelot_split / "cell"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"")  # native curation never decodes pixels
    with (ann_dir / f"{stem}.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(points)


def _make_raw(tmp_path: Path) -> Path:
    raw_root = tmp_path / "ocelot_raw"
    # train: 2 samples (one with a BC+TC mix, one empty), val: 1, test: 1.
    _write_raw_sample(raw_root, "train", "001", [(42, 516, 2), (10, 20, 1)])
    _write_raw_sample(raw_root, "train", "002", [])  # empty annotation is valid
    _write_raw_sample(raw_root, "val", "010", [(100, 100, 1)])
    _write_raw_sample(raw_root, "test", "020", [(5, 5, 2), (6, 6, 2)])
    return raw_root


def test_curate_ocelot_emits_detection_manifest_and_remaps_labels(tmp_path: Path):
    raw_root = _make_raw(tmp_path)
    out = tmp_path / "curated"

    manifest = curate_ocelot_detection(raw_root, out)

    assert manifest.dataset_csv == out / "dataset.csv"
    assert manifest.splits_csv == out / "splits.csv"

    # Manifest loads through Soma's detection manifest (required cols present, ids safe).
    df = pd.read_csv(manifest.dataset_csv)
    assert list(df.columns) == [
        "sample_id",
        "image_path",
        "points_path",
        "spacing_at_level_0",
    ]
    detection = DetectionManifest(manifest.dataset_csv)
    assert set(detection.sample_ids) == {"train_001", "train_002", "val_010", "test_020"}
    assert detection.samples["train_001"].spacing_at_level_0 == OCELOT_NATIVE_SPACING_UM

    # OCELOT 1-based labels are remapped to Soma 0-based: 1=BC->0, 2=TC->1.
    pts = pd.read_csv(out / "points" / "train_001.csv")
    assert list(pts.columns) == ["x", "y", "class"]
    assert sorted(pts["class"].tolist()) == [0, 1]

    # Empty annotation -> a valid header-only point CSV (no rows).
    empty = pd.read_csv(out / "points" / "train_002.csv")
    assert len(empty) == 0


def test_curate_ocelot_maps_splits_to_soma_roles(tmp_path: Path):
    raw_root = _make_raw(tmp_path)
    out = tmp_path / "curated"
    manifest = curate_ocelot_detection(raw_root, out)

    detection = DetectionManifest(manifest.dataset_csv)
    fold = Splits(manifest.splits_csv, detection).folds[0]

    # OCELOT train->train, val->tune, test->test (Soma emits the split verbatim).
    assert OCELOT_SPLIT_ROLE == {"train": "train", "val": "tune", "test": "test"}
    assert set(fold.train) == {"train_001", "train_002"}
    assert set(fold.tune) == {"val_010"}
    assert set(fold.tests["test"]) == {"test_020"}


def test_curate_ocelot_summary_counts(tmp_path: Path):
    raw_root = _make_raw(tmp_path)
    out = tmp_path / "curated"
    curate_ocelot_detection(raw_root, out)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_classes"] == 2
    assert summary["class_names"] == ["BC", "TC"]
    assert summary["total_samples"] == 4
    assert summary["splits"]["train"]["num_empty"] == 1
    assert summary["splits"]["train"]["points_per_class"] == {"BC": 1, "TC": 1}


def test_curate_ocelot_has_one_native_manifest_interface():
    assert tuple(inspect.signature(curate_ocelot_detection).parameters) == (
        "raw_root",
        "output_dir",
    )


def test_curate_ocelot_native_path_declares_native_source_spacing(tmp_path: Path):
    raw_root = _make_raw(tmp_path)
    out = tmp_path / "curated"

    manifest = curate_ocelot_detection(raw_root, out)
    df = pd.read_csv(manifest.dataset_csv)
    assert df["spacing_at_level_0"].tolist() == [OCELOT_NATIVE_SPACING_UM] * len(df)
    assert Path(df.loc[df["sample_id"] == "train_001", "image_path"].item()) == (
        raw_root / "images" / "train" / "cell" / "001.jpg"
    ).resolve()
    points = pd.read_csv(out / "points" / "train_001.csv")
    assert points.to_dict("records") == [
        {"x": 42.0, "y": 516.0, "class": 1},
        {"x": 10.0, "y": 20.0, "class": 0},
    ]
    assert not (out / "images").exists()
    assert "render_spacing_um" not in json.loads((out / "summary.json").read_text())
