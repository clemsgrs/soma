"""Tests for MONKEY curation on the public-bucket layout.

Each test builds a tiny raw root shaped like the ``aws s3 sync`` download: one small
pyramidal TIFF per case under ``images/pas-cpg/`` and per-class Grand-Challenge JSONs
(points and ROI polygons in mm) under ``annotations/json_mm/``. At 0.25 µm/px one mm is
4000 px and the 5 µm valid margin is 20 px.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from soma.benchmarks.detection_benchmark import DetectionBenchmark
from soma.curation.monkey import (
    DOWNLOAD_COMMAND,
    MONKEY_SPACING_LEVEL0,
    curate_monkey_detection,
)
from soma.curation.tile_detection import tile_detection_manifest
from soma.dataset import DetectionManifest, Splits
from tests.eva_slide_raw import write_pyramid

SPACING = 0.25
MM = 1000.0 / SPACING  # px per mm


def _mm(*px: float) -> list[float]:
    return [v / MM for v in px]


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """A rectangular ROI polygon in mm from px corners."""
    return [_mm(x0, y0), _mm(x1, y0), _mm(x1, y1), _mm(x0, y1)]


def _write_case(
    raw: Path,
    case_id: str,
    *,
    rois: list[list[list[float]]],
    lymphocytes: list[tuple[float, float]] = (),
    monocytes: list[tuple[float, float]] = (),
    size: tuple[int, int] = (400, 300),
    spacing: float = SPACING,
) -> None:
    """One case in the bucket layout; points and ROI corners are given in level-0 px."""
    write_pyramid(
        raw / "images" / "pas-cpg" / f"{case_id}_PAS_CPG.tif",
        size=size,
        mpp=spacing,
        levels=2,
        seed=sum(map(ord, case_id)),
    )
    ann = raw / "annotations" / "json_mm"
    ann.mkdir(parents=True, exist_ok=True)
    # Corners and points are px at SPACING; rescale the mm when the slide is at another one.
    scale = spacing / SPACING
    rois = [[[v * scale for v in corner] for corner in polygon] for polygon in rois]
    lymphocytes = [(x * scale, y * scale) for x, y in lymphocytes]
    monocytes = [(x * scale, y * scale) for x, y in monocytes]
    area_px = sum(
        abs((p[2][0] - p[0][0]) * (p[2][1] - p[0][1])) * MM * MM for p in rois
    ) / scale**2
    for name, points in (
        ("lymphocytes", lymphocytes),
        ("monocytes", monocytes),
        ("inflammatory-cells", [*lymphocytes, *monocytes]),
    ):
        doc = {
            "name": name,
            "type": "Multiple points",
            "points": [
                {"name": f"Point {i}", "point": _mm(x, y)} for i, (x, y) in enumerate(points)
            ],
            "area_rois": area_px,
            "rois": [{"name": f"ROI {i}", "polygon": p} for i, p in enumerate(rois)],
        }
        (ann / f"{case_id}_{name}.json").write_text(json.dumps(doc))


def _one_case_raw(tmp_path: Path) -> Path:
    """A 400x300 slide with two ROIs and points inside, near and far from them.

    Four empty 64x64 cases make up the five patients the cross-validation needs.
    """
    raw = tmp_path / "raw"
    for patient in range(2, 6):
        _write_case(raw, f"B_P{patient:06d}", rois=[_box(10, 10, 50, 50)], size=(64, 64))
    _write_case(
        raw,
        "A_P000001",
        rois=[_box(100, 100, 200, 200), _box(300, 40, 360, 100)],
        lymphocytes=[
            (150.0, 150.0),  # inside ROI 0
            (210.0, 150.0),  # 10 px = 2.5 µm right of ROI 0: kept
            (150.0, 260.0),  # 60 px = 15 µm below ROI 0: dropped
        ],
        monocytes=[
            (330.0, 70.0),  # inside ROI 1
            (20.0, 20.0),  # far from both: dropped
        ],
    )
    return raw


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_curator_writes_one_sample_per_roi(tmp_path: Path):
    raw = _one_case_raw(tmp_path)
    out = tmp_path / "curated"

    manifest = curate_monkey_detection(raw, out, spacing_at_level_0=SPACING)

    every_roi = pd.read_csv(manifest.dataset_csv).set_index("sample_id")
    assert every_roi.index.tolist()[:3] == ["A_P000001_roi0", "A_P000001_roi1", "B_P000002_roi0"]
    df = every_roi.loc[["A_P000001_roi0", "A_P000001_roi1"]]
    assert set(df["source_slide"]) == {"images/pas-cpg/A_P000001_PAS_CPG.tif"}
    assert df["spacing_at_level_0"].tolist() == [SPACING, SPACING]
    assert df["patient_id"].tolist() == ["A_P000001", "A_P000001"]
    assert df["centre"].tolist() == ["A", "A"]
    # The crop is the polygon's bounding box grown by the 20 px margin.
    roi0 = df.loc["A_P000001_roi0"]
    assert (roi0["crop_x"], roi0["crop_y"]) == (80, 80)

    # The crop holds the slide's level-0 pixels at the recorded offset.
    slide = tifffile.imread(raw / "images" / "pas-cpg" / "A_P000001_PAS_CPG.tif")
    crop = np.asarray(Image.open(roi0["image_path"]))
    assert crop.shape == (141, 141, 3)
    assert np.array_equal(crop, slide[80:221, 80:221])

    # Points are in crop coordinates; lymphocytes are class 0, monocytes class 1.
    pts0 = pd.read_csv(roi0["points_path"]).values.tolist()
    assert pts0 == [[70.0, 70.0, 0.0], [130.0, 70.0, 0.0]]
    roi1 = df.loc["A_P000001_roi1"]
    pts1 = pd.read_csv(roi1["points_path"])
    assert pts1.values.tolist() == [[330.0 - roi1["crop_x"], 70.0 - roi1["crop_y"], 1.0]]

    # The ignore mask is 0 on the polygon and its 5 µm margin, 255 beyond it.
    mask = np.asarray(Image.open(roi0["ignore_mask_path"]))
    assert mask.shape == crop.shape[:2] and set(np.unique(mask)) == {0, 255}
    assert mask[70, 70] == 0  # polygon interior
    assert mask[70, 139] == 0  # 19 px right of the polygon edge
    assert mask[0, 0] == 255  # crop corner, 20·√2 px from the polygon corner

    detection = DetectionManifest(manifest.dataset_csv)
    assert detection.samples["A_P000001_roi0"].ignore_mask_path == Path(roi0["ignore_mask_path"])

    summary = json.loads(manifest.summary_json.read_text())
    assert (summary["total_slides"], summary["total_rois"]) == (5, 6)
    assert summary["points_total"] == 5
    assert summary["points_kept"] == 3
    assert summary["points_dropped"] == 2
    assert summary["points_kept_outside_polygon"] == 1
    assert summary["points_per_class"] == {"lymphocytes": 2, "monocytes": 1}
    assert summary["area_rois_px"] == pytest.approx(100 * 100 + 60 * 60 + 4 * 40 * 40)
    valid_px = sum(
        int((np.asarray(Image.open(p)) == 0).sum()) for p in every_roi["ignore_mask_path"]
    )
    assert summary["valid_area_px"] == valid_px > summary["area_rois_px"]


@pytest.mark.parametrize("missing", ["images/pas-cpg", "annotations/json_mm"])
def test_missing_bucket_folder_shows_the_download_command(tmp_path: Path, missing: str):
    raw = _one_case_raw(tmp_path)
    shutil.rmtree(raw / missing)

    with pytest.raises(FileNotFoundError, match="aws s3 sync s3://monkey-training") as err:
        curate_monkey_detection(raw, tmp_path / "out", spacing_at_level_0=SPACING)
    assert DOWNLOAD_COMMAND in str(err.value)


def test_missing_slide_shows_the_download_command(tmp_path: Path):
    raw = _one_case_raw(tmp_path)
    (raw / "images" / "pas-cpg" / "A_P000001_PAS_CPG.tif").unlink()

    with pytest.raises(FileNotFoundError, match="aws s3 sync s3://monkey-training"):
        curate_monkey_detection(raw, tmp_path / "out", spacing_at_level_0=SPACING)


def test_slide_spacing_must_match_the_declaration(tmp_path: Path):
    raw = _one_case_raw(tmp_path)

    with pytest.raises(ValueError, match="level-0 spacing"):
        curate_monkey_detection(raw, tmp_path / "out", spacing_at_level_0=0.5)


def _cohort_raw(tmp_path: Path) -> Path:
    """13 one-ROI patients over four centres (A: 5, B: 4, C: 2, D: 2)."""
    raw = tmp_path / "raw"
    cases = [f"A_P{i:06d}" for i in range(1, 6)] + [f"B_P{i:06d}" for i in range(10, 14)]
    cases += ["C_P000020", "C_P000021", "D_P000030", "D_P000031"]
    for case_id in cases:
        _write_case(
            raw, case_id, rois=[_box(20, 20, 80, 80)], lymphocytes=[(50.0, 50.0)],
            size=(128, 128),
        )
    return raw


def test_splits_are_a_five_fold_patient_cross_validation(tmp_path: Path):
    raw = _cohort_raw(tmp_path)
    manifest = curate_monkey_detection(raw, tmp_path / "curated", spacing_at_level_0=SPACING)

    dataset = pd.read_csv(manifest.dataset_csv)
    patient_of = dict(zip(dataset["sample_id"], dataset["patient_id"]))
    centre_of = dict(zip(dataset["patient_id"], dataset["centre"]))
    splits = Splits(manifest.splits_csv, DetectionManifest(manifest.dataset_csv))
    assert len(splits.folds) == 5

    test_fold_of: dict[str, int] = {}
    for k, fold in enumerate(splits.folds):
        roles = {
            "train": {patient_of[s] for s in fold.train},
            "tune": {patient_of[s] for s in fold.tune},
            "test": {patient_of[s] for s in fold.tests["test"]},
        }
        # No patient leaks across train, tune and test within a fold.
        assert not roles["train"] & roles["tune"]
        assert not roles["train"] & roles["test"]
        assert not roles["tune"] & roles["test"]
        assert set.union(*roles.values()) == set(patient_of.values())
        for patient in roles["test"]:
            assert patient not in test_fold_of  # test exactly once
            test_fold_of[patient] = k
    # Every slide is test exactly once.
    assert set(test_fold_of) == set(patient_of.values())

    # BEETLE rotation: fold k's tune patients are fold (k + 1)'s test patients.
    for k, fold in enumerate(splits.folds):
        tune = {patient_of[s] for s in fold.tune}
        assert tune == {p for p, f in test_fold_of.items() if f == (k + 1) % 5}

    # Each centre is spread evenly, and fold sizes differ by at most one patient.
    sizes = [list(test_fold_of.values()).count(k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    for centre in "ABCD":
        per_fold = [
            sum(1 for p, f in test_fold_of.items() if f == k and centre_of[p] == centre)
            for k in range(5)
        ]
        assert max(per_fold) - min(per_fold) <= 1


def test_curation_is_byte_identical(tmp_path: Path):
    raw = _one_case_raw(tmp_path)
    out = tmp_path / "curated"
    curate_monkey_detection(raw, out, spacing_at_level_0=SPACING)
    first = _tree_digest(out)

    curate_monkey_detection(raw, out, spacing_at_level_0=SPACING)

    assert _tree_digest(out) == first
    assert {name.split("/")[0] for name in first} >= {"images", "points", "ignore_masks"}


def test_tiling_carries_the_ignore_masks(tmp_path: Path):
    raw = _one_case_raw(tmp_path)
    curated = tmp_path / "curated"
    curate_monkey_detection(raw, curated, spacing_at_level_0=SPACING)

    summary = tile_detection_manifest(curated, tmp_path / "tiled", tile_size=64, overlap=8)

    tiles = pd.read_csv(tmp_path / "tiled" / "dataset.csv")
    assert tiles["ignore_mask_path"].notna().all()
    assert all(np.asarray(Image.open(p)).shape == (64, 64) for p in tiles["ignore_mask_path"])
    # The curator keeps only points on valid pixels, so the tiler drops none.
    assert summary["points_in_ignored_region"] == 0
    rois = pd.read_csv(curated / "dataset.csv").set_index("sample_id")
    for roi_id, valid_px in tiles.groupby("source_wsi")["roi_valid_area_px"].first().items():
        mask = np.asarray(Image.open(rois.loc[roi_id, "ignore_mask_path"]))
        assert valid_px == int((mask == 0).sum())


def test_benchmark_curate_tiles_the_roi_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    for case_id in ("A_P000001", "A_P000002", "B_P000003", "B_P000004", "C_P000005"):
        _write_case(
            raw, case_id, rois=[_box(20, 20, 80, 80)], lymphocytes=[(50.0, 50.0)],
            size=(128, 128), spacing=MONKEY_SPACING_LEVEL0,
        )
    out = tmp_path / "curated"

    manifest = DetectionBenchmark().curate(raw, out, dataset="monkey")

    # The ROI-level output is kept under roi/; the run-ready manifest is the tiled one.
    assert (out / "roi" / "dataset.csv").is_file()
    tiles = pd.read_csv(manifest.dataset_csv)
    assert len(tiles) == 5  # each small ROI is padded up to one 1024² tile
    assert tiles["source_wsi"].str.endswith("_roi0").all()
    assert [Image.open(p).size for p in tiles["image_path"]] == [(1024, 1024)] * 5
    assert tiles["ignore_mask_path"].notna().all()
    summary = json.loads(manifest.summary_json.read_text())
    assert (summary["tile_size"], summary["overlap"]) == (1024, 128)
    assert set(pd.read_csv(manifest.splits_csv)["fold"]) == set(range(5))
