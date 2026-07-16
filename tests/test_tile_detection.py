"""Tests for the detection ROI tiler (soma.curation.tile_detection)."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from soma.curation.tile_detection import cover_origins, tile_detection_manifest


def test_cover_origins_edge_flush():
    # stride 96 over extent 300: [0, 96] leaves a gap (224 < 300) -> flush 172 appended.
    assert cover_origins(300, 128, 96) == [0, 96, 172]
    # exact fit (extent == size) -> a single origin.
    assert cover_origins(128, 128, 96) == [0]
    # evenly divisible -> no extra flush window.
    assert cover_origins(256, 128, 128) == [0, 128]
    with pytest.raises(ValueError):
        cover_origins(100, 128, 96)  # ROI smaller than a tile


def test_cover_origins_rejects_stride_larger_than_tile():
    with pytest.raises(ValueError, match="stride must be <= tile size"):
        cover_origins(300, 100, 150)


def _write_curated(root: Path, *, points: list[tuple[float, float, int]], size=(300, 250)) -> Path:
    """A one-ROI curated detection manifest with a synthetic image + point CSV."""
    curated = root / "curated"
    (curated / "points").mkdir(parents=True)
    img_path = curated / "roi_A.png"
    Image.new("RGB", size, (127, 127, 127)).save(img_path)
    pts_path = curated / "points" / "roi_A.csv"
    with pts_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "class"])
        for x, y, c in points:
            w.writerow([x, y, c])
    pd.DataFrame(
        [{"sample_id": "roi_A", "image_path": str(img_path), "points_path": str(pts_path),
          "domain": "d1", "level0_spacing": 0.25}]
    ).to_csv(curated / "dataset.csv", index=False)
    pd.DataFrame([{"sample_id": "roi_A", "split": "tune", "fold": 0}]).to_csv(
        curated / "splits.csv", index=False
    )
    return curated


def test_tiler_coverage_labels_and_stitch_columns(tmp_path):
    # (10,10) sits only in the top-left tile; (160,160) sits in a y-overlap band -> 2 tiles.
    curated = _write_curated(tmp_path, points=[(10.0, 10.0, 0), (160.0, 160.0, 0)])
    out = tmp_path / "tiled"
    summary = tile_detection_manifest(curated, out, tile_size=128, overlap=32)

    assert summary["num_tiles"] == 9  # 3 x-origins * 3 y-origins
    ds = pd.read_csv(out / "dataset.csv")
    assert len(ds) == 9
    # stitching + carried columns are present on every tile.
    for col in ("source_wsi", "tile_x", "tile_y", "domain", "level0_spacing"):
        assert col in ds.columns
    assert set(ds["source_wsi"]) == {"roi_A"}
    assert (ds["domain"] == "d1").all()

    # Every emitted tile PNG is exactly tile_size square.
    for p in ds["image_path"]:
        assert Image.open(p).size == (128, 128)

    # Reconstruct every (tile-local point + tile origin) back to ROI coords: no point lost,
    # and the overlap point appears in exactly two tiles.
    recon = []
    per_point_tiles = {(10.0, 10.0): 0, (160.0, 160.0): 0}
    for _, row in ds.iterrows():
        for x, y, c in _read_local(row["points_path"]):
            gx, gy = x + row["tile_x"], y + row["tile_y"]
            recon.append((gx, gy))
            if (gx, gy) in per_point_tiles:
                per_point_tiles[(gx, gy)] += 1
    assert (10.0, 10.0) in recon and (160.0, 160.0) in recon
    assert per_point_tiles[(10.0, 10.0)] == 1
    assert per_point_tiles[(160.0, 160.0)] == 2  # duplicated across the y-overlap
    assert summary["total_points_in_tiles"] == 3  # 1 + 2

    # Tiles inherit the parent ROI split/fold.
    sp = pd.read_csv(out / "splits.csv")
    assert set(sp["split"]) == {"tune"} and set(sp["fold"]) == {0}
    assert len(sp) == 9


def test_tiler_deterministic(tmp_path):
    curated = _write_curated(tmp_path, points=[(50.0, 50.0, 0), (160.0, 160.0, 0)])
    a, b = tmp_path / "a", tmp_path / "b"
    tile_detection_manifest(curated, a, tile_size=128, overlap=32)
    tile_detection_manifest(curated, b, tile_size=128, overlap=32)
    # image_path/points_path embed the output dir, so compare the location-independent
    # content: tile identity + origins, the split table, and each tile's points.
    loc = ["sample_id", "source_wsi", "tile_x", "tile_y", "domain", "level0_spacing"]
    da = pd.read_csv(a / "dataset.csv")[loc]
    db = pd.read_csv(b / "dataset.csv")[loc]
    pd.testing.assert_frame_equal(da, db)
    assert (a / "splits.csv").read_text() == (b / "splits.csv").read_text()
    for tid in da["sample_id"]:
        assert (a / "points" / f"{tid}.csv").read_text() == (b / "points" / f"{tid}.csv").read_text()


def _read_local(points_path) -> list[tuple[float, float, int]]:
    with Path(points_path).open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [(float(r[0]), float(r[1]), int(r[2])) for r in reader if r]


def _write_manifest(root: Path, rois: list[dict], splits: list[dict], *, size=(300, 250)) -> Path:
    """Multi-ROI curated manifest. Each ROI dict: sample_id + optional level0_spacing + points."""
    curated = root / "curated"
    (curated / "points").mkdir(parents=True, exist_ok=True)
    ds_rows = []
    for roi in rois:
        sid = roi["sample_id"]
        img = curated / f"{sid}.png"
        Image.new("RGB", roi.get("size", size), (127, 127, 127)).save(img)
        pts = curated / "points" / f"{sid}.csv"
        with pts.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y", "class"])
            for x, y, c in roi.get("points", []):
                w.writerow([x, y, c])
        row = {"sample_id": sid, "image_path": str(img), "points_path": str(pts), "domain": "d1"}
        if "level0_spacing" in roi:
            row["level0_spacing"] = roi["level0_spacing"]
        ds_rows.append(row)
    pd.DataFrame(ds_rows).to_csv(curated / "dataset.csv", index=False)
    pd.DataFrame(splits).to_csv(curated / "splits.csv", index=False)
    return curated


# --- F4: exact split coverage --------------------------------------------------------


def test_missing_split_row_raises(tmp_path):
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "points": [(10.0, 10.0, 0)]},
         {"sample_id": "roi_B", "points": [(10.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "train", "fold": 0}],  # roi_B has no split row
    )
    with pytest.raises(ValueError, match="missing a split row"):
        tile_detection_manifest(curated, tmp_path / "tiled", tile_size=128, overlap=32)


def test_duplicate_sample_id_raises(tmp_path):
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "points": [(10.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "train", "fold": 0}],
    )
    ds = pd.read_csv(curated / "dataset.csv")
    pd.concat([ds, ds]).to_csv(curated / "dataset.csv", index=False)  # roi_A twice
    with pytest.raises(ValueError, match="duplicate sample_id"):
        tile_detection_manifest(curated, tmp_path / "tiled", tile_size=128, overlap=32)


def test_duplicate_split_sample_id_raises(tmp_path):
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "points": [(10.0, 10.0, 0)]}],
        [
            {"sample_id": "roi_A", "split": "train", "fold": 0},
            {"sample_id": "roi_A", "split": "test", "fold": 0},
        ],
    )
    with pytest.raises(ValueError, match="splits.csv has duplicate sample_id"):
        tile_detection_manifest(curated, tmp_path / "tiled", tile_size=128, overlap=32)


# --- F2: uniform spacing -------------------------------------------------------------


def test_heterogeneous_spacing_raises(tmp_path):
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "level0_spacing": 0.25, "points": [(10.0, 10.0, 0)]},
         {"sample_id": "roi_B", "level0_spacing": 0.50, "points": [(10.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "train", "fold": 0},
         {"sample_id": "roi_B", "split": "train", "fold": 0}],
    )
    with pytest.raises(ValueError, match="uniform level0_spacing"):
        tile_detection_manifest(curated, tmp_path / "tiled", tile_size=128, overlap=32)


def test_target_spacing_mismatch_raises_and_match_recorded(tmp_path):
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "level0_spacing": 0.25, "points": [(10.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "train", "fold": 0}],
    )
    with pytest.raises(ValueError, match="target-spacing"):
        tile_detection_manifest(curated, tmp_path / "bad", tile_size=128, overlap=32, target_spacing=0.5)
    summary = tile_detection_manifest(
        curated, tmp_path / "ok", tile_size=128, overlap=32, target_spacing=0.25
    )
    assert summary["level0_spacing"] == 0.25


# --- #5 padding + F3b out-of-bounds --------------------------------------------------


def test_roi_smaller_than_tile_is_padded(tmp_path):
    # 300x250 ROI, tile 512 -> padded to one 512² tile; original dims kept, point preserved.
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "size": (300, 250), "points": [(120.0, 90.0, 0)]}],
        [{"sample_id": "roi_A", "split": "test", "fold": 0}],
    )
    out = tmp_path / "tiled"
    summary = tile_detection_manifest(curated, out, tile_size=512, overlap=128)
    assert summary["num_tiles"] == 1
    ds = pd.read_csv(out / "dataset.csv")
    assert Image.open(ds.iloc[0]["image_path"]).size == (512, 512)
    assert int(ds.iloc[0]["roi_width"]) == 300 and int(ds.iloc[0]["roi_height"]) == 250
    assert _read_local(ds.iloc[0]["points_path"]) == [(120.0, 90.0, 0)]


def test_out_of_bounds_point_dropped_not_written_on_padding(tmp_path):
    # (400,10) is outside the 300-wide ROI: it must be dropped/counted, never written onto
    # the padded canvas; the in-bounds point survives.
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "size": (300, 250), "points": [(10.0, 10.0, 0), (400.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "test", "fold": 0}],
    )
    out = tmp_path / "tiled"
    summary = tile_detection_manifest(curated, out, tile_size=512, overlap=128)
    assert summary["dropped_out_of_bounds"] == 1
    ds = pd.read_csv(out / "dataset.csv")
    all_local = [p for path in ds["points_path"] for p in _read_local(path)]
    assert all_local == [(10.0, 10.0, 0)]  # only the in-bounds point, in tile-local coords


def test_max_image_pixels_not_disabled_globally(tmp_path):
    # Importing/using the tiler must not leave PIL's decompression-bomb guard disabled.
    curated = _write_manifest(
        tmp_path,
        [{"sample_id": "roi_A", "points": [(10.0, 10.0, 0)]}],
        [{"sample_id": "roi_A", "split": "train", "fold": 0}],
    )
    tile_detection_manifest(curated, tmp_path / "tiled", tile_size=128, overlap=32)
    assert Image.MAX_IMAGE_PIXELS is not None  # restored, not left at None
