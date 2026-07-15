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
