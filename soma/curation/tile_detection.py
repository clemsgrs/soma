"""Tile a large-ROI detection manifest into fixed-size tiles for the dense path.

soma's dense **detection** path (:func:`soma.dense_extraction.extract_dense_grids`) requires
every sample to be one fixed ``target_size`` tile. OCELOT ships fixed 1024² cell patches so
it needs no tiling; MIDOG 2022 and MONKEY ship **large, variable-size ROIs** (thousands of
px, many distinct sizes), so we cut each ROI into fixed ``tile_size²`` tiles here. The tiled
manifest is then extracted/trained exactly like OCELOT, and the score path stitches per-tile
predictions back to the ROI frame for the native per-ROI metric.

**Labeling — every point in the window, no single-assignment.** Each tile carries every GT
point whose *center* falls inside its window, overlap copies included. Single-assignment
(each point to one tile) would leave a mitosis *visible in the pixels but unlabeled* in a
neighboring overlapping tile, and the foreground-weighted heatmap loss would then punish the
model for correctly firing there. Duplicate positives across overlapping tiles are correct —
each tile is an independent training image and the object genuinely appears in both. The
prediction duplicates they cause are removed once, at inference, by the per-ROI stitch (NMS).
A residual near-seam effect (a cell whose center is just *outside* a window but whose body
pokes in is an unlabeled partial object) is bounded by ``overlap >= object diameter`` and is
the subject of the deferred ignore-band refinement (issue #287).

**Placement.** Tile origins use the same edge-flush walk as the encoder's sliding windows:
stride ``= tile_size - overlap``; the last tile in each dimension is shifted flush to the ROI
edge so coverage is complete with no partial tail (every emitted tile is exactly
``tile_size²``). Each ROI dimension must be ``>= tile_size``.

**Stitching hooks.** Each tile row carries ``source_wsi`` (the parent ROI ``sample_id``),
``tile_x`` and ``tile_y`` (the window origin in ROI pixels) — the columns
:class:`~soma.dataset.DetectionManifest` already reserves for WSI stitching — so the
score-time stitch lifts per-tile predicted points back to ROI coordinates. Tiles inherit
their parent ROI's ``split``/``fold``; per-ROI metadata columns (``domain`` / ``tumor_type``
/ ``scanner`` / ``level0_spacing`` / ``patient_id``) are carried through unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

# Reading a ~35 MP ROI as a PIL image trips the decompression-bomb guard; these are trusted
# local dataset ROIs, so lift the cap.
Image.MAX_IMAGE_PIXELS = None

_REQUIRED = ("sample_id", "image_path", "points_path")


def cover_origins(extent: int, size: int, stride: int) -> list[int]:
    """Start offsets of ``size``-wide tiles that fully cover ``[0, extent)``.

    Walk ``[0, extent - size]`` in ``stride`` steps; if the last step leaves a gap, append a
    final origin flush to the far edge (``extent - size``) so coverage is complete with no
    partial tail. Mirrors the encoder-window placement in ``slide2vec``'s ``cover_origins``.
    """
    if extent < size:
        raise ValueError(f"ROI extent {extent} smaller than tile size {size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    starts = list(range(0, extent - size + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def _read_points(points_path: Path) -> list[tuple[float, float, int]]:
    """Read a level-0 ``x,y,class`` point CSV (header row required)."""
    out: list[tuple[float, float, int]] = []
    with Path(points_path).open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            out.append((float(row[0]), float(row[1]), int(row[2])))
    return out


def _tile_windows(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Deterministic (x0, y0) origins covering a ``width x height`` ROI (row-major)."""
    return [
        (x0, y0)
        for y0 in cover_origins(height, tile, stride)
        for x0 in cover_origins(width, tile, stride)
    ]


def tile_detection_manifest(
    curated_dir: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
) -> dict:
    """Tile every ROI of a curated detection manifest into fixed ``tile_size²`` tiles.

    Reads ``curated_dir/{dataset.csv, splits.csv}`` (the ROI-level manifest emitted by a
    detection curator), writes ``output_dir/{dataset.csv, splits.csv, summary.json}`` plus
    ``output_dir/tiles/<tile_id>.png`` and ``output_dir/points/<tile_id>.csv``.

    Returns a small summary dict (also written to ``summary.json``).
    """
    curated_dir = Path(curated_dir)
    output_dir = Path(output_dir)
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must be in [0, tile_size); got {overlap} vs {tile_size}")
    stride = tile_size - overlap

    roi_df = pd.read_csv(curated_dir / "dataset.csv")
    for col in _REQUIRED:
        if col not in roi_df.columns:
            raise ValueError(f"curated dataset.csv missing required column {col!r}")
    split_df = pd.read_csv(curated_dir / "splits.csv")
    split_of = {str(r["sample_id"]): (str(r["split"]), int(r["fold"])) for _, r in split_df.iterrows()}

    tiles_dir = output_dir / "tiles"
    points_dir = output_dir / "points"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    # Columns carried unchanged onto every tile of an ROI (everything but the per-tile paths).
    carry_cols = [c for c in roi_df.columns if c not in ("sample_id", "image_path", "points_path")]

    tile_rows: list[dict] = []
    split_rows: list[dict] = []
    per_split_tiles: Counter[str] = Counter()
    total_points = 0
    empty_tiles = 0

    for _, roi in roi_df.sort_values("sample_id").iterrows():
        roi_id = str(roi["sample_id"])
        points = _read_points(roi["points_path"])
        with Image.open(roi["image_path"]) as im:
            im = im.convert("RGB")
            width, height = im.size
            windows = _tile_windows(width, height, tile_size, stride)
            for i, (x0, y0) in enumerate(windows):
                tile_id = f"{roi_id}_t{i:04d}"
                # Crop is [x0, x0+tile) x [y0, y0+tile); every window is exactly tile_size².
                tile_img = im.crop((x0, y0, x0 + tile_size, y0 + tile_size))
                tile_png = tiles_dir / f"{tile_id}.png"
                # Lossless but fast: default PNG compression (level 6) dominates the runtime
                # at ~17k tiles; level 1 is ~3-5x faster and the tiles are a throwaway cache.
                tile_img.save(tile_png, compress_level=1)

                # All points whose CENTER falls in this window, in tile-local coordinates.
                local = [
                    (x - x0, y - y0, c)
                    for (x, y, c) in points
                    if x0 <= x < x0 + tile_size and y0 <= y < y0 + tile_size
                ]
                tile_csv = points_dir / f"{tile_id}.csv"
                with tile_csv.open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["x", "y", "class"])
                    for x, y, c in local:
                        w.writerow([x, y, c])
                total_points += len(local)
                empty_tiles += int(len(local) == 0)

                row = {
                    "sample_id": tile_id,
                    "image_path": str(tile_png.resolve()),
                    "points_path": str(tile_csv.resolve()),
                    "source_wsi": roi_id,   # parent ROI (DetectionManifest stitching column)
                    "tile_x": x0,
                    "tile_y": y0,
                    "roi_width": width,     # parent ROI dims -> stitched-ROI area (FROC per-mm²)
                    "roi_height": height,
                }
                for c in carry_cols:
                    row[c] = roi[c]
                tile_rows.append(row)

                split, fold = split_of.get(roi_id, ("train", 0))
                split_rows.append({"sample_id": tile_id, "split": split, "fold": fold})
                per_split_tiles[split] += 1

    if not tile_rows:
        raise ValueError(f"no tiles produced from {curated_dir}")

    pd.DataFrame(tile_rows).to_csv(output_dir / "dataset.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "splits.csv", index=False)
    summary = {
        "source_manifest": str(curated_dir),
        "dataset_type": "detection",
        "tile_size": tile_size,
        "overlap": overlap,
        "num_rois": int(len(roi_df)),
        "num_tiles": len(tile_rows),
        "total_points_in_tiles": total_points,
        "empty_tiles": empty_tiles,
        "tiles_per_split": dict(per_split_tiles),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m soma.curation.tile_detection",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--curated-dir", type=Path, required=True, help="ROI-level curated manifest dir")
    ap.add_argument("--output-dir", type=Path, required=True, help="tiled manifest output dir")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args(argv)
    summary = tile_detection_manifest(
        args.curated_dir, args.output_dir, tile_size=args.tile_size, overlap=args.overlap
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
