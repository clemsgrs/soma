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
``tile_size²``). An ROI dimension smaller than ``tile_size`` is padded up to it with a white
(glass-background) fill so it still yields one full tile; the original ROI dimensions are
retained (``roi_width`` / ``roi_height``) for the stitch coordinate frame and per-mm² area.
Every in-bounds source point is asserted to land in ≥1 tile (a coverage-correctness guard);
any point outside the ROI is dropped and counted (curation should have clamped/rejected it).

**Spacing (flat images).** These tiles are flat PNG crops, which the dense reader reads at
**native pixel resolution** — the run's ``requested_spacing_um`` is a *label*, not a resample
(``soma.dense.reader._is_flat``). So a tile's ``level0_spacing`` must equal the spacing the
run reads it at; to change physical spacing you must **resample the pixels**, never merely
restamp ``level0_spacing`` (that scales the point transform while the pixels stay put →
misregistration). This tiler therefore requires the source manifest's ``level0_spacing`` to
be **uniform** across ROIs (a heterogeneous flat-image manifest can't share one geometry) and
carries that single value onto every tile; ``target_spacing`` optionally asserts it equals an
expected value (``0.25`` for MIDOG).

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

_REQUIRED = ("sample_id", "image_path", "points_path")

# White (glass-background) fill for ROIs padded up to a full tile — see the padding note in
# ``tile_detection_manifest``.
_PAD_FILL = (255, 255, 255)


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


def _resolve_uniform_spacing(roi_df: pd.DataFrame, target_spacing: float | None) -> float | None:
    """The single ``level0_spacing`` all ROIs share, or ``None`` when the column is absent.

    Flat PNG tiles are read at native pixel resolution, so a tile's ``level0_spacing`` must
    equal the spacing the run reads it at — which is only well-defined if every ROI shares
    one spacing. A heterogeneous flat-image manifest (e.g. re-stamped with true per-scanner
    mpp) can't share one geometry and would misregister points against unresampled pixels, so
    it is rejected here; ``target_spacing`` additionally pins the expected value.
    """
    if "level0_spacing" not in roi_df.columns:
        if target_spacing is not None:
            raise ValueError(
                f"--target-spacing {target_spacing} given but the manifest has no "
                f"'level0_spacing' column to check against."
            )
        return None
    values = sorted({round(float(v), 6) for v in roi_df["level0_spacing"]})
    if len(values) != 1:
        raise ValueError(
            f"tiling requires a uniform level0_spacing (flat PNG tiles are read at native "
            f"resolution, not resampled), but the manifest carries {values}. Resample the "
            f"ROI pixels to a common spacing before tiling — do not merely restamp spacing."
        )
    spacing = values[0]
    if target_spacing is not None and round(float(target_spacing), 6) != spacing:
        raise ValueError(
            f"manifest level0_spacing {spacing} != --target-spacing {target_spacing}; the "
            f"flat tiles would be read at the run spacing but labelled a different one."
        )
    return spacing


def tile_detection_manifest(
    curated_dir: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    target_spacing: float | None = None,
) -> dict:
    """Tile every ROI of a curated detection manifest into fixed ``tile_size²`` tiles.

    Reads ``curated_dir/{dataset.csv, splits.csv}`` (the ROI-level manifest emitted by a
    detection curator), writes ``output_dir/{dataset.csv, splits.csv, summary.json}`` plus
    ``output_dir/tiles/<tile_id>.png`` and ``output_dir/points/<tile_id>.csv``.

    ``target_spacing`` (optional) asserts the manifest's uniform ``level0_spacing`` equals an
    expected value (e.g. ``0.25`` for MIDOG); the spacing is always required to be uniform
    (see the module docstring's *Spacing* note). Returns a small summary dict (also written
    to ``summary.json``).
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
    uniform_spacing = _resolve_uniform_spacing(roi_df, target_spacing)

    split_df = pd.read_csv(curated_dir / "splits.csv")
    split_of = {str(r["sample_id"]): (str(r["split"]), int(r["fold"])) for _, r in split_df.iterrows()}
    # F4: require exact, unique sample-id coverage between dataset.csv and splits.csv. A
    # missing split row must NOT silently default a held-out ROI into training.
    roi_ids = [str(s) for s in roi_df["sample_id"]]
    roi_id_set, split_id_set = set(roi_ids), set(split_of)
    if len(roi_ids) != len(roi_id_set):
        dupes = sorted({s for s in roi_ids if roi_ids.count(s) > 1})
        raise ValueError(f"dataset.csv has duplicate sample_id(s): {dupes[:5]}")
    missing, extra = roi_id_set - split_id_set, split_id_set - roi_id_set
    if missing or extra:
        raise ValueError(
            f"dataset.csv and splits.csv sample_ids must match exactly; "
            f"{len(missing)} ROI(s) missing a split row (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} split row(s) with no ROI (e.g. {sorted(extra)[:3]})."
        )

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
    dropped_out_of_bounds = 0

    # Trusted local dataset ROIs are large (tens of MP) and trip PIL's decompression-bomb
    # guard; lift it for the read loop only, then restore (importing this module must not
    # change PIL's global state).
    saved_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        for _, roi in roi_df.sort_values("sample_id").iterrows():
            roi_id = str(roi["sample_id"])
            raw_points = _read_points(roi["points_path"])
            with Image.open(roi["image_path"]) as opened:
                im = opened.convert("RGB")
            width, height = im.size  # original ROI dims (stitch frame + per-mm² area)

            # Drop any point outside the ROI up front so it is never written onto the padded
            # region below (curation should have clamped/rejected these; last-line guard).
            points = [(x, y, c) for (x, y, c) in raw_points if 0 <= x < width and 0 <= y < height]
            dropped_out_of_bounds += len(raw_points) - len(points)

            # #5: an ROI smaller than a tile is padded up to one full tile (white glass fill)
            # so every emitted tile is exactly tile_size²; points and original dims are kept.
            canvas_w, canvas_h = max(width, tile_size), max(height, tile_size)
            if (canvas_w, canvas_h) != (width, height):
                canvas = Image.new("RGB", (canvas_w, canvas_h), _PAD_FILL)
                canvas.paste(im, (0, 0))
                im = canvas

            windows = _tile_windows(canvas_w, canvas_h, tile_size, stride)
            covered: set[int] = set()
            for i, (x0, y0) in enumerate(windows):
                tile_id = f"{roi_id}_t{i:04d}"
                # Crop is [x0, x0+tile) x [y0, y0+tile); every window is exactly tile_size².
                tile_img = im.crop((x0, y0, x0 + tile_size, y0 + tile_size))
                tile_png = tiles_dir / f"{tile_id}.png"
                # Lossless but fast: default PNG compression (level 6) dominates the runtime
                # at ~17k tiles; level 1 is ~3-5x faster and the tiles are a throwaway cache.
                tile_img.save(tile_png, compress_level=1)

                # All points whose CENTER falls in this window, in tile-local coordinates.
                local: list[tuple[float, float, int]] = []
                for pi, (x, y, c) in enumerate(points):
                    if x0 <= x < x0 + tile_size and y0 <= y < y0 + tile_size:
                        local.append((x - x0, y - y0, c))
                        covered.add(pi)
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
                    "roi_width": width,     # original ROI dims -> stitched-ROI area (FROC per-mm²)
                    "roi_height": height,
                }
                for c in carry_cols:
                    row[c] = roi[c]
                tile_rows.append(row)

                split, fold = split_of[roi_id]
                split_rows.append({"sample_id": tile_id, "split": split, "fold": fold})
                per_split_tiles[split] += 1

            # F3b: every (in-bounds) source point must land in ≥1 tile. cover_origins covers
            # [0, extent) with no gap, so a point that reaches no tile is a coverage bug.
            uncovered = set(range(len(points))) - covered
            if uncovered:
                pi = sorted(uncovered)[0]
                raise ValueError(
                    f"ROI {roi_id}: {len(uncovered)} source point(s) reached no tile "
                    f"(e.g. {points[pi]} in {width}x{height}); tiling coverage is broken."
                )
    finally:
        Image.MAX_IMAGE_PIXELS = saved_max_pixels

    if not tile_rows:
        raise ValueError(f"no tiles produced from {curated_dir}")
    if dropped_out_of_bounds:
        print(
            f"warning: dropped {dropped_out_of_bounds} out-of-image source point(s) while "
            f"tiling {curated_dir} (curation should clamp/reject these)."
        )

    pd.DataFrame(tile_rows).to_csv(output_dir / "dataset.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "splits.csv", index=False)
    summary = {
        "source_manifest": str(curated_dir),
        "dataset_type": "detection",
        "tile_size": tile_size,
        "overlap": overlap,
        "level0_spacing": uniform_spacing,
        "num_rois": int(len(roi_df)),
        "num_tiles": len(tile_rows),
        "total_points_in_tiles": total_points,
        "empty_tiles": empty_tiles,
        "dropped_out_of_bounds": dropped_out_of_bounds,
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
    ap.add_argument("--target-spacing", type=float, default=None,
                    help="assert the manifest's uniform level0_spacing equals this (e.g. 0.25)")
    args = ap.parse_args(argv)
    summary = tile_detection_manifest(
        args.curated_dir, args.output_dir, tile_size=args.tile_size, overlap=args.overlap,
        target_spacing=args.target_spacing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
