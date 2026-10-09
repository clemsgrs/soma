"""Curator for the MONKEY kidney-biopsy inflammatory-cell detection challenge.

Like the EVA / OCELOT / MIDOG curators, this emits Soma's unified Manifest
(``dataset.csv`` + ``splits.csv`` + ``summary.json``) from locally downloaded raw data and
**does not download anything**. The MONKEY training set (81 PAS-stained kidney-transplant
biopsy WSIs, centres A–D, one slide per patient) is in a public bucket; download the
images and the mm-frame annotations once with::

    aws s3 sync s3://monkey-training <raw_root> --no-sign-request \\
      --exclude "*" --include "images/pas-cpg/*" --include "annotations/json_mm/*"

which gives::

    <raw_root>/images/pas-cpg/<case>_PAS_CPG.tif
    <raw_root>/annotations/json_mm/<case>_{lymphocytes,monocytes,inflammatory-cells}.json

Each case ``<centre>_P<patient>`` (e.g. ``A_P000001``) has one Grand-Challenge JSON per
class. Its ``points[].point`` and ``rois[].polygon`` are in **millimetres** in the slide's
level-0 frame (``px = mm * 1000 / spacing``), and ``area_rois`` is the summed polygon area
in px². Cells are annotated only inside the ROI polygons.

**One sample per ROI.** Each polygon becomes one flat sample, cropped at curation time:
the polygon's level-0 bounding box, grown by the valid-region margin and clipped to the
slide, is read with hs2p's slide reader and written as ``images/<sample_id>.png``. The
row records the slide (``source_slide``, relative to ``raw_root``) and the crop's level-0
offset (``crop_x`` / ``crop_y``).

**Valid region.** The valid region of an ROI is its polygon dilated by
``valid_margin_um`` (5 µm) on the crop's pixel grid. It is written as an ignore mask
(``ignore_masks/<sample_id>.png``: 255 outside, 0 inside; see
:func:`soma.detection.io.read_ignore_mask`), so the detection path has no loss, no ground
truth and no scored prediction outside it, and FROC divides false positives by its area.
A point is kept when it lands on a valid pixel, i.e. within 5 µm of its polygon; the rest
are dropped, and ``summary.json`` counts both. Lymphocytes become class ``0`` and
monocytes class ``1``, written in crop coordinates to ``points/<sample_id>.csv`` (the
merged inflammatory-cells class is pooled at score time, so it is not stored).

**Splits.** The public data ships no partition and the official test set is hidden, so the
curator writes a 5-fold cross-validation by patient. Within each centre, patients are
ordered by a stable SHA1 of their id and dealt round-robin over the folds, the deal
continuing from one centre to the next, so every centre is spread evenly and fold sizes
differ by at most one patient. The folds rotate as in BEETLE: in fold ``k``, fold ``k``
is ``test``, fold ``(k + 1) % 5`` is ``tune`` and the other three are ``train``, so every
slide is ``test`` exactly once. These numbers are **not comparable** to the published
MONKEY leaderboard (a different test set).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from soma.curation.manifest import CuratedManifest, write_manifest
from soma.detection.encode import IGNORE_VALUE, points_on_valid

# Level-0 µm/px of the MONKEY PAS-CPG WSIs (the challenge evaluator's SPACING_LEVEL0).
MONKEY_SPACING_LEVEL0 = 0.24199951445730394

# 0-based class ids for the stored points CSV (the merged MNL class is derived at scoring
# time by pooling both, so it is not stored as a third class).
MONKEY_CLASS_NAMES = ("lymphocytes", "monocytes")
MONKEY_LABEL_REMAP = {name: idx for idx, name in enumerate(MONKEY_CLASS_NAMES)}
MONKEY_NUM_CLASSES = len(MONKEY_CLASS_NAMES)

# The valid region is the ROI polygon dilated by this distance; points beyond it are dropped.
VALID_MARGIN_UM = 5.0
NUM_FOLDS = 5

# A slide whose own level-0 spacing differs from the declaration by more than this
# relative amount would misplace the mm annotations.
_SPACING_RTOL = 1e-3

DOWNLOAD_COMMAND = (
    "aws s3 sync s3://monkey-training <raw_root> --no-sign-request "
    '--exclude "*" --include "images/pas-cpg/*" --include "annotations/json_mm/*"'
)
_IMAGES_SUBDIR = Path("images") / "pas-cpg"
_ANNOTATIONS_SUBDIR = Path("annotations") / "json_mm"


def _missing_input(message: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{message}. Download the MONKEY training data first:\n    {DOWNLOAD_COMMAND}"
    )


def parse_case_id(case_id: str) -> tuple[str, str]:
    """Map a MONKEY ``<centre>_P<patient>[_<slide>]`` case id to ``(centre, patient_id)``.

    The centre is the leading token; the patient is the first two underscore-joined
    tokens (``A_P000001``), so any extra trailing slide suffix still groups a patient's
    slides together. Ids without an underscore fall back to being their own patient.
    """
    tokens = case_id.split("_")
    centre = tokens[0]
    patient_id = "_".join(tokens[:2]) if len(tokens) >= 2 else case_id
    return centre, patient_id


def assign_patient_folds(patients: dict[str, str], *, num_folds: int = NUM_FOLDS) -> dict[str, int]:
    """Deal patients over ``num_folds`` folds, spreading each centre evenly.

    ``patients`` maps ``patient_id -> centre``. Centres are taken in sorted order and their
    patients in stable-SHA1 order; one running counter deals them round-robin, so each
    centre lands evenly on the folds and fold sizes differ by at most one patient.
    Returns ``patient_id -> fold`` and is a pure function of its inputs.
    """
    if num_folds < 3:
        raise ValueError(f"num_folds must be >= 3 (a test, a tune and a train fold); got {num_folds}.")
    if len(patients) < num_folds:
        raise ValueError(
            f"{len(patients)} patient(s) cannot fill {num_folds} folds; every fold needs "
            "a test patient."
        )
    by_centre: dict[str, list[str]] = defaultdict(list)
    for patient_id, centre in patients.items():
        by_centre[centre].append(patient_id)

    fold_of: dict[str, int] = {}
    position = 0
    for centre in sorted(by_centre):
        ordered = sorted(by_centre[centre], key=lambda p: (hashlib.sha1(p.encode()).hexdigest(), p))
        for patient_id in ordered:
            fold_of[patient_id] = position % num_folds
            position += 1
    return fold_of


def build_split_rows(
    sample_patients: list[tuple[str, str]],
    patient_fold: dict[str, int],
    *,
    num_folds: int = NUM_FOLDS,
) -> list[dict]:
    """One ``splits.csv`` row per sample per fold, rotated as in BEETLE.

    ``sample_patients`` lists ``(sample_id, patient_id)``. In fold ``k`` a sample whose
    patient is in fold ``k`` is ``test``, one in fold ``(k + 1) % num_folds`` is ``tune``,
    and the rest are ``train``.
    """
    rows: list[dict] = []
    for k in range(num_folds):
        tune_fold = (k + 1) % num_folds
        for sample_id, patient_id in sample_patients:
            fold = patient_fold[patient_id]
            split = "test" if fold == k else "tune" if fold == tune_fold else "train"
            rows.append({"sample_id": sample_id, "split": split, "fold": k})
    return rows


def _read_annotation(json_path: Path) -> dict:
    if not json_path.is_file():
        raise _missing_input(f"missing MONKEY annotation file {json_path}")
    return json.loads(json_path.read_text())


def _crop_window(
    polygon_px: np.ndarray, margin_px: float, slide_dims: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Level-0 ``(x0, y0, width, height)`` holding the polygon grown by ``margin_px``."""
    slide_w, slide_h = slide_dims
    x0 = max(0, math.floor(polygon_px[:, 0].min() - margin_px))
    y0 = max(0, math.floor(polygon_px[:, 1].min() - margin_px))
    x1 = min(slide_w, math.ceil(polygon_px[:, 0].max() + margin_px) + 1)
    y1 = min(slide_h, math.ceil(polygon_px[:, 1].max() + margin_px) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI polygon lies outside the {slide_w}x{slide_h} slide.")
    return x0, y0, x1 - x0, y1 - y0


def _rasterize_valid_region(
    polygon_local: np.ndarray, shape: tuple[int, int], margin_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, valid)`` rasters of the polygon and of its ``margin_px`` dilation.

    A pixel is inside when its centre is inside the polygon; pixel ``(r, c)`` is centred
    on ``(x=c, y=r)``, the frame :func:`~soma.detection.encode.points_on_valid` reads
    points in. A pixel is valid when its centre lies within ``margin_px`` of an inside
    pixel's centre.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import polygon as fill_polygon

    inside = np.zeros(shape, dtype=bool)
    rows, cols = fill_polygon(polygon_local[:, 1], polygon_local[:, 0], shape=shape)
    inside[rows, cols] = True
    if not inside.any():
        raise ValueError("ROI polygon covers no pixel centre.")
    valid = distance_transform_edt(~inside) <= margin_px
    return inside, valid


def _write_points_csv(path: Path, points: list[tuple[float, float, int]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "class"])
        for x, y, c in points:
            writer.writerow([x, y, c])


def curate_monkey_detection(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    spacing_at_level_0: float = MONKEY_SPACING_LEVEL0,
    valid_margin_um: float = VALID_MARGIN_UM,
    num_folds: int = NUM_FOLDS,
    backend: str = "auto",
) -> CuratedManifest:
    """Curate the public MONKEY training set into one detection sample per ROI.

    Args:
        raw_root: The ``aws s3 sync`` download: ``images/pas-cpg/<case>_PAS_CPG.tif`` and
            ``annotations/json_mm/<case>_{lymphocytes,monocytes}.json``.
        output_dir: Where ``images/``, ``points/`` and ``ignore_masks/`` (one file per
            ROI each), ``dataset.csv``, ``splits.csv`` and ``summary.json`` are written.
        spacing_at_level_0: Level-0 µm/px that converts the mm annotations to pixels.
            Each slide's own spacing must agree with it.
        valid_margin_um: The valid region is the ROI polygon dilated by this distance;
            points beyond it are dropped.
        num_folds: Number of patient cross-validation folds.
        backend: hs2p slide-reader backend used for the crops.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` for the generated files.
    """
    from hs2p.wsi.reader import open_slide

    raw_root = Path(raw_root)
    img_root = raw_root / _IMAGES_SUBDIR
    ann_root = raw_root / _ANNOTATIONS_SUBDIR
    for folder in (_IMAGES_SUBDIR, _ANNOTATIONS_SUBDIR):
        if not (raw_root / folder).is_dir():
            raise _missing_input(f"expected {folder.as_posix()}/ under the MONKEY root {raw_root}")
    if spacing_at_level_0 <= 0:
        raise ValueError(f"spacing_at_level_0 must be > 0, got {spacing_at_level_0}.")
    if valid_margin_um < 0:
        raise ValueError(f"valid_margin_um must be >= 0, got {valid_margin_um}.")
    mm_to_px = 1000.0 / float(spacing_at_level_0)  # mm -> µm (×1000) -> px (÷spacing)
    margin_px = float(valid_margin_um) / float(spacing_at_level_0)

    case_ids = sorted(p.name[: -len("_lymphocytes.json")] for p in ann_root.glob("*_lymphocytes.json"))
    if not case_ids:
        raise _missing_input(f"no MONKEY cases (*_lymphocytes.json) under {ann_root}")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    points_dir = output_dir / "points"
    masks_dir = output_dir / "ignore_masks"
    for folder in (images_dir, points_dir, masks_dir):
        folder.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict] = []
    patients: dict[str, str] = {}  # patient_id -> centre
    kept_per_class: Counter[int] = Counter()
    points_total = 0
    points_kept = 0
    kept_outside_polygon = 0
    area_rois_px = 0.0
    valid_area_px = 0
    num_empty = 0

    for case_id in case_ids:
        wsi_path = img_root / f"{case_id}_PAS_CPG.tif"
        if not wsi_path.is_file():
            raise _missing_input(f"missing PAS WSI for case {case_id}: {wsi_path}")
        centre, patient_id = parse_case_id(case_id)
        patients[patient_id] = centre

        docs = {
            name: _read_annotation(ann_root / f"{case_id}_{name}.json")
            for name in MONKEY_CLASS_NAMES
        }
        polygons_mm = [roi["polygon"] for roi in docs["lymphocytes"]["rois"]]
        if [roi["polygon"] for roi in docs["monocytes"]["rois"]] != polygons_mm:
            raise ValueError(f"case {case_id}: the lymphocyte and monocyte files list different ROIs.")
        if not polygons_mm:
            raise ValueError(f"case {case_id}: the annotation files list no ROI.")
        area_rois_px += float(docs["lymphocytes"]["area_rois"])

        # Every point of the case in level-0 px, with its class id.
        xy_parts, class_parts = [], []
        for class_name in MONKEY_CLASS_NAMES:
            pts_mm = [p["point"][:2] for p in docs[class_name]["points"]]
            xy_parts.append(np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2) * mm_to_px)
            class_parts.append(np.full(len(pts_mm), MONKEY_LABEL_REMAP[class_name], dtype=np.int64))
        case_xy = np.concatenate(xy_parts)
        case_class = np.concatenate(class_parts)
        points_total += len(case_xy)
        claimed = np.zeros(len(case_xy), dtype=bool)

        with open_slide(wsi_path, backend=backend) as slide:
            slide_spacing = float(slide.spacing)
            if abs(slide_spacing - spacing_at_level_0) > _SPACING_RTOL * spacing_at_level_0:
                raise ValueError(
                    f"{wsi_path} has level-0 spacing {slide_spacing} µm/px but the "
                    f"annotations are converted at {spacing_at_level_0} µm/px."
                )
            slide_dims = (int(slide.dimensions[0]), int(slide.dimensions[1]))

            for roi_index, polygon_mm in enumerate(polygons_mm):
                sample_id = f"{case_id}_roi{roi_index}"
                polygon_px = np.asarray(polygon_mm, dtype=np.float64)[:, :2] * mm_to_px
                x0, y0, width, height = _crop_window(polygon_px, margin_px, slide_dims)
                origin = np.array([x0, y0], dtype=np.float64)
                inside, valid = _rasterize_valid_region(
                    polygon_px - origin, (height, width), margin_px
                )

                local_xy = case_xy - origin
                on_valid = points_on_valid(local_xy, valid)
                if (on_valid & claimed).any():
                    raise ValueError(
                        f"{sample_id}: a point lies in the valid region of two ROIs; the "
                        "dilated ROI polygons must not overlap."
                    )
                claimed |= on_valid
                kept_outside_polygon += int((on_valid & ~points_on_valid(local_xy, inside)).sum())
                kept = [
                    (float(x), float(y), int(c))
                    for (x, y), c in zip(local_xy[on_valid], case_class[on_valid])
                ]
                kept_per_class.update(c for _, _, c in kept)
                num_empty += int(not kept)
                valid_area_px += int(valid.sum())

                # PNG level 1 writes ~3x faster than the default for ~6% more disk.
                crop = np.asarray(slide.read_region((x0, y0), 0, (width, height)))[..., :3]
                image_path = images_dir / f"{sample_id}.png"
                Image.fromarray(np.ascontiguousarray(crop)).save(image_path, compress_level=1)
                ignore = np.where(valid, 0, IGNORE_VALUE).astype(np.uint8)
                mask_path = masks_dir / f"{sample_id}.png"
                Image.fromarray(ignore).save(mask_path, compress_level=1)
                points_path = points_dir / f"{sample_id}.csv"
                _write_points_csv(points_path, kept)

                dataset_rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(image_path.resolve()),
                        "points_path": str(points_path.resolve()),
                        "ignore_mask_path": str(mask_path.resolve()),
                        "patient_id": patient_id,
                        "spacing_at_level_0": float(spacing_at_level_0),
                        "centre": centre,
                        "source_slide": wsi_path.relative_to(raw_root).as_posix(),
                        "crop_x": x0,
                        "crop_y": y0,
                    }
                )
        points_kept += int(claimed.sum())

    patient_fold = assign_patient_folds(patients, num_folds=num_folds)
    split_rows = build_split_rows(
        [(row["sample_id"], row["patient_id"]) for row in dataset_rows],
        patient_fold,
        num_folds=num_folds,
    )

    px_to_mm2 = float(spacing_at_level_0) ** 2 / 1_000_000.0
    centres_per_fold: dict[int, Counter[str]] = {k: Counter() for k in range(num_folds)}
    for patient_id, fold in patient_fold.items():
        centres_per_fold[fold][patients[patient_id]] += 1
    rois_per_fold = Counter(patient_fold[row["patient_id"]] for row in dataset_rows)

    summary = {
        "dataset": "MONKEY (kidney-biopsy inflammatory-cell detection)",
        "dataset_type": "detection",
        "source": "s3://monkey-training (images/pas-cpg, annotations/json_mm)",
        "stain": "PAS",
        "num_classes": MONKEY_NUM_CLASSES,
        "class_names": list(MONKEY_CLASS_NAMES),
        "mnl_merged_class": "inflammatory-cells",
        "native_metric": "FROC",
        "spacing_at_level_0": float(spacing_at_level_0),
        "total_slides": len(case_ids),
        "total_patients": len(patients),
        "total_rois": len(dataset_rows),
        "num_empty": num_empty,
        "valid_margin_um": float(valid_margin_um),
        "points_total": points_total,
        "points_kept": points_kept,
        "points_dropped": points_total - points_kept,
        "points_kept_outside_polygon": kept_outside_polygon,
        "points_per_class": {
            MONKEY_CLASS_NAMES[c]: kept_per_class[c] for c in range(MONKEY_NUM_CLASSES)
        },
        "area_rois_px": area_rois_px,
        "area_rois_mm2": area_rois_px * px_to_mm2,
        "valid_area_px": valid_area_px,
        "valid_area_mm2": valid_area_px * px_to_mm2,
        "num_folds": num_folds,
        "fold_rotation": "fold k: test = fold k, tune = fold (k + 1) % num_folds, train = the rest",
        "patients_per_fold": {
            str(k): dict(sorted(counts.items())) for k, counts in centres_per_fold.items()
        },
        "rois_per_fold": {str(k): rois_per_fold[k] for k in range(num_folds)},
        "leaderboard_comparable": False,
        "split_note": (
            "Patient cross-validation over the public training data; the official MONKEY "
            "test set is hidden, so these numbers are NOT comparable to the published "
            "leaderboard (show it only as a reference band)."
        ),
    }
    return write_manifest(
        output_dir,
        dataset_type="detection",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m soma.curation.monkey",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw-root", type=Path, required=True, help="MONKEY download root")
    ap.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    ap.add_argument(
        "--spacing-at-level-0",
        type=float,
        default=MONKEY_SPACING_LEVEL0,
        help="level-0 µm/px that converts the mm annotations to pixels",
    )
    ap.add_argument("--valid-margin-um", type=float, default=VALID_MARGIN_UM)
    ap.add_argument("--num-folds", type=int, default=NUM_FOLDS)
    ap.add_argument("--backend", default="auto", help="hs2p slide-reader backend")
    args = ap.parse_args(argv)
    manifest = curate_monkey_detection(
        args.raw_root,
        args.output_dir,
        spacing_at_level_0=args.spacing_at_level_0,
        valid_margin_um=args.valid_margin_um,
        num_folds=args.num_folds,
        backend=args.backend,
    )
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")


if __name__ == "__main__":
    main()
