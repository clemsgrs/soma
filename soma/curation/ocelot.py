"""Curator for the OCELOT 2023 cell-detection dataset.

Like the EVA curators, this emits Soma's standard ``dataset.csv`` + ``splits.csv``
manifests from a locally prepared raw dataset and **does not download anything**:
OCELOT requires accepting the Zenodo terms first (see ``examples/ocelot/README.md``
for the download step).

OCELOT (Zenodo record 8417503, ``ocelot2023_v1.0.1.zip``) ships paired *cell* and
*tissue* patches cropped from TCGA WSIs. detection-v1 uses the **cell** patches only:
1024x1024 RGB JPEGs at ~0.2 µm/px, each paired with a headerless point CSV
``x,y,label`` where ``label`` is OCELOT's 1-based cell class::

    1 = BC (background cell, i.e. every non-tumor cell)
    2 = TC (tumor cell)

Soma's :class:`~soma.tasks.detection.DetectionHead` wants **0-based** class ids, so we
remap ``1 -> 0`` (BC) and ``2 -> 1`` (TC) and write one ``x,y,class`` point CSV per
sample under ``points/``. Points stay in the cell-patch pixel frame (0..1023) because
the JPEGs are referenced at native resolution. The curator therefore declares each
source image's physical scale as ``spacing_at_level_0`` in the Manifest.

OCELOT's own train/val/test split maps onto Soma's roles as train -> ``train``,
val -> ``tune`` (the threshold-sweep / monitor split), test -> ``test``. Soma never
partitions data itself; the split membership is emitted verbatim as a single fold.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from soma.curation.manifest import CuratedManifest, write_manifest

# OCELOT 1-based cell label -> Soma 0-based class id.
OCELOT_LABEL_REMAP = {1: 0, 2: 1}  # 1=BC -> 0, 2=TC -> 1
OCELOT_CELL_CLASS_NAMES = ("BC", "TC")  # index = 0-based class id
OCELOT_NUM_CLASSES = len(OCELOT_CELL_CLASS_NAMES)

# Native µm/px of the OCELOT cell patches (1024x1024 JPEGs at ~0.2 µm/px).
OCELOT_NATIVE_SPACING_UM = 0.2

# OCELOT split name -> Soma split role.
OCELOT_SPLIT_ROLE = {"train": "train", "val": "tune", "test": "test"}


def _read_ocelot_points(csv_path: Path) -> list[tuple[float, float, int]]:
    """Read a headerless OCELOT ``x,y,label`` CSV, remap to 0-based ``(x, y, class)``."""
    rows: list[tuple[float, float, int]] = []
    with csv_path.open(newline="") as fh:
        for line in csv.reader(fh):
            if not line or not line[0].strip():
                continue
            x, y, label = float(line[0]), float(line[1]), int(float(line[2]))
            if label not in OCELOT_LABEL_REMAP:
                raise ValueError(
                    f"{csv_path}: unexpected cell label {label!r} (expected 1=BC or 2=TC)"
                )
            rows.append((x, y, OCELOT_LABEL_REMAP[label]))
    return rows


def curate_ocelot_detection(
    raw_root: str | Path,
    output_dir: str | Path,
) -> CuratedManifest:
    """Curate the OCELOT 2023 cell-detection dataset into Soma's detection format.

    Args:
        raw_root: The unzipped ``ocelot2023_v1.0.1`` directory, i.e. the parent of
            ``images/`` and ``annotations/``.
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, the per-sample
            ``points/<sample_id>.csv`` files, and ``summary.json`` are written.
    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` pointing at the generated
        ``dataset.csv`` and ``splits.csv``.
    """
    raw_root = Path(raw_root)
    images_root = raw_root / "images"
    ann_root = raw_root / "annotations"
    if not images_root.is_dir() or not ann_root.is_dir():
        raise FileNotFoundError(
            f"expected images/ and annotations/ under the unzipped OCELOT root: {raw_root}"
        )

    output_dir = Path(output_dir)
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict] = []
    split_rows: list[dict] = []  # sample_id, split, fold
    per_split: dict[str, dict] = {}

    for ocelot_split, role in OCELOT_SPLIT_ROLE.items():
        cell_img_dir = images_root / ocelot_split / "cell"
        cell_ann_dir = ann_root / ocelot_split / "cell"
        if not cell_img_dir.is_dir():
            continue
        class_counter: Counter[int] = Counter()
        n_samples = n_empty = 0
        for img_path in sorted(cell_img_dir.glob("*.jpg")):
            stem = img_path.stem  # e.g. "001"
            ann_path = cell_ann_dir / f"{stem}.csv"
            if not ann_path.exists():
                raise FileNotFoundError(f"missing annotation for {img_path}: {ann_path}")
            sample_id = f"{ocelot_split}_{stem}"
            pts = _read_ocelot_points(ann_path)
            out_csv = points_dir / f"{sample_id}.csv"
            with out_csv.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["x", "y", "class"])
                for x, y, c in pts:
                    w.writerow([x, y, c])
                    class_counter[c] += 1
            n_samples += 1
            n_empty += int(len(pts) == 0)
            row = {
                "sample_id": sample_id,
                "image_path": str(img_path.resolve()),
                "points_path": str(out_csv.resolve()),
                "spacing_at_level_0": OCELOT_NATIVE_SPACING_UM,
            }
            dataset_rows.append(row)
            split_rows.append({"sample_id": sample_id, "split": role, "fold": 0})
        per_split[role] = {
            "ocelot_split": ocelot_split,
            "num_samples": n_samples,
            "num_empty": n_empty,
            "points_per_class": {
                OCELOT_CELL_CLASS_NAMES[c]: class_counter[c] for c in range(OCELOT_NUM_CLASSES)
            },
        }

    if not dataset_rows:
        raise ValueError(f"No OCELOT cell patches found under {images_root}")

    summary = {
        "dataset": "OCELOT 2023 (cell detection)",
        "dataset_type": "detection",
        "num_classes": OCELOT_NUM_CLASSES,
        "class_names": list(OCELOT_CELL_CLASS_NAMES),
        "label_remap": {str(k): v for k, v in OCELOT_LABEL_REMAP.items()},
        "native_spacing_um": OCELOT_NATIVE_SPACING_UM,
        "total_samples": len(dataset_rows),
        "splits": per_split,
    }
    return write_manifest(
        output_dir,
        dataset_type="detection",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-root", type=Path, required=True, help="unzipped ocelot2023_v1.0.1 dir"
    )
    ap.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    args = ap.parse_args()

    manifest = curate_ocelot_detection(args.raw_root, args.output_dir)
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")


if __name__ == "__main__":
    main()
