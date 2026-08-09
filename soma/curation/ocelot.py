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
sample under ``points/``. Points stay in the cell-patch pixel frame (0..1023); because
the JPEGs are read flat at native resolution. The curator therefore declares each
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

from PIL import Image

from soma.curation.manifest import CuratedManifest, write_manifest

# OCELOT 1-based cell label -> Soma 0-based class id.
OCELOT_LABEL_REMAP = {1: 0, 2: 1}  # 1=BC -> 0, 2=TC -> 1
OCELOT_CELL_CLASS_NAMES = ("BC", "TC")  # index = 0-based class id
OCELOT_NUM_CLASSES = len(OCELOT_CELL_CLASS_NAMES)

# Native µm/px of the OCELOT cell patches (1024x1024 JPEGs at ~0.2 µm/px). The render
# variant materializes a coarser spacing at curation time because the dense reader reads
# flat JPEGs with PIL and ignores any requested spacing.
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


def _render_patch(src: Path, dst: Path, factor: float) -> None:
    """Downsample ``src`` by ``factor`` (area resampling) and write it to ``dst``."""
    with Image.open(src) as image:
        image = image.convert("RGB")
        w, h = image.size
        new_size = (max(1, round(w / factor)), max(1, round(h / factor)))
        image.resize(new_size, resample=Image.Resampling.BOX).save(dst)


def curate_ocelot_detection(
    raw_root: str | Path,
    output_dir: str | Path,
    render_spacing_um: float | None = None,
) -> CuratedManifest:
    """Curate the OCELOT 2023 cell-detection dataset into Soma's detection format.

    Args:
        raw_root: The unzipped ``ocelot2023_v1.0.1`` directory, i.e. the parent of
            ``images/`` and ``annotations/``.
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, the per-sample
            ``points/<sample_id>.csv`` files, and ``summary.json`` are written.
        render_spacing_um: Optional target µm/px at which to materialize the cell
            patches. When ``None`` (default), patches are referenced in place at their
            native :data:`OCELOT_NATIVE_SPACING_UM`. When
            set, every patch is downsampled by ``render_spacing_um / native`` into
            ``output_dir/images/`` and its point coordinates are scaled by the same
            factor, so annotations stay on the cells; the manifest then carries a
            ``spacing_at_level_0`` declaration equal to ``render_spacing_um`` per sample. This
            materialization is required because the dense reader reads flat JPEGs with PIL
            and ignores any requested spacing, so magnification must be baked in here.

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

    if render_spacing_um is not None:
        render_spacing_um = float(render_spacing_um)
        if render_spacing_um < OCELOT_NATIVE_SPACING_UM:
            raise ValueError(
                f"render_spacing_um={render_spacing_um} is finer than the native "
                f"{OCELOT_NATIVE_SPACING_UM} µm/px; flat JPEGs can only be downsampled"
            )
    factor = render_spacing_um / OCELOT_NATIVE_SPACING_UM if render_spacing_um is not None else None

    output_dir = Path(output_dir)
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    rendered_img_dir = output_dir / "images"
    if factor is not None:
        rendered_img_dir.mkdir(parents=True, exist_ok=True)

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
            if factor is not None:
                pts = [(x / factor, y / factor, c) for x, y, c in pts]
            out_csv = points_dir / f"{sample_id}.csv"
            with out_csv.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["x", "y", "class"])
                for x, y, c in pts:
                    w.writerow([x, y, c])
                    class_counter[c] += 1
            if factor is not None:
                rendered_path = rendered_img_dir / f"{sample_id}.jpg"
                _render_patch(img_path, rendered_path, factor)
                image_path = rendered_path
            else:
                image_path = img_path
            n_samples += 1
            n_empty += int(len(pts) == 0)
            row = {
                "sample_id": sample_id,
                "image_path": str(image_path.resolve()),
                "points_path": str(out_csv.resolve()),
                "spacing_at_level_0": (
                    render_spacing_um
                    if render_spacing_um is not None
                    else OCELOT_NATIVE_SPACING_UM
                ),
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
        "render_spacing_um": render_spacing_um,
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
    ap.add_argument(
        "--render-spacing-um",
        type=float,
        default=None,
        help=(
            "optional target µm/px to materialize the patches at (>= native "
            f"{OCELOT_NATIVE_SPACING_UM}); omit to reference native patches in place"
        ),
    )
    args = ap.parse_args()

    manifest = curate_ocelot_detection(
        args.raw_root, args.output_dir, render_spacing_um=args.render_spacing_um
    )
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")


if __name__ == "__main__":
    main()
