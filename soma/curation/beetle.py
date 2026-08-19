"""Curator for the BEETLE breast-cancer segmentation challenge (slide-manifest path).

Like the EVA and OCELOT curators, this emits Soma's unified Manifest
(``dataset.csv`` + ``splits.csv`` + ``summary.json``) from locally prepared raw data and
**does not download anything** (see ``data/beetle/download.sh``).

BEETLE is a *segmentation* dataset, so the supervision column is ``label_mask_path``: one row per
development WSI pairs the slide (``image_path``) with its multiresolution annotation raster
(``label_mask_path``). No tiles are materialized here — soma runs hs2p annotation sampling over
these slides at train time (``masks:`` / ``sampling:`` in ``examples/segmentation_beetle.yaml``)
to derive ROIs, so the cached slide-manifest path is the sole BEETLE recipe.

Splits preserve BEETLE's predefined CV folds at the SLIDE level (soma never partitions; the
sampled ROIs inherit their parent slide's split). For fold ``k``: a slide whose
``validation_fold == k`` is ``test``, ``== (k+1) % n_folds`` is ``tune``, else ``train``.

Run as ``python -m soma.curation.beetle`` (or via ``examples/make_beetle_manifest.py``).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from soma.curation.manifest import CuratedManifest, write_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BEETLE_ROOT = _REPO_ROOT / "data" / "beetle"

# BEETLE pixel vocabulary (label_map.json): name -> raw mask pixel value. This is the
# masks.pixel_mapping the soma config carries; "background" (0, unannotated) is the ignore
# label. Kept here so the manifest's coverage scan uses the same class scheme as training.
PIXEL_MAPPING = {
    "background": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}
ANNOTATED_FRACTION_MIN = 0.05  # >=5% annotated to keep a tile
CANONICAL_MPP = 0.5
CROP_SIZE = 512
IGNORE_INDEX = 255


def read_usable_dev_rows(overview_csv: Path, beetle_root: Path) -> list[dict]:
    """Dev rows whose WSI AND mask both exist on disk (the pairing rule).

    A blank ``wsi_path`` makes ``ROOT / ""`` resolve to ROOT (a real dir), so guard on a
    non-empty path + ``is_file()``.
    """
    rows = [r for r in csv.DictReader(overview_csv.open()) if r["split"] == "development"]
    if not rows:
        raise RuntimeError(f"No development rows found in {overview_csv}")

    def has(rel: str) -> bool:
        return bool(rel.strip()) and (beetle_root / rel).is_file()

    usable = [r for r in rows if has(r["wsi_path"]) and has(r["annotation_mask_path"])]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"Note: {dropped}/{len(rows)} dev WSIs lack an image on disk — using {len(usable)}.")
    if not usable:
        raise RuntimeError("No development WSIs have both image and mask on disk.")
    return usable


def select_subset(rows: list[dict], n: int | None) -> list[dict]:
    """Deterministically pick ~n dev WSIs balanced across folds."""
    if n is None:
        return rows
    by_fold: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fold[r["validation_fold"]].append(r)
    folds = sorted(by_fold)
    per_fold = max(1, round(n / len(folds)))
    selected: list[dict] = []
    for fold in folds:
        bucket = sorted(by_fold[fold], key=lambda r: (r["source"], r["name"]))
        selected.extend(bucket[:per_fold])
    return selected[:n]


def build_dataset_rows(rows: list[dict], beetle_root: Path) -> list[dict]:
    """One unified-schema dataset row per slide (supervision column = ``label_mask_path``)."""
    dataset_rows: list[dict] = []
    for r in rows:
        wsi = (beetle_root / r["wsi_path"]).resolve()
        mask = (beetle_root / r["annotation_mask_path"]).resolve()
        dataset_rows.append(
            {
                "sample_id": r["name"],
                "image_path": str(wsi),
                "label_mask_path": str(mask),
                "patient_id": r["patient_id"],
                "source": r["source"],
                "specimen_type": r["specimen_type"],
                "validation_fold": r["validation_fold"],
            }
        )
    return dataset_rows


def build_split_rows(dataset_rows: list[dict]) -> list[dict]:
    """Slide-level CV splits from BEETLE's fold rotation (test/tune/train)."""
    fold_ids = sorted({r["validation_fold"] for r in dataset_rows})
    fold_nums = sorted(int(f.replace("fold", "")) for f in fold_ids)
    n_folds = len(fold_nums)
    split_rows: list[dict] = []
    for k in fold_nums:
        tune_fold = (k + 1) % n_folds
        for r in dataset_rows:
            wf = int(r["validation_fold"].replace("fold", ""))
            if wf == k:
                split = "test"
            elif wf == tune_fold:
                split = "tune"
            else:
                split = "train"
            split_rows.append({"sample_id": r["sample_id"], "split": split, "fold": k})
    return split_rows


def _write_coverage(dataset_csv: Path, out_path: Path) -> None:
    """Per-slide per-class annotation coverage (soma.curation.segmentation_coverage)."""
    from soma.curation.segmentation_coverage import summarize_coverage, write_coverage_csv

    min_cov = {k: ANNOTATED_FRACTION_MIN for k in PIXEL_MAPPING if k != "background"}
    coverage = summarize_coverage(
        dataset_csv,
        pixel_mapping=PIXEL_MAPPING,
        min_coverage=min_cov,
        tile_size_px=CROP_SIZE,
        spacing_um=CANONICAL_MPP,
    )
    write_coverage_csv(out_path, coverage)
    print(f"Wrote coverage for {len(coverage)} slides to {out_path}")


def curate_beetle_slide_manifest(
    overview_csv: str | Path,
    beetle_root: str | Path,
    output_dir: str | Path,
    *,
    slides: int | None = None,
    coverage: bool = False,
) -> CuratedManifest:
    """Curate the BEETLE development slides into Soma's unified segmentation Manifest.

    Args:
        overview_csv: ``data_overview.csv`` listing every BEETLE slide + annotation.
        beetle_root: Root the CSV's relative ``wsi_path`` / ``annotation_mask_path`` resolve against.
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, ``summary.json`` (and,
            when ``coverage`` is set, ``coverage.csv``) are written.
        slides: Optionally cap the number of dev WSIs (balanced across folds) for smoke runs.
        coverage: When ``True``, also scan per-slide per-class annotation coverage
            (requires hs2p + readable WSIs); informs split authoring only.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` for the generated files.
    """
    overview_csv = Path(overview_csv)
    beetle_root = Path(beetle_root)

    rows = read_usable_dev_rows(overview_csv, beetle_root)
    chosen = select_subset(rows, slides)
    dataset_rows = build_dataset_rows(chosen, beetle_root)
    split_rows = build_split_rows(dataset_rows)

    summary = {
        "dataset": "BEETLE (breast-cancer segmentation, slide manifest)",
        "dataset_type": "segmentation",
        "num_slides": len(dataset_rows),
        "num_classes": len(PIXEL_MAPPING) - 1,
        "ignore_index": IGNORE_INDEX,
        "pixel_mapping": PIXEL_MAPPING,
        "annotated_fraction_min": ANNOTATED_FRACTION_MIN,
        "canonical_mpp": CANONICAL_MPP,
        "crop_size": CROP_SIZE,
        "cv_folds": sorted({int(r["validation_fold"].replace("fold", "")) for r in dataset_rows}),
        "slides_per_fold": dict(Counter(r["validation_fold"] for r in dataset_rows)),
        "slides_per_source": dict(Counter(r["source"] for r in dataset_rows)),
    }

    manifest = write_manifest(
        output_dir,
        dataset_type="segmentation",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )
    print(f"Wrote slide manifest ({len(dataset_rows)} slides) + splits to {output_dir}")

    if coverage:
        _write_coverage(manifest.dataset_csv, Path(output_dir) / "coverage.csv")

    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m soma.curation.beetle",
        description="Curate the BEETLE development slides into a Soma segmentation Manifest.",
    )
    parser.add_argument(
        "--beetle-root",
        type=Path,
        default=_DEFAULT_BEETLE_ROOT,
        help="root the overview CSV's relative paths resolve against",
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=None,
        help="data_overview.csv (default: <beetle-root>/data_overview.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <beetle-root>/curated_slide_manifest)",
    )
    parser.add_argument("--slides", type=int, default=None, help="cap dev WSIs (balanced across folds)")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="also run the (slow) per-slide annotation coverage scan",
    )
    args = parser.parse_args(argv)

    overview_csv = args.overview_csv or (args.beetle_root / "data_overview.csv")
    out = args.out or (args.beetle_root / "curated_slide_manifest")
    curate_beetle_slide_manifest(
        overview_csv,
        args.beetle_root,
        out,
        slides=args.slides,
        coverage=args.coverage,
    )


if __name__ == "__main__":
    main()
