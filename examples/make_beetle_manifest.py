"""Emit a SLIDE-LEVEL BEETLE manifest + splits for soma's slide-manifest segmentation path.

The slide-level counterpart of ``data/beetle/curate_beetle.py``: instead of cropping tiles,
it pairs each development WSI with its multiresolution annotation mask (reusing the exact
pairing + fold logic from ``curate_beetle.py``) and writes one row per slide —
``sample_id, image_path (WSI), mask_path (annotation raster)`` — plus slide-level CV splits.
soma then runs hs2p annotation sampling over these slides (``masks:``/``sampling:`` in
``examples/segmentation_beetle.yaml``), so no tiles are materialized here.

Splits preserve BEETLE's 5 predefined CV folds at the SLIDE level (soma never partitions;
the sampled ROIs inherit their parent slide's split). For fold ``k``: a slide whose
``validation_fold == k`` is ``test``, ``== (k+1) % 5`` is ``tune``, else ``train`` — the
same rotation ``curate_beetle.py`` applies to tiles, so the slide assignment matches the
standalone baseline.

Outputs (under ``--out``, default ``data/beetle/curated_slide_manifest/`` — gitignored):
``manifest.csv``, ``splits.csv``, ``coverage.csv`` (per-slide per-class annotation
coverage, informs split authoring), ``summary.json``.

Run (after ``data/beetle/download.sh`` populated ``images/`` + ``annotations/``)::

    python examples/make_beetle_manifest.py                  # all dev WSIs with image+mask on disk
    python examples/make_beetle_manifest.py --slides 4       # tiny subset (smoke)
    python examples/make_beetle_manifest.py --no-coverage    # skip the (slow) coverage scan
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Prefer the in-repo soma over any stale installed copy (the coverage scan uses
# soma.curation.segmentation_coverage, which an older site-packages build may lack).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BEETLE_ROOT = _REPO_ROOT / "data" / "beetle"
OVERVIEW_CSV = BEETLE_ROOT / "data_overview.csv"
DEFAULT_OUT = BEETLE_ROOT / "curated_slide_manifest"

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
ANNOTATED_FRACTION_MIN = 0.05  # >=5% annotated to keep a tile (curate_beetle.py)
CANONICAL_MPP = 0.5
CROP_SIZE = 512
N_FOLDS = 5


def read_usable_dev_rows() -> list[dict]:
    """Dev rows whose WSI AND mask both exist on disk (curate_beetle.py's pairing rule).

    A blank ``wsi_path`` makes ``ROOT / ""`` resolve to ROOT (a real dir), so guard on a
    non-empty path + ``is_file()`` — the same guard curate_beetle.py uses.
    """
    rows = [r for r in csv.DictReader(OVERVIEW_CSV.open()) if r["split"] == "development"]
    if not rows:
        raise RuntimeError(f"No development rows found in {OVERVIEW_CSV}")

    def has(rel: str) -> bool:
        return bool(rel.strip()) and (BEETLE_ROOT / rel).is_file()

    usable = [r for r in rows if has(r["wsi_path"]) and has(r["annotation_mask_path"])]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"Note: {dropped}/{len(rows)} dev WSIs lack an image on disk — using {len(usable)}.")
    if not usable:
        raise RuntimeError("No development WSIs have both image and mask on disk.")
    return usable


def select_subset(rows: list[dict], n: int | None) -> list[dict]:
    """Deterministically pick ~n dev WSIs balanced across folds (curate_beetle.py's scheme)."""
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
    return selected[:n] if n is not None else selected


def build_manifest_rows(rows: list[dict]) -> list[dict]:
    manifest: list[dict] = []
    for r in rows:
        wsi = (BEETLE_ROOT / r["wsi_path"]).resolve()
        mask = (BEETLE_ROOT / r["annotation_mask_path"]).resolve()
        manifest.append(
            {
                "sample_id": r["name"],
                "image_path": str(wsi),
                "mask_path": str(mask),
                "patient_id": r["patient_id"],
                "source": r["source"],
                "specimen_type": r["specimen_type"],
                "validation_fold": r["validation_fold"],
            }
        )
    return manifest


def build_split_rows(manifest: list[dict]) -> list[dict]:
    """Slide-level CV splits matching curate_beetle.py's fold rotation (test/tune/train)."""
    fold_ids = sorted({r["validation_fold"] for r in manifest})
    fold_nums = sorted(int(f.replace("fold", "")) for f in fold_ids)
    n_folds = len(fold_nums)
    split_rows: list[dict] = []
    for k in fold_nums:
        tune_fold = (k + 1) % n_folds
        for r in manifest:
            wf = int(r["validation_fold"].replace("fold", ""))
            if wf == k:
                split = "test"
            elif wf == tune_fold:
                split = "tune"
            else:
                split = "train"
            split_rows.append({"sample_id": r["sample_id"], "split": split, "fold": k})
    return split_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_coverage(manifest_csv: Path, out_path: Path) -> None:
    """Per-slide per-class annotation coverage (soma.curation.segmentation_coverage)."""
    from soma.curation.segmentation_coverage import summarize_coverage, write_coverage_csv

    min_cov = {k: ANNOTATED_FRACTION_MIN for k in PIXEL_MAPPING if k != "background"}
    coverage = summarize_coverage(
        manifest_csv,
        pixel_mapping=PIXEL_MAPPING,
        min_coverage=min_cov,
        tile_size_px=CROP_SIZE,
        spacing_um=CANONICAL_MPP,
    )
    write_coverage_csv(out_path, coverage)
    print(f"Wrote coverage for {len(coverage)} slides to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--slides", type=int, help="number of development WSIs (default: all on disk)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--no-coverage", action="store_true", help="skip the per-slide coverage scan")
    args = ap.parse_args()

    rows = read_usable_dev_rows()
    chosen = select_subset(rows, args.slides)
    manifest = build_manifest_rows(chosen)
    splits = build_split_rows(manifest)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_csv = out / "manifest.csv"
    write_csv(
        manifest_csv,
        ["sample_id", "image_path", "mask_path", "patient_id", "source", "specimen_type", "validation_fold"],
        manifest,
    )
    write_csv(out / "splits.csv", ["sample_id", "split", "fold"], splits)

    summary = {
        "num_slides": len(manifest),
        "num_classes": len(PIXEL_MAPPING) - 1,
        "ignore_index": 255,
        "pixel_mapping": PIXEL_MAPPING,
        "annotated_fraction_min": ANNOTATED_FRACTION_MIN,
        "canonical_mpp": CANONICAL_MPP,
        "crop_size": CROP_SIZE,
        "cv_folds": sorted({int(r["validation_fold"].replace("fold", "")) for r in manifest}),
        "slides_per_fold": dict(Counter(r["validation_fold"] for r in manifest)),
        "slides_per_source": dict(Counter(r["source"] for r in manifest)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote slide manifest ({len(manifest)} slides) + splits to {out}")

    if not args.no_coverage:
        write_coverage(manifest_csv, out / "coverage.csv")


if __name__ == "__main__":
    main()
