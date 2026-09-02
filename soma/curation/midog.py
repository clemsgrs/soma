"""Curator for the MIDOG 2022 mitosis-detection dataset.

Like the OCELOT and EVA curators, this emits Soma's unified Manifest (``dataset.csv`` +
``splits.csv`` + ``summary.json``) from a locally prepared raw dataset and **does not
download anything** (MIDOG 2022 ships via grand-challenge / the DeepMicroscopy group).

MIDOG 2022 is the Mitosis DOmain Generalization challenge: whole-region H&E images from
several tumor types digitized on several scanners (the challenge's *domain* axis), each
paired with point/box annotations of **mitotic figures** (the single rare positive class)
alongside hard-negative "not-mitotic-figure" imposters. The raw annotations are a
**COCO-format** JSON (``MIDOG2022_training.json``)::

    {
      "categories":  [{"id": 1, "name": "mitotic figure"},
                      {"id": 2, "name": "not mitotic figure"}],
      "images":      [{"id": 1, "file_name": "001.tiff", "width": W, "height": H,
                       "tumortype": "...", "scanner": "...", "patient_id": "...",
                       "spacing": 0.23}, ...],
      "annotations": [{"id": .., "image_id": 1, "category_id": 1,
                       "bbox": [x1, y1, x2, y2]}, ...]   # MIDOG: corner boxes (xyxy)
    }

Each mitosis box is a fixed ~50 px **corner** box ``[x1, y1, x2, y2]`` (``bbox_format="xyxy"``,
the curator default) — *not* COCO ``[x, y, w, h]``. Every resolved centre is checked against
its image bounds. The size and bounds checks catch common wrong-format cases, but cannot
infer the convention when both interpretations happen to be geometrically plausible; the
canonical MIDOG path therefore fixes ``xyxy`` rather than relying on inference.

Soma's detection path wants **0-based** class ids and a per-sample ``x,y,class`` point CSV,
so each mitotic-figure box is reduced to its **centre** and written as class ``0`` under
``points/``; the hard-negative imposters (a distinct category) are excluded because mitosis
is a single-class detection task. Per-image tumor-type / scanner metadata is carried into
the manifest so a robustness sub-analysis can stratify detection F1 by domain.

**Splits.** MIDOG 2022's official test set is **held out**, so this curator carves a
*fixed*, **patient/domain-stratified** local held-out split (train / tune / test) from the
public labeled data. The assignment is deterministic (stable SHA1 ordering within each
domain, all of a patient's images kept together), so re-curation is byte-identical. These
numbers are **not comparable** to the published MIDOG 2022 leaderboard (different test
set) — render the leaderboard as a reference band only.

``spacing_at_level_0`` (µm/px of the stored image's level-0 frame) is emitted when known —
either from a per-image ``spacing`` in the JSON (MIDOG scanners differ) or from the
``spacing_at_level_0`` argument — so extraction can resolve the source's physical scale.
The detection benchmark instead passes ``force_spacing_at_level_0=0.25`` to express its
explicit single-nominal-spacing approximation while retaining every ROI in its native
pixel grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from soma.curation.manifest import CuratedManifest, write_manifest

# The single positive class. MIDOG's COCO export canonically uses category id 1 for the
# mitotic figure; the curator still resolves the id by name (robust to re-numbering) and
# records the resolved value in the summary.
MIDOG_CLASS_NAMES = ("mitotic figure",)
MIDOG_NUM_CLASSES = len(MIDOG_CLASS_NAMES)
MIDOG_MITOTIC_CATEGORY_ID = 1

# A resolved mitosis centre this far *outside* its image is a misplacement (wrong
# ``bbox_format`` -> centres land ~2x out, thousands of px past the edge; or a small-image
# xyxy-as-xywh read -> tens of px out). MIDOG 2022's genuine annotations sit at most ~2 px
# past an edge (a single edge mitosis in the labeled set), so an 8 px tolerance clamps that
# real sub-pixel overhang while still catching gross misplacements. Beyond it the curator
# raises rather than silently accepting points the median-side guard can miss.
_MAX_CENTER_OVERSHOOT_PX = 8.0

# Default local held-out split fractions (patient/domain-stratified). Train = the rest.
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_TUNE_FRACTION = 0.2

_DEFAULT_ANNOTATIONS_NAME = "MIDOG2022_training.json"

_NON_LEADERBOARD_NOTE = (
    "Local held-out split carved from MIDOG 2022 public training data; the official test "
    "set is held out, so these numbers are NOT comparable to the published MIDOG 2022 "
    "leaderboard (different test set). Render the leaderboard as a reference band only."
)


def _resolve_mitotic_category_id(categories: list[dict]) -> int:
    """Return the COCO category id of the mitotic-figure (positive) class.

    Resolved by name so a re-numbered export still works: the positive class is the
    category whose name mentions "mitot" and is not the "not mitotic figure" imposter.
    """
    positives = [
        int(c["id"])
        for c in categories
        if "mitot" in str(c["name"]).lower() and not str(c["name"]).lower().strip().startswith("not")
    ]
    if len(positives) != 1:
        names = [str(c.get("name")) for c in categories]
        raise ValueError(
            f"expected exactly one mitotic-figure category in MIDOG categories {names}, "
            f"found {len(positives)}."
        )
    return positives[0]


def _validate_bbox_sizes(
    annotations: list[dict], mitotic_id: int, bbox_format: str, *, max_median_side: float = 400.0
) -> None:
    """Guard against common wrong-``bbox_format`` failures and implausible boxes.

    MIDOG annotates each mitosis with a fixed ~50 px box encoded as **corner**
    coordinates ``[x1, y1, x2, y2]``. Reading those as COCO ``[x, y, w, h]`` makes the
    "width" the far corner (~thousands of px), so the computed centre lands ~2x out —
    past the image edge — and the error is otherwise silent. This raises loudly if the
    resolved boxes are implausible for a mitosis (median side ``<= 0`` or very large).
    """
    import statistics

    sides: list[float] = []
    for ann in annotations:
        if int(ann["category_id"]) != mitotic_id:
            continue
        x0, y0, a, b = (float(v) for v in ann["bbox"][:4])
        w, h = (a, b) if bbox_format == "xywh" else (a - x0, b - y0)
        sides.extend((w, h))
    if not sides:
        return
    med = statistics.median(sides)
    if med <= 0 or med > max_median_side:
        raise ValueError(
            f"MIDOG mitosis boxes have an implausible median side {med:.0f}px under "
            f"bbox_format={bbox_format!r} (expected ~50px). The MIDOG COCO uses corner "
            f"boxes [x1,y1,x2,y2] — pass bbox_format='xyxy'. A wrong format silently "
            f"mis-places every point past the image edge. (Raise max_median_side if your "
            f"data genuinely has large boxes.)"
        )


def _bbox_center(bbox: list[float], bbox_format: str) -> tuple[float, float]:
    """Centre of a COCO ``[x, y, w, h]`` (``xywh``) or corner ``[x1, y1, x2, y2]`` box."""
    x0, y0, a, b = (float(v) for v in bbox[:4])
    if bbox_format == "xywh":
        return x0 + a / 2.0, y0 + b / 2.0
    if bbox_format == "xyxy":
        return (x0 + a) / 2.0, (y0 + b) / 2.0
    raise ValueError(f"unknown bbox_format {bbox_format!r}; use 'xywh' or 'xyxy'.")


def _build_split_rows(
    dataset_rows: list[dict], *, test_fraction: float, tune_fraction: float
) -> list[dict]:
    """Fixed, patient/domain-stratified train/tune/test rows (single fold).

    Patients (all images of a ``patient_id`` share a split — no leakage) are grouped by
    ``domain`` and, within each domain, ordered by a stable SHA1 of the patient id so the
    split is decorrelated from file order yet fully reproducible. The leading
    ``test_fraction`` of each domain's patients become ``test``, the next ``tune_fraction``
    become ``tune``, the rest ``train`` — so every sufficiently populated domain is
    represented in each split.
    """
    patients: dict[str, dict] = {}
    for row in dataset_rows:
        entry = patients.setdefault(row["patient_id"], {"domain": row["domain"], "sample_ids": []})
        entry["sample_ids"].append(row["sample_id"])

    by_domain: dict[str, list[str]] = defaultdict(list)
    for pid, entry in patients.items():
        by_domain[entry["domain"]].append(pid)

    split_of_sample: dict[str, str] = {}
    for domain in sorted(by_domain):
        pids = sorted(by_domain[domain], key=lambda p: (hashlib.sha1(p.encode()).hexdigest(), p))
        n = len(pids)
        n_test = min(round(n * test_fraction), n)
        n_tune = min(round(n * tune_fraction), n - n_test)
        for i, pid in enumerate(pids):
            if i < n_test:
                role = "test"
            elif i < n_test + n_tune:
                role = "tune"
            else:
                role = "train"
            for sid in patients[pid]["sample_ids"]:
                split_of_sample[sid] = role

    # Emit in dataset order (already sorted by sample_id) so splits.csv is byte-stable.
    return [
        {"sample_id": row["sample_id"], "split": split_of_sample[row["sample_id"]], "fold": 0}
        for row in dataset_rows
    ]


def _midog_sample_id(file_name: str) -> str:
    """``midog_<stem>``: the one sample-id rule shared by sorting and row construction."""
    return f"midog_{Path(str(file_name)).stem}"


def curate_midog_detection(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    annotations_json: str | Path | None = None,
    spacing_at_level_0: float | None = None,
    force_spacing_at_level_0: float | None = None,
    bbox_format: str = "xyxy",
    test_fraction: float = DEFAULT_TEST_FRACTION,
    tune_fraction: float = DEFAULT_TUNE_FRACTION,
) -> CuratedManifest:
    """Curate MIDOG 2022 into Soma's detection Manifest (COCO JSON -> point CSVs).

    Args:
        raw_root: Directory holding ``images/`` and the COCO annotations JSON.
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, the per-sample
            ``points/<sample_id>.csv`` files, and ``summary.json`` are written.
        annotations_json: Path to the COCO annotations file (default
            ``raw_root/MIDOG2022_training.json``).
        spacing_at_level_0: Optional fallback µm/px declaration for the source image,
            stamped per sample; a per-image ``spacing`` in the JSON overrides it because
            MIDOG scanners differ. When neither spacing argument is given, no declaration
            is emitted.
        force_spacing_at_level_0: Optional nominal µm/px applied to every image, overriding
            per-image JSON metadata. The detection benchmark uses ``0.25`` here because it
            deliberately evaluates MIDOG in its native pixel grid under one nominal frame.
        bbox_format: Box convention in the JSON. MIDOG 2022 encodes each mitosis as a fixed
            ~50 px **corner** box ``[x1, y1, x2, y2]`` (``"xyxy"``, the default); pass
            ``"xywh"`` only for a genuine COCO ``[x, y, w, h]`` export. The size and
            per-centre bounds guards catch common wrong values, but no geometric check can
            distinguish every plausible input; the canonical MIDOG benchmark keeps the
            default ``"xyxy"``.
        test_fraction / tune_fraction: Local held-out split fractions per domain.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` for the generated files.
    """
    raw_root = Path(raw_root)
    if spacing_at_level_0 is not None and force_spacing_at_level_0 is not None:
        raise ValueError(
            "spacing_at_level_0 is a per-image fallback while force_spacing_at_level_0 is "
            "an override; pass only one."
        )
    images_root = raw_root / "images"
    ann_path = Path(annotations_json) if annotations_json is not None else raw_root / _DEFAULT_ANNOTATIONS_NAME
    if not images_root.is_dir():
        raise FileNotFoundError(f"expected an images/ directory under the MIDOG root: {raw_root}")
    if not ann_path.is_file():
        raise FileNotFoundError(
            f"MIDOG COCO annotations not found: {ann_path} (pass annotations_json= to override)."
        )

    coco = json.loads(ann_path.read_text())
    mitotic_id = _resolve_mitotic_category_id(coco["categories"])
    _validate_bbox_sizes(coco["annotations"], mitotic_id, bbox_format)

    # image_id -> list of mitotic-figure centre points (and a hard-negative tally). Each
    # centre is validated against its image's bounds: a wrong bbox_format (or corrupt
    # coordinates) places the centre outside the image, which the median-side guard alone
    # can miss on a small image. Grossly out-of-image centres raise; a genuine sub-pixel
    # edge overhang (MIDOG has one) is clamped into the frame and counted.
    dims_by_id: dict[int, tuple[int, int]] = {
        int(im["id"]): (int(im["width"]), int(im["height"])) for im in coco["images"]
    }
    mitoses_by_image: dict[int, list[tuple[float, float]]] = defaultdict(list)
    num_hard_negatives = 0
    num_edge_clamped = 0
    gross_oob: list[tuple[int, float, float, int, int, float]] = []
    for ann in coco["annotations"]:
        if int(ann["category_id"]) != mitotic_id:
            num_hard_negatives += 1
            continue
        image_id = int(ann["image_id"])
        cx, cy = _bbox_center(ann["bbox"], bbox_format)
        if not (math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError(
                f"MIDOG annotation {ann.get('id')} on image {image_id} has a non-finite "
                f"centre ({cx}, {cy}) under bbox_format={bbox_format!r}."
            )
        width, height = dims_by_id[image_id]
        overshoot = max(0.0, -cx, cx - (width - 1), -cy, cy - (height - 1))
        if overshoot > _MAX_CENTER_OVERSHOOT_PX:
            gross_oob.append((image_id, round(cx, 1), round(cy, 1), width, height, round(overshoot, 1)))
        if overshoot > 0.0:
            cx = min(max(cx, 0.0), float(width - 1))
            cy = min(max(cy, 0.0), float(height - 1))
            num_edge_clamped += 1
        mitoses_by_image[image_id].append((cx, cy))

    if gross_oob:
        raise ValueError(
            f"{len(gross_oob)} MIDOG mitosis centre(s) fall > {_MAX_CENTER_OVERSHOOT_PX:.0f}px "
            f"outside their image under bbox_format={bbox_format!r} "
            f"(image_id, cx, cy, W, H, overshoot): {gross_oob[:3]}. This means a wrong "
            f"bbox_format (MIDOG uses corner boxes [x1,y1,x2,y2] -> pass 'xyxy') or corrupt "
            f"coordinates — a wrong format silently mis-places every point past the image edge."
        )

    output_dir = Path(output_dir)
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict] = []
    total_mitoses = 0
    num_empty = 0
    per_domain_samples: Counter[str] = Counter()
    mitoses_per_sample: dict[str, int] = {}

    # Sort by sample_id so dataset.csv / points ordering is deterministic regardless of the
    # order images appear in the JSON.
    images_sorted = sorted(coco["images"], key=lambda im: _midog_sample_id(im["file_name"]))
    for img in images_sorted:
        file_name = str(img["file_name"])
        sample_id = _midog_sample_id(file_name)
        image_path = (images_root / file_name).resolve()

        tumor_type = img.get("tumortype", img.get("tumor_type"))
        scanner = img.get("scanner")
        # A present-but-null patient_id must not collapse every image onto the patient
        # "None" (which would make the leakage check group the whole set together).
        patient_id = str(img.get("patient_id") or sample_id)
        domain = str(tumor_type or scanner or "unknown")
        spacing = (
            force_spacing_at_level_0
            if force_spacing_at_level_0 is not None
            else img.get("spacing", spacing_at_level_0)
        )

        centers = sorted(mitoses_by_image.get(int(img["id"]), []))
        out_csv = points_dir / f"{sample_id}.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "class"])
            for x, y in centers:
                writer.writerow([x, y, 0])
        total_mitoses += len(centers)
        num_empty += int(len(centers) == 0)
        per_domain_samples[domain] += 1
        mitoses_per_sample[sample_id] = len(centers)

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "points_path": str(out_csv.resolve()),
            "patient_id": patient_id,
            "domain": domain,
        }
        if tumor_type is not None:
            row["tumor_type"] = str(tumor_type)
        if scanner is not None:
            row["scanner"] = str(scanner)
        if spacing is not None:
            row["spacing_at_level_0"] = float(spacing)
        dataset_rows.append(row)

    if not dataset_rows:
        raise ValueError(f"No MIDOG images found in {ann_path}")

    split_rows = _build_split_rows(
        dataset_rows, test_fraction=test_fraction, tune_fraction=tune_fraction
    )

    role_of = {r["sample_id"]: r["split"] for r in split_rows}
    domain_of = {r["sample_id"]: r["domain"] for r in dataset_rows}
    per_split: dict[str, dict] = {}
    for role in ("train", "tune", "test"):
        members = [sid for sid in role_of if role_of[sid] == role]
        per_split[role] = {
            "num_samples": len(members),
            "num_mitoses": sum(mitoses_per_sample[sid] for sid in members),
            "domains": dict(Counter(domain_of[sid] for sid in members)),
        }

    summary = {
        "dataset": "MIDOG 2022 (mitosis detection)",
        "dataset_type": "detection",
        "num_classes": MIDOG_NUM_CLASSES,
        "class_names": list(MIDOG_CLASS_NAMES),
        "mitotic_category_id": mitotic_id,
        "total_samples": len(dataset_rows),
        "total_mitoses": total_mitoses,
        "num_hard_negatives": num_hard_negatives,
        "num_edge_clamped": num_edge_clamped,
        "num_empty": num_empty,
        "domains": dict(per_domain_samples),
        "spacing_at_level_0": spacing_at_level_0,
        "forced_spacing_at_level_0": force_spacing_at_level_0,
        "bbox_format": bbox_format,
        "split_fractions": {
            "test": test_fraction,
            "tune": tune_fraction,
            "train": round(1.0 - test_fraction - tune_fraction, 6),
        },
        "splits": per_split,
        "leaderboard_comparable": False,
        "note": _NON_LEADERBOARD_NOTE,
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
        prog="python -m soma.curation.midog", description=__doc__
    )
    ap.add_argument("--raw-root", type=Path, required=True, help="MIDOG root (images/ + COCO json)")
    ap.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    ap.add_argument("--annotations-json", type=Path, default=None, help="COCO json path override")
    ap.add_argument(
        "--spacing-at-level-0", type=float, default=None,
        help="source-image µm/px declaration to stamp (per-image JSON spacing wins)",
    )
    ap.add_argument("--bbox-format", choices=("xywh", "xyxy"), default="xyxy",
                    help="box convention (MIDOG uses corner boxes -> 'xyxy', the default)")
    ap.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    ap.add_argument("--tune-fraction", type=float, default=DEFAULT_TUNE_FRACTION)
    args = ap.parse_args(argv)

    manifest = curate_midog_detection(
        args.raw_root,
        args.output_dir,
        annotations_json=args.annotations_json,
        spacing_at_level_0=args.spacing_at_level_0,
        bbox_format=args.bbox_format,
        test_fraction=args.test_fraction,
        tune_fraction=args.tune_fraction,
    )
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")


if __name__ == "__main__":
    main()
