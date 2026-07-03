"""Curator for the MONKEY kidney-biopsy inflammatory-cell detection challenge.

Like the EVA / OCELOT / BEETLE curators, this emits Soma's unified Manifest
(``dataset.csv`` + ``splits.csv`` + ``summary.json``) from locally prepared raw data and
**does not download anything** (MONKEY requires a Grand-Challenge account; the user has
institutional access — see the challenge page for the download step).

MONKEY ships PAS-stained kidney-transplant biopsy WSIs with **dot** annotations for two
mononuclear-leukocyte classes — lymphocytes and monocytes — plus a merged
"inflammatory-cells" (MNL) leaderboard. Each case ``<dept>_P<patient>`` (e.g.
``A_P000001``) provides per-class JSON files whose ``points`` are in **millimetres** at
the slide's level-0 spacing::

    annotations/json/<case>_lymphocytes.json
    annotations/json/<case>_monocytes.json
    images/pas-cpg/<case>_PAS_CPG.tif

This curator reads the lymphocyte + monocyte JSONs, converts each mm point to the level-0
**pixel** frame (``px = mm * 1000 / spacing_um``), and writes one ``x,y,class`` point CSV
per case with lymphocytes → class ``0`` and monocytes → class ``1`` (the merged MNL class
is recovered at score time by pooling both, so it is *not* a third stored class). The WSI
is referenced in place and never decoded, exactly like OCELOT's native path.

FROC — MONKEY's native metric — normalizes false positives per **mm²**, so every row also
carries ``level0_spacing`` (µm/px) and, when the annotation ROIs give it, ``roi_area_mm2``
(``spacing² · area_rois / 1e6``), the physical area the FROC sweep divides by.

**Splits.** MONKEY's official test set is hidden, and the public labelled data ships no
train/val/test partition, so — unlike every other soma curator, which inherits a
predefined split — this one **carves a fixed, deterministic patient-stratified local
held-out split** (``train`` / ``tune`` / ``test``). Whole patients (all their slides) stay
in one split, and departments are interleaved so each split spans the available domains.
The resulting ``test`` split is a *local* held-out set: numbers on it are **not
comparable** to the published MONKEY leaderboard (a different, hidden test set), which
should be shown only as a reference band. See ``design`` §2.6.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from soma.curation.manifest import CuratedManifest, write_manifest

# Level-0 µm/px of the MONKEY PAS-CPG WSIs (the challenge evaluator's SPACING_LEVEL0).
MONKEY_SPACING_LEVEL0 = 0.24199951445730394

# 0-based class ids for the stored points CSV (the merged MNL class is derived at scoring
# time by pooling both, so it is not stored as a third class).
MONKEY_CLASS_NAMES = ("lymphocytes", "monocytes")
MONKEY_LABEL_REMAP = {name: idx for idx, name in enumerate(MONKEY_CLASS_NAMES)}
MONKEY_NUM_CLASSES = len(MONKEY_CLASS_NAMES)

# Default train / tune / test fractions for the carved local held-out split.
DEFAULT_SPLIT_FRACTIONS = (0.7, 0.15, 0.15)
_SPLIT_ROLES = ("train", "tune", "test")


def _read_points_mm(json_path: Path) -> tuple[list[tuple[float, float]], float | None]:
    """Read a MONKEY per-class dot-annotation JSON: return mm points + ROI area in px²."""
    doc = json.loads(json_path.read_text())
    points = [(float(p["point"][0]), float(p["point"][1])) for p in doc.get("points", [])]
    area_rois = doc.get("area_rois")
    return points, (float(area_rois) if area_rois is not None else None)


def parse_case_id(case_id: str) -> tuple[str, str]:
    """Map a MONKEY ``<dept>_P<patient>[_<slide>]`` case id to ``(department, patient_id)``.

    The department is the leading token; the patient is the first two underscore-joined
    tokens (``A_P000001``), so any extra trailing slide suffix still groups a patient's
    slides together. Ids without an underscore fall back to being their own patient.
    """
    tokens = case_id.split("_")
    department = tokens[0]
    patient_id = "_".join(tokens[:2]) if len(tokens) >= 2 else case_id
    return department, patient_id


def assign_patient_splits(
    patients: dict[str, str],
    *,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
) -> dict[str, str]:
    """Deterministic patient-stratified, department-interleaved split assignment.

    ``patients`` maps ``patient_id -> department``. Patients are ordered by
    round-robin across departments (rank 0 of each dept, then rank 1, …) so the splits
    span all domains, then assigned by position along the ``train/tune/test`` fractions.
    Returns ``patient_id -> split`` and is a pure function of its inputs (no randomness).
    """
    if len(fractions) != 3:
        raise ValueError(f"fractions must be (train, tune, test); got {fractions}.")
    if any(f < 0 for f in fractions) or sum(fractions) <= 0:
        raise ValueError(f"fractions must be non-negative and sum to > 0; got {fractions}.")

    by_dept: dict[str, list[str]] = defaultdict(list)
    for patient_id, dept in patients.items():
        by_dept[dept].append(patient_id)
    for dept in by_dept:
        by_dept[dept].sort()

    depts = sorted(by_dept)
    max_len = max((len(v) for v in by_dept.values()), default=0)
    ordered: list[str] = []
    for rank in range(max_len):
        for dept in depts:
            if rank < len(by_dept[dept]):
                ordered.append(by_dept[dept][rank])

    total = len(ordered)
    f_train, f_tune, _ = (f / sum(fractions) for f in fractions)
    cum_train, cum_tune = f_train, f_train + f_tune
    assignment: dict[str, str] = {}
    for k, patient_id in enumerate(ordered):
        pos = (k + 0.5) / total
        if pos < cum_train:
            assignment[patient_id] = "train"
        elif pos < cum_tune:
            assignment[patient_id] = "tune"
        else:
            assignment[patient_id] = "test"
    return assignment


def curate_monkey_detection(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    spacing_um: float = MONKEY_SPACING_LEVEL0,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
) -> CuratedManifest:
    """Curate the MONKEY detection dataset into Soma's unified detection Manifest.

    Args:
        raw_root: Directory holding ``images/pas-cpg/<case>_PAS_CPG.tif`` and
            ``annotations/json/<case>_{lymphocytes,monocytes}.json``.
        output_dir: Where ``dataset.csv``, ``splits.csv``, the per-case
            ``points/<case>.csv`` files, and ``summary.json`` are written.
        spacing_um: Level-0 µm/px used to convert mm dot annotations to level-0 pixels and
            emitted per row as ``level0_spacing`` (defaults to :data:`MONKEY_SPACING_LEVEL0`).
        split_fractions: ``(train, tune, test)`` fractions for the carved deterministic
            patient-stratified local held-out split.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` for the generated files.
    """
    raw_root = Path(raw_root)
    ann_root = raw_root / "annotations" / "json"
    img_root = raw_root / "images" / "pas-cpg"
    if not ann_root.is_dir() or not img_root.is_dir():
        raise FileNotFoundError(
            f"expected images/pas-cpg/ and annotations/json/ under the MONKEY root: {raw_root}"
        )

    if spacing_um <= 0:
        raise ValueError(f"spacing_um must be > 0, got {spacing_um}.")
    mm_to_px = 1000.0 / float(spacing_um)  # mm -> µm (×1000) -> px (÷spacing)

    output_dir = Path(output_dir)
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    # Discover cases from the lymphocyte annotation files (every case has both classes).
    case_ids = sorted(p.name[: -len("_lymphocytes.json")] for p in ann_root.glob("*_lymphocytes.json"))
    if not case_ids:
        raise ValueError(f"No MONKEY cases (*_lymphocytes.json) found under {ann_root}")

    dataset_rows: list[dict] = []
    patients: dict[str, str] = {}  # patient_id -> department
    case_patient: dict[str, str] = {}  # case_id -> patient_id
    class_counter: Counter[int] = Counter()
    n_empty = 0

    for case_id in case_ids:
        wsi_path = img_root / f"{case_id}_PAS_CPG.tif"
        if not wsi_path.exists():
            raise FileNotFoundError(f"missing PAS WSI for case {case_id}: {wsi_path}")
        department, patient_id = parse_case_id(case_id)
        patients[patient_id] = department
        case_patient[case_id] = patient_id

        # Merge lymphocytes (class 0) + monocytes (class 1) into one x,y,class CSV, in
        # the level-0 pixel frame.
        merged: list[tuple[float, float, int]] = []
        roi_area_px: float | None = None
        for class_name in MONKEY_CLASS_NAMES:
            class_id = MONKEY_LABEL_REMAP[class_name]
            pts_mm, area_rois = _read_points_mm(ann_root / f"{case_id}_{class_name}.json")
            if area_rois is not None and roi_area_px is None:
                roi_area_px = area_rois
            for x_mm, y_mm in pts_mm:
                merged.append((x_mm * mm_to_px, y_mm * mm_to_px, class_id))
                class_counter[class_id] += 1

        out_csv = points_dir / f"{case_id}.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "class"])
            for x, y, c in merged:
                writer.writerow([x, y, c])
        n_empty += int(len(merged) == 0)

        row = {
            "sample_id": case_id,
            "image_path": str(wsi_path.resolve()),
            "points_path": str(out_csv.resolve()),
            "patient_id": patient_id,
            "level0_spacing": float(spacing_um),
            "department": department,
        }
        if roi_area_px is not None:
            # Physical ROI area for FROC's per-mm² false-positive normalization.
            row["roi_area_mm2"] = float(spacing_um) * float(spacing_um) * roi_area_px / 1_000_000.0
        dataset_rows.append(row)

    patient_split = assign_patient_splits(patients, fractions=split_fractions)
    split_rows = [
        {"sample_id": case_id, "split": patient_split[case_patient[case_id]], "fold": 0}
        for case_id in case_ids
    ]

    per_split_patients: dict[str, set[str]] = defaultdict(set)
    for patient_id, split in patient_split.items():
        per_split_patients[split].add(patient_id)

    summary = {
        "dataset": "MONKEY (kidney-biopsy inflammatory-cell detection)",
        "dataset_type": "detection",
        "stain": "PAS",
        "num_classes": MONKEY_NUM_CLASSES,
        "class_names": list(MONKEY_CLASS_NAMES),
        "mnl_merged_class": "inflammatory-cells",
        "native_metric": "FROC",
        "level0_spacing_um": float(spacing_um),
        "total_cases": len(dataset_rows),
        "total_patients": len(patients),
        "num_empty": n_empty,
        "points_per_class": {
            MONKEY_CLASS_NAMES[c]: class_counter[c] for c in range(MONKEY_NUM_CLASSES)
        },
        "split_fractions": {
            role: frac for role, frac in zip(_SPLIT_ROLES, split_fractions)
        },
        "cases_per_split": dict(Counter(r["split"] for r in split_rows)),
        "patients_per_split": {role: len(per_split_patients.get(role, set())) for role in _SPLIT_ROLES},
        "held_out_split": "test",
        "leaderboard_comparable": False,
        "split_note": (
            "Local patient-stratified held-out split carved from public labelled data; the "
            "official MONKEY test set is hidden, so 'test' numbers are NOT comparable to the "
            "published leaderboard (show it only as a reference band)."
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
        prog="python -m soma.curation.monkey", description=__doc__
    )
    ap.add_argument("--raw-root", type=Path, required=True, help="MONKEY raw root")
    ap.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    ap.add_argument(
        "--spacing-um", type=float, default=MONKEY_SPACING_LEVEL0, help="level-0 µm/px"
    )
    args = ap.parse_args(argv)
    manifest = curate_monkey_detection(
        args.raw_root, args.output_dir, spacing_um=args.spacing_um
    )
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")


if __name__ == "__main__":
    main()
