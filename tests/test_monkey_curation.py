"""Tests for MONKEY kidney-biopsy cell-detection curation (built on a synthetic layout).

The real MONKEY data is never downloaded; these build a tiny synthetic raw tree that
mirrors the challenge's on-disk shape — a PAS WSI per case plus per-class dot-annotation
JSONs (coordinates in mm) — and assert the curator emits Soma's unified detection
manifest, converts the mm points to the level-0 pixel frame, and carves a deterministic
patient-stratified local held-out split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.curation.monkey import (
    MONKEY_SPACING_LEVEL0,
    curate_monkey_detection,
)
from soma.dataset import DetectionManifest, Splits


def _write_case(
    raw_root: Path,
    case_id: str,
    lymph_mm: list[tuple[float, float]],
    mono_mm: list[tuple[float, float]],
    *,
    area_rois_px: float = 4_000_000.0,
) -> None:
    """Write one synthetic MONKEY case: an (undecoded) PAS WSI + per-class JSON dots."""
    img_dir = raw_root / "images" / "pas-cpg"
    ann_dir = raw_root / "annotations" / "json"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{case_id}_PAS_CPG.tif").write_bytes(b"")  # curator never decodes it

    def _dump(name: str, points_mm: list[tuple[float, float]]) -> None:
        doc = {
            "name": name,
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [
                {"name": f"{name}-{i}", "point": [float(x), float(y), 0.0]}
                for i, (x, y) in enumerate(points_mm)
            ],
            "rois": [{"polygon": [[0, 0], [2000, 0], [2000, 2000], [0, 2000]]}],
            "area_rois": area_rois_px,
        }
        (ann_dir / f"{case_id}_{name}.json").write_text(json.dumps(doc))

    _dump("lymphocytes", lymph_mm)
    _dump("monocytes", mono_mm)


def _make_raw(tmp_path: Path) -> Path:
    """Six cases across two departments (A, B), a couple sharing a patient."""
    raw = tmp_path / "monkey_raw"
    _write_case(raw, "A_P000001", [(0.01, 0.02)], [(0.05, 0.05)])
    _write_case(raw, "A_P000002", [(0.03, 0.03), (0.04, 0.04)], [])
    _write_case(raw, "A_P000003", [], [(0.06, 0.06)])
    _write_case(raw, "B_P000010", [(0.10, 0.10)], [(0.11, 0.11)])
    _write_case(raw, "B_P000011", [(0.12, 0.12)], [])
    _write_case(raw, "B_P000012", [(0.13, 0.13)], [(0.14, 0.14)])
    return raw


def test_curate_monkey_emits_detection_manifest_with_merged_points(tmp_path: Path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "curated"

    manifest = curate_monkey_detection(raw, out)

    assert manifest.dataset_csv == out / "dataset.csv"
    df = pd.read_csv(manifest.dataset_csv)
    # Unified detection schema: sample_id, image_path, points_path, then recognized
    # optional columns (patient_id, spacing_at_level_0) and curator metadata.
    assert df.columns[:3].tolist() == ["sample_id", "image_path", "points_path"]
    assert "patient_id" in df.columns
    assert "spacing_at_level_0" in df.columns

    detection = DetectionManifest(manifest.dataset_csv)
    assert set(detection.sample_ids) == {
        "A_P000001", "A_P000002", "A_P000003", "B_P000010", "B_P000011", "B_P000012",
    }

    # Lymphocytes -> class 0, monocytes -> class 1, merged into one x,y,class CSV.
    pts = pd.read_csv(out / "points" / "A_P000001.csv")
    assert pts.columns.tolist() == ["x", "y", "class"]
    assert sorted(pts["class"].tolist()) == [0, 1]

    # The source declaration is emitted (FROC also needs physical spacing per mm²).
    assert df["spacing_at_level_0"].tolist() == pytest.approx(
        [MONKEY_SPACING_LEVEL0] * len(df)
    )
    assert detection.samples["A_P000001"].spacing_at_level_0 == pytest.approx(
        MONKEY_SPACING_LEVEL0
    )


def test_curate_monkey_converts_mm_points_to_level0_pixels(tmp_path: Path):
    raw = tmp_path / "monkey_raw"
    # spacing 0.25 µm/px -> mm_to_px = 1000 / 0.25 = 4000.
    _write_case(raw, "A_P000001", [(0.01, 0.02)], [(0.05, 0.05)])
    out = tmp_path / "curated"

    curate_monkey_detection(raw, out, spacing_at_level_0=0.25)

    pts = pd.read_csv(out / "points" / "A_P000001.csv")
    rows = sorted((r.x, r.y, int(r["class"])) for _, r in pts.iterrows())
    # lymph (0.01,0.02)mm -> (40,80)px class 0 ; mono (0.05,0.05)mm -> (200,200)px class 1.
    assert rows == [(40.0, 80.0, 0), (200.0, 200.0, 1)]


def test_curate_monkey_emits_roi_area_mm2_for_froc(tmp_path: Path):
    raw = tmp_path / "monkey_raw"
    _write_case(raw, "A_P000001", [(0.01, 0.01)], [], area_rois_px=4_000_000.0)
    out = tmp_path / "curated"

    curate_monkey_detection(raw, out, spacing_at_level_0=0.25)

    df = pd.read_csv(out / "dataset.csv")
    # roi_area_mm2 = spacing² · area_rois / 1e6 = 0.0625 · 4e6 / 1e6 = 0.25 mm².
    assert df["roi_area_mm2"].iloc[0] == pytest.approx(0.25)


def test_curate_monkey_split_is_patient_stratified_local_holdout(tmp_path: Path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "curated"
    manifest = curate_monkey_detection(raw, out)

    detection = DetectionManifest(manifest.dataset_csv)
    fold = Splits(manifest.splits_csv, detection).folds[0]

    splits = {"train": set(fold.train), "tune": set(fold.tune), "test": set(fold.tests["test"])}
    # All three roles are populated and disjoint (a real held-out test split exists).
    assert all(splits.values())
    all_ids = [sid for ids in splits.values() for sid in ids]
    assert len(all_ids) == len(set(all_ids)) == 6

    # No patient straddles two splits (whole patients are stratified).
    from soma.curation.monkey import parse_case_id

    patient_to_split = {}
    for split_name, ids in splits.items():
        for sid in ids:
            _, patient = parse_case_id(sid)
            assert patient_to_split.setdefault(patient, split_name) == split_name


def test_curate_monkey_keeps_a_patients_slides_in_one_split(tmp_path: Path):
    raw = tmp_path / "monkey_raw"
    # Two slides for patient A_P000001, plus other patients to populate every split.
    _write_case(raw, "A_P000001", [(0.01, 0.01)], [])
    _write_case(raw, "A_P000001_02", [(0.02, 0.02)], [])
    _write_case(raw, "A_P000002", [(0.03, 0.03)], [])
    _write_case(raw, "B_P000010", [(0.10, 0.10)], [])
    _write_case(raw, "B_P000011", [(0.11, 0.11)], [])
    _write_case(raw, "B_P000012", [(0.12, 0.12)], [])
    out = tmp_path / "curated"

    manifest = curate_monkey_detection(raw, out)
    splits_df = pd.read_csv(manifest.splits_csv).set_index("sample_id")["split"]
    # Both slides of patient A_P000001 land in the same split.
    assert splits_df["A_P000001"] == splits_df["A_P000001_02"]


def test_curate_monkey_is_deterministic(tmp_path: Path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "curated"
    curate_monkey_detection(raw, out)
    first = {name: (out / name).read_bytes() for name in ("dataset.csv", "splits.csv", "summary.json")}
    # Re-curating the same raw data into the same location is byte-identical (row order,
    # column order, split assignment and sorted-key summary are all deterministic).
    curate_monkey_detection(raw, out)
    for name, data in first.items():
        assert (out / name).read_bytes() == data


def test_curate_monkey_summary_flags_non_leaderboard_holdout(tmp_path: Path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "curated"
    curate_monkey_detection(raw, out)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_classes"] == 2
    assert summary["class_names"] == ["lymphocytes", "monocytes"]
    assert summary["native_metric"] == "FROC"
    assert summary["total_cases"] == 6
    assert summary["total_patients"] == 6
    assert summary["points_per_class"] == {"lymphocytes": 6, "monocytes": 4}
    # The carved test split is explicitly flagged as not leaderboard-comparable.
    assert summary["held_out_split"] == "test"
    assert summary["leaderboard_comparable"] is False
    assert "not comparable" in summary["split_note"].lower()
