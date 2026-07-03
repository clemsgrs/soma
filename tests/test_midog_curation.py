"""Tests for MIDOG 2022 mitosis-detection curation (synthetic COCO layout).

Built + tested against a synthetic on-disk layout — the real MIDOG download is never
needed (wiring to real data + level0_spacing is a run-time step). The layout mirrors
MIDOG 2022's COCO export: an ``images/`` dir of TIFFs and a ``MIDOG2022_training.json``
with ``images`` / ``categories`` / ``annotations``. Each image entry carries the
per-domain (tumor type / scanner) metadata the challenge ships.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from soma.curation import Curator, curate_midog_detection
from soma.curation.midog import (
    MIDOG_CLASS_NAMES,
    MIDOG_MITOTIC_CATEGORY_ID,
)
from soma.dataset import DetectionManifest, Splits


# --------------------------------------------------------------------------- helpers


def _write_midog_raw(raw_root: Path, images: list[dict]) -> Path:
    """Write a synthetic MIDOG 2022 COCO layout under ``raw_root``.

    ``images`` is a list of specs, each with ``file_name`` and optional
    ``tumortype`` / ``scanner`` / ``patient_id`` / ``spacing`` and a ``boxes`` list of
    ``(x, y, w, h, category_id)`` COCO boxes.
    """
    images_dir = raw_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    for image_id, spec in enumerate(images, start=1):
        (images_dir / spec["file_name"]).write_bytes(b"")  # never decoded here
        entry = {"id": image_id, "file_name": spec["file_name"], "width": 100, "height": 100}
        for key in ("tumortype", "scanner", "patient_id", "spacing"):
            if key in spec:
                entry[key] = spec[key]
        coco_images.append(entry)
        for (x, y, w, h, category_id) in spec.get("boxes", []):
            coco_annotations.append(
                {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h]}
            )
            ann_id += 1
    coco = {
        "categories": [
            {"id": 1, "name": "mitotic figure"},
            {"id": 2, "name": "not mitotic figure"},
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (raw_root / "MIDOG2022_training.json").write_text(json.dumps(coco))
    return raw_root


def _stratified_images(n_domains: int = 3, per_domain: int = 5) -> list[dict]:
    """One image per patient, ``per_domain`` patients across ``n_domains`` domains."""
    images: list[dict] = []
    for d in range(n_domains):
        tumor = f"tumor{d}"
        for k in range(per_domain):
            stem = f"{d:01d}{k:02d}"
            images.append(
                {
                    "file_name": f"{stem}.tiff",
                    "tumortype": tumor,
                    "scanner": f"scanner{d}",
                    "patient_id": f"case_{stem}",
                    "boxes": [(10, 20, 4, 4, 1)],  # one mitosis
                }
            )
    return images


# ------------------------------------------------------------- schema + point emission


def test_curate_emits_detection_manifest_and_center_points(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw",
        [
            {
                "file_name": "001.tiff",
                "tumortype": "breast",
                "scanner": "XR",
                "boxes": [(10, 20, 4, 4, 1), (30, 30, 6, 6, 2)],  # 1 mitosis + 1 hard-negative
            },
        ],
    )
    out = tmp_path / "curated"
    manifest = curate_midog_detection(raw, out)

    assert manifest.dataset_csv == out / "dataset.csv"
    assert manifest.splits_csv == out / "splits.csv"

    df = pd.read_csv(manifest.dataset_csv)
    assert list(df.columns)[:3] == ["sample_id", "image_path", "points_path"]

    detection = DetectionManifest(manifest.dataset_csv)
    (sid,) = detection.sample_ids
    assert sid == "midog_001"

    # COCO bbox [x, y, w, h] -> centre (x + w/2, y + h/2); hard-negative (cat 2) excluded.
    pts = pd.read_csv(out / "points" / f"{sid}.csv")
    assert list(pts.columns) == ["x", "y", "class"]
    rows = sorted((r.x, r.y, r["class"]) for _, r in pts.iterrows())
    assert rows == [(12.0, 22.0, 0)]  # single mitosis class, 0-based


def test_curate_writes_header_only_csv_for_image_without_mitoses(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw",
        [{"file_name": "empty.tiff", "tumortype": "breast", "boxes": [(1, 1, 2, 2, 2)]}],
    )
    out = tmp_path / "curated"
    curate_midog_detection(raw, out)
    pts = pd.read_csv(out / "points" / "midog_empty.csv")
    assert list(pts.columns) == ["x", "y", "class"]
    assert len(pts) == 0


# --------------------------------------------------------- per-domain metadata carried


def test_per_domain_metadata_propagated(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw",
        [{"file_name": "007.tiff", "tumortype": "melanoma", "scanner": "Aperio", "patient_id": "p7", "boxes": []}],
    )
    out = tmp_path / "curated"
    detection = DetectionManifest(curate_midog_detection(raw, out).dataset_csv)
    rec = detection.samples["midog_007"]
    assert rec.patient_id == "p7"
    assert rec.metadata["tumor_type"] == "melanoma"
    assert rec.metadata["scanner"] == "Aperio"
    # A single ``domain`` column groups robustness stratification (defaults to tumor type).
    assert rec.metadata["domain"] == "melanoma"


# ------------------------------------------------------ fixed patient/domain-stratified


def test_local_split_is_domain_stratified(tmp_path: Path):
    raw = _write_midog_raw(tmp_path / "raw", _stratified_images(n_domains=3, per_domain=5))
    out = tmp_path / "curated"
    manifest = curate_midog_detection(raw, out)

    dataset = pd.read_csv(manifest.dataset_csv)
    splits = pd.read_csv(manifest.splits_csv)
    merged = splits.merge(dataset[["sample_id", "domain"]], on="sample_id")

    # Every domain contributes to all three roles (5 per domain -> 1 test / 1 tune / 3 train).
    for domain, group in merged.groupby("domain"):
        assert set(group["split"]) == {"train", "tune", "test"}, domain
    # Single fold, and every sample lands in exactly one split.
    assert set(splits["fold"]) == {0}
    assert len(splits) == len(dataset)


def test_local_split_keeps_a_patient_in_one_split(tmp_path: Path):
    # Two images share a patient: they must not leak across splits.
    images = _stratified_images(n_domains=2, per_domain=5)
    images.append(
        {"file_name": "shared_b.tiff", "tumortype": "tumor0", "scanner": "scanner0",
         "patient_id": "case_000", "boxes": [(5, 5, 4, 4, 1)]}  # same patient as file "000.tiff"
    )
    raw = _write_midog_raw(tmp_path / "raw", images)
    out = tmp_path / "curated"
    manifest = curate_midog_detection(raw, out)

    detection = DetectionManifest(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, detection)
    # The no-leakage invariant holds across every fold.
    splits.validate_no_patient_leakage(detection)


def test_split_is_fixed_across_recuration(tmp_path: Path):
    raw = _write_midog_raw(tmp_path / "raw", _stratified_images())
    a = curate_midog_detection(raw, tmp_path / "a")
    b = curate_midog_detection(raw, tmp_path / "b")
    assert a.splits_csv.read_bytes() == b.splits_csv.read_bytes()


# ------------------------------------------------------------------- level0 spacing


def test_level0_spacing_param_stamped_per_sample(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw", [{"file_name": "001.tiff", "tumortype": "breast", "boxes": []}]
    )
    out = tmp_path / "curated"
    manifest = curate_midog_detection(raw, out, level0_spacing_um=0.5)
    df = pd.read_csv(manifest.dataset_csv)
    assert "level0_spacing" in df.columns
    assert (df["level0_spacing"] == 0.5).all()


def test_per_image_spacing_overrides_param(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw",
        [
            {"file_name": "001.tiff", "tumortype": "breast", "spacing": 0.23, "boxes": []},
            {"file_name": "002.tiff", "tumortype": "breast", "boxes": []},
        ],
    )
    out = tmp_path / "curated"
    detection = DetectionManifest(
        curate_midog_detection(raw, out, level0_spacing_um=0.5).dataset_csv
    )
    # Per-image spacing (from the JSON) wins; the param fills in where absent.
    assert float(detection.samples["midog_001"].metadata["level0_spacing"]) == 0.23
    assert float(detection.samples["midog_002"].metadata["level0_spacing"]) == 0.5


def test_native_path_has_no_spacing_column(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw", [{"file_name": "001.tiff", "tumortype": "breast", "boxes": []}]
    )
    out = tmp_path / "curated"
    df = pd.read_csv(curate_midog_detection(raw, out).dataset_csv)
    assert "level0_spacing" not in df.columns


# ------------------------------------------------------------------- summary + protocol


def test_summary_documents_non_leaderboard_split_and_counts(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw",
        [
            {"file_name": "001.tiff", "tumortype": "breast", "boxes": [(10, 20, 4, 4, 1), (0, 0, 2, 2, 2)]},
            {"file_name": "002.tiff", "tumortype": "lung", "boxes": [(5, 5, 4, 4, 1)]},
        ],
    )
    out = tmp_path / "curated"
    curate_midog_detection(raw, out)
    summary = json.loads((out / "summary.json").read_text())

    assert summary["dataset_type"] == "detection"
    assert summary["num_classes"] == 1
    assert summary["class_names"] == list(MIDOG_CLASS_NAMES)
    assert summary["mitotic_category_id"] == MIDOG_MITOTIC_CATEGORY_ID
    assert summary["total_samples"] == 2
    assert summary["total_mitoses"] == 2
    assert summary["num_hard_negatives"] == 1
    # The local split is explicitly flagged as not comparable to the official leaderboard.
    assert summary["leaderboard_comparable"] is False
    assert "leaderboard" in summary["note"].lower()
    assert set(summary["domains"]) == {"breast", "lung"}


def test_curator_satisfies_structural_protocol(tmp_path: Path):
    assert isinstance(curate_midog_detection, Curator)


def test_recuration_is_byte_identical(tmp_path: Path):
    # Re-curating the same raw data into the same output dir is byte-identical: rows are
    # emitted in a fixed (sample_id-sorted) order and the split is deterministic. (Like the
    # OCELOT detection path, dataset.csv embeds the absolute per-sample points_path, so the
    # determinism check re-curates into one dir rather than comparing two different dirs.)
    raw = _write_midog_raw(tmp_path / "raw", _stratified_images())
    out = tmp_path / "curated"
    a = curate_midog_detection(raw, out, level0_spacing_um=0.5)
    before = {name: (out / name).read_bytes() for name in ("dataset.csv", "splits.csv", "summary.json")}
    curate_midog_detection(raw, out, level0_spacing_um=0.5)
    for name, data in before.items():
        assert (out / name).read_bytes() == data, f"{name} is not byte-identical across re-curation"
    assert a.dataset_csv == out / "dataset.csv"

    # The path-free artifacts (the fixed split + summary) are byte-identical even across a
    # fresh output directory — the local held-out split does not depend on where it lands.
    other = tmp_path / "curated_elsewhere"
    curate_midog_detection(raw, other, level0_spacing_um=0.5)
    for name in ("splits.csv", "summary.json"):
        assert (other / name).read_bytes() == before[name], f"{name} differs across dirs"
