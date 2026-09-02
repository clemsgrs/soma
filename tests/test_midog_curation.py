"""Tests for MIDOG 2022 mitosis-detection curation (synthetic COCO layout).

Built + tested against a synthetic on-disk layout — the real MIDOG download is never
needed (wiring to real data + source-spacing declaration is a run-time step). The layout mirrors
MIDOG 2022's COCO export: an ``images/`` dir of TIFFs and a ``MIDOG2022_training.json``
with ``images`` / ``categories`` / ``annotations``. Each image entry carries the
per-domain (tumor type / scanner) metadata the challenge ships.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.curation import Curator, curate_midog_detection
from soma.curation.midog import (
    MIDOG_CLASS_NAMES,
    MIDOG_MITOTIC_CATEGORY_ID,
)
from soma.dataset import DetectionManifest, Splits


# --------------------------------------------------------------------------- helpers


def _write_midog_raw(raw_root: Path, images: list[dict]) -> Path:
    """Write a synthetic MIDOG 2022 COCO layout under ``raw_root``.

    ``images`` is a list of specs, each with ``file_name``, optional ``width`` / ``height``
    (default 100), optional ``tumortype`` / ``scanner`` / ``patient_id`` / ``spacing`` and a
    ``boxes`` list of ``(x, y, w, h, category_id)`` where ``(x, y)`` is the top-left and
    ``w, h`` the box extent. Boxes are written to the JSON as MIDOG **corner** boxes
    ``[x, y, x+w, y+h]`` (``xyxy``, the curator's real convention); the xyxy centre equals
    the xywh centre so centre assertions are unchanged.
    """
    images_dir = raw_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    for image_id, spec in enumerate(images, start=1):
        (images_dir / spec["file_name"]).write_bytes(b"")  # never decoded here
        entry = {
            "id": image_id, "file_name": spec["file_name"],
            "width": spec.get("width", 100), "height": spec.get("height", 100),
        }
        for key in ("tumortype", "scanner", "patient_id", "spacing"):
            if key in spec:
                entry[key] = spec[key]
        coco_images.append(entry)
        for (x, y, w, h, category_id) in spec.get("boxes", []):
            coco_annotations.append(
                {"id": ann_id, "image_id": image_id, "category_id": category_id,
                 "bbox": [x, y, x + w, y + h]}  # corner box (xyxy)
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


# --------------------------------------------------------- source-spacing declaration


def test_spacing_at_level_0_param_stamped_per_sample(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw", [{"file_name": "001.tiff", "tumortype": "breast", "boxes": []}]
    )
    out = tmp_path / "curated"
    manifest = curate_midog_detection(raw, out, spacing_at_level_0=0.5)
    df = pd.read_csv(manifest.dataset_csv)
    assert "spacing_at_level_0" in df.columns
    assert (df["spacing_at_level_0"] == 0.5).all()


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
        curate_midog_detection(raw, out, spacing_at_level_0=0.5).dataset_csv
    )
    # Per-image spacing (from the JSON) wins; the param fills in where absent.
    assert detection.samples["midog_001"].spacing_at_level_0 == 0.23
    assert detection.samples["midog_002"].spacing_at_level_0 == 0.5


def test_native_path_has_no_spacing_column(tmp_path: Path):
    raw = _write_midog_raw(
        tmp_path / "raw", [{"file_name": "001.tiff", "tumortype": "breast", "boxes": []}]
    )
    out = tmp_path / "curated"
    df = pd.read_csv(curate_midog_detection(raw, out).dataset_csv)
    assert "spacing_at_level_0" not in df.columns


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
    a = curate_midog_detection(raw, out, spacing_at_level_0=0.5)
    before = {name: (out / name).read_bytes() for name in ("dataset.csv", "splits.csv", "summary.json")}
    curate_midog_detection(raw, out, spacing_at_level_0=0.5)
    for name, data in before.items():
        assert (out / name).read_bytes() == data, f"{name} is not byte-identical across re-curation"
    assert a.dataset_csv == out / "dataset.csv"

    # The path-free artifacts (the fixed split + summary) are byte-identical even across a
    # fresh output directory — the local held-out split does not depend on where it lands.
    other = tmp_path / "curated_elsewhere"
    curate_midog_detection(raw, other, spacing_at_level_0=0.5)
    for name in ("splits.csv", "summary.json"):
        assert (other / name).read_bytes() == before[name], f"{name} differs across dirs"


def test_bbox_format_guard_rejects_corner_boxes_read_as_xywh(tmp_path: Path):
    # MIDOG ships ~50px corner boxes [x1,y1,x2,y2]; on a real (large) ROI the far corner is a
    # ~thousands-px coordinate, so reading them as COCO xywh treats that coordinate as the box
    # width -> absurd sizes. Default (xyxy) curation proceeds; forcing xywh must fail loudly.
    raw = _write_midog_raw(
        tmp_path / "raw",
        [{"file_name": "001.png", "patient_id": "p1", "width": 4000, "height": 4000, "boxes": [
            (500, 500, 50, 50, 1), (1000, 1000, 50, 50, 1), (2000, 2000, 50, 50, 1)]}],
    )
    with pytest.raises(ValueError, match="implausible median side"):
        curate_midog_detection(raw, tmp_path / "bad", bbox_format="xywh")
    # The curator default is xyxy, so no bbox_format need be passed for real MIDOG data.
    curate_midog_detection(raw, tmp_path / "ok")
    assert (tmp_path / "ok" / "dataset.csv").is_file()


def test_bbox_bounds_guard_catches_small_image_misread(tmp_path: Path):
    # The median-side guard alone can miss a wrong format on a small image (the far corner is
    # a small number). The per-centre in-bounds check is the backstop: a 40px corner box on a
    # 200px image read as xywh puts the centre 45px outside the image -> gross-OOB raise.
    raw = _write_midog_raw(
        tmp_path / "raw",
        [{"file_name": "001.png", "patient_id": "p1", "width": 200, "height": 200,
          "boxes": [(150, 150, 40, 40, 1)]}],  # xyxy [150,150,190,190]; xywh centre (245,245)
    )
    with pytest.raises(ValueError, match="outside their image"):
        curate_midog_detection(raw, tmp_path / "bad", bbox_format="xywh")
    curate_midog_detection(raw, tmp_path / "ok")  # xyxy centre (170,170) is in-bounds
    assert (tmp_path / "ok" / "dataset.csv").is_file()


def test_edge_annotation_clamped_and_counted(tmp_path: Path):
    # A genuine sub-pixel edge overhang (centre a hair past an edge) is clamped into the frame
    # and counted, not rejected — mirrors MIDOG's single edge mitosis.
    raw = _write_midog_raw(
        tmp_path / "raw",
        [{"file_name": "001.png", "patient_id": "p1", "width": 100, "height": 100,
          "boxes": [(-3, 40, 2, 4, 1)]}],  # xyxy [-3,40,-1,44] -> centre (-2, 42), 2px past left
    )
    out = tmp_path / "curated"
    curate_midog_detection(raw, out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["num_edge_clamped"] == 1
    pts = pd.read_csv(out / "points" / "midog_001.csv")
    assert (pts["x"] >= 0).all() and (pts["y"] >= 0).all()  # clamped into frame


def test_null_patient_id_falls_back_to_the_sample_id(tmp_path: Path):
    """A present-but-null patient_id must not collapse every image onto patient "None"."""
    raw = _write_midog_raw(
        tmp_path / "raw",
        [
            {"file_name": "007.tiff", "tumortype": "melanoma", "scanner": "Aperio", "patient_id": None, "boxes": []},
            {"file_name": "008.tiff", "tumortype": "melanoma", "scanner": "Aperio", "patient_id": None, "boxes": []},
        ],
    )
    detection = DetectionManifest(curate_midog_detection(raw, tmp_path / "curated").dataset_csv)
    assert detection.samples["midog_007"].patient_id == "midog_007"
    assert detection.samples["midog_008"].patient_id == "midog_008"
