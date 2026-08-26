"""Config-validation tests for the BEETLE slide-manifest segmentation deliverable (#93).

These are config-only (no slide/mask I/O): they assert the tracked example config
``examples/segmentation_beetle.yaml`` loads through soma's loader and encodes the
BEETLE recipe — the masks ``pixel_mapping`` (BEETLE's raw vocabulary),
the 5%% min-coverage rule, 512 px @ 0.5 µm/px spacing-aware, phikon sliding-224 dense
window, lightweight_conv decoder, num_classes=4, and the three metrics — and that the
derived raw-pixel → class-index remap matches BEETLE's pixel→class contract exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, TiffImagePlugin

from soma.config import load_config
from soma.curation.beetle import curate_beetle_slide_manifest
from soma.dense.reader import build_label_remap

REPO_ROOT = Path(__file__).resolve().parents[1]
BEETLE_CONFIG = REPO_ROOT / "examples" / "segmentation_beetle.yaml"

# BEETLE's pixel -> soma class contract (255 = ignore).
EXPECTED_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 0: 255}

COARSE_TCGA_SLIDES = {
    "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25": "fold0",
    "TCGA-OL-A66P-01Z-00-DX1.5ADD0D6D-37C6-4BC9-8C2B-64DB18BE99B3": "fold2",
    "TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014": "fold1",
}
COARSE_SPACING_UM_PER_PX = 0.657476464
TIGER_IMAGE_PREFIX = "wsirois/wsi-level-annotations/images"
TIGER_IMAGE_BASE_URL = "https://tiger-training.s3.amazonaws.com"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_resolution_tiff(path: Path, spacing_um_per_px: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = 10_000 / spacing_um_per_px
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[282] = resolution
    tags[283] = resolution
    tags[296] = 3  # centimetres
    Image.new("L", (1, 1)).save(path, tiffinfo=tags)


def _full_cohort(root: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    """A compact on-disk 587-slide/527-patient cohort with released naming patterns."""
    rows: list[dict[str, str]] = []

    def add(
        name: str,
        *,
        source: str,
        fold: str,
        patient_id: str = "",
        wsi_path: str | None = None,
    ) -> None:
        image_rel = wsi_path if wsi_path is not None else f"images/development/wsis/{name}.tif"
        mask_rel = f"annotations/masks/{name}.tif"
        if name in COARSE_TCGA_SLIDES:
            _write_resolution_tiff(root / image_rel, COARSE_SPACING_UM_PER_PX)
        else:
            _touch(root / image_rel)
        _touch(root / mask_rel)
        rows.append(
            {
                "patient_id": patient_id,
                "wsi_id": "",
                "name": name,
                "source": source,
                "specimen_type": "resection",
                "scanner": "fixture",
                # Public TIGER rows intentionally leave this blank in the released overview.
                "wsi_path": image_rel if wsi_path is not None else "",
                "annotation_mask_path": mask_rel,
                "annotation_xml_path": "",
                "annotation_json_path": "",
                "split": "development",
                "validation_fold": fold,
            }
        )

    # 319 released anonymous patients over 379 slides (60 repeated-patient slides).
    for patient in range(319):
        add(
            f"patient{patient + 1}_wsi1",
            source="nki",
            fold=f"fold{patient % 5}",
            patient_id=f"patient{patient + 1}",
            wsi_path=f"images/development/wsis/patient{patient + 1}_wsi1.tif",
        )
    for patient in range(60):
        add(
            f"patient{patient + 1}_wsi2",
            source="nki",
            fold=f"fold{patient % 5}",
            patient_id=f"patient{patient + 1}",
            wsi_path=f"images/development/wsis/patient{patient + 1}_wsi2.tif",
        )

    tcga_names = list(COARSE_TCGA_SLIDES)
    for patient in range(161):
        tcga_names.append(
            f"TCGA-AA-{patient:04X}-01Z-00-DX1.00000000-0000-0000-0000-{patient:012d}"
        )
    for patient, name in enumerate(tcga_names):
        add(name, source="tcga", fold=COARSE_TCGA_SLIDES.get(name, f"fold{patient % 5}"))

    for patient in range(26):
        add(
            f"TC_S01_P{patient + 1:06d}_C0001_B101",
            source="rumc",
            fold=f"fold{patient % 5}",
        )
    for patient in range(18):
        add(f"{patient + 100}B", source="jb", fold=f"fold{patient % 5}")

    assert len(rows) == 587
    overview = root / "data_overview.csv"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(overview, index=False)
    return overview, root, rows


def test_beetle_config_loads_and_validates():
    cfg = load_config(BEETLE_CONFIG)
    assert cfg.dataset_type == "segmentation"
    assert cfg.task.name == "segmentation"
    assert cfg.task.params["num_classes"] == 4


def test_beetle_config_encodes_masks_contract():
    cfg = load_config(BEETLE_CONFIG)
    masks = cfg.preprocessing.masks
    assert masks is not None
    # masks.pixel_mapping is the BEETLE raw vocabulary; must include background (unannotated).
    assert masks.pixel_mapping["background"] == 0
    assert masks.pixel_mapping["other"] == 1
    assert masks.pixel_mapping["non_invasive_epithelium"] == 2
    assert masks.pixel_mapping["invasive_epithelium"] == 3
    assert masks.pixel_mapping["necrosis"] == 4
    # >=5%% min-coverage rule on the annotated (non-background) classes.
    assert all(v == 0.05 for v in masks.min_coverage.values())
    assert set(masks.min_coverage) == {"other", "non_invasive_epithelium", "invasive_epithelium", "necrosis"}


def test_beetle_config_encodes_recipe():
    cfg = load_config(BEETLE_CONFIG)
    pp = cfg.preprocessing
    assert pp.requested_tile_size_px == 512
    assert pp.requested_spacing_um == 0.5
    # phikon native-224 sliding window @ 0.5 overlap.
    assert pp.dense_window_size == 224
    assert pp.dense_window_overlap == 0.5
    assert cfg.encoder.name == "phikon"
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.evaluation.metrics == ["mean_dice", "mean_iou", "dice_per_class"]
    assert cfg.preprocessing.sampling.output_mode == "merged"


def test_beetle_remap_matches_curation_contract():
    cfg = load_config(BEETLE_CONFIG)
    lut, num_classes = build_label_remap(cfg.preprocessing.masks.pixel_mapping, ignore_index=255)
    assert num_classes == 4
    raw = np.array(sorted(EXPECTED_REMAP), dtype=np.int64)
    expected = np.array([EXPECTED_REMAP[int(v)] for v in raw], dtype=lut.dtype)
    np.testing.assert_array_equal(lut[raw], expected)


@pytest.mark.parametrize(
    ("sample_id", "patient_id", "provenance"),
    [
        ("patient1_wsi1", "patient1", "released_data_overview"),
        (
            "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25",
            "TCGA-OL-A66I",
            "derived_tcga_barcode",
        ),
        ("TC_S01_P000001_C0001_B101", "TC_S01_P000001", "derived_rumc_name"),
        ("100B", "100", "derived_jb_name"),
    ],
)
def test_beetle_curator_resolves_patient_ids_with_provenance(
    tmp_path: Path, sample_id: str, patient_id: str, provenance: str
):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    dataset = pd.read_csv(manifest.dataset_csv, dtype=str, keep_default_na=False).set_index("sample_id")
    assert dataset.loc[sample_id, ["patient_id", "patient_id_provenance"]].tolist() == [
        patient_id,
        provenance,
    ]


def test_beetle_curator_records_exact_patient_id_resolution_metadata(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    resolution = json.loads(manifest.summary_json.read_text())["patient_id_resolution"]
    assert resolution == {
        "derived_jb_name": {
            "num_slides": 18,
            "rule": "use the numeric patient prefix before the B/S specimen suffix",
            "source_url": "https://tiger.grand-challenge.org/Data/",
        },
        "derived_rumc_name": {
            "num_slides": 26,
            "rule": "use the TC_S##_P###### prefix before case/block fields",
            "source_url": "https://tiger.grand-challenge.org/Data/",
        },
        "derived_tcga_barcode": {
            "num_slides": 164,
            "rule": "use the first three hyphen-separated TCGA barcode fields",
            "source_url": "https://tiger.grand-challenge.org/Data/",
        },
        "released_data_overview": {
            "num_slides": 379,
            "rule": "preserve each non-empty released patient_id verbatim",
            "source_url": "https://zenodo.org/records/16812932/files/data_overview.csv",
        },
    }


def test_beetle_curator_requires_full_cohort_before_writing(tmp_path: Path):
    overview, root, rows = _full_cohort(tmp_path / "raw")
    pd.DataFrame(rows[:-1]).to_csv(overview, index=False)
    output = tmp_path / "curated"

    with pytest.raises(ValueError, match=r"exactly 587 slides / 527 patients"):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_curator_requires_every_local_slide_and_annotation(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")
    (root / "images/development/wsis/patient319_wsi1.tif").unlink()
    output = tmp_path / "curated"

    with pytest.raises(
        ValueError,
        match=r"missing local WSI or annotation.*patient319_wsi1",
    ):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_curator_rejects_patient_cross_fold_leakage(tmp_path: Path):
    overview, root, rows = _full_cohort(tmp_path / "raw")
    repeated = next(row for row in rows if row["name"] == "patient1_wsi2")
    repeated["validation_fold"] = "fold4"
    pd.DataFrame(rows).to_csv(overview, index=False)
    output = tmp_path / "curated"

    with pytest.raises(ValueError, match=r"patient1.*fold0.*fold4"):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_curator_rejects_recovered_patient_id_collisions(tmp_path: Path):
    overview, root, rows = _full_cohort(tmp_path / "raw")
    for row in rows:
        if row["patient_id"] == "patient319":
            row["patient_id"] = "100"
    pd.DataFrame(rows).to_csv(overview, index=False)
    output = tmp_path / "curated"

    with pytest.raises(
        ValueError,
        match=r"collision.*100.*derived_jb_name.*released_data_overview",
    ):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_curator_records_exact_native_spacing_rows(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    dataset = pd.read_csv(manifest.dataset_csv, keep_default_na=False)
    columns = [
        "sample_id",
        "validation_fold",
        "spacing_at_level_0",
        "read_policy",
        "spacing_provenance",
        "in_native_spacing_sensitivity_subset",
    ]
    exceptions = dataset[dataset["read_policy"] == "native_level_0_no_upsample"][columns]
    expected = [
        [
            sample_id,
            fold,
            str(COARSE_SPACING_UM_PER_PX),
            "native_level_0_no_upsample",
            "local_tiff_level_0_resolution_tags",
            False,
        ]
        for sample_id, fold in COARSE_TCGA_SLIDES.items()
    ]
    assert exceptions.values.tolist() == expected


def test_beetle_curator_preserves_primary_cohort_identity(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["cohort"] == {
        "kind": "primary_full_cohort",
        "num_patients": 527,
        "num_slides": 587,
    }


def test_beetle_curator_derives_exact_evaluation_sensitivity_subset(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    dataset = pd.read_csv(manifest.dataset_csv, keep_default_na=False)
    sensitivity = dataset[dataset["in_native_spacing_sensitivity_subset"]]
    assert len(sensitivity) == 584
    assert sensitivity["patient_id"].nunique() == 524

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["derived_evaluation_subsets"]["native_spacing_sensitivity"] == {
        "excluded_patient_ids": ["TCGA-OL-A66I", "TCGA-OL-A66P", "TCGA-OL-A6VO"],
        "excluded_sample_ids": list(COARSE_TCGA_SLIDES),
        "intended_use": "evaluation_only",
        "num_patients": 524,
        "num_slides": 584,
    }


def test_beetle_curator_records_exact_native_spacing_provenance(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "curated")

    spacing_exceptions = json.loads(manifest.summary_json.read_text())["spacing_exceptions"]
    assert spacing_exceptions == [
        {
            "measurement": "local level-0 TIFF XResolution/YResolution + ResolutionUnit tags",
            "measured_spacing_um_per_px": COARSE_SPACING_UM_PER_PX,
            "organizer_fold": fold,
            "patient_id": patient_id,
            "read_policy": "native_level_0_no_upsample",
            "sample_id": sample_id,
            "source_object": f"{TIGER_IMAGE_PREFIX}/{sample_id}.tif",
            "source_url": f"{TIGER_IMAGE_BASE_URL}/{TIGER_IMAGE_PREFIX}/{sample_id}.tif",
        }
        for sample_id, (fold, patient_id) in {
            "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25": (
                "fold0",
                "TCGA-OL-A66I",
            ),
            "TCGA-OL-A66P-01Z-00-DX1.5ADD0D6D-37C6-4BC9-8C2B-64DB18BE99B3": (
                "fold2",
                "TCGA-OL-A66P",
            ),
            "TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014": (
                "fold1",
                "TCGA-OL-A6VO",
            ),
        }.items()
    ]


def test_beetle_curator_rejects_wrong_native_spacing_metadata(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")
    coarse_name = next(iter(COARSE_TCGA_SLIDES))
    _write_resolution_tiff(
        root / f"images/development/wsis/{coarse_name}.tif",
        0.5,
    )
    output = tmp_path / "curated"

    with pytest.raises(ValueError, match=r"native-spacing exception.*0\.5"):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_smoke_selects_available_rows_before_local_coverage_check(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")
    available = {"100B", "101B", "102B", "103B"}
    for path in (root / "images/development/wsis").glob("*.tif"):
        if path.stem not in available:
            path.unlink()
    for path in (root / "annotations/masks").glob("*.tif"):
        if path.stem not in available:
            path.unlink()

    manifest = curate_beetle_slide_manifest(overview, root, tmp_path / "smoke", slides=4)

    assert pd.read_csv(manifest.dataset_csv)["sample_id"].tolist() == [
        "100B",
        "101B",
        "102B",
        "103B",
    ]


def test_beetle_smoke_cannot_publish_to_canonical_output(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")
    canonical_output = root / "curated_slide_manifest"

    with pytest.raises(ValueError, match=r"non-publication smoke.*canonical"):
        curate_beetle_slide_manifest(overview, root, canonical_output, slides=4)

    assert not canonical_output.exists()


def test_beetle_curator_rejects_native_spacing_exception_fold_drift(tmp_path: Path):
    overview, root, rows = _full_cohort(tmp_path / "raw")
    coarse_name = next(iter(COARSE_TCGA_SLIDES))
    next(row for row in rows if row["name"] == coarse_name)["validation_fold"] = "fold4"
    pd.DataFrame(rows).to_csv(overview, index=False)
    output = tmp_path / "curated"

    with pytest.raises(ValueError, match=r"native-spacing exception.*organizer_fold"):
        curate_beetle_slide_manifest(overview, root, output)

    assert not output.exists()


def test_beetle_recuration_is_byte_identical(tmp_path: Path):
    overview, root, _ = _full_cohort(tmp_path / "raw")

    first = curate_beetle_slide_manifest(overview, root, tmp_path / "first")
    second = curate_beetle_slide_manifest(overview, root, tmp_path / "second")

    for filename in ("dataset.csv", "splits.csv", "summary.json"):
        assert (first.dataset_csv.parent / filename).read_bytes() == (
            second.dataset_csv.parent / filename
        ).read_bytes()
