"""Tests for the unified Manifest schema, shared writer, and Protocol-typed curators."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soma.curation import (
    SUPERVISION_COLUMN,
    CuratedManifest,
    Curator,
    curate_beetle_slide_manifest,
    curate_eva_patch_dataset,
    curate_ocelot_detection,
    write_manifest,
)
from soma.dataset import (
    Dataset,
    DetectionManifest,
    SegmentationManifest,
    SpatialExpressionManifest,
    load_manifest,
)


# --------------------------------------------------------------------------- helpers


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_mhist_raw(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    annotations = pd.DataFrame(
        {
            "Image Name": [
                "ssa_train_0.png",
                "ssa_train_1.png",
                "hp_train_0.png",
                "hp_train_1.png",
                "ssa_test.png",
                "hp_test.png",
            ],
            "Majority Vote Label": ["SSA", "SSA", "HP", "HP", "SSA", "HP"],
            "Partition": ["train", "train", "train", "train", "test", "test"],
        }
    )
    annotations.to_csv(root / "annotations.csv", index=False)
    for image_name in annotations["Image Name"]:
        _touch(root / "images" / image_name)
    return root


def _make_ocelot_raw(root: Path) -> Path:
    for split, stem, points in [
        ("train", "001", [(42, 516, 2), (10, 20, 1)]),
        ("val", "010", [(100, 100, 1)]),
        ("test", "020", [(5, 5, 2)]),
    ]:
        img_dir = root / "images" / split / "cell"
        ann_dir = root / "annotations" / split / "cell"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{stem}.jpg").write_bytes(b"")
        with (ann_dir / f"{stem}.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows(points)
    return root


def _make_beetle_raw(root: Path, *, n_folds: int = 3) -> tuple[Path, Path]:
    """A synthetic BEETLE data_overview.csv + touched WSI/mask files."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_folds):
        wsi_rel = f"images/slide_{i}.tif"
        mask_rel = f"annotations/slide_{i}_mask.tif"
        _touch(root / wsi_rel)
        _touch(root / mask_rel)
        rows.append(
            {
                "name": f"slide_{i}",
                "split": "development",
                "wsi_path": wsi_rel,
                "annotation_mask_path": mask_rel,
                "patient_id": f"p{i}",
                "source": "srcA",
                "specimen_type": "biopsy",
                "validation_fold": f"fold{i}",
            }
        )
    overview = root / "data_overview.csv"
    pd.DataFrame(rows).to_csv(overview, index=False)
    return overview, root


# ------------------------------------------------------------------- write_manifest


def test_write_manifest_orders_columns_and_selects_supervision(tmp_path: Path):
    manifest = write_manifest(
        tmp_path / "out",
        dataset_type="tile",
        dataset_rows=[
            {"label": 1, "image_path": "/a.png", "sample_id": "s0", "class_name": "x"},
            {"label": 0, "image_path": "/b.png", "sample_id": "s1", "class_name": "y"},
        ],
        split_rows=[
            {"sample_id": "s0", "split": "train"},
            {"sample_id": "s1", "split": "test"},
        ],
        summary={"note": "hi"},
    )
    assert isinstance(manifest, CuratedManifest)
    df = pd.read_csv(manifest.dataset_csv)
    # sample_id, image_path, then the supervision column (label), then extras.
    assert list(df.columns) == ["sample_id", "image_path", "label", "class_name"]
    splits = pd.read_csv(manifest.splits_csv)
    # fold is always present; defaulted to 0 when the curator omits it.
    assert list(splits.columns) == ["sample_id", "split", "fold"]
    assert set(splits["fold"]) == {0}
    assert manifest.summary_json.exists()


def test_write_manifest_supervision_column_by_dataset_type(tmp_path: Path):
    seg = write_manifest(
        tmp_path / "seg",
        dataset_type="segmentation",
        dataset_rows=[{"sample_id": "s0", "image_path": "/a.tif", "label_mask_path": "/m.tif"}],
        split_rows=[{"sample_id": "s0", "split": "test"}],
        summary={},
    )
    det = write_manifest(
        tmp_path / "det",
        dataset_type="detection",
        dataset_rows=[{"sample_id": "s0", "image_path": "/a.jpg", "points_path": "/p.csv"}],
        split_rows=[{"sample_id": "s0", "split": "test"}],
        summary={},
    )
    assert list(pd.read_csv(seg.dataset_csv).columns)[:3] == ["sample_id", "image_path", "label_mask_path"]
    assert list(pd.read_csv(det.dataset_csv).columns)[:3] == ["sample_id", "image_path", "points_path"]


def test_write_manifest_rejects_missing_supervision_column(tmp_path: Path):
    with pytest.raises(ValueError, match="missing required column"):
        write_manifest(
            tmp_path / "out",
            dataset_type="segmentation",
            dataset_rows=[{"sample_id": "s0", "image_path": "/a.tif"}],  # no label_mask_path
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


def test_write_manifest_rejects_two_supervision_columns(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one supervision column"):
        write_manifest(
            tmp_path / "out",
            dataset_type="tile",
            dataset_rows=[
                {"sample_id": "s0", "image_path": "/a.png", "label": 1, "label_mask_path": "/m.tif"}
            ],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


def test_write_manifest_accepts_tissue_mask_path_for_every_dataset_type(tmp_path: Path):
    # mask_path is the optional tissue mask, not supervision: it sits right after
    # image_path and is valid beside label / label_mask_path / points_path alike.
    for dtype, supervision, value in [
        ("slide", "label", 1),
        ("segmentation", "label_mask_path", "/m.tif"),
        ("detection", "points_path", "/p.csv"),
    ]:
        out = write_manifest(
            tmp_path / dtype,
            dataset_type=dtype,
            dataset_rows=[
                {"sample_id": "s0", "image_path": "/a.tif", supervision: value, "mask_path": "/t.tif"}
            ],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )
        df = pd.read_csv(out.dataset_csv)
        assert list(df.columns)[:4] == ["sample_id", "image_path", "mask_path", supervision]
        record = next(iter(load_manifest(out.dataset_csv, dtype).samples.values()))
        assert record.mask_path == Path("/t.tif")


def test_segmentation_loader_rejects_pre_rename_manifest(tmp_path: Path):
    # A stale manifest carrying the supervision raster as mask_path must fail loud
    # naming the rename, never be reinterpreted as a tissue-mask row missing supervision.
    stale = tmp_path / "stale.csv"
    stale.write_text("sample_id,image_path,mask_path\ns0,/a.tif,/m.tif\n")
    with pytest.raises(ValueError, match="pre-rename segmentation manifest"):
        SegmentationManifest(stale)


def test_segmentation_row_carries_both_tissue_and_label_masks(tmp_path: Path):
    seg = tmp_path / "seg.csv"
    seg.write_text("sample_id,image_path,mask_path,label_mask_path\ns0,/a.tif,/t.tif,/m.tif\n")
    record = SegmentationManifest(seg).samples["s0"]
    assert record.mask_path == Path("/t.tif")
    assert record.label_mask_path == Path("/m.tif")


def test_write_manifest_rejects_unknown_dataset_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        write_manifest(
            tmp_path / "out",
            dataset_type="bogus",
            dataset_rows=[{"sample_id": "s0", "image_path": "/a.png", "label": 1}],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


@pytest.mark.parametrize("spacing", [0.0, -0.25, float("inf"), True, "invalid"])
def test_write_manifest_rejects_invalid_source_spacing(tmp_path: Path, spacing):
    with pytest.raises(
        ValueError,
        match=r"spacing_at_level_0 must be a positive, finite number or blank",
    ):
        write_manifest(
            tmp_path / "out",
            dataset_type="detection",
            dataset_rows=[
                {
                    "sample_id": "s0",
                    "image_path": "/a.jpg",
                    "points_path": "/p.csv",
                    "spacing_at_level_0": spacing,
                }
            ],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
        )


# ------------------------------------------------------------- load-time validation


def test_load_manifest_selects_loader_by_dataset_type(tmp_path: Path):
    ds = tmp_path / "cls.csv"
    ds.write_text("sample_id,image_path,label\ns0,/a.png,1\n")
    assert isinstance(load_manifest(ds, "tile"), Dataset)
    assert isinstance(load_manifest(ds, "slide"), Dataset)

    seg = tmp_path / "seg.csv"
    seg.write_text("sample_id,image_path,label_mask_path\ns0,/a.tif,/m.tif\n")
    assert isinstance(load_manifest(seg, "segmentation"), SegmentationManifest)

    det = tmp_path / "det.csv"
    det.write_text("sample_id,image_path,points_path\ns0,/a.jpg,/p.csv\n")
    assert isinstance(load_manifest(det, "detection"), DetectionManifest)


def test_load_manifest_rejects_malformed_supervision_column(tmp_path: Path):
    # A segmentation manifest missing label_mask_path is rejected fail-fast with a clear message.
    bad = tmp_path / "bad.csv"
    bad.write_text("sample_id,image_path,label\ns0,/a.tif,1\n")
    with pytest.raises(ValueError, match="label_mask_path"):
        load_manifest(bad, "segmentation")


@pytest.mark.parametrize(
    ("dataset_type", "supervision_column", "supervision_value"),
    [
        ("tile", "label", 1),
        ("segmentation", "label_mask_path", "/m.png"),
        ("detection", "points_path", "/p.csv"),
        ("spatial_expression", "target_index", 0),
    ],
)
def test_manifest_round_trips_typed_source_spacing(
    tmp_path: Path, dataset_type: str, supervision_column: str, supervision_value
):
    sidecars = (
        {"target_matrix": np.array([[1.0]]), "genes": ["GENE"]}
        if dataset_type == "spatial_expression"
        else {}
    )
    manifest = write_manifest(
        tmp_path / dataset_type,
        dataset_type=dataset_type,
        dataset_rows=[
            {
                "sample_id": "s0",
                "image_path": "/a.jpg",
                supervision_column: supervision_value,
                "spacing_at_level_0": 0.25,
            }
        ],
        split_rows=[{"sample_id": "s0", "split": "test"}],
        summary={},
        **sidecars,
    )

    record = load_manifest(manifest.dataset_csv, dataset_type).samples["s0"]
    assert record.spacing_at_level_0 == 0.25
    assert record.metadata == {}


@pytest.mark.parametrize(
    ("dataset_type", "supervision_column", "supervision_value"),
    [
        ("tile", "label", "1"),
        ("segmentation", "label_mask_path", "/m.png"),
        ("detection", "points_path", "/p.csv"),
        ("spatial_expression", "target_index", "0"),
    ],
)
def test_load_manifest_rejects_invalid_source_spacing(
    tmp_path: Path, dataset_type: str, supervision_column: str, supervision_value: str
):
    manifest = tmp_path / "dataset.csv"
    manifest.write_text(
        f"sample_id,image_path,{supervision_column},spacing_at_level_0\n"
        f"s0,/a.jpg,{supervision_value},0\n"
    )
    if dataset_type == "spatial_expression":
        np.save(tmp_path / "targets.npy", np.array([[1.0]]))
        (tmp_path / "genes.json").write_text('["GENE"]')

    with pytest.raises(
        ValueError,
        match=r"spacing_at_level_0 must be a positive, finite number or blank",
    ):
        load_manifest(manifest, dataset_type)


@pytest.mark.parametrize(
    ("dataset_type", "supervision_column", "supervision_value"),
    [
        ("tile", "label", "1"),
        ("segmentation", "label_mask_path", "/m.png"),
        ("detection", "points_path", "/p.csv"),
        ("spatial_expression", "target_index", "0"),
    ],
)
def test_load_manifest_rejects_retired_source_spacing(
    tmp_path: Path, dataset_type: str, supervision_column: str, supervision_value: str
):
    manifest = tmp_path / "dataset.csv"
    manifest.write_text(
        f"sample_id,image_path,{supervision_column},level0_spacing\n"
        f"s0,/a.jpg,{supervision_value},0.25\n"
    )
    if dataset_type == "spatial_expression":
        np.save(tmp_path / "targets.npy", np.array([[1.0]]))
        (tmp_path / "genes.json").write_text('["GENE"]')

    with pytest.raises(
        ValueError,
        match="Manifest column 'level0_spacing' is retired; use 'spacing_at_level_0'",
    ):
        load_manifest(manifest, dataset_type)


# ----------------------------------------------------------------- Curator Protocol


def test_curators_satisfy_structural_protocol_without_base_class():
    # ADR 0004: no base class — curators are just deterministic functions that return a
    # CuratedManifest, so they satisfy the structural Curator Protocol via isinstance.
    for curator in (
        curate_eva_patch_dataset,
        curate_ocelot_detection,
        curate_beetle_slide_manifest,
    ):
        assert isinstance(curator, Curator)
    # None of them inherit from Curator (it is a Protocol, never a base class).
    assert Curator not in type(curate_eva_patch_dataset).__mro__


def test_curated_manifest_lives_in_neutral_module():
    from soma.curation import manifest as neutral

    assert CuratedManifest is neutral.CuratedManifest
    # It is NOT re-exported from the EVA curator anymore.
    import soma.curation.eva as eva

    assert getattr(eva, "CuratedManifest", None) is neutral.CuratedManifest


# --------------------------------------------------- every curator emits the schema


def test_all_curators_emit_identical_core_schema(tmp_path: Path):
    eva = curate_eva_patch_dataset(
        "mhist", _make_mhist_raw(tmp_path / "mhist_raw"), tmp_path / "eva", tune_fraction=0.5
    )
    ocelot = curate_ocelot_detection(_make_ocelot_raw(tmp_path / "oce_raw"), tmp_path / "oce")
    overview, beetle_root = _make_beetle_raw(tmp_path / "beetle_raw")
    beetle = curate_beetle_slide_manifest(
        overview, beetle_root, tmp_path / "beetle", slides=3
    )

    for manifest, supervision in [
        (eva, "label"),
        (ocelot, "points_path"),
        (beetle, "label_mask_path"),
    ]:
        ds = pd.read_csv(manifest.dataset_csv)
        assert list(ds.columns)[:3] == ["sample_id", "image_path", supervision]
        splits = pd.read_csv(manifest.splits_csv)
        assert list(splits.columns)[:3] == ["sample_id", "split", "fold"]
        assert manifest.summary_json is not None and manifest.summary_json.exists()


def test_beetle_curator_emits_dataset_csv_not_manifest_csv(tmp_path: Path):
    overview, beetle_root = _make_beetle_raw(tmp_path / "beetle_raw")
    out = tmp_path / "beetle"
    manifest = curate_beetle_slide_manifest(overview, beetle_root, out, slides=3)

    assert manifest.dataset_csv == out / "dataset.csv"
    assert (out / "dataset.csv").exists()
    assert not (out / "manifest.csv").exists()  # the old shape is retired
    assert json.loads(manifest.summary_json.read_text())["cohort"]["kind"] == (
        "non_publication_smoke_subset"
    )
    # Loads through the segmentation loader (label_mask_path supervision).
    seg = SegmentationManifest(manifest.dataset_csv)
    assert set(seg.sample_ids) == {"slide_0", "slide_1", "slide_2"}


# ------------------------------------------------------------- determinism contract


def test_recuration_is_byte_identical(tmp_path: Path):
    raw = _make_mhist_raw(tmp_path / "mhist_raw")
    a = curate_eva_patch_dataset("mhist", raw, tmp_path / "a", tune_fraction=0.5)
    b = curate_eva_patch_dataset("mhist", raw, tmp_path / "b", tune_fraction=0.5)

    for name in ("dataset.csv", "splits.csv", "summary.json"):
        assert (a.dataset_csv.parent / name).read_bytes() == (
            b.dataset_csv.parent / name
        ).read_bytes(), f"{name} is not byte-identical across re-curation"


# ------------------------------------------------ spatial_expression multi-target shape


def _spatial_expression_inputs():
    """A tiny hand-built spatial_expression Manifest: 3 spots, 4 genes."""
    genes = ["GENE_C", "GENE_A", "GENE_B", "GENE_D"]  # deliberately non-alphabetical
    matrix = np.array(
        [
            [0.0, 1.5, 2.25, 3.125],
            [4.0, 5.0, 6.0, 7.0],
            [0.125, 0.25, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    dataset_rows = [
        {"sample_id": "spot0", "image_path": "/img/spot0.png", "target_index": 0},
        {"sample_id": "spot1", "image_path": "/img/spot1.png", "target_index": 1},
        {"sample_id": "spot2", "image_path": "/img/spot2.png", "target_index": 2},
    ]
    split_rows = [
        {"sample_id": "spot0", "split": "train"},
        {"sample_id": "spot1", "split": "train"},
        {"sample_id": "spot2", "split": "test"},
    ]
    return genes, matrix, dataset_rows, split_rows


def test_spatial_expression_supervision_column():
    # The new dataset_type carries exactly one supervision column: target_index.
    assert SUPERVISION_COLUMN["spatial_expression"] == "target_index"
    # Existing dataset_types are unaffected.
    assert SUPERVISION_COLUMN["tile"] == "label"
    assert SUPERVISION_COLUMN["slide"] == "label"
    assert SUPERVISION_COLUMN["patient"] == "label"
    assert SUPERVISION_COLUMN["segmentation"] == "label_mask_path"
    assert SUPERVISION_COLUMN["detection"] == "points_path"


def test_write_manifest_spatial_expression_emits_sidecars(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    manifest = write_manifest(
        tmp_path / "se",
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={"task": "IDC"},
        target_matrix=matrix,
        genes=genes,
    )
    # dataset.csv carries the target_index supervision column in canonical order.
    df = pd.read_csv(manifest.dataset_csv)
    assert list(df.columns)[:3] == ["sample_id", "image_path", "target_index"]
    # Sidecars are written beside dataset.csv and surfaced on the CuratedManifest.
    out = manifest.dataset_csv.parent
    assert (out / "targets.npy").exists()
    assert (out / "genes.json").exists()
    assert manifest.target_matrix_path == out / "targets.npy"
    assert manifest.genes_path == out / "genes.json"
    # Round-trip the sidecar artifacts.
    np.testing.assert_array_equal(np.load(out / "targets.npy"), matrix)
    assert json.loads((out / "genes.json").read_text()) == genes


def test_write_manifest_spatial_expression_requires_sidecar_inputs(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    with pytest.raises(ValueError, match="target_matrix"):
        write_manifest(
            tmp_path / "no_matrix",
            dataset_type="spatial_expression",
            dataset_rows=dataset_rows,
            split_rows=split_rows,
            summary={},
            genes=genes,
        )
    with pytest.raises(ValueError, match="genes"):
        write_manifest(
            tmp_path / "no_genes",
            dataset_type="spatial_expression",
            dataset_rows=dataset_rows,
            split_rows=split_rows,
            summary={},
            target_matrix=matrix,
        )


def test_write_manifest_rejects_out_of_range_target_index(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    dataset_rows[2]["target_index"] = 99  # matrix only has 3 rows
    with pytest.raises(ValueError, match="out of range"):
        write_manifest(
            tmp_path / "oob",
            dataset_type="spatial_expression",
            dataset_rows=dataset_rows,
            split_rows=split_rows,
            summary={},
            target_matrix=matrix,
            genes=genes,
        )


def test_write_manifest_rejects_gene_count_mismatch(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    with pytest.raises(ValueError, match="gene"):
        write_manifest(
            tmp_path / "mismatch",
            dataset_type="spatial_expression",
            dataset_rows=dataset_rows,
            split_rows=split_rows,
            summary={},
            target_matrix=matrix,
            genes=genes[:-1],  # one fewer gene than matrix columns
        )


def test_write_manifest_rejects_sidecars_for_non_spatial_dataset_type(tmp_path: Path):
    genes, matrix, _, _ = _spatial_expression_inputs()
    with pytest.raises(ValueError, match="spatial_expression"):
        write_manifest(
            tmp_path / "tile",
            dataset_type="tile",
            dataset_rows=[{"sample_id": "s0", "image_path": "/a.png", "label": 1}],
            split_rows=[{"sample_id": "s0", "split": "test"}],
            summary={},
            target_matrix=matrix,
            genes=genes,
        )


def test_load_manifest_selects_spatial_expression_loader(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    manifest = write_manifest(
        tmp_path / "se",
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={},
        target_matrix=matrix,
        genes=genes,
    )
    loaded = load_manifest(manifest.dataset_csv, "spatial_expression")
    assert isinstance(loaded, SpatialExpressionManifest)


def test_spatial_expression_targets_round_trip_exactly(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    manifest = write_manifest(
        tmp_path / "se",
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={},
        target_matrix=matrix,
        genes=genes,
    )
    loaded = SpatialExpressionManifest(manifest.dataset_csv)
    # Gene order is preserved verbatim from the sidecar.
    assert loaded.genes == genes
    # Each sample record carries the resolved vector target for its target_index row.
    for row in dataset_rows:
        rec = loaded.samples[row["sample_id"]]
        np.testing.assert_array_equal(rec.target, matrix[row["target_index"]])
        assert rec.label is None
    # The whole target matrix is exposed and byte-exact.
    np.testing.assert_array_equal(loaded.target_matrix, matrix)


def test_spatial_expression_loader_rejects_missing_sidecars(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    manifest = write_manifest(
        tmp_path / "se",
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={},
        target_matrix=matrix,
        genes=genes,
    )
    out = manifest.dataset_csv.parent
    (out / "targets.npy").unlink()
    with pytest.raises(ValueError, match="targets.npy"):
        SpatialExpressionManifest(manifest.dataset_csv)


def test_spatial_expression_loader_rejects_missing_column(tmp_path: Path):
    # A dataset.csv without target_index cannot be loaded as spatial_expression.
    bad = tmp_path / "dataset.csv"
    bad.write_text("sample_id,image_path,label\nspot0,/a.png,1\n")
    np.save(tmp_path / "targets.npy", np.zeros((1, 2)))
    (tmp_path / "genes.json").write_text(json.dumps(["A", "B"]))
    with pytest.raises(ValueError, match="target_index"):
        SpatialExpressionManifest(bad)


def test_spatial_expression_recuration_is_byte_identical(tmp_path: Path):
    genes, matrix, dataset_rows, split_rows = _spatial_expression_inputs()
    kwargs = dict(
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={"task": "IDC"},
        target_matrix=matrix,
        genes=genes,
    )
    a = write_manifest(tmp_path / "a", **kwargs)
    b = write_manifest(tmp_path / "b", **kwargs)
    for name in ("dataset.csv", "splits.csv", "summary.json", "targets.npy", "genes.json"):
        assert (a.dataset_csv.parent / name).read_bytes() == (
            b.dataset_csv.parent / name
        ).read_bytes(), f"{name} is not byte-identical across re-write"
