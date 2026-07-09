"""Tests for HEST-Benchmark spatial-expression curation (curate_hest).

The curator turns a pre-provisioned hest-bench task tree (per-slide patch HDF5 +
gene-expression ``.h5ad`` + per-fold slide split CSVs + a task gene list) into soma's
unified ``spatial_expression`` Manifest. These tests drive it against a tiny synthetic
hest-bench fixture built in-process (no network, no real HEST data); prior art is the
EVA/OCELOT curator determinism tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# The curator's readers (anndata for .h5ad, h5py for the patch HDF5) live in the
# soma[hest] extra; the fixture builder needs them too. Skip the whole module when the
# extra is absent so core CI (which does not install it) stays green.
anndata = pytest.importorskip("anndata")
h5py = pytest.importorskip("h5py")

from soma.curation.hest import curate_hest  # noqa: E402
from soma.dataset import SpatialExpressionManifest, Splits  # noqa: E402

# Silence anndata's pandas>=3 Arrow-string write guard when building fixtures.
anndata.settings.allow_write_nullable_strings = True


# ------------------------------------------------------------------- synthetic fixture

# Task gene list (kept tiny; the code is general to HEST's 50 genes). The .h5ad var set
# is a *superset* in a *different* order, so a correct curator must subselect + reorder.
FIXTURE_GENES = ["GENA", "GENB", "GENC"]
FIXTURE_VAR_NAMES = ["GEXTRA", "GENC", "GENA", "GENB"]  # shuffled superset

# Per-slide spots: barcode -> raw integer counts over FIXTURE_VAR_NAMES (var order).
SLIDES: dict[str, dict[str, list[int]]] = {
    "S1": {"AAAA-1": [7, 3, 0, 11], "CCCC-1": [1, 4, 9, 2]},
    "S2": {"GGGG-1": [0, 0, 5, 6], "TTTT-1": [8, 2, 3, 1]},
    "S3": {"ACGT-1": [4, 10, 1, 0]},
}

# HEST ships slide-level train_i/test_i per fold (no tune). Two folds here.
FOLDS = {
    0: {"train": ["S1", "S2"], "test": ["S3"]},
    1: {"train": ["S1", "S3"], "test": ["S2"]},
}

TILE_H, TILE_W = 5, 4  # non-square on purpose to catch H/W transposition bugs


def _spot_image(slide: str, barcode: str) -> np.ndarray:
    """A deterministic per-spot RGB tile (stable across runs, distinct per spot)."""
    seed = abs(hash((slide, barcode))) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(TILE_H, TILE_W, 3), dtype=np.uint8)


def _write_patches_h5(path: Path, barcodes: list[str], images: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = np.stack(images).astype(np.uint8)  # [N, H, W, 3]
    coords = np.arange(len(barcodes) * 2, dtype=np.int64).reshape(-1, 2)
    bc = np.array([[b.encode("utf-8")] for b in barcodes])  # [N, 1] bytes, like HEST
    with h5py.File(path, "w") as f:
        f.create_dataset("img", data=imgs)
        f.create_dataset("coords", data=coords)
        f.create_dataset("barcode", data=bc)


def _write_adata_h5ad(
    path: Path, barcodes: list[str], counts_by_bc: dict[str, list[int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # obs (spot) order deliberately reversed vs the patch order to force barcode-based
    # alignment rather than positional alignment.
    obs_order = list(reversed(barcodes))
    X = np.array([counts_by_bc[b] for b in obs_order], dtype=np.float32)
    obs = pd.DataFrame(index=pd.Index(obs_order, dtype=object))
    var = pd.DataFrame(index=pd.Index(list(FIXTURE_VAR_NAMES), dtype=object))
    anndata.AnnData(X=X, obs=obs, var=var).write_h5ad(path)


def _make_fixture(root: Path, task: str = "IDC") -> Path:
    """Build a tiny hest-bench task tree under ``root/<task>`` and return that dir."""
    task_dir = root / task
    (task_dir / "patches").mkdir(parents=True, exist_ok=True)
    (task_dir / "adata").mkdir(parents=True, exist_ok=True)
    (task_dir / "splits").mkdir(parents=True, exist_ok=True)

    for slide, counts_by_bc in SLIDES.items():
        barcodes = list(counts_by_bc.keys())
        images = [_spot_image(slide, b) for b in barcodes]
        _write_patches_h5(task_dir / "patches" / f"{slide}.h5", barcodes, images)
        _write_adata_h5ad(task_dir / "adata" / f"{slide}.h5ad", barcodes, counts_by_bc)

    (task_dir / "var_50genes.json").write_text(json.dumps({"genes": FIXTURE_GENES}))

    for fold, members in FOLDS.items():
        for split in ("train", "test"):
            rows = [
                {
                    "sample_id": slide,
                    "patches_path": f"patches/{slide}.h5",
                    "expr_path": f"adata/{slide}.h5ad",
                }
                for slide in members[split]
            ]
            pd.DataFrame(rows).to_csv(task_dir / "splits" / f"{split}_{fold}.csv", index=False)
    return task_dir


def _reference_target(slide: str, barcode: str) -> np.ndarray:
    """log1p(raw counts) for FIXTURE_GENES, computed straight from the fixture."""
    counts = SLIDES[slide][barcode]
    gene_pos = [FIXTURE_VAR_NAMES.index(g) for g in FIXTURE_GENES]
    raw = np.array([counts[p] for p in gene_pos], dtype=np.float64)
    return np.log1p(raw)


# --------------------------------------------------------------------------- schema


def test_curate_hest_emits_spatial_expression_manifest(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"

    manifest = curate_hest(task_dir, out)

    assert manifest.dataset_csv == out / "dataset.csv"
    assert manifest.splits_csv == out / "splits.csv"
    assert manifest.target_matrix_path == out / "targets.npy"
    assert manifest.genes_path == out / "genes.json"
    assert (out / "summary.json").exists()

    # Loads through soma's spatial_expression loader (schema + sidecars validated there).
    loaded = SpatialExpressionManifest(manifest.dataset_csv)
    n_spots = sum(len(v) for v in SLIDES.values())
    assert len(loaded.sample_ids) == n_spots
    assert loaded.genes == FIXTURE_GENES
    assert loaded.target_matrix.shape == (n_spots, len(FIXTURE_GENES))

    df = pd.read_csv(manifest.dataset_csv)
    assert {"sample_id", "image_path", "target_index"} <= set(df.columns)
    # target_index is the supervision column; no scalar/label/mask columns leak in.
    assert "label" not in df.columns and "mask_path" not in df.columns


# ------------------------------------------------------------------- baked targets


def test_targets_are_log1p_of_raw_counts_in_gene_order(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"
    curate_hest(task_dir, out)

    genes = json.loads((out / "genes.json").read_text())
    assert genes == FIXTURE_GENES  # order preserved, matches the task gene list

    df = pd.read_csv(out / "dataset.csv")
    targets = np.load(out / "targets.npy")
    assert targets.dtype == np.float64
    for _, row in df.iterrows():
        expected = _reference_target(str(row["slide_id"]), str(row["barcode"]))
        np.testing.assert_allclose(targets[int(row["target_index"])], expected, rtol=0, atol=0)


# --------------------------------------------------------------- per-fold expansion


def test_per_fold_spot_row_expansion(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"
    curate_hest(task_dir, out)

    df = pd.read_csv(out / "dataset.csv")
    splits = pd.read_csv(out / "splits.csv")

    spots_by_slide = df.groupby("slide_id")["sample_id"].apply(set).to_dict()

    assert set(splits["fold"]) == set(FOLDS)  # fold count matches the fixture
    assert set(splits["split"]) == {"train", "test"}  # no tune rows

    for fold, members in FOLDS.items():
        fold_rows = splits[splits["fold"] == fold]
        for split in ("train", "test"):
            expected_spots: set[str] = set()
            for slide in members[split]:
                expected_spots |= spots_by_slide[slide]
            got = set(fold_rows[fold_rows["split"] == split]["sample_id"])
            assert got == expected_spots, f"fold {fold} {split} mismatch"

    # The Manifest + splits load and build folds through soma unchanged (no tune needed).
    loaded = SpatialExpressionManifest(out / "dataset.csv")
    folds = Splits(out / "splits.csv", loaded).folds
    assert len(folds) == len(FOLDS)
    for i, members in FOLDS.items():
        expected_test: set[str] = set()
        for slide in members["test"]:
            expected_test |= spots_by_slide[slide]
        assert set(folds[i].tests["test"]) == expected_test
        assert folds[i].tune == ()  # HEST has no tune split


# --------------------------------------------------------------- lossless tiles


def test_exploded_tiles_are_pixel_identical_to_source(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"
    curate_hest(task_dir, out)

    df = pd.read_csv(out / "dataset.csv")
    for _, row in df.iterrows():
        slide, barcode = str(row["slide_id"]), str(row["barcode"])
        source = _spot_image(slide, barcode)
        with Image.open(row["image_path"]) as im:
            assert im.format == "PNG"  # lossless format
            decoded = np.array(im)
        np.testing.assert_array_equal(decoded, source)


# --------------------------------------------------------------- determinism


def test_recuration_is_byte_identical(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"

    # image_path is absolute (downstream opens it directly), so re-curating into the
    # *same* dir is the location-independent determinism check: identical bytes twice.
    curate_hest(task_dir, out)
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    curate_hest(task_dir, out)
    second = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}

    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], f"{name} is not byte-identical across re-curation"


def test_summary_reports_folds_and_spot_counts(tmp_path: Path):
    task_dir = _make_fixture(tmp_path / "raw")
    out = tmp_path / "curated"
    curate_hest(task_dir, out)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["dataset_type"] == "spatial_expression"
    assert summary["task"] == "IDC"
    assert summary["num_genes"] == len(FIXTURE_GENES)
    assert summary["num_folds"] == len(FOLDS)
    assert summary["num_slides"] == len(SLIDES)
    assert summary["num_spots"] == sum(len(v) for v in SLIDES.values())


# --------------------------------------------------------------- packaging / lazy import


def test_hest_module_import_does_not_eagerly_load_curation_deps():
    """Importing the curator module must not import anndata/h5py (lazy inside the fn)."""
    code = (
        "import sys, soma.curation.hest;"
        "print('anndata' in sys.modules, 'h5py' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_pyproject_declares_hest_extra():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "hest" in extras
    joined = " ".join(extras["hest"]).lower()
    assert "anndata" in joined and "h5py" in joined
