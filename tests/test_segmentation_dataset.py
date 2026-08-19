"""Tests for the segmentation data plane: SegmentationManifest, Dataset, collate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from soma.dataset import SampleRecord, SegmentationManifest
from soma.dense import DenseFeatureStore, compute_dense_geometry, dense_grid_metadata, write_dense_grid
from soma.training.segmentation_dataset import (
    SegmentationBatch,
    SegmentationDataset,
    segmentation_collate_fn,
)

_TARGET = 32
_GRID = 2  # 32 / patch16
_DIM = 8


def _write_grid(store_dir: Path, sample_id: str, *, target: int = _TARGET) -> None:
    geom = compute_dense_geometry(target_size=target, patch_size=16)
    meta = dense_grid_metadata(geom, feature_dim=_DIM, pad_mode="reflect")
    write_dense_grid(store_dir, sample_id, torch.randn(_DIM, target // 16, target // 16), meta)


def _write_mask(path: Path, *, size: int = _TARGET) -> None:
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[: size // 2] = 1
    arr[0, 0] = 255  # ignore_index sentinel
    Image.fromarray(arr).save(path)


def _mask_target_fn(record: SampleRecord) -> dict[str, torch.Tensor]:
    arr = np.array(Image.open(record.label_mask_path))
    return {"mask": torch.from_numpy(arr).long()}


def _build(tmp_path: Path, n: int = 3) -> tuple[SegmentationManifest, DenseFeatureStore]:
    store_dir = tmp_path / "dense_embeddings"
    rows = []
    for i in range(n):
        sid = f"s{i}"
        _write_grid(store_dir, sid)
        label_mask_path = tmp_path / f"{sid}_mask.png"
        _write_mask(label_mask_path)
        rows.append(
            {"sample_id": sid, "image_path": f"/tiles/{sid}.png", "label_mask_path": str(label_mask_path)}
        )
    csv = tmp_path / "seg.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return SegmentationManifest(csv), DenseFeatureStore(store_dir)


# --------------------------------------------------------------------------- #
# SegmentationManifest
# --------------------------------------------------------------------------- #


def test_manifest_loads_without_label(tmp_path: Path):
    manifest, _ = _build(tmp_path, n=2)
    assert sorted(manifest.sample_ids) == ["s0", "s1"]
    rec = manifest.samples["s0"]
    assert rec.label is None  # segmentation has no scalar label
    assert rec.label_mask_path == tmp_path / "s0_mask.png"


def test_manifest_requires_mask_path_column(tmp_path: Path):
    csv = tmp_path / "seg.csv"
    pd.DataFrame([{"sample_id": "s0", "image_path": "/a.png"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Required column 'label_mask_path'"):
        SegmentationManifest(csv)


def test_manifest_rejects_null_mask_path(tmp_path: Path):
    csv = tmp_path / "seg.csv"
    pd.DataFrame(
        [
            {"sample_id": "s0", "image_path": "/a.png", "label_mask_path": "/m0.png"},
            {"sample_id": "s1", "image_path": "/b.png", "label_mask_path": None},
        ]
    ).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="label_mask_path is required"):
        SegmentationManifest(csv)


def test_manifest_rejects_unsafe_sample_id(tmp_path: Path):
    csv = tmp_path / "seg.csv"
    pd.DataFrame(
        [{"sample_id": "../escape", "image_path": "/a.png", "label_mask_path": "/m.png"}]
    ).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Unsafe sample_id"):
        SegmentationManifest(csv)


def test_manifest_keeps_optional_label(tmp_path: Path):
    csv = tmp_path / "seg.csv"
    pd.DataFrame(
        [{"sample_id": "s0", "image_path": "/a.png", "label_mask_path": "/m.png", "label": "tumor"}]
    ).to_csv(csv, index=False)
    assert SegmentationManifest(csv).samples["s0"].label == "tumor"


# --------------------------------------------------------------------------- #
# SegmentationDataset + collate
# --------------------------------------------------------------------------- #


def test_dataset_and_collate_roundtrip(tmp_path: Path):
    manifest, store = _build(tmp_path, n=3)
    records = list(manifest.samples.values())
    dataset = SegmentationDataset(records, store, _mask_target_fn)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: segmentation_collate_fn(b, target_dtypes={"mask": torch.long}),
    )
    batches = list(loader)
    assert isinstance(batches[0], SegmentationBatch)
    feats = batches[0].features
    masks = batches[0].targets["mask"]
    assert tuple(feats.shape) == (2, _DIM, _GRID, _GRID)
    assert tuple(masks.shape) == (2, _TARGET, _TARGET)
    assert masks.dtype == torch.long
    assert (masks == 255).any()  # ignore_index preserved through collation


def test_dataset_rejects_mask_size_mismatch(tmp_path: Path):
    manifest, store = _build(tmp_path, n=1)
    records = list(manifest.samples.values())

    def _wrong_size(record):  # mask 28x28 != recorded target_size 32x32
        return {"mask": torch.zeros(28, 28, dtype=torch.long)}

    dataset = SegmentationDataset(records, store, _wrong_size)
    with pytest.raises(ValueError, match="records target_size"):
        _ = dataset[0]


def test_dataset_requires_mask_key(tmp_path: Path):
    manifest, store = _build(tmp_path, n=1)
    records = list(manifest.samples.values())
    dataset = SegmentationDataset(records, store, lambda r: {"not_mask": torch.zeros(32, 32)})
    with pytest.raises(ValueError, match="must return a 'mask'"):
        _ = dataset[0]


def test_collate_fails_loud_on_nonuniform_grids():
    batch = [
        (torch.randn(_DIM, 2, 2), {"mask": torch.zeros(32, 32, dtype=torch.long)}, "s0"),
        (torch.randn(_DIM, 3, 3), {"mask": torch.zeros(32, 32, dtype=torch.long)}, "s1"),
    ]
    with pytest.raises(ValueError, match="dense grids in a batch must share shape"):
        segmentation_collate_fn(batch, target_dtypes={"mask": torch.long})


def test_collate_casts_mask_to_long_from_uint8():
    batch = [
        (torch.randn(_DIM, 2, 2), {"mask": torch.zeros(32, 32, dtype=torch.uint8)}, "s0"),
    ]
    out = segmentation_collate_fn(batch, target_dtypes={"mask": torch.long})
    assert out.targets["mask"].dtype == torch.long
