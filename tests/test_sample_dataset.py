"""Tests for soma.training — SampleDataset and sample_collate_fn."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.dataset import SampleRecord
from soma.features import FeatureStore
from soma.training.sample_dataset import SampleDataset, SampleBatch, sample_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_store(tmp_path: Path, sample_ids: list[str], feature_dim: int) -> FeatureStore:
    for sid in sample_ids:
        torch.save(torch.randn(feature_dim), tmp_path / f"{sid}.pt")
    return FeatureStore(tmp_path)


def _make_record(sample_id: str, label: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        image_path=Path(f"/slides/{sample_id}.svs"),
        label=label,
    )


LABEL_MAP = {"normal": 0, "tumor": 1}
TARGET_DTYPES = {"label": torch.long}


def _target_fn(record: SampleRecord) -> dict[str, int]:
    return {"label": LABEL_MAP[record.label]}


# ---------------------------------------------------------------------------
# SampleDataset
# ---------------------------------------------------------------------------


class TestSampleDataset:
    def test_getitem_returns_1d_features(self, tmp_path: Path):
        store = _create_store(tmp_path, ["s1", "s2"], feature_dim=512)
        records = [_make_record("s1", "normal"), _make_record("s2", "tumor")]
        ds = SampleDataset(records, store, _target_fn)

        features, targets, sample_id = ds[0]
        assert features.shape == (512,)
        assert targets == {"label": 0}
        assert sample_id == "s1"

    def test_getitem_label_mapping(self, tmp_path: Path):
        store = _create_store(tmp_path, ["s1"], feature_dim=256)
        records = [_make_record("s1", "tumor")]
        ds = SampleDataset(records, store, _target_fn)

        _, targets, _ = ds[0]
        assert targets == {"label": 1}

    def test_len(self, tmp_path: Path):
        store = _create_store(tmp_path, ["s1", "s2", "s3"], feature_dim=64)
        records = [_make_record(f"s{i}", "normal") for i in range(1, 4)]
        ds = SampleDataset(records, store, _target_fn)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# sample_collate_fn
# ---------------------------------------------------------------------------


class TestSampleCollateFn:
    def test_stacks_into_batch(self):
        D = 512
        batch = [
            (torch.randn(D), {"label": 0}, "s1"),
            (torch.randn(D), {"label": 1}, "s2"),
            (torch.randn(D), {"label": 0}, "s3"),
        ]
        result = sample_collate_fn(batch, TARGET_DTYPES)
        assert isinstance(result, SampleBatch)
        assert result.features.shape == (3, D)
        assert result.targets["label"].shape == (3,)
        assert result.sample_ids == ("s1", "s2", "s3")

    def test_targets_tensor(self):
        D = 8
        batch = [
            (torch.randn(D), {"label": 0}, "s1"),
            (torch.randn(D), {"label": 1}, "s2"),
            (torch.randn(D), {"label": 2}, "s3"),
        ]
        result = sample_collate_fn(batch, TARGET_DTYPES)
        assert torch.equal(result.targets["label"], torch.tensor([0, 1, 2]))

    def test_no_mask_attribute(self):
        D = 8
        batch = [(torch.randn(D), {"label": 0}, "s1")]
        result = sample_collate_fn(batch, TARGET_DTYPES)
        assert not hasattr(result, "mask")
