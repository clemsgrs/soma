"""Tests for soma.training — SlideDataset and slide_collate_fn."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.dataset import SampleRecord
from soma.features import FeatureStore
from soma.training.slide_dataset import SlideDataset, SlideBatch, slide_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_slide_store(tmp_path: Path, sample_ids: list[str], feature_dim: int) -> FeatureStore:
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


# ---------------------------------------------------------------------------
# SlideDataset
# ---------------------------------------------------------------------------


class TestSlideDataset:
    def test_getitem_returns_1d_features(self, tmp_path: Path):
        store = _create_slide_store(tmp_path, ["s1", "s2"], feature_dim=512)
        records = [_make_record("s1", "normal"), _make_record("s2", "tumor")]
        ds = SlideDataset(records, store, LABEL_MAP)

        features, label, sample_id = ds[0]
        assert features.shape == (512,)
        assert label == 0
        assert sample_id == "s1"

    def test_getitem_label_mapping(self, tmp_path: Path):
        store = _create_slide_store(tmp_path, ["s1"], feature_dim=256)
        records = [_make_record("s1", "tumor")]
        ds = SlideDataset(records, store, LABEL_MAP)

        _, label, _ = ds[0]
        assert label == 1

    def test_len(self, tmp_path: Path):
        store = _create_slide_store(tmp_path, ["s1", "s2", "s3"], feature_dim=64)
        records = [_make_record(f"s{i}", "normal") for i in range(1, 4)]
        ds = SlideDataset(records, store, LABEL_MAP)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# slide_collate_fn
# ---------------------------------------------------------------------------


class TestSlideCollateFn:
    def test_stacks_into_batch(self):
        D = 512
        batch = [
            (torch.randn(D), 0, "s1"),
            (torch.randn(D), 1, "s2"),
            (torch.randn(D), 0, "s3"),
        ]
        result = slide_collate_fn(batch)
        assert isinstance(result, SlideBatch)
        assert result.features.shape == (3, D)
        assert result.labels.shape == (3,)
        assert result.sample_ids == ("s1", "s2", "s3")

    def test_labels_tensor(self):
        D = 8
        batch = [
            (torch.randn(D), 0, "s1"),
            (torch.randn(D), 1, "s2"),
            (torch.randn(D), 2, "s3"),
        ]
        result = slide_collate_fn(batch)
        assert torch.equal(result.labels, torch.tensor([0, 1, 2]))

    def test_no_mask_attribute(self):
        D = 8
        batch = [(torch.randn(D), 0, "s1")]
        result = slide_collate_fn(batch)
        assert not hasattr(result, "mask")
