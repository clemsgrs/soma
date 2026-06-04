"""Tests for soma.training — BagDataset and bag_collate_fn."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.dataset import SampleRecord
from soma.features import FeatureStore
from soma.training.bag_dataset import BagDataset
from soma.training.collate import BagBatch, bag_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_feature_store(tmp_path: Path, samples: dict[str, tuple[int, int]]) -> FeatureStore:
    """Create a feature store with synthetic features.

    Args:
        samples: {sample_id: (num_tiles, feature_dim)}
    """
    for sample_id, (n, d) in samples.items():
        torch.save(torch.randn(n, d), tmp_path / f"{sample_id}.pt")
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
# BagDataset
# ---------------------------------------------------------------------------


class TestBagDataset:
    def test_getitem(self, tmp_path: Path):
        torch.manual_seed(0)
        store = _create_feature_store(tmp_path, {"s1": (10, 8), "s2": (20, 8)})
        records = [_make_record("s1", "normal"), _make_record("s2", "tumor")]
        ds = BagDataset(records, store, _target_fn)

        features, targets, sample_id = ds[0]
        assert features.shape == (10, 8)
        assert targets == {"label": 0}  # "normal" → 0
        assert sample_id == "s1"

        features, targets, sample_id = ds[1]
        assert features.shape == (20, 8)
        assert targets == {"label": 1}  # "tumor" → 1
        assert sample_id == "s2"

    def test_len(self, tmp_path: Path):
        store = _create_feature_store(tmp_path, {"s1": (5, 4), "s2": (5, 4), "s3": (5, 4)})
        records = [_make_record(f"s{i}", "normal") for i in range(1, 4)]
        ds = BagDataset(records, store, _target_fn)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# bag_collate_fn
# ---------------------------------------------------------------------------


class TestBagCollateFn:
    def test_pads_to_max_length(self):
        """Three bags of sizes 10, 20, 5 should pad to (3, 20, D)."""
        D = 8
        batch = [
            (torch.randn(10, D), {"label": 0}, "s1"),
            (torch.randn(20, D), {"label": 1}, "s2"),
            (torch.randn(5, D), {"label": 0}, "s3"),
        ]
        result = bag_collate_fn(batch, TARGET_DTYPES)
        assert isinstance(result, BagBatch)
        assert result.features.shape == (3, 20, D)
        assert result.mask.shape == (3, 20)
        assert result.targets["label"].shape == (3,)
        assert result.sample_ids == ("s1", "s2", "s3")

    def test_mask_correctness(self):
        """Mask should be True for valid tiles, False for padding."""
        D = 4
        batch = [
            (torch.randn(10, D), {"label": 0}, "s1"),
            (torch.randn(20, D), {"label": 1}, "s2"),
            (torch.randn(5, D), {"label": 0}, "s3"),
        ]
        result = bag_collate_fn(batch, TARGET_DTYPES)
        assert result.mask[0].sum().item() == 10
        assert result.mask[1].sum().item() == 20
        assert result.mask[2].sum().item() == 5

    def test_padded_values_are_zero(self):
        """Padding should be filled with zeros."""
        D = 4
        batch = [
            (torch.ones(3, D), {"label": 0}, "s1"),
            (torch.ones(5, D), {"label": 1}, "s2"),
        ]
        result = bag_collate_fn(batch, TARGET_DTYPES)
        assert torch.equal(result.features[0, 3:], torch.zeros(2, D))

    def test_single_bag_no_padding(self):
        """Single bag should not be padded."""
        D = 4
        batch = [(torch.randn(7, D), {"label": 1}, "s1")]
        result = bag_collate_fn(batch, TARGET_DTYPES)
        assert result.features.shape == (1, 7, D)
        assert result.mask.all()

    def test_targets_tensor(self):
        D = 4
        batch = [
            (torch.randn(3, D), {"label": 0}, "s1"),
            (torch.randn(3, D), {"label": 1}, "s2"),
            (torch.randn(3, D), {"label": 2}, "s3"),
        ]
        result = bag_collate_fn(batch, TARGET_DTYPES)
        assert torch.equal(result.targets["label"], torch.tensor([0, 1, 2]))
