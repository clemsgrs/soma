"""SlideDataset — dataset and collation for slide-level (pre-aggregated) features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore


class SlideDataset(Dataset):
    """Dataset for slide-level features (one embedding per sample).

    Unlike BagDataset, each item is a 1-D tensor of shape (feature_dim,)
    rather than a variable-length bag of tile features.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        feature_store: FeatureStore,
        label_map: dict[str | int, int],
    ) -> None:
        self._records = records
        self._store = feature_store
        self._label_map = label_map

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        record = self._records[idx]
        features = self._store.load(record.sample_id)  # (D,)
        label = self._label_map[record.label]
        return features, label, record.sample_id


@dataclass
class SlideBatch:
    """A batch of slide-level features."""

    features: Tensor  # (B, D)
    labels: Tensor  # (B,)
    sample_ids: tuple[str, ...]


def slide_collate_fn(batch: list[tuple[Tensor, int, str]]) -> SlideBatch:
    """Collate a list of slide-level items into a SlideBatch."""
    features, labels, sample_ids = zip(*batch)
    return SlideBatch(
        features=torch.stack(features),
        labels=torch.tensor(labels, dtype=torch.long),
        sample_ids=tuple(sample_ids),
    )
