"""SampleDataset — dataset and collation for single-embedding samples.

Used by slide-level, patient (slide-encoder output), and tile-level pipelines
where each sample is already represented by a single 1-D feature vector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore


class SampleDataset(Dataset):
    """Dataset for samples represented by a single feature vector (one embedding per sample).

    Unlike BagDataset, each item is a 1-D tensor of shape (feature_dim,)
    rather than a variable-length bag of tile features.

    Args:
        records: List of SampleRecords with sample_id and label.
        feature_store: FeatureStore for loading precomputed embeddings.
        label_map: Mapping from raw labels to integer indices. Ignored when
            label_fn is provided.
        label_fn: Optional callable that maps a SampleRecord to its label
            value (int or float). When provided, label_map is not used.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        feature_store: FeatureStore,
        label_map: dict[str | int, int],
        label_fn: Callable[[SampleRecord], int | float] | None = None,
    ) -> None:
        self._records = records
        self._store = feature_store
        self._label_map = label_map
        self._label_fn = label_fn or (lambda record: label_map[record.label])

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, int | float, str]:
        record = self._records[idx]
        features = self._store.load(record.sample_id)  # (D,)
        label = self._label_fn(record)
        return features, label, record.sample_id


@dataclass
class SampleBatch:
    """A batch of single-embedding samples."""

    features: Tensor  # (B, D)
    labels: Tensor  # (B,)
    sample_ids: tuple[str, ...]


def sample_collate_fn(
    batch: list[tuple[Tensor, int | float, str]],
    label_dtype: torch.dtype = torch.long,
) -> SampleBatch:
    """Collate a list of single-embedding items into a SampleBatch."""
    features, labels, sample_ids = zip(*batch)
    return SampleBatch(
        features=torch.stack(features),
        labels=torch.tensor(labels, dtype=label_dtype),
        sample_ids=tuple(sample_ids),
    )
