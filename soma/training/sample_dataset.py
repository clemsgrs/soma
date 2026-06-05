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
from soma.training.collate import stack_targets


class SampleDataset(Dataset):
    """Dataset for samples represented by a single feature vector (one embedding per sample).

    Unlike BagDataset, each item is a 1-D tensor of shape (feature_dim,)
    rather than a variable-length bag of tile features.

    Args:
        records: List of SampleRecords with sample_id and label.
        feature_store: FeatureStore for loading precomputed embeddings.
        target_fn: Callable mapping a SampleRecord to its targets dict
            (the head's ``extract_targets``).
    """

    def __init__(
        self,
        records: list[SampleRecord],
        feature_store: FeatureStore,
        target_fn: Callable[[SampleRecord], dict[str, int | float]],
    ) -> None:
        self._records = records
        self._store = feature_store
        self._target_fn = target_fn

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, int | float], str]:
        record = self._records[idx]
        features = self._store.load(record.sample_id)  # (D,)
        return features, self._target_fn(record), record.sample_id


@dataclass
class SampleBatch:
    """A batch of single-embedding samples."""

    features: Tensor  # (B, D)
    targets: dict[str, Tensor]  # each (B,)
    sample_ids: tuple[str, ...]


def sample_collate_fn(
    batch: list[tuple[Tensor, dict[str, int | float], str]],
    target_dtypes: dict[str, torch.dtype],
) -> SampleBatch:
    """Collate a list of single-embedding items into a SampleBatch."""
    features, target_dicts, sample_ids = zip(*batch)
    return SampleBatch(
        features=torch.stack(features),
        targets=stack_targets(target_dicts, target_dtypes),
        sample_ids=tuple(sample_ids),
    )
