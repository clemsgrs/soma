"""BagDataset — maps sample records to feature bags for MIL training."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore


class BagDataset(Dataset):
    """PyTorch Dataset that loads tile features per sample.

    Each item returns (features, targets, sample_id) where features
    is a variable-length tensor of shape (N_tiles, D).

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
        features = self._store.load(record.sample_id)
        if features.ndim != 2:
            raise ValueError(
                f"BagDataset expects 2-D tile features, got rank {features.ndim} for "
                f"sample_id={record.sample_id}"
            )
        return features, self._target_fn(record), record.sample_id


class HierarchicalBagDataset(Dataset):
    """PyTorch Dataset that loads hierarchical features per sample.

    Each item returns (features, targets, sample_id) where features
    is a tensor of shape (num_regions, num_tiles_per_region, D).

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
        features = self._store.load(record.sample_id)
        if features.ndim != 3:
            raise ValueError(
                f"HierarchicalBagDataset expects 3-D hierarchical features, got rank "
                f"{features.ndim} for sample_id={record.sample_id}"
            )
        return features, self._target_fn(record), record.sample_id
