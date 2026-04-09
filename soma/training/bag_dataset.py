"""BagDataset — maps sample records to feature bags for MIL training."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore


class BagDataset(Dataset):
    """PyTorch Dataset that loads tile features per sample.

    Each item returns (features, label, sample_id) where features
    is a variable-length tensor of shape (N_tiles, D).

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
        features = self._store.load(record.sample_id)
        if features.ndim != 2:
            raise ValueError(
                f"BagDataset expects 2-D tile features, got rank {features.ndim} for "
                f"sample_id={record.sample_id}"
            )
        label = self._label_fn(record)
        return features, label, record.sample_id


class HierarchicalBagDataset(Dataset):
    """PyTorch Dataset that loads hierarchical features per sample.

    Each item returns (features, label, sample_id) where features
    is a tensor of shape (num_regions, num_tiles_per_region, D).

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
        features = self._store.load(record.sample_id)
        if features.ndim != 3:
            raise ValueError(
                f"HierarchicalBagDataset expects 3-D hierarchical features, got rank "
                f"{features.ndim} for sample_id={record.sample_id}"
            )
        label = self._label_fn(record)
        return features, label, record.sample_id
