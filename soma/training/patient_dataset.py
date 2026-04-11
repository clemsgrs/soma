"""PatientDataset — dataset and collation for patient-level (pre-aggregated) features."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore


class PatientDataset(Dataset):
    """Dataset for patient-level features (one embedding per patient).

    Each item is a 1-D tensor of shape (feature_dim,) loaded from the feature
    store by patient_id. Used when a pretrained patient encoder (e.g. MOOZY)
    has produced one embedding per patient.

    Args:
        patient_ids: Ordered list of patient IDs for this split.
        patient_label_map: Mapping from patient_id to raw label.
        feature_store: FeatureStore for loading precomputed patient embeddings,
            indexed by patient_id.
        label_map: Mapping from raw labels to integer indices. Ignored when
            label_fn is provided.
        label_fn: Optional callable that maps a (patient_id, raw_label) pair to
            its label value (int or float). When provided, label_map is not used.
    """

    def __init__(
        self,
        patient_ids: list[str],
        patient_label_map: dict[str, str | int],
        feature_store: FeatureStore,
        label_map: dict[str | int, int],
        label_fn: Callable[[str, str | int], int | float] | None = None,
    ) -> None:
        self._patient_ids = patient_ids
        self._patient_label_map = patient_label_map
        self._store = feature_store
        self._label_map = label_map
        self._label_fn = label_fn or (lambda pid, raw: label_map[raw])

    def __len__(self) -> int:
        return len(self._patient_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, int | float, str]:
        patient_id = self._patient_ids[idx]
        features = self._store.load(patient_id)  # (D,)
        raw_label = self._patient_label_map[patient_id]
        label = self._label_fn(patient_id, raw_label)
        return features, label, patient_id


@dataclass
class PatientBatch:
    """A batch of patient-level features."""

    features: Tensor  # (B, D)
    labels: Tensor  # (B,)
    sample_ids: tuple[str, ...]  # patient_ids, named sample_ids for API compatibility


def patient_collate_fn(
    batch: list[tuple[Tensor, int | float, str]],
    label_dtype: torch.dtype = torch.long,
) -> PatientBatch:
    """Collate a list of patient-level items into a PatientBatch."""
    features, labels, patient_ids = zip(*batch)
    return PatientBatch(
        features=torch.stack(features),
        labels=torch.tensor(labels, dtype=label_dtype),
        sample_ids=tuple(patient_ids),
    )
