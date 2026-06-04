"""PatientDataset — dataset and collation for patient-level (pre-aggregated) features."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.features import FeatureStore
from soma.training.collate import stack_targets


class PatientDataset(Dataset):
    """Dataset for patient-level features (one embedding per patient).

    Each item is a 1-D tensor of shape (feature_dim,) loaded from the feature
    store by patient_id. Used when a pretrained patient encoder (e.g. MOOZY)
    has produced one embedding per patient.

    Targets are extracted from a representative SampleRecord per patient, so the
    head's ``extract_targets`` behaves identically to the slide-level path.

    Args:
        patient_ids: Ordered list of patient IDs for this split.
        patient_record_map: Mapping from patient_id to a representative
            SampleRecord (carrying the patient's label and metadata).
        feature_store: FeatureStore for loading precomputed patient embeddings,
            indexed by patient_id.
        target_fn: Callable mapping a SampleRecord to its targets dict
            (the head's ``extract_targets``).
    """

    def __init__(
        self,
        patient_ids: list[str],
        patient_record_map: dict[str, SampleRecord],
        feature_store: FeatureStore,
        target_fn: Callable[[SampleRecord], dict[str, int | float]],
    ) -> None:
        self._patient_ids = patient_ids
        self._patient_record_map = patient_record_map
        self._store = feature_store
        self._target_fn = target_fn

    def __len__(self) -> int:
        return len(self._patient_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, int | float], str]:
        patient_id = self._patient_ids[idx]
        features = self._store.load(patient_id)  # (D,)
        record = self._patient_record_map[patient_id]
        return features, self._target_fn(record), patient_id


@dataclass
class PatientBatch:
    """A batch of patient-level features."""

    features: Tensor  # (B, D)
    targets: dict[str, Tensor]  # each (B,)
    sample_ids: tuple[str, ...]  # patient_ids, named sample_ids for API compatibility


def patient_collate_fn(
    batch: list[tuple[Tensor, dict[str, int | float], str]],
    target_dtypes: dict[str, torch.dtype],
) -> PatientBatch:
    """Collate a list of patient-level items into a PatientBatch."""
    features, target_dicts, patient_ids = zip(*batch)
    return PatientBatch(
        features=torch.stack(features),
        targets=stack_targets(target_dicts, target_dtypes),
        sample_ids=tuple(patient_ids),
    )
