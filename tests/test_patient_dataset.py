"""Tests for soma.training — PatientDataset and patient_collate_fn."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.features import FeatureStore
from soma.training.patient_dataset import PatientBatch, PatientDataset, patient_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_patient_store(tmp_path: Path, patient_ids: list[str], feature_dim: int) -> FeatureStore:
    for pid in patient_ids:
        torch.save(torch.randn(feature_dim), tmp_path / f"{pid}.pt")
    return FeatureStore(tmp_path)


LABEL_MAP = {"normal": 0, "tumor": 1}
PATIENT_LABEL_MAP = {"p1": "tumor", "p2": "normal", "p3": "tumor"}


# ---------------------------------------------------------------------------
# PatientDataset
# ---------------------------------------------------------------------------


class TestPatientDataset:
    def test_len(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2", "p3"], feature_dim=64)
        ds = PatientDataset(["p1", "p2", "p3"], PATIENT_LABEL_MAP, store, LABEL_MAP)
        assert len(ds) == 3

    def test_getitem_shape(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=768)
        ds = PatientDataset(["p1", "p2"], PATIENT_LABEL_MAP, store, LABEL_MAP)
        features, label, patient_id = ds[0]
        assert features.shape == (768,)

    def test_getitem_label_mapped(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=32)
        ds = PatientDataset(["p1", "p2"], PATIENT_LABEL_MAP, store, LABEL_MAP)
        _, label_p1, pid = ds[0]
        assert pid == "p1"
        assert label_p1 == LABEL_MAP["tumor"]  # 1

    def test_getitem_returns_patient_id(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=16)
        ds = PatientDataset(["p1", "p2"], PATIENT_LABEL_MAP, store, LABEL_MAP)
        _, _, patient_id = ds[0]
        assert patient_id == "p1"

    def test_custom_label_fn(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1"], feature_dim=8)
        label_fn = lambda pid, raw: 99.0
        ds = PatientDataset(["p1"], PATIENT_LABEL_MAP, store, LABEL_MAP, label_fn=label_fn)
        _, label, _ = ds[0]
        assert label == 99.0

    def test_preserves_order(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2", "p3"], feature_dim=8)
        ds = PatientDataset(["p3", "p1", "p2"], PATIENT_LABEL_MAP, store, LABEL_MAP)
        _, _, pid0 = ds[0]
        _, _, pid1 = ds[1]
        _, _, pid2 = ds[2]
        assert pid0 == "p3"
        assert pid1 == "p1"
        assert pid2 == "p2"


# ---------------------------------------------------------------------------
# patient_collate_fn
# ---------------------------------------------------------------------------


class TestPatientCollateFn:
    def test_stacks_features(self):
        D = 128
        batch = [
            (torch.randn(D), 0, "p1"),
            (torch.randn(D), 1, "p2"),
        ]
        result = patient_collate_fn(batch)
        assert isinstance(result, PatientBatch)
        assert result.features.shape == (2, D)

    def test_labels_tensor(self):
        D = 8
        batch = [
            (torch.randn(D), 0, "p1"),
            (torch.randn(D), 1, "p2"),
            (torch.randn(D), 0, "p3"),
        ]
        result = patient_collate_fn(batch)
        assert torch.equal(result.labels, torch.tensor([0, 1, 0]))

    def test_label_dtype_float(self):
        D = 8
        batch = [(torch.randn(D), 2.5, "p1")]
        result = patient_collate_fn(batch, label_dtype=torch.float)
        assert result.labels.dtype == torch.float

    def test_sample_ids_tuple(self):
        D = 8
        batch = [
            (torch.randn(D), 0, "p1"),
            (torch.randn(D), 1, "p2"),
        ]
        result = patient_collate_fn(batch)
        assert result.sample_ids == ("p1", "p2")
