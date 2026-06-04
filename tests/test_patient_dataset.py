"""Tests for soma.training — PatientDataset and patient_collate_fn."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from soma.dataset import SampleRecord
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
TARGET_DTYPES = {"label": torch.long}
PATIENT_RAW_LABEL = {"p1": "tumor", "p2": "normal", "p3": "tumor"}
PATIENT_RECORD_MAP = {
    pid: SampleRecord(sample_id=pid, image_path=Path(f"/{pid}.svs"), label=raw, patient_id=pid)
    for pid, raw in PATIENT_RAW_LABEL.items()
}


def _target_fn(record: SampleRecord) -> dict[str, int]:
    return {"label": LABEL_MAP[record.label]}


# ---------------------------------------------------------------------------
# PatientDataset
# ---------------------------------------------------------------------------


class TestPatientDataset:
    def test_len(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2", "p3"], feature_dim=64)
        ds = PatientDataset(["p1", "p2", "p3"], PATIENT_RECORD_MAP, store, _target_fn)
        assert len(ds) == 3

    def test_getitem_shape(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=768)
        ds = PatientDataset(["p1", "p2"], PATIENT_RECORD_MAP, store, _target_fn)
        features, targets, patient_id = ds[0]
        assert features.shape == (768,)

    def test_getitem_label_mapped(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=32)
        ds = PatientDataset(["p1", "p2"], PATIENT_RECORD_MAP, store, _target_fn)
        _, targets, pid = ds[0]
        assert pid == "p1"
        assert targets == {"label": LABEL_MAP["tumor"]}  # 1

    def test_getitem_returns_patient_id(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2"], feature_dim=16)
        ds = PatientDataset(["p1", "p2"], PATIENT_RECORD_MAP, store, _target_fn)
        _, _, patient_id = ds[0]
        assert patient_id == "p1"

    def test_custom_target_fn(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1"], feature_dim=8)
        ds = PatientDataset(["p1"], PATIENT_RECORD_MAP, store, lambda record: {"value": 99.0})
        _, targets, _ = ds[0]
        assert targets == {"value": 99.0}

    def test_preserves_order(self, tmp_path: Path):
        store = _create_patient_store(tmp_path, ["p1", "p2", "p3"], feature_dim=8)
        ds = PatientDataset(["p3", "p1", "p2"], PATIENT_RECORD_MAP, store, _target_fn)
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
            (torch.randn(D), {"label": 0}, "p1"),
            (torch.randn(D), {"label": 1}, "p2"),
        ]
        result = patient_collate_fn(batch, TARGET_DTYPES)
        assert isinstance(result, PatientBatch)
        assert result.features.shape == (2, D)

    def test_targets_tensor(self):
        D = 8
        batch = [
            (torch.randn(D), {"label": 0}, "p1"),
            (torch.randn(D), {"label": 1}, "p2"),
            (torch.randn(D), {"label": 0}, "p3"),
        ]
        result = patient_collate_fn(batch, TARGET_DTYPES)
        assert torch.equal(result.targets["label"], torch.tensor([0, 1, 0]))

    def test_target_dtype_float(self):
        D = 8
        batch = [(torch.randn(D), {"value": 2.5}, "p1")]
        result = patient_collate_fn(batch, {"value": torch.float})
        assert result.targets["value"].dtype == torch.float

    def test_sample_ids_tuple(self):
        D = 8
        batch = [
            (torch.randn(D), {"label": 0}, "p1"),
            (torch.randn(D), {"label": 1}, "p2"),
        ]
        result = patient_collate_fn(batch, TARGET_DTYPES)
        assert result.sample_ids == ("p1", "p2")
