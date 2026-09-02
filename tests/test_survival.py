"""Tests for discrete-time survival: head, loss, metrics, validation, e2e."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from soma.config import AggregatorConfig, PipelineConfig, TaskConfig, TrainingConfig
from soma.dataset import Dataset, SampleRecord
from soma.evaluation.metrics import compute_survival_metrics, resolve_metrics
from soma.pipeline import Pipeline, PipelineResult
from soma.tasks.registry import task_registry
from soma.tasks.survival import (
    SurvivalHead,
    _risk_from_logits,
    survival_nll_loss,
    validate_survival_dataset,
)

D = 16
FIXED_RUN_ID = "20990101_000000"


def _record(sample_id: str, time: float, event: int, bin_: int, patient_id: str | None = None) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        image_path=Path(f"/slides/{sample_id}.svs"),
        label=time,
        patient_id=patient_id,
        metadata={"event": event, "bin": bin_},
    )


# ---------------------------------------------------------------------------
# Loss (reference parity is checked numerically against the HIPT formula
# recomputed independently here)
# ---------------------------------------------------------------------------


def _reference_nll(logits, bins, events, alpha):
    """Independent re-implementation of the HIPT NLLSurvLoss for cross-check."""
    eps = 1e-7
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=1)
    b = len(bins)
    Y = bins.view(b, 1)
    c = (1.0 - events).view(b, 1).float()
    surv_pad = torch.cat([torch.ones_like(c), survival], dim=1)
    unc = -(1 - c) * (
        torch.log(torch.gather(surv_pad, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    cen = -c * torch.log(torch.gather(surv_pad, 1, Y + 1).clamp(min=eps))
    neg_l = cen + unc
    return ((1 - alpha) * neg_l + alpha * unc).mean()


class TestSurvivalLoss:
    def test_matches_reference_formula(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 4)
        bins = torch.tensor([0, 1, 2, 3, 1])
        events = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        loss = survival_nll_loss(logits, bins, events, alpha=0.15)
        ref = _reference_nll(logits, bins, events, alpha=0.15)
        assert torch.allclose(loss, ref)

    def test_is_finite_for_saturated_logits(self):
        # Extreme logits → hazards ~0/1; clamps must keep the loss finite.
        logits = torch.tensor([[-40.0, 40.0, -40.0, 40.0]])
        loss_event = survival_nll_loss(logits, torch.tensor([3]), torch.tensor([1.0]))
        loss_censored = survival_nll_loss(logits, torch.tensor([0]), torch.tensor([0.0]))
        assert torch.isfinite(loss_event) and torch.isfinite(loss_censored)

    def test_alpha_reweights_uncensored(self):
        torch.manual_seed(1)
        logits = torch.randn(4, 3)
        bins = torch.tensor([0, 1, 2, 1])
        events = torch.tensor([1.0, 1.0, 0.0, 0.0])
        a0 = survival_nll_loss(logits, bins, events, alpha=0.0)
        a1 = survival_nll_loss(logits, bins, events, alpha=0.5)
        assert not torch.allclose(a0, a1)


class TestRisk:
    def test_risk_is_negative_restricted_mean_survival(self):
        logits = torch.randn(3, 4)
        risk = _risk_from_logits(logits)
        surv = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
        assert torch.allclose(risk, -surv.sum(dim=1))
        # Risk is strictly negative (survival probs are positive).
        assert (risk < 0).all()


# ---------------------------------------------------------------------------
# SurvivalHead
# ---------------------------------------------------------------------------


class TestSurvivalHead:
    def test_registered(self):
        assert task_registry.get("survival") is SurvivalHead

    def test_target_dtypes(self):
        assert SurvivalHead.target_dtypes == {
            "bin": torch.long,
            "event": torch.float,
            "time": torch.float,
        }

    def test_forward_shape(self):
        head = SurvivalHead(input_dim=8, num_bins=4)
        assert head(torch.randn(3, 8)).shape == (3, 4)

    def test_extract_targets(self):
        head = SurvivalHead(input_dim=8, num_bins=4)
        record = _record("s1", time=2.5, event=1, bin_=2)
        assert head.extract_targets(record) == {"bin": 2, "event": 1.0, "time": 2.5}

    def test_auto_params_infers_num_bins(self):
        ds = _FakeDataset([
            _record("s0", 0.5, 1, 0),
            _record("s1", 1.5, 0, 1),
            _record("s2", 3.5, 1, 3),
        ])
        assert SurvivalHead.auto_params(ds) == {"num_bins": 4}

    def test_postprocess_returns_risk_scores(self):
        head = SurvivalHead(input_dim=8, num_bins=4)
        out = head.postprocess(torch.randn(5, 4))
        assert "risk_scores" in out and out["risk_scores"].shape == (5,)

    def test_compute_metrics_returns_c_index(self):
        head = SurvivalHead(input_dim=8, num_bins=4)
        logits = torch.randn(6, 4)
        targets = {
            "bin": torch.tensor([0, 1, 2, 3, 1, 2]),
            "event": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            "time": torch.tensor([0.4, 1.2, 2.6, 3.1, 1.0, 2.0]),
        }
        m = head.compute_metrics(logits, targets)
        assert set(m) == {"c_index"}
        assert isinstance(m["c_index"], float)

    def test_default_metrics(self):
        assert SurvivalHead(input_dim=8, num_bins=4).metrics == ["c_index"]

    def test_gradient_flows(self):
        head = SurvivalHead(input_dim=8, num_bins=4)
        X = torch.randn(3, 8, requires_grad=True)
        targets = {"bin": torch.tensor([0, 1, 3]), "event": torch.tensor([1.0, 0.0, 1.0])}
        head.compute_loss(head(X), targets).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0


class _FakeDataset:
    def __init__(self, records):
        self.samples = {r.sample_id: r for r in records}


# ---------------------------------------------------------------------------
# compute_survival_metrics
# ---------------------------------------------------------------------------


class TestSurvivalMetrics:
    def test_registered_in_families(self):
        assert resolve_metrics("survival", []) == ["c_index"]

    def test_perfect_ranking_c_index_is_one(self):
        # risk perfectly anti-correlated with survival time → c-index = 1.
        event = np.array([1, 1, 1])
        time = np.array([1.0, 2.0, 3.0])
        risk = np.array([3.0, 2.0, 1.0])
        m = compute_survival_metrics(["c_index"], event, time, risk)
        assert m["c_index"] == pytest.approx(1.0)

    def test_all_censored_returns_nan_not_crash(self):
        m = compute_survival_metrics(["c_index"], np.zeros(3), np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        assert np.isnan(m["c_index"])

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown survival metric"):
            compute_survival_metrics(["nonsense"], np.array([1]), np.array([1.0]), np.array([0.5]))


# ---------------------------------------------------------------------------
# validate_survival_dataset
# ---------------------------------------------------------------------------


def _survival_dataset(tmp_path: Path, rows: list[dict]) -> Dataset:
    df = pd.DataFrame(rows)
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return Dataset(path)


class TestValidateSurvivalDataset:
    def _rows(self):
        return [
            {"sample_id": f"s{i}", "image_path": f"/s{i}.svs", "label": float(i), "event": i % 2, "bin": i}
            for i in range(4)
        ]

    def test_valid_passes(self, tmp_path: Path):
        ds = _survival_dataset(tmp_path, self._rows())
        validate_survival_dataset(ds, "slide")  # no raise

    def test_missing_event_column_raises(self, tmp_path: Path):
        rows = [{"sample_id": "s0", "image_path": "/s0.svs", "label": 1.0, "bin": 0}]
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="'event'"):
            validate_survival_dataset(ds, "slide")

    def test_bad_event_value_raises(self, tmp_path: Path):
        rows = self._rows()
        rows[0]["event"] = 2
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="event"):
            validate_survival_dataset(ds, "slide")

    def test_non_contiguous_bins_raise(self, tmp_path: Path):
        rows = self._rows()
        rows[2]["bin"] = 9  # bins become {0,1,9,3} → not contiguous
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="contiguous"):
            validate_survival_dataset(ds, "slide")

    def test_negative_time_raises(self, tmp_path: Path):
        rows = self._rows()
        rows[1]["label"] = -1.0
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="time"):
            validate_survival_dataset(ds, "slide")

    def test_patient_inconsistent_targets_raise(self, tmp_path: Path):
        rows = [
            {"sample_id": "s0", "image_path": "/s0.svs", "label": 1.0, "event": 1, "bin": 0, "patient_id": "p0"},
            {"sample_id": "s1", "image_path": "/s1.svs", "label": 2.0, "event": 1, "bin": 1, "patient_id": "p0"},
        ]
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="inconsistent survival targets"):
            validate_survival_dataset(ds, "patient")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSurvivalConfigValidation:
    def _base(self, tmp_path, **kw):
        return dict(
            dataset_csv=tmp_path / "d.csv",
            splits_csv=tmp_path / "s.csv",
            output_root=tmp_path / "out",
            task=TaskConfig(name="survival"),
            **kw,
        )

    def test_rejects_tile(self, tmp_path: Path):
        with pytest.raises(ValueError, match="tile.*not supported for survival|survival"):
            PipelineConfig(**self._base(tmp_path, dataset_type="tile"))

    def test_rejects_clam(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not supported for survival"):
            PipelineConfig(**self._base(tmp_path, dataset_type="slide", aggregator=AggregatorConfig(name="clam_sb")))

    def test_allows_slide_with_abmil(self, tmp_path: Path):
        PipelineConfig(**self._base(tmp_path, dataset_type="slide", aggregator=AggregatorConfig(name="abmil")))


# ---------------------------------------------------------------------------
# End-to-end through Pipeline.run (also exercises report generation)
# ---------------------------------------------------------------------------


def _setup_survival_slide_data(tmp_path: Path):
    n = 8
    bins = [0, 1, 2, 3, 0, 1, 2, 3]
    events = [1, 1, 0, 1, 0, 1, 1, 0]
    times = [float(b) + 0.5 for b in bins]
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/slides/s{i}.svs" for i in range(n)],
            "label": times,
            "event": events,
            "bin": bins,
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * n,
            "sample_id": [f"s{i}" for i in range(n)],
            "split": ["train"] * 4 + ["tune", "tune", "test", "test"],
        }
    ).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(0)
    for i in range(n):
        torch.save(torch.randn(5 + i, D), feature_dir / f"s{i}.pt")
    return dataset_csv, splits_csv, feature_dir


class TestSurvivalEndToEnd:
    def test_slide_level_survival_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_survival_slide_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="survival"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        test_report = result.fold_results[0].test_reports["test"]
        assert "c_index" in test_report.metrics
        # Predictions CSV carries survival columns.
        preds_csv = next(result.run_dir.rglob("predictions_test.csv"))
        cols = pd.read_csv(preds_csv).columns
        assert {"sample_id", "true_label", "event", "risk_score"} <= set(cols)

    def test_patient_level_survival_run(self, tmp_path: Path):
        n = 8
        sample_ids = [f"s{i}" for i in range(n)]
        patient_ids = [f"p{i // 2}" for i in range(n)]  # p0 p0 p1 p1 p2 p2 p3 p3
        # Per-patient consistent (time, event, bin); bins 0..3 contiguous.
        per_patient = {"p0": (0.5, 1, 0), "p1": (1.5, 0, 1), "p2": (2.5, 1, 2), "p3": (3.5, 1, 3)}
        dataset_csv = tmp_path / "dataset.csv"
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "image_path": [f"/{s}.svs" for s in sample_ids],
                "label": [per_patient[p][0] for p in patient_ids],
                "event": [per_patient[p][1] for p in patient_ids],
                "bin": [per_patient[p][2] for p in patient_ids],
                "patient_id": patient_ids,
            }
        ).to_csv(dataset_csv, index=False)

        splits_csv = tmp_path / "splits.csv"
        split_for = {"p0": "train", "p1": "train", "p2": "tune", "p3": "test"}
        pd.DataFrame(
            {
                "fold": [0] * n,
                "sample_id": sample_ids,
                "split": [split_for[p] for p in patient_ids],
            }
        ).to_csv(splits_csv, index=False)

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        torch.manual_seed(0)
        for pid in ["p0", "p1", "p2", "p3"]:
            torch.save(torch.randn(D), feature_dir / f"{pid}.pt")

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="patient",
            aggregator=None,
            task=TaskConfig(name="survival"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_reports["test"].metrics
