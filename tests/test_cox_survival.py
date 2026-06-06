"""Tests for continuous-time CoxPH survival (phase 1: slide/patient, batched).

Covers the Breslow partial-likelihood loss, raw-risk polarity/sign, the
event-balanced batch sampler, the full-cohort tune-loss invariance, the config
guards, and slide + patient end-to-end runs through ``Pipeline.run``.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from soma.config import AggregatorConfig, PipelineConfig, TaskConfig, TrainingConfig
from soma.dataset import Dataset, SampleRecord
from soma.evaluation.metrics import compute_survival_metrics
from soma.pipeline import Pipeline, PipelineResult
from soma.tasks.survival import (
    CoxSurvivalHead,
    SurvivalHead,
    cox_breslow_loss,
    resolve_survival_head,
    validate_survival_dataset,
)
from soma.training.model import EmbeddingModel
from soma.training.sample_dataset import SampleBatch
from soma.training.survival_sampler import EventBalancedBatchSampler
from soma.training.trainer import Trainer

D = 16
FIXED_RUN_ID = "20990101_000000"


# ---------------------------------------------------------------------------
# Breslow loss — numerical parity with an independent hand computation
# ---------------------------------------------------------------------------


def _reference_breslow(risk, time, events) -> float:
    """Independent no-ties Breslow partial likelihood for cross-check."""
    order = sorted(range(len(time)), key=lambda i: time[i], reverse=True)
    r = [float(risk[i]) for i in order]
    e = [float(events[i]) for i in order]
    lcse = []
    acc: list[float] = []
    for x in r:
        acc.append(x)
        m = max(acc)
        lcse.append(m + math.log(sum(math.exp(v - m) for v in acc)))
    n_events = sum(e)
    contrib = sum((r[i] - lcse[i]) * e[i] for i in range(len(r)))
    return -contrib / n_events


class TestCoxBreslowLoss:
    def test_matches_reference_no_ties(self):
        risk = torch.tensor([2.0, 0.5, 1.0, -0.7])
        time = torch.tensor([10.0, 5.0, 8.0, 3.0])
        events = torch.tensor([1.0, 1.0, 0.0, 1.0])
        loss = float(cox_breslow_loss(risk, time, events))
        assert loss == pytest.approx(_reference_breslow(risk, time, events), abs=1e-5)

    def test_gradient_flows_and_graph_connected(self):
        risk = torch.randn(6, requires_grad=True)
        time = torch.arange(6, 0, -1).float()
        events = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        loss = cox_breslow_loss(risk, time, events)
        assert loss.requires_grad and loss.grad_fn is not None
        loss.backward()
        assert risk.grad is not None and risk.grad.abs().sum() > 0

    def test_no_events_returns_graph_connected_zero(self):
        # An event-free risk set has no numerator; the loss is a graph-connected
        # zero so a (rare, avoidable) event-free batch never crashes training.
        risk = torch.randn(4, requires_grad=True)
        time = torch.tensor([4.0, 3.0, 2.0, 1.0])
        events = torch.zeros(4)
        loss = cox_breslow_loss(risk, time, events)
        assert float(loss) == 0.0 and loss.grad_fn is not None
        loss.backward()  # does not raise

    def test_is_finite_for_extreme_risks(self):
        risk = torch.tensor([60.0, -60.0, 60.0])
        time = torch.tensor([3.0, 2.0, 1.0])
        events = torch.tensor([1.0, 1.0, 0.0])
        assert torch.isfinite(cox_breslow_loss(risk, time, events))


# ---------------------------------------------------------------------------
# Polarity / sign — guards event semantics and raw-risk-into-C-index together
# ---------------------------------------------------------------------------


class TestPolarity:
    def test_correct_risk_assignment_has_lower_loss(self):
        # Times ascending; all events. Higher risk should go to earlier events.
        time = torch.tensor([1.0, 2.0, 3.0, 4.0])
        events = torch.ones(4)
        correct = torch.tensor([4.0, 3.0, 2.0, 1.0])  # high risk = early event
        reversed_ = torch.tensor([1.0, 2.0, 3.0, 4.0])  # high risk = late event
        loss_correct = cox_breslow_loss(correct, time, events)
        loss_reversed = cox_breslow_loss(reversed_, time, events)
        assert float(loss_correct) < float(loss_reversed)

    def test_c_index_uses_raw_risk_polarity(self):
        # Same cohort: correct raw risk -> c-index 1; reversed -> 0. This pins the
        # "higher risk = earlier event" convention shared by loss and metric.
        event = np.ones(4)
        time = np.array([1.0, 2.0, 3.0, 4.0])
        correct_risk = np.array([4.0, 3.0, 2.0, 1.0])
        reversed_risk = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_survival_metrics(["c_index"], event, time, correct_risk)["c_index"] == pytest.approx(1.0)
        assert compute_survival_metrics(["c_index"], event, time, reversed_risk)["c_index"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CoxSurvivalHead
# ---------------------------------------------------------------------------


class TestCoxSurvivalHead:
    def test_resolve_routes_by_loss(self):
        assert resolve_survival_head("nll") is SurvivalHead
        assert resolve_survival_head("cox") is CoxSurvivalHead

    def test_resolve_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown survival loss"):
            resolve_survival_head("efron")

    def test_target_dtypes_have_no_bin(self):
        assert CoxSurvivalHead.target_dtypes == {"event": torch.float, "time": torch.float}

    def test_flags(self):
        head = CoxSurvivalHead(input_dim=8)
        assert head.full_cohort_eval_loss is True
        assert head.needs_event_balanced_batches is True

    def test_forward_emits_single_scalar(self):
        head = CoxSurvivalHead(input_dim=8)
        assert head(torch.randn(5, 8)).shape == (5, 1)

    def test_extract_targets(self):
        head = CoxSurvivalHead(input_dim=8)
        record = SampleRecord(
            sample_id="s1",
            image_path=Path("/s1.svs"),
            label=2.5,
            metadata={"event": 1},
        )
        assert head.extract_targets(record) == {"event": 1.0, "time": 2.5}

    def test_postprocess_returns_raw_risk(self):
        head = CoxSurvivalHead(input_dim=8)
        raw = torch.randn(5, 1)
        out = head.postprocess(raw)
        assert "risk_scores" in out and out["risk_scores"].shape == (5,)
        # Risk is the raw logit, NOT the NLL head's -sum(surv) derivation.
        assert np.allclose(out["risk_scores"], raw.squeeze(-1).numpy())

    def test_invalid_ties_raises(self):
        with pytest.raises(ValueError, match="breslow"):
            CoxSurvivalHead(input_dim=8, ties="efron")

    def test_auto_params_empty(self):
        # No binning -> nothing to infer from the dataset.
        assert CoxSurvivalHead.auto_params(object()) == {}

    def test_gradient_flows(self):
        head = CoxSurvivalHead(input_dim=8)
        X = torch.randn(4, 8, requires_grad=True)
        targets = {"time": torch.tensor([4.0, 3.0, 2.0, 1.0]), "event": torch.tensor([1.0, 0.0, 1.0, 1.0])}
        head.compute_loss(head(X), targets).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Event-balanced batch sampler
# ---------------------------------------------------------------------------


class TestEventBalancedSampler:
    def test_every_batch_meets_event_floor(self):
        events = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=1, seed=0)
        for batch in sampler:
            assert sum(events[i] for i in batch) >= 1

    def test_higher_min_events_floor(self):
        events = [1] * 6 + [0] * 6
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=2, seed=0)
        for batch in sampler:
            assert sum(events[i] for i in batch) >= 2

    def test_covers_all_samples_each_epoch(self):
        events = [1, 0, 0, 1, 0, 0, 1, 0]
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=1, seed=0)
        for _ in range(3):
            flat = sorted(i for batch in sampler for i in batch)
            assert flat == list(range(len(events)))

    def test_never_exceeds_batch_size(self):
        events = [1, 1, 0, 0, 0, 0, 0]
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=1, seed=0)
        batches = [list(batch) for batch in sampler]
        assert [len(batch) for batch in batches] == [4, 3]
        assert sorted(i for batch in batches for i in batch) == list(range(len(events)))

    def test_avoids_singleton_risk_sets(self):
        events = [1, 1, 0, 0, 0]
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=1, seed=0)
        batches = [list(batch) for batch in sampler]
        assert [len(batch) for batch in batches] == [3, 2]
        assert sorted(i for batch in batches for i in batch) == list(range(len(events)))

    def test_rejects_when_capped_windows_would_force_singleton(self):
        with pytest.raises(ValueError, match="singleton"):
            EventBalancedBatchSampler([1, 1, 0], batch_size=2, min_events_per_window=1, seed=0)

    def test_reshuffles_across_epochs(self):
        events = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
        sampler = EventBalancedBatchSampler(events, batch_size=4, min_events_per_window=1, seed=0)
        epoch1 = [list(b) for b in sampler]
        epoch2 = [list(b) for b in sampler]
        assert epoch1 != epoch2

    def test_deterministic_given_seed(self):
        events = [1, 0, 0, 1, 0, 0, 1, 0]
        a = [list(b) for b in EventBalancedBatchSampler(events, batch_size=4, seed=7)]
        b = [list(b) for b in EventBalancedBatchSampler(events, batch_size=4, seed=7)]
        assert a == b

    def test_raises_when_too_few_events(self):
        with pytest.raises(ValueError, match="at least"):
            EventBalancedBatchSampler([1, 0, 0, 0, 0, 0, 0, 0], batch_size=2, min_events_per_window=1)

    def test_rejects_small_batch_size(self):
        with pytest.raises(ValueError, match="batch_size >= 2"):
            EventBalancedBatchSampler([1, 0, 1, 0], batch_size=1)


# ---------------------------------------------------------------------------
# Full-cohort tune loss — invariant to tune batch size (deterministic signal)
# ---------------------------------------------------------------------------


def _sample_batches(X, time, event, batch_sizes):
    """Slice (X, time, event) into SampleBatches of the given sizes, in order."""
    batches = []
    start = 0
    for bs in batch_sizes:
        sl = slice(start, start + bs)
        batches.append(
            SampleBatch(
                features=X[sl],
                targets={"time": time[sl], "event": event[sl]},
                sample_ids=tuple(f"s{i}" for i in range(start, start + bs)),
            )
        )
        start += bs
    return batches


class TestFullCohortTuneLoss:
    def test_tune_loss_invariant_to_batch_partition(self, tmp_path: Path):
        torch.manual_seed(0)
        model = EmbeddingModel(task_head=CoxSurvivalHead(input_dim=D))
        n = 6
        X = torch.randn(n, D)
        time = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        # Note: the first partition's leading batch is all-censored — full-cohort
        # eval must not be derailed by an event-free tune batch.
        event = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])

        loaders = {
            "whole": _sample_batches(X, time, event, [6]),
            "pairs": _sample_batches(X, time, event, [2, 2, 2]),
            "uneven": _sample_batches(X, time, event, [1, 3, 2]),
        }

        device = torch.device("cpu")
        losses = {}
        for name, batches in loaders.items():
            trainer = Trainer(
                model=model,
                train_loader=batches,
                tune_loader=batches,
                config=TrainingConfig(epochs=1, batch_size=2),
                fold_dir=tmp_path / name,
                device=device,
            )
            loss, _ = trainer._tune()
            losses[name] = loss

        # Direct full-cohort reference.
        with torch.no_grad():
            risk = model(X).logits.squeeze(-1)
        reference = float(cox_breslow_loss(risk, time, event))

        for name, loss in losses.items():
            assert loss == pytest.approx(reference, abs=1e-5), name


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


class TestCoxConfigGuards:
    def _base(self, tmp_path, **kw):
        return dict(
            dataset_csv=tmp_path / "d.csv",
            splits_csv=tmp_path / "s.csv",
            output_root=tmp_path / "out",
            dataset_type="slide",
            task=TaskConfig(name="survival", params={"loss": "cox"}),
            **kw,
        )

    def test_padded_mil_aggregator_allowed(self, tmp_path: Path):
        # Phase 2 lifted the phase-1 "Cox rejects any aggregator" rule: padded MIL
        # Cox (masking, batch_size >= 2) is now valid. Accumulation-mode guards are
        # exercised in test_cox_accumulation.py.
        for name in ("abmil", "transmil", "mean_pool"):
            PipelineConfig(
                **self._base(
                    tmp_path,
                    aggregator=AggregatorConfig(name=name),
                    training=TrainingConfig(batch_size=4),
                )
            )

    def test_rejects_gradient_accumulation(self, tmp_path: Path):
        with pytest.raises(ValueError, match="gradient_accumulation = 1"):
            PipelineConfig(
                **self._base(tmp_path, training=TrainingConfig(batch_size=4, gradient_accumulation=2))
            )

    def test_rejects_batch_size_below_two(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size >= 2"):
            PipelineConfig(**self._base(tmp_path, training=TrainingConfig(batch_size=1)))

    def test_rejects_unknown_loss(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown survival loss"):
            PipelineConfig(
                **dict(
                    self._base(tmp_path, training=TrainingConfig(batch_size=4)),
                    task=TaskConfig(name="survival", params={"loss": "foo"}),
                )
            )

    def test_valid_cox_config_passes(self, tmp_path: Path):
        PipelineConfig(**self._base(tmp_path, training=TrainingConfig(batch_size=4)))


# ---------------------------------------------------------------------------
# validate_survival_dataset — Cox does not require the bin column
# ---------------------------------------------------------------------------


def _dataset(tmp_path: Path, rows: list[dict]) -> Dataset:
    path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return Dataset(path)


class TestValidateCoxDataset:
    def test_cox_passes_without_bin(self, tmp_path: Path):
        rows = [
            {"sample_id": f"s{i}", "image_path": f"/s{i}.svs", "label": float(i + 1), "event": i % 2}
            for i in range(4)
        ]
        validate_survival_dataset(_dataset(tmp_path, rows), "slide", loss="cox")  # no raise

    def test_cox_still_requires_event(self, tmp_path: Path):
        rows = [{"sample_id": "s0", "image_path": "/s0.svs", "label": 1.0}]
        with pytest.raises(ValueError, match="'event'"):
            validate_survival_dataset(_dataset(tmp_path, rows), "slide", loss="cox")

    def test_cox_patient_inconsistent_targets_raise(self, tmp_path: Path):
        rows = [
            {"sample_id": "s0", "image_path": "/s0.svs", "label": 1.0, "event": 1, "patient_id": "p0"},
            {"sample_id": "s1", "image_path": "/s1.svs", "label": 2.0, "event": 1, "patient_id": "p0"},
        ]
        with pytest.raises(ValueError, match="inconsistent survival targets"):
            validate_survival_dataset(_dataset(tmp_path, rows), "patient", loss="cox")


# ---------------------------------------------------------------------------
# End-to-end through Pipeline.run
# ---------------------------------------------------------------------------


class TestCoxEndToEnd:
    def test_slide_level_cox_run(self, tmp_path: Path):
        n = 8
        # Slide-LEVEL features (one 1-D vector per slide) -> SampleDataset path,
        # no aggregator needed.
        events = [1, 1, 0, 1, 0, 1, 1, 0]
        times = [float(i + 1) for i in range(n)]
        dataset_csv = tmp_path / "dataset.csv"
        pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(n)],
                "image_path": [f"/slides/s{i}.svs" for i in range(n)],
                "label": times,
                "event": events,
            }
        ).to_csv(dataset_csv, index=False)

        splits_csv = tmp_path / "splits.csv"
        pd.DataFrame(
            {
                "fold": [0] * n,
                "sample_id": [f"s{i}" for i in range(n)],
                "split": ["train", "train", "train", "train", "tune", "tune", "test", "test"],
            }
        ).to_csv(splits_csv, index=False)

        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        torch.manual_seed(0)
        for i in range(n):
            torch.save(torch.randn(D), feature_dir / f"s{i}.pt")  # (D,) -> slide-level

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="slide",
            aggregator=None,
            task=TaskConfig(name="survival", params={"loss": "cox"}),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        test_report = result.fold_results[0].test_reports["test"]
        assert "c_index" in test_report.metrics
        preds_csv = next(result.run_dir.rglob("predictions_test.csv"))
        cols = pd.read_csv(preds_csv).columns
        assert {"sample_id", "true_label", "event", "risk_score"} <= set(cols)

    def test_patient_level_cox_run(self, tmp_path: Path):
        n = 8
        sample_ids = [f"s{i}" for i in range(n)]
        patient_ids = [f"p{i // 2}" for i in range(n)]
        # Per-patient consistent (time, event); no bin needed for Cox.
        per_patient = {"p0": (1.0, 1), "p1": (2.0, 1), "p2": (3.0, 1), "p3": (4.0, 0)}
        dataset_csv = tmp_path / "dataset.csv"
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "image_path": [f"/{s}.svs" for s in sample_ids],
                "label": [per_patient[p][0] for p in patient_ids],
                "event": [per_patient[p][1] for p in patient_ids],
                "patient_id": patient_ids,
            }
        ).to_csv(dataset_csv, index=False)

        splits_csv = tmp_path / "splits.csv"
        # Train needs >= 2 patients with >= 1 event for batch_size=2 sampling.
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
            task=TaskConfig(name="survival", params={"loss": "cox"}),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_reports["test"].metrics
