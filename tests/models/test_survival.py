"""Tests for discrete-time survival: head, loss, metrics, validation, e2e."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from soma.data import CachedSetSource
from soma.config import FeaturesConfig
from soma.plan import check
from tests.experiment_configs import experiment
from soma.data import SampleRecord
from soma.data import Cohort
from soma.evaluation import Predictions
from soma.evaluation.metrics import compute_survival_metrics, resolve_metrics
from soma.pipeline import Pipeline, PipelineResult
from soma.models.heads.registry import task_registry
from soma.models.heads.survival import DiscreteSurvivalHead, survival_nll_loss, validate_survival_dataset

D = 16
FIXED_RUN_ID = "20990101_000000"


def _record(sample_id: str, time: float, event: int, bin_: int, patient_id: str | None = None) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        targets={"time": time, "event": event, "bin": bin_},
        patient_id=patient_id,
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


# ---------------------------------------------------------------------------
# DiscreteSurvivalHead
# ---------------------------------------------------------------------------


class TestSurvivalHead:
    def test_registered(self):
        assert task_registry.get("discrete_survival") is DiscreteSurvivalHead

    def test_target_dtypes(self):
        assert DiscreteSurvivalHead.target_dtypes == {
            "bin": torch.long,
            "event": torch.float,
            "time": torch.float,
        }

    def test_forward_shape(self):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4)
        assert head(torch.randn(3, 8)).shape == (3, 4)

    def test_extract_targets(self):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4)
        record = _record("s1", time=2.5, event=1, bin_=2)
        assert head.extract_targets(record) == {"bin": 2, "event": 1.0, "time": 2.5}

    def test_infer_reads_num_bins_from_the_cohort(self):
        records = [
            _record("s0", 0.5, 1, 0),
            _record("s1", 1.5, 0, 1),
            _record("s2", 3.5, 1, 3),
        ]
        assert task_registry.infer("discrete_survival", SimpleNamespace(records=records), {}) == {"num_bins": 4}

    def test_postprocess_returns_risk_scores(self):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4)
        out = head.postprocess(torch.randn(5, 4))
        assert "risk_scores" in out and out["risk_scores"].shape == (5,)

    def test_compute_metrics_returns_c_index(self):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4)
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
        assert DiscreteSurvivalHead(input_dim=8, num_bins=4).metrics == ["c_index"]

    def test_gradient_flows(self):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4)
        X = torch.randn(3, 8, requires_grad=True)
        targets = {"bin": torch.tensor([0, 1, 3]), "event": torch.tensor([1.0, 0.0, 1.0])}
        head.compute_loss(head(X), targets).backward()
        assert X.grad is not None and X.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Bins derived from the training split (binning: train_quantiles)
# ---------------------------------------------------------------------------


def _timed(sample_id: str, time: float, event: int) -> SampleRecord:
    return SampleRecord(sample_id=sample_id, targets={"time": time, "event": event})


QUANTILES = {"binning": "train_quantiles", "num_bins": 4}


class TestTrainQuantileBins:
    def test_edges_are_the_inner_quantiles_of_the_uncensored_training_times(self):
        # Six uncensored times; the 1/4, 2/4, 3/4 quantiles (linear) sit at positions
        # 1.25, 2.5 and 3.75 of the sorted times: 10 + .25 * 10, 20 + .5 * 10, 30 + .75 * 10.
        # The censored times (1 and 1000) take no part.
        times = [5.0, 10.0, 20.0, 30.0, 40.0, 100.0]
        records = [_timed(f"u{i}", t, 1) for i, t in enumerate(times)]
        records += [_timed("c0", 1.0, 0), _timed("c1", 1000.0, 0)]

        assert task_registry.fit("discrete_survival", records, QUANTILES) == {"bin_edges": [12.5, 25.0, 37.5]}

    def test_a_column_head_fits_nothing(self):
        records = [_timed(f"u{i}", float(i), 1) for i in range(8)]
        assert task_registry.fit("discrete_survival", records, {"num_bins": 4}) == {}

    @pytest.mark.parametrize(
        ("time", "event", "bin_"),
        [
            (2.0, 1, 0),  # below the training range: the open first bin
            (12.5, 1, 1),  # equal to a cut point: the upper bin
            (25.0, 0, 2),  # censored samples are binned like the others
            (37.4, 1, 2),
            (500.0, 1, 3),  # above the training range: the open last bin
        ],
    )
    def test_a_head_with_edges_bins_each_time_left_closed(self, time, event, bin_):
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4, binning="train_quantiles", bin_edges=[12.5, 25.0, 37.5])

        assert head.extract_targets(_timed("s", time, event)) == {"bin": bin_, "event": float(event), "time": time}

    def test_a_train_quantiles_head_reads_no_bin_column(self):
        assert DiscreteSurvivalHead.resolve_target_schema(QUANTILES) == {"time": float, "event": int}
        assert DiscreteSurvivalHead.resolve_target_schema({"num_bins": 4}) == {"time": float, "event": int, "bin": int}
        head = DiscreteSurvivalHead(input_dim=8, num_bins=4, binning="train_quantiles", bin_edges=[12.5, 25.0, 37.5])
        assert head.target_schema == {"time": float, "event": int}

    def test_too_few_uncensored_times_cannot_form_the_bins(self):
        records = [_timed(f"u{i}", float(i), 1) for i in range(3)] + [_timed("c", 9.0, 0)]
        with pytest.raises(ValueError, match=r"num_bins=4 .* 3 uncensored training time\(s\).*lower num_bins"):
            task_registry.fit("discrete_survival", records, QUANTILES)

    def test_tied_cut_points_cannot_form_the_bins(self):
        # Positions 5/3 and 10/3 of [1, 1, 1, 1, 1, 2] both fall among the ones.
        records = [_timed(f"u{i}", t, 1) for i, t in enumerate([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])]
        with pytest.raises(ValueError, match=r"num_bins=3 .* 6 uncensored training time\(s\).*cut point 1 .*lower num_bins"):
            task_registry.fit("discrete_survival", records, {"binning": "train_quantiles", "num_bins": 3})


def test_a_fixed_head_needs_its_bin_edges():
    with pytest.raises(ValueError, match=r"binning 'fixed' needs bin_edges"):
        DiscreteSurvivalHead(input_dim=8, num_bins=4, binning="fixed")
    with pytest.raises(ValueError, match=r"bin_edges must be num_bins - 1 = 3 strictly increasing cut points"):
        DiscreteSurvivalHead(input_dim=8, num_bins=4, binning="fixed", bin_edges=[12.0, 24.0])


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


def _survival_dataset(tmp_path: Path, rows: list[dict]) -> Cohort:
    """The survival records of ``rows``: their ``time`` / ``event`` / ``bin`` columns as
    targets (a cohort read straight from the table)."""
    frame = pd.DataFrame(rows)
    targets = [key for key in ("time", "event", "bin") if key in frame.columns]
    return Cohort.from_frames(frame.assign(split="test"), targets=targets)


class TestValidateSurvivalDataset:
    def _rows(self):
        return [
            {"sample_id": f"s{i}", "image_path": f"/s{i}.svs", "time": float(i), "event": i % 2, "bin": i}
            for i in range(4)
        ]

    def test_valid_passes(self, tmp_path: Path):
        ds = _survival_dataset(tmp_path, self._rows())
        validate_survival_dataset(ds.records, "slide")  # no raise

    def test_missing_event_column_raises(self, tmp_path: Path):
        rows = [{"sample_id": "s0", "image_path": "/s0.svs", "time": 1.0, "bin": 0}]
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="'event'"):
            validate_survival_dataset(ds.records, "slide")

    def test_bad_event_value_raises(self, tmp_path: Path):
        rows = self._rows()
        rows[0]["event"] = 2
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="event"):
            validate_survival_dataset(ds.records, "slide")

    def test_non_contiguous_bins_raise(self, tmp_path: Path):
        rows = self._rows()
        rows[2]["bin"] = 9  # bins become {0,1,9,3} → not contiguous
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="contiguous"):
            validate_survival_dataset(ds.records, "slide")

    def test_negative_time_raises(self, tmp_path: Path):
        rows = self._rows()
        rows[1]["time"] = -1.0
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="time"):
            validate_survival_dataset(ds.records, "slide")

    @pytest.mark.parametrize("num_bins", [None, 4])
    def test_patient_inconsistent_targets_raise(self, tmp_path: Path, num_bins):
        rows = [
            {"sample_id": "s0", "image_path": "/s0.svs", "time": 1.0, "event": 1, "bin": 0, "patient_id": "p0"},
            {"sample_id": "s1", "image_path": "/s1.svs", "time": 2.0, "event": 1, "bin": 1, "patient_id": "p0"},
        ]
        ds = _survival_dataset(tmp_path, rows)
        with pytest.raises(ValueError, match="inconsistent survival targets"):
            validate_survival_dataset(ds.records, "patient", num_bins=num_bins)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSurvivalConfigValidation:
    def _check(self, tmp_path, **fields):
        check(experiment(tmp_path / "d.csv", tmp_path / "s.csv", head="discrete_survival", **fields))

    def test_rejects_tile(self, tmp_path: Path):
        # A tile encoder without an aggregator embeds each tile: a tile-level sample.
        with pytest.raises(ValueError, match="not a single tile"):
            self._check(tmp_path, features=FeaturesConfig(encoder="phikon"))

    def test_rejects_clam(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not supported for survival"):
            self._check(tmp_path, aggregator="clam_sb")

    def test_allows_slide_with_abmil(self, tmp_path: Path):
        self._check(tmp_path, aggregator="abmil")


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
            "time": times,
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
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'out',
            aggregator='mean_pool',
            head="discrete_survival",
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert isinstance(result, PipelineResult)
        test_report = result.fold_results[0].test_evaluations["test"]
        assert "c_index" in test_report.metrics
        # The predictions file carries the survival head's column, never the targets.
        predictions = Predictions.from_parquet(next(result.run_dir.rglob("predictions_test.parquet")))
        assert predictions.prediction_columns == ("risk_score",)
        assert not {"time", "event"} & set(predictions.frame.columns)

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
                "time": [per_patient[p][0] for p in patient_ids],
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

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'out',
            head="discrete_survival",
            cohort={"collapse": "patient_id"},
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_evaluations["test"].metrics


def test_a_patient_run_fits_the_bin_edges_on_its_training_patients_once_each(tmp_path: Path, caplog):
    """p0's three slides count once: the training patients' uncensored times are 1, 2 and
    4, whose median is 2 (the five slides' would be 1)."""
    patients = {  # patient: (slides, time, event, split)
        "p0": (3, 1.0, 1, "train"),
        "p1": (1, 2.0, 1, "train"),
        "p2": (1, 4.0, 1, "train"),
        "p3": (1, 9.0, 0, "train"),
        "p4": (1, 3.0, 1, "tune"),
        "p5": (1, 5.0, 1, "test"),
    }
    rows = [
        {"sample_id": f"{pid}_{k}", "patient_id": pid, "image_path": f"/{pid}_{k}.svs", "time": time, "event": event}
        for pid, (slides, time, event, _) in patients.items()
        for k in range(slides)
    ]
    dataset_csv, splits_csv = tmp_path / "dataset.csv", tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    pd.DataFrame(
        {"sample_id": [r["sample_id"] for r in rows], "split": [patients[r["patient_id"]][3] for r in rows]}
    ).to_csv(splits_csv, index=False)
    features = tmp_path / "features"
    features.mkdir()
    torch.manual_seed(0)
    for pid in patients:
        torch.save(torch.randn(D), features / f"{pid}.pt")
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "out",
        run={"run_id": "patient-quantiles"},
        head="discrete_survival",
        head_params={"binning": "train_quantiles", "num_bins": 2},
        cohort={"collapse": "patient_id"},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1, "callbacks": [{"name": "checkpoint", "params": {"selection": "last"}}]},
    )
    with caplog.at_level("INFO", logger="soma"):
        result = Pipeline(config, source=CachedSetSource(features)).run()

    import json

    import yaml

    import soma

    bundle = result.run_dir / "bundle"
    head_params = json.loads((bundle / "model.json").read_text())["params"]["head"]["params"]
    assert (head_params["binning"], head_params["bin_edges"]) == ("train_quantiles", [2.0])
    head = soma.load(bundle).model.task_head
    assert (head.binning, head.bin_edges) == ("train_quantiles", [2.0])
    plan_head = yaml.safe_load((result.run_dir / "plan.yaml").read_text())["learner"]["params"]["head"]["params"]
    assert plan_head["bin_edges"] is None
    assert "fitted {'bin_edges': [2.0]} on 4 training patients" in caplog.text


def test_a_training_slide_without_tissue_takes_no_part_in_the_bin_edges(tmp_path: Path):
    """c is ``empty`` in the feature manifest, so it never trains: the edge is the median
    of a's and b's times, 1.5, not of a's, b's and c's, 2."""
    times = {"a": (1.0, "train"), "b": (2.0, "train"), "c": (10.0, "train"), "d": (3.0, "tune"), "e": (4.0, "test")}
    dataset_csv, splits_csv = tmp_path / "dataset.csv", tmp_path / "splits.csv"
    pd.DataFrame(
        {"sample_id": list(times), "image_path": [f"/{s}.svs" for s in times], "time": [t for t, _ in times.values()], "event": 1}
    ).to_csv(dataset_csv, index=False)
    pd.DataFrame({"sample_id": list(times), "split": [split for _, split in times.values()]}).to_csv(splits_csv, index=False)
    features = tmp_path / "features"
    features.mkdir()
    for sid in "abde":
        torch.save(torch.randn(D), features / f"{sid}.pt")
    pd.DataFrame(
        {"sample_id": list(times), "feature_status": ["empty" if s == "c" else "success" for s in times], "feature_path": ""}
    ).to_csv(features / "process_list.csv", index=False)
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "out",
        run={"run_id": "empty-quantiles"},
        head="discrete_survival",
        head_params={"binning": "train_quantiles", "num_bins": 2},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1, "callbacks": [{"name": "checkpoint", "params": {"selection": "last"}}]},
    )
    result = Pipeline(config, source=CachedSetSource(features)).run()

    import soma

    assert soma.load(result.run_dir / "bundle").model.task_head.bin_edges == [1.5]


@pytest.mark.parametrize("aggregation", [None, "mean_pool"])
def test_explicit_width_trains_sparse_whole_dataset_with_four_outputs(tmp_path, aggregation):
    """Cached embeddings/bags use bins 0 and 3 without dummy Cases or runtime patches."""
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,time,event,bin\n"
        "a,/a.svs,1,1,0\nb,/b.svs,7,0,3\n"
        "c,/c.svs,2,1,0\nd,/d.svs,8,0,3\n"
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text("sample_id,split\na,train\nb,train\nc,test\nd,test\n")
    features = tmp_path / "features"
    features.mkdir()
    for sid, values in [("a", [1., 0., 2., 0.]), ("b", [2., 1., 1., 0.5]),
                        ("c", [3., 2., 0., 1.]), ("d", [4., 0., -1., 1.5])]:
        tensor = torch.tensor(values)
        torch.save(tensor.unsqueeze(0) if aggregation else tensor, features / f"{sid}.pt")
    original = {p.name: p.read_bytes() for p in features.iterdir()}
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / 'out',
        aggregator=aggregation or None,
        head="discrete_survival",
        head_params={'num_bins': 4},
        seed=0,
        cohort={"allow_missing_tune": True},
        run={"run_id": 'sparse-nll'},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1, "callbacks": [{"name": "checkpoint", "params": {"selection": 'last'}}]},
        scheduler='cosine',
    )
    result = Pipeline(config, source=CachedSetSource(features)).run()
    fold = result.fold_results[0]
    assert 0 <= fold.test_evaluations["test"].metrics["c_index"] <= 1
    from safetensors.torch import load_file

    weights = load_file(result.run_dir / "bundle" / "weights.safetensors")
    assert tuple(weights["task_head.fc.weight"].shape) == (4, 4)
    assert pd.read_csv(dataset_csv)["bin"].tolist() == [0, 3, 0, 3]
    assert {name: (features / name).read_bytes() for name in original} == original


def test_explicit_bin_width_accepts_sparse_indices_unchanged(tmp_path):
    dataset = _survival_dataset(tmp_path, [
        {"sample_id": "a", "image_path": "/a.svs", "time": 1., "event": 1, "bin": 0},
        {"sample_id": "b", "image_path": "/b.svs", "time": 7., "event": 0, "bin": 3},
    ])
    validate_survival_dataset(dataset.records, "slide", num_bins=4)
    assert [r.targets["bin"] for r in dataset.records] == [0, 3]


def test_explicit_bin_width_rejects_an_index_at_the_width(tmp_path):
    dataset = _survival_dataset(tmp_path, [
        {"sample_id": "a", "image_path": "/a.svs", "time": 1., "event": 1, "bin": 0},
        {"sample_id": "b", "image_path": "/b.svs", "time": 7., "event": 0, "bin": 4},
    ])
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        validate_survival_dataset(dataset.records, "slide", num_bins=4)


@pytest.mark.parametrize("width", [0, -1, True, False, 4.0, 2.5, "4"])
def test_explicit_bin_width_must_be_a_positive_integer(tmp_path, width):
    dataset = _survival_dataset(tmp_path, [
        {"sample_id": "a", "image_path": "/a.svs", "time": 1., "event": 1, "bin": 0},
    ])
    with pytest.raises(ValueError, match="num_bins.*positive integer"):
        validate_survival_dataset(dataset.records, "slide", num_bins=width)


@pytest.mark.parametrize("bins", [[0, 3], [1, 2]])
def test_absent_bin_width_preserves_contiguous_from_zero_requirement(tmp_path, bins):
    dataset = _survival_dataset(tmp_path, [
        {"sample_id": "a", "image_path": "/a.svs", "time": 1., "event": 1, "bin": bins[0]},
        {"sample_id": "b", "image_path": "/b.svs", "time": 7., "event": 0, "bin": bins[1]},
    ])
    with pytest.raises(ValueError, match="contiguous integers starting at 0"):
        validate_survival_dataset(dataset.records, "slide")
