"""Tests for Cox prediction-accumulation (phase 2: large variable-size MIL bags).

Covers the window as a training batch (graph connectivity, no-detach), padded-vs-
accumulation parity, the no-pad window collate, full-cohort MIL eval, the revised
config guards, and MIL accumulation + padded end-to-end runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from soma.data import CachedSetSource
from soma.models.aggregators.registry import aggregator_registry
from tests.experiment_configs import experiment
from tests.extraction_configs import extraction
from soma.config import FeaturesConfig
from soma.plan import check
from soma.evaluation import Predictions
from soma.models.heads.survival import CoxSurvivalHead, cox_breslow_loss
from soma.pipeline import Pipeline, PipelineResult
from soma.learn.collate import CoxWindowBatch, bag_collate_fn, cox_window_collate
from soma.models.model import MILModel
from soma.learn import MILLearner

D = 16
FIXED_RUN_ID = "20990101_000000"


def _mil_cox_model(cox_window: int = 1) -> MILModel:
    agg = aggregator_registry.get("abmil")(input_dim=D)
    head = CoxSurvivalHead(input_dim=agg.output_dim, cox_window=cox_window)
    return MILModel(aggregator=agg, task_head=head)


def _transmil_cox_model(cox_window: int = 1) -> MILModel:
    agg = aggregator_registry.get("transmil")(
        input_dim=D,
        att_dim=16,
        n_layers=2,
        n_heads=2,
        n_landmarks=8,
        dropout=0.0,
    )
    head = CoxSurvivalHead(input_dim=agg.output_dim, cox_window=cox_window)
    return MILModel(aggregator=agg, task_head=head)


def _hipt_cox_model(cox_window: int = 1) -> MILModel:
    """HIPT over regions of 2 x 2 tiles; no dropout, so a forward is deterministic."""
    agg = aggregator_registry.get("hipt")(
        input_dim=D,
        region_size=8,
        patch_size=4,
        embed_dim_region=12,
        embed_dim_slide=12,
        num_heads=2,
        dropout=0.0,
    )
    head = CoxSurvivalHead(input_dim=agg.output_dim, cox_window=cox_window)
    return MILModel(aggregator=agg, task_head=head)


def _bags(sizes, seed=0):
    """Return a list of (n_i, D) feature tensors."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, D, generator=g) for n in sizes]


def _region_bags(regions, seed=0):
    """Return a list of (m_i, 4, D) region grids: m_i regions of 2 x 2 tiles."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(m, 4, D, generator=g) for m in regions]


def _window_batch(bags, time, event):
    return CoxWindowBatch(
        bags=bags,
        targets={"time": torch.tensor(time), "event": torch.tensor(event)},
        sample_ids=tuple(f"s{i}" for i in range(len(bags))),
    )


def _bag_batch(bags, time, event):
    """Pad a list of bags into a BagBatch (with mask) via the real collate."""
    items = [
        (bag, {"time": time[i], "event": event[i]}, f"s{i}")
        for i, bag in enumerate(bags)
    ]
    return bag_collate_fn(items, target_dtypes={"time": torch.float, "event": torch.float})


# ---------------------------------------------------------------------------
# Windowed training loop — graph connectivity and no-detach
# ---------------------------------------------------------------------------


class TestWindowedLoop:
    def test_head_flags_in_accumulation_mode(self):
        head = CoxSurvivalHead(input_dim=D, cox_window=4)
        assert head.accumulates_predictions is True
        assert head.accumulation_window == 4

    def test_head_flags_in_padded_mode(self):
        head = CoxSurvivalHead(input_dim=D)  # cox_window defaults to 1
        assert head.accumulates_predictions is False
        assert head.accumulation_window == 1

    def test_risks_stay_graph_connected_before_loss(self):
        # The crux of prediction accumulation: per-bag risks must keep their graph
        # (no detach/item/no_grad) so one Cox loss couples them.
        model = _mil_cox_model(cox_window=3)
        model.train()
        bags = _bags([5, 8, 3])
        risks = [model(bag.unsqueeze(0)).logits.view(1) for bag in bags]
        for r in risks:
            assert r.requires_grad and r.grad_fn is not None
        risk = torch.cat(risks)
        assert risk.requires_grad and risk.grad_fn is not None

    def test_window_step_produces_gradients(self, tmp_path: Path):
        model = _mil_cox_model(cox_window=3)
        window = _window_batch(_bags([5, 8, 3]), [3.0, 2.0, 1.0], [1.0, 0.0, 1.0])
        loss = MILLearner(model).training_step(window)
        assert loss.ndim == 0
        loss.backward()
        agg_grads = [p.grad for p in model.aggregator.parameters() if p.grad is not None]
        assert agg_grads and any(g.abs().sum() > 0 for g in agg_grads)

    def test_window_step_passes_all_true_masks(self, tmp_path: Path):
        model = _transmil_cox_model(cox_window=3)
        seen_masks = []
        original_forward = model.forward

        def capture_forward(X, mask=None):
            seen_masks.append(None if mask is None else mask.detach().cpu().clone())
            return original_forward(X, mask=mask)

        model.forward = capture_forward
        window = _window_batch(_bags([5, 8, 3]), [3.0, 2.0, 1.0], [1.0, 0.0, 1.0])
        MILLearner(model).training_step(window)

        assert [mask.tolist() if mask is not None else None for mask in seen_masks] == [
            [[True] * 5],
            [[True] * 8],
            [[True] * 3],
        ]

    def test_hipt_window_step_reaches_every_bag_with_the_breslow_loss(self):
        torch.manual_seed(0)
        model = _hipt_cox_model(cox_window=8)
        bags = [bag.requires_grad_() for bag in _region_bags([2, 1, 3, 2, 1, 4, 2, 3])]
        # The earliest time is an event, so every bag is at risk at some event time and
        # enters the loss (a censored bag earlier than every event would get no gradient).
        time = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        event = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        loss = MILLearner(model).training_step(_window_batch(bags, time, event))
        loss.backward()

        for i, bag in enumerate(bags):
            assert bag.grad is not None and bag.grad.abs().sum() > 0, f"bag {i}"
        with torch.no_grad():
            risks = torch.cat([model(bag.unsqueeze(0)).logits.view(1) for bag in bags])
        expected = cox_breslow_loss(risks, torch.tensor(time), torch.tensor(event))
        assert torch.allclose(loss.detach(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Padded-vs-accumulation parity — the key correctness guard
# ---------------------------------------------------------------------------


class TestParity:
    def test_padded_and_accumulation_risks_match(self):
        # Masking makes a padded forward equal to an un-padded one, so the two
        # modes must produce the same risks (and hence the same Cox loss).
        torch.manual_seed(0)
        model = _mil_cox_model()
        model.eval()  # disable dropout for a deterministic comparison
        bags = _bags([5, 8, 3, 6])
        time = [4.0, 3.0, 2.0, 1.0]
        event = [1.0, 0.0, 1.0, 1.0]

        with torch.no_grad():
            # Accumulation: forward each bag un-padded.
            acc_risk = torch.cat([model(b.unsqueeze(0)).logits.view(1) for b in bags])
            # Padded: one masked batch.
            batch = _bag_batch(bags, time, event)
            pad_risk = model(batch.features, mask=batch.mask).logits.squeeze(-1)

        assert torch.allclose(acc_risk, pad_risk, atol=1e-5)

        t = torch.tensor(time)
        e = torch.tensor(event)
        assert torch.allclose(
            cox_breslow_loss(acc_risk, t, e), cox_breslow_loss(pad_risk, t, e), atol=1e-5
        )


# ---------------------------------------------------------------------------
# Window collate
# ---------------------------------------------------------------------------


class TestCoxWindowCollate:
    def test_returns_unpadded_bags_and_stacked_targets(self):
        bags = _bags([5, 8, 3])
        items = [
            (bags[0], {"time": 3.0, "event": 1}, "s0"),
            (bags[1], {"time": 2.0, "event": 0}, "s1"),
            (bags[2], {"time": 1.0, "event": 1}, "s2"),
        ]
        batch = cox_window_collate(items, target_dtypes={"time": torch.float, "event": torch.float})
        assert isinstance(batch, CoxWindowBatch)
        # Bags kept at their own lengths (no padding to a common max).
        assert [b.shape[0] for b in batch.bags] == [5, 8, 3]
        assert batch.targets["time"].tolist() == [3.0, 2.0, 1.0]
        assert batch.targets["event"].tolist() == [1.0, 0.0, 1.0]
        assert batch.sample_ids == ("s0", "s1", "s2")


# ---------------------------------------------------------------------------
# Full-cohort MIL eval — invariant to tune batch partition
# ---------------------------------------------------------------------------


class TestFullCohortMILEval:
    def test_tune_loss_invariant_to_partition(self, tmp_path: Path):
        torch.manual_seed(0)
        model = _mil_cox_model(cox_window=2)  # accumulation head; eval still full-cohort
        model.eval()
        bags = _bags([5, 8, 3, 6, 4, 7])
        time = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        event = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]  # leading tune batch all-censored

        whole = [_bag_batch(bags, time, event)]
        pairs = [
            _bag_batch(bags[i : i + 2], time[i : i + 2], event[i : i + 2])
            for i in range(0, 6, 2)
        ]

        learner = MILLearner(model)
        learner.eval()
        losses = {}
        for name, loader in {"whole": whole, "pairs": pairs}.items():
            losses[name] = learner.validation_summary(
                [learner.validation_step(batch) for batch in loader]
            )["loss"]

        with torch.no_grad():
            risk = torch.cat([model(b.unsqueeze(0)).logits.view(1) for b in bags])
        reference = float(cox_breslow_loss(risk, torch.tensor(time), torch.tensor(event)))

        for name, loss in losses.items():
            assert loss == pytest.approx(reference, abs=1e-5), name


# ---------------------------------------------------------------------------
# Revised config guards
# ---------------------------------------------------------------------------


class TestRevisedCoxGuards:
    def _check(self, tmp_path, *, aggregator="abmil", batch_size, accumulate=1, **head_params):
        check(
            experiment(
                tmp_path / "d.csv",
                tmp_path / "s.csv",
                output_dir=tmp_path / "out",
                aggregator=aggregator,
                head="cox_survival",
                head_params=head_params,
                loaders={"batch_size": batch_size},
                trainer={"accumulate_grad_batches": accumulate},
            )
        )

    def test_padded_mil_now_allowed(self, tmp_path: Path):
        # Phase-2 change: Cox + MIL aggregator at batch_size >= 2 is valid.
        self._check(tmp_path, batch_size=4)

    def test_accumulation_requires_batch_size_one(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size = 1"):
            self._check(tmp_path, batch_size=2, cox_window=8)

    def test_accumulation_requires_aggregator(self, tmp_path: Path):
        with pytest.raises(ValueError, match="needs an aggregator"):
            self._check(tmp_path, aggregator=None, batch_size=1, cox_window=8)

    def test_accumulation_valid_config_passes(self, tmp_path: Path):
        self._check(tmp_path, batch_size=1, cox_window=8)

    def test_gradient_accumulation_still_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="accumulate_grad_batches = 1"):
            self._check(tmp_path, batch_size=1, accumulate=2, cox_window=8)

    def test_single_embedding_cox_still_valid(self, tmp_path: Path):
        # Phase-1 path unchanged: no aggregator, no cox_window, batch_size >= 2.
        self._check(tmp_path, aggregator=None, batch_size=4)

    def test_invalid_cox_window_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cox_window"):
            self._check(tmp_path, batch_size=1, cox_window=0)


# ---------------------------------------------------------------------------
# End-to-end through Pipeline.run (MIL bags)
# ---------------------------------------------------------------------------


def _setup_mil_survival_data(tmp_path: Path):
    n = 8
    events = [1, 1, 0, 1, 1, 0, 1, 0]
    times = [float(i + 1) for i in range(n)]
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/slides/s{i}.svs" for i in range(n)],
            "time": times,
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
        torch.save(torch.randn(5 + i, D), feature_dir / f"s{i}.pt")  # (n_tiles, D) -> MIL bag
    return dataset_csv, splits_csv, feature_dir


class TestMILCoxEndToEnd:
    def test_accumulation_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'out',
            aggregator='abmil',
            head='cox_survival',
            head_params={'cox_window': 2},
            loaders={"batch_size": 1},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()
        assert isinstance(result, PipelineResult)
        report = result.fold_results[0].test_evaluations["test"]
        assert "c_index" in report.metrics
        predictions = Predictions.from_parquet(next(result.run_dir.rglob("predictions_test.parquet")))
        assert predictions.task_family == "survival"
        assert predictions.prediction_columns == ("risk_score",)

    def test_padded_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'out',
            aggregator='abmil',
            head='cox_survival',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()
        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_evaluations["test"].metrics

    def test_hierarchical_accumulation_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        for i in range(8):
            torch.save(torch.randn(2 + (i % 2), 4, D), feature_dir / f"s{i}.pt")
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'out',
            features=FeaturesConfig(extraction=extraction(0.5, 4, requested_region_size_px=8)),
            aggregator={"name": 'hipt', "params": {'embed_dim_region': 12, 'embed_dim_slide': 12, 'num_heads': 2, 'dropout': 0.0}},
            head='cox_survival',
            head_params={'cox_window': 2},
            loaders={"batch_size": 1},
            trainer={"max_epochs": 1},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()
        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_evaluations["test"].metrics

    def test_abmil_accumulation_mode_runs_on_a_region_grid_store(self, tmp_path: Path):
        """ABMIL reads each region grid as one flat bag through the Cox window, and
        scores exactly as on the same tiles stored as a flat bag."""
        risks = {}
        for layout in ("regions", "flat"):
            (tmp_path / layout).mkdir()
            dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path / layout)
            torch.manual_seed(1)
            for i in range(8):
                grid = torch.randn(2 + (i % 2), 4, D)
                torch.save(grid if layout == "regions" else grid.reshape(-1, D), feature_dir / f"s{i}.pt")
            config = experiment(
                dataset_csv,
                splits_csv,
                output_dir=tmp_path / layout / 'out',
                aggregator='abmil',
                head='cox_survival',
                head_params={'cox_window': 8},
                loaders={"batch_size": 1},
                trainer={"max_epochs": 2},
                scheduler='cosine',
            )
            with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
                result = Pipeline(config, source=CachedSetSource(feature_dir)).run()
            assert "c_index" in result.fold_results[0].test_evaluations["test"].metrics
            risks[layout] = Predictions.from_parquet(next(result.run_dir.rglob("predictions_test.parquet")))
        pd.testing.assert_frame_equal(risks["regions"].frame, risks["flat"].frame)
