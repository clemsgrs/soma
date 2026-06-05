"""Tests for Cox prediction-accumulation (phase 2: large variable-size MIL bags).

Covers the windowed training loop (graph connectivity, no-detach), padded-vs-
accumulation parity, the no-pad window collate, full-cohort MIL eval, the revised
config guards, and MIL accumulation + padded end-to-end runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from soma.aggregators.registry import aggregator_registry
from soma.config import (
    AggregatorConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.tasks.survival import CoxSurvivalHead, cox_breslow_loss
from soma.pipeline import Pipeline, PipelineResult
from soma.training.collate import CoxWindowBatch, bag_collate_fn, cox_window_collate
from soma.training.model import MILModel
from soma.training.trainer import Trainer

D = 16
FIXED_RUN_ID = "20990101_000000"


def _mil_cox_model(cox_window: int = 1) -> MILModel:
    agg = aggregator_registry.get("abmil")(input_dim=D)
    head = CoxSurvivalHead(input_dim=agg.output_dim, cox_window=cox_window)
    return MILModel(aggregator=agg, task_head=head)


def _bags(sizes, seed=0):
    """Return a list of (n_i, D) feature tensors."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, D, generator=g) for n in sizes]


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

    def test_windowed_epoch_produces_gradients(self, tmp_path: Path):
        model = _mil_cox_model(cox_window=3)
        window = _window_batch(_bags([5, 8, 3]), [3.0, 2.0, 1.0], [1.0, 0.0, 1.0])
        trainer = Trainer(
            model=model,
            train_loader=[window],
            tune_loader=[window],
            config=TrainingConfig(epochs=1, batch_size=1),
            fold_dir=tmp_path,
            device=torch.device("cpu"),
        )
        loss = trainer._train_epoch()
        assert isinstance(loss, float)
        agg_grads = [p.grad for p in model.aggregator.parameters() if p.grad is not None]
        assert agg_grads and any(g.abs().sum() > 0 for g in agg_grads)


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

        losses = {}
        for name, loader in {"whole": whole, "pairs": pairs}.items():
            trainer = Trainer(
                model=model,
                train_loader=loader,
                tune_loader=loader,
                config=TrainingConfig(epochs=1, batch_size=1),
                fold_dir=tmp_path / name,
                device=torch.device("cpu"),
            )
            losses[name], _ = trainer._tune()

        with torch.no_grad():
            risk = torch.cat([model(b.unsqueeze(0)).logits.view(1) for b in bags])
        reference = float(cox_breslow_loss(risk, torch.tensor(time), torch.tensor(event)))

        for name, loss in losses.items():
            assert loss == pytest.approx(reference, abs=1e-5), name


# ---------------------------------------------------------------------------
# Revised config guards
# ---------------------------------------------------------------------------


class TestRevisedCoxGuards:
    def _base(self, tmp_path, **kw):
        return dict(
            dataset_csv=tmp_path / "d.csv",
            splits_csv=tmp_path / "s.csv",
            output_root=tmp_path / "out",
            dataset_type="slide",
            **kw,
        )

    def test_padded_mil_now_allowed(self, tmp_path: Path):
        # Phase-2 change: Cox + MIL aggregator at batch_size >= 2 is valid.
        PipelineConfig(
            **self._base(
                tmp_path,
                aggregator=AggregatorConfig(name="abmil"),
                task=TaskConfig(name="survival", params={"loss": "cox"}),
                training=TrainingConfig(batch_size=4),
            )
        )

    def test_accumulation_requires_batch_size_one(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size = 1"):
            PipelineConfig(
                **self._base(
                    tmp_path,
                    aggregator=AggregatorConfig(name="abmil"),
                    task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 8}),
                    training=TrainingConfig(batch_size=2),
                )
            )

    def test_accumulation_requires_aggregator(self, tmp_path: Path):
        with pytest.raises(ValueError, match="requires an aggregator"):
            PipelineConfig(
                **self._base(
                    tmp_path,
                    aggregator=None,
                    task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 8}),
                    training=TrainingConfig(batch_size=1),
                )
            )

    def test_accumulation_valid_config_passes(self, tmp_path: Path):
        PipelineConfig(
            **self._base(
                tmp_path,
                aggregator=AggregatorConfig(name="abmil"),
                task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 8}),
                training=TrainingConfig(batch_size=1),
            )
        )

    def test_gradient_accumulation_still_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="gradient_accumulation = 1"):
            PipelineConfig(
                **self._base(
                    tmp_path,
                    aggregator=AggregatorConfig(name="abmil"),
                    task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 8}),
                    training=TrainingConfig(batch_size=1, gradient_accumulation=2),
                )
            )

    def test_single_embedding_cox_still_valid(self, tmp_path: Path):
        # Phase-1 path unchanged: no aggregator, no cox_window, batch_size >= 2.
        PipelineConfig(
            **self._base(
                tmp_path,
                aggregator=None,
                task=TaskConfig(name="survival", params={"loss": "cox"}),
                training=TrainingConfig(batch_size=4),
            )
        )

    def test_invalid_cox_window_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cox_window"):
            PipelineConfig(
                **self._base(
                    tmp_path,
                    aggregator=AggregatorConfig(name="abmil"),
                    task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 0}),
                    training=TrainingConfig(batch_size=1),
                )
            )


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
        torch.save(torch.randn(5 + i, D), feature_dir / f"s{i}.pt")  # (n_tiles, D) -> MIL bag
    return dataset_csv, splits_csv, feature_dir


class TestMILCoxEndToEnd:
    def test_accumulation_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 2}),
            training=TrainingConfig(epochs=2, patience=10, batch_size=1),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()
        assert isinstance(result, PipelineResult)
        report = result.fold_results[0].test_reports["test"]
        assert "c_index" in report.metrics
        preds_csv = next(result.run_dir.rglob("predictions_test.csv"))
        cols = pd.read_csv(preds_csv).columns
        assert {"sample_id", "true_label", "event", "risk_score"} <= set(cols)

    def test_padded_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="survival", params={"loss": "cox"}),  # padded mode
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()
        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_reports["test"].metrics

    def test_hierarchical_accumulation_mode_run(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_mil_survival_data(tmp_path)
        for i in range(8):
            torch.save(torch.randn(2 + (i % 2), 4, D), feature_dir / f"s{i}.pt")
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=tmp_path / "out",
            dataset_type="slide",
            preprocessing=PreprocessingConfig(
                requested_tile_size_px=4,
                requested_region_size_px=8,
            ),
            aggregator=AggregatorConfig(
                name="hipt",
                params={
                    "embed_dim_region": 12,
                    "embed_dim_slide": 12,
                    "num_heads": 2,
                    "dropout": 0.0,
                },
            ),
            task=TaskConfig(name="survival", params={"loss": "cox", "cox_window": 2}),
            training=TrainingConfig(epochs=1, patience=10, batch_size=1),
        )
        with patch("soma.output_layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, feature_dir=feature_dir).run()
        assert isinstance(result, PipelineResult)
        assert "c_index" in result.fold_results[0].test_reports["test"].metrics
