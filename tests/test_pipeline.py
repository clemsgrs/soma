"""Tests for soma.pipeline — the ``Pipeline`` orchestrator and its run panels."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest
import yaml
from rich.console import Console

import slide2vec

from soma import ExperimentConfig, Plan, resolve
from soma.config import CacheOptions, FeaturesConfig
from soma.plan import load_cohort
from tests.experiment_configs import experiment
from tests.extraction_configs import annotation_masks, extraction
from soma.data._legacy import legacy_samples_from_csv, legacy_folds_from_csv
from soma.evaluation import Predictions
from soma.data import CachedSetSource
from soma.pipeline.layout import (
    build_experiment_spec,
    read_test_results,
)
from soma.pipeline.layout import test_identity_digest as compute_test_digest
from soma.pipeline import Pipeline, PipelineResult
from soma.pipeline.console import build_completed_run_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


D = 16
NUM_SAMPLES = 8
FIXED_RUN_ID = "2026-04-09_16-22-10__local"


def _plan(config: ExperimentConfig) -> Plan:
    return resolve(config, load_cohort(config))


def _expected_run_dir(config: ExperimentConfig) -> Path:
    spec = build_experiment_spec(_plan(config))
    return Path(config.run.output_dir) / "experiments" / spec.experiment_dirname / "runs" / FIXED_RUN_ID


def _setup_synthetic_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset.csv, splits.csv, and feature .pt files.

    Returns (dataset_csv, splits_csv, feature_dir).
    """
    # Dataset CSV
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [f"/slides/s{i}.svs" for i in range(NUM_SAMPLES)],
            "label": ["tumor" if i % 2 == 0 else "normal" for i in range(NUM_SAMPLES)],
        }
    ).to_csv(dataset_csv, index=False)

    # Splits CSV — single fold: 6 train, 1 tune, 1 test
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * NUM_SAMPLES,
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "split": ["train"] * 6 + ["tune", "test"],
        }
    ).to_csv(splits_csv, index=False)

    # Feature files
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(42)
    for i in range(NUM_SAMPLES):
        n_tiles = 5 + i
        torch.save(torch.randn(n_tiles, D), feature_dir / f"s{i}.pt")

    return dataset_csv, splits_csv, feature_dir


def _write_feature_manifest(feature_dir: Path, statuses: dict[str, str]) -> None:
    rows = []
    for sample_id, status in statuses.items():
        rows.append(
            {
                "sample_id": sample_id,
                "feature_status": status,
                "feature_path": str((feature_dir / f"{sample_id}.pt").resolve()) if status == "success" else "",
                "num_tiles": 1 if status == "success" else 0,
                "feature_rank": 2,
                "feature_dim": D,
            }
        )
    pd.DataFrame(rows).to_csv(feature_dir / "process_list.csv", index=False)


def _setup_multifold_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset with 2 folds."""
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [f"/slides/s{i}.svs" for i in range(NUM_SAMPLES)],
            "label": ["tumor" if i % 2 == 0 else "normal" for i in range(NUM_SAMPLES)],
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    rows = []
    for fold in [0, 1]:
        for i in range(NUM_SAMPLES):
            if fold == 0:
                split = "train" if i < 6 else ("tune" if i == 6 else "test")
            else:
                split = "train" if i >= 2 else ("tune" if i == 0 else "test")
            rows.append({"fold": fold, "sample_id": f"s{i}", "split": split})
    pd.DataFrame(rows).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(42)
    for i in range(NUM_SAMPLES):
        torch.save(torch.randn(5 + i, D), feature_dir / f"s{i}.pt")

    return dataset_csv, splits_csv, feature_dir


def _setup_hierarchical_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a small dataset with hierarchical (3-D) feature tensors."""
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"/slides/s{i}.svs" for i in range(4)],
            "label": ["tumor", "normal", "tumor", "normal"],
        }
    ).to_csv(dataset_csv, index=False)

    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "fold": [0] * 4,
            "sample_id": ["s0", "s1", "s2", "s3"],
            "split": ["train", "train", "tune", "test"],
        }
    ).to_csv(splits_csv, index=False)

    feature_dir = tmp_path / "hier_features"
    feature_dir.mkdir()
    torch.manual_seed(21)
    torch.save(torch.randn(2, 4, D), feature_dir / "s0.pt")
    torch.save(torch.randn(3, 4, D), feature_dir / "s1.pt")
    torch.save(torch.randn(1, 4, D), feature_dir / "s2.pt")
    torch.save(torch.randn(2, 4, D), feature_dir / "s3.pt")

    return dataset_csv, splits_csv, feature_dir


# ---------------------------------------------------------------------------
# The completed-run panel
# ---------------------------------------------------------------------------


class TestCompletedRunPanel:
    def test_completed_run_panel_includes_coverage_summary(self):
        panel = build_completed_run_panel(
            summary_metrics={
                "test/auroc_mean": 0.75,
                "test/auroc_std": 0.02,
                "test/coverage_mean": 0.9,
                "test/num_samples_mean": 10.0,
                "test/num_real_samples_mean": 9.0,
                "test/num_placeholder_samples_mean": 1.0,
            }
        )
        console = Console(record=True, width=120)
        console.print(panel)
        rendered = console.export_text()

        assert "coverage" in rendered.lower()
        assert "auroc=0.7500" in rendered
        assert "90.0% (9/10)" in rendered
        assert "placeholder=1" in rendered

    def test_completed_run_panel_includes_single_fold_coverage_summary(self):
        panel = build_completed_run_panel(
            summary_metrics={
                "test/auroc": 0.75,
                "test/coverage": 0.9,
                "test/num_samples": 10.0,
                "test/num_real_samples": 9.0,
                "test/num_placeholder_samples": 1.0,
            }
        )
        console = Console(record=True, width=120)
        console.print(panel)
        rendered = console.export_text()

        assert "coverage" in rendered.lower()
        assert "auroc=0.7500" in rendered
        assert "90.0% (9/10)" in rendered
        assert "placeholder=1" in rendered

# ---------------------------------------------------------------------------
# Pipeline (Layer 2 — orchestrator)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pooled_mask_config_can_read_preextracted_features(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            features=FeaturesConfig(extraction=extraction(None, None, masks=annotation_masks({'tumor': 1}, {'tumor': 0.5}))),
            aggregator='mean_pool',
            loaders={"batch_size": 2, "pin_memory": False},
            trainer={"max_epochs": 1},
            scheduler='cosine',
        )
        source = CachedSetSource(feature_dir)

        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=source).run()

        # A pooled masks config trains on the supplied features; nothing is extracted.
        assert "test/auroc" in result.summary
        assert not (_expected_run_dir(config) / "features").exists()

    def test_run_single_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        pipeline = Pipeline(config, source=CachedSetSource(feature_dir))
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.run_dir == _expected_run_dir(config)

    def test_run_trains_on_a_supplied_in_memory_source(self, tmp_path: Path):
        """A supplied source without a feature manifest (``from_arrays``) trains and
        scores like a cache: every listed sample must simply have features."""
        from soma.data import from_arrays

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        features = {path.stem: torch.load(path) for path in sorted(feature_dir.glob("*.pt"))}
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            aggregator='mean_pool',
            loaders={"batch_size": 2, "pin_memory": False},
            trainer={"max_epochs": 1},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=from_arrays(features)).run()

        assert "test/auroc" in result.summary
        assert (result.run_dir / "bundle" / "model.json").is_file()

    def test_run_result_carries_the_checksums_its_metadata_files_record(
        self, tmp_path: Path
    ):
        import json

        import yaml

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        run_dir = _expected_run_dir(config)
        recorded = json.loads(
            (run_dir.parents[1] / "experiment.json").read_text(encoding="utf-8")
        )
        run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
        assert result.dataset_checksum == recorded["dataset_checksum"] != ""
        assert result.splits_checksum == recorded["splits_checksum"] != ""
        assert result.test_checksum == run["test_checksum"] != ""

    def test_run_ignores_samples_without_features(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        (feature_dir / "s5.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "empty",
                "s6": "success",
                "s7": "success",
            },
        )
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert (_expected_run_dir(config) / "bundle" / "weights.safetensors").is_file()
        assert not (_expected_run_dir(config) / "best_model.pt").exists()

    def test_run_saves_config_yaml(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert (_expected_run_dir(config) / "config.yaml").exists()

    def test_run_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, source=CachedSetSource(feature_dir)).run()

        summary_path = _expected_run_dir(config) / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "test/auroc" in summary

    def test_test_invariant_identity_lets_test_sets_coexist_and_guards_rescore(
        self, tmp_path: Path
    ):
        """Issue #247 end-to-end: same train/tune + different test sets → same experiment
        dir (the checkpoint is reusable), results coexist keyed by test identity, and
        re-scoring the same test set is refused unless overwrite_test."""
        # dataset with a shared train/tune core (s0..s2) and two candidate test rows.
        dataset_csv = tmp_path / "dataset.csv"
        pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "sA", "sB"],
                "image_path": [f"/slides/{s}.svs" for s in ["s0", "s1", "s2", "sA", "sB"]],
                "label": ["tumor", "normal", "tumor", "normal", "tumor"],
            }
        ).to_csv(dataset_csv, index=False)
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        torch.manual_seed(0)
        for sid in ["s0", "s1", "s2", "sA", "sB"]:
            torch.save(torch.randn(6, D), feature_dir / f"{sid}.pt")

        def _splits(test_sample: str) -> Path:
            path = tmp_path / f"splits_{test_sample}.csv"
            pd.DataFrame(
                {
                    "fold": [0, 0, 0, 0],
                    "sample_id": ["s0", "s1", "s2", test_sample],
                    "split": ["train", "train", "tune", "test"],
                }
            ).to_csv(path, index=False)
            return path

        output_root = tmp_path / "output"

        def _config(test_sample: str, **eval_kwargs) -> ExperimentConfig:
            return experiment(
                dataset_csv,
                _splits(test_sample),
                output_dir=output_root,
                aggregator='mean_pool',
                loaders={"batch_size": 2},
                trainer={"max_epochs": 2},
                evaluation=eval_kwargs,
                scheduler='cosine',
            )

        config_a = _config("sA")
        config_b = _config("sB")

        # Same train/tune, different test set → identical experiment id (checkpoint reuse).
        assert (
            build_experiment_spec(_plan(config_a)).experiment_id
            == build_experiment_spec(_plan(config_b)).experiment_id
        )

        run_a = "2026-04-09_16-22-10__local"
        run_b = "2026-04-09_16-23-10__local"
        with patch("soma.pipeline.layout.make_run_id", return_value=run_a):
            result_a = Pipeline(config_a, source=CachedSetSource(feature_dir)).run()
        with patch("soma.pipeline.layout.make_run_id", return_value=run_b):
            result_b = Pipeline(config_b, source=CachedSetSource(feature_dir)).run()

        # Both runs land under ONE experiment dir; each records its own test identity.
        assert result_a.run_dir.parent == result_b.run_dir.parent
        reg_a = read_test_results(result_a.run_dir)
        reg_b = read_test_results(result_b.run_dir)
        assert list(reg_a) == [compute_test_digest(_plan(config_a))]
        assert list(reg_b) == [compute_test_digest(_plan(config_b))]
        assert set(reg_a) != set(reg_b)  # distinct test sets coexist
        assert "test" in result_a.fold_results[0].test_evaluations

        # Re-scoring the SAME test set into the SAME run dir is refused (loud skip) — the
        # test splits are held out so the prior result is not clobbered.
        with patch("soma.pipeline.layout.make_run_id", return_value=run_a):
            result_a_again = Pipeline(config_a, source=CachedSetSource(feature_dir)).run()
        assert result_a_again.fold_results[0].test_evaluations == {}

        # overwrite_test re-scores it.
        with patch("soma.pipeline.layout.make_run_id", return_value=run_a):
            result_a_over = Pipeline(
                _config("sA", overwrite_test=True), source=CachedSetSource(feature_dir)
            ).run()
        assert "test" in result_a_over.fold_results[0].test_evaluations

    def test_run_saves_summary_json_with_coverage_aggregates(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        for sample_id in ("s6", "s7"):
            (feature_dir / f"{sample_id}.pt").unlink()
        _write_feature_manifest(
            feature_dir,
            {
                "s0": "success",
                "s1": "success",
                "s2": "success",
                "s3": "success",
                "s4": "success",
                "s5": "success",
                "s6": "empty",
                "s7": "empty",
            },
        )
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            Pipeline(config, source=CachedSetSource(feature_dir)).run()

        summary = json.loads((_expected_run_dir(config) / "summary.json").read_text())
        assert "test/coverage" in summary
        assert "test/num_samples" in summary
        assert "test/num_real_samples" in summary
        assert "test/num_placeholder_samples" in summary

    def test_run_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        output_root = tmp_path / "output"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert len(result.fold_results) == 2
        assert (_expected_run_dir(config) / "fold_0" / "bundle").is_dir()
        assert (_expected_run_dir(config) / "fold_1" / "bundle").is_dir()

        summary = json.loads((_expected_run_dir(config) / "summary.json").read_text())
        assert "test/auroc_mean" in summary
        assert "test/auroc_std" in summary

    def test_run_slide_level_features(self, tmp_path: Path):
        """Pipeline with aggregator=None should work with slide-level (1-D) features."""
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)

        # Create slide-level features (1-D per sample)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        torch.manual_seed(7)
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        output_root = tmp_path / "output_slide"
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(slide_feature_dir)).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].test_evaluations["test"] is not None
        assert "auroc" in result.fold_results[0].test_evaluations["test"].metrics

    def test_run_slide_level_features_can_omit_aggregator(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)

        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        torch.manual_seed(7)
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        output_root = tmp_path / "output_slide_omitted"
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(slide_feature_dir)).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert "auroc" in result.fold_results[0].test_evaluations["test"].metrics

    def test_run_hierarchical_features_with_hipt(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_hierarchical_data(tmp_path)
        output_root = tmp_path / "output_hier"

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            features=FeaturesConfig(encoder='uni2', extraction=extraction(0.5, 224, requested_region_size_px=448, region_tile_multiple=2)),
            aggregator={"name": 'hipt', "params": {'embed_dim_region': 12, 'embed_dim_slide': 12, 'num_heads': 2, 'dropout': 0.0}},
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert (_expected_run_dir(config) / "bundle" / "weights.safetensors").is_file()
        assert not (_expected_run_dir(config) / "best_model.pt").exists()
        assert "auroc" in result.fold_results[0].test_evaluations["test"].metrics

    def test_resolve_preprocessing_populates_hipt_geometry(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output_resolve',
            features=FeaturesConfig(encoder='uni2', extraction=slide2vec.PreprocessingConfig()),
            aggregator={"name": 'hipt', "params": {'tile_multiple': 6}},
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        from soma.preprocessing.resolution import resolve_pipeline_preprocessing

        resolved = resolve_pipeline_preprocessing(config)

        assert resolved.tiling.requested_tile_size_px == 224
        assert resolved.region_tile_multiple == 6
        assert resolved.requested_region_size_px == 1344

    def test_run_auto_extracts_slide_features_without_feature_dir(self, tmp_path: Path):
        pytest.importorskip("soma.extraction")
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        output_root = tmp_path / "output_prism"

        def _fake_extract(self):
            out = self._output_root / "features"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.randn(D), out / f"s{i}.pt")
            return SimpleNamespace(
                source=CachedSetSource(out),
                dataset=self._dataset,
            )

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=output_root,
            features=FeaturesConfig(encoder='prism'),
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )

        with patch("soma.extraction.FeatureExtractor.extract", new=_fake_extract):
            with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
                result = Pipeline(config).run()

        assert isinstance(result, PipelineResult)
        assert (_expected_run_dir(config) / "features" / "s0.pt").exists()

    def test_pipeline_rejects_aggregator_for_slide_features(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output_slide_error',
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        with pytest.raises(ValueError, match="an aggregator pools a bag"):
            with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(config, source=CachedSetSource(slide_feature_dir)).run()

        run_dir = _expected_run_dir(config)
        latest = run_dir.parents[1] / "latest"
        run_yaml = run_dir / "run.yaml"
        payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))

        assert payload["status"] == "failed"
        assert latest.is_symlink()
        assert latest.resolve() == run_dir.resolve()

    def test_pipeline_reuses_shared_cache_for_sibling_runs(self, tmp_path: Path):
        pytest.importorskip("slide2vec")
        from types import SimpleNamespace

        from hs2p import SlideSpec
        from slide2vec.encoders.base import SlideEncoder, TileEncoder
        from slide2vec.encoders.registry import encoder_registry
        from soma.slide2vec_adapter import LoadedTiling
        from soma.cache import record_feature_dim, record_sample_identity_signatures
        from soma.extraction.extractor import _PooledFeatureExtractor
        from tests.test_extraction import _save_committable

        test_tile = "_test_pipeline_cache_tile"
        test_slide = "_test_pipeline_cache_slide"

        class _PipelineTileEncoder(TileEncoder):
            def __init__(self, **kwargs):
                self._device = torch.device("cpu")
                self._output_variant = kwargs.get("output_variant") or "default"

            def get_transform(self):
                return lambda x: x

            def get_normalization_transform(self):
                return lambda x: x

            def encode_tiles(self, batch):
                return torch.ones(batch.shape[0], D)

            @property
            def encode_dim(self):
                return D

            @property
            def device(self):
                return self._device

            def to(self, device):
                self._device = torch.device(device)
                return self

        class _PipelineSlideEncoder(SlideEncoder):
            def __init__(self, **kwargs):
                self._device = torch.device("cpu")
                self._output_variant = kwargs.get("output_variant") or "default"

            def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0: int | None = None):
                return tile_features.mean(dim=0)

            @property
            def encode_dim(self):
                return D

            @property
            def device(self):
                return self._device

            def to(self, device):
                self._device = torch.device(device)
                return self

        if test_tile not in encoder_registry:
            encoder_registry.register(
                test_tile,
                _PipelineTileEncoder,
                metadata={
                    "level": "tile",
                    "input_size": 256,
                    "output_variants": {"default": {"encode_dim": D}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                },
            )
        if test_slide not in encoder_registry:
            encoder_registry.register(
                test_slide,
                _PipelineSlideEncoder,
                metadata={
                    "level": "slide",
                    "tile_encoder": test_tile,
                    "tile_encoder_output_variant": "default",
                    "output_variants": {"default": {"encode_dim": D}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                },
            )

        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        shared_cache = tmp_path / "shared-cache"

        loaded_tilings = [
            LoadedTiling(
                slide=SlideSpec(
                    sample_id=f"s{i}",
                    image_path=Path(f"/slides/s{i}.svs"),
                    mask_path=None,
                    spacing_at_level_0=None,
                ),
                tiling_result=SimpleNamespace(
                    x=np.array([0, 256], dtype=np.int64),
                    y=np.array([0, 256], dtype=np.int64),
                    tissue_fractions=np.array([1.0, 1.0], dtype=np.float32),
                    requested_tile_size_px=256,
                    requested_spacing_um=0.5,
                    read_tile_size_px=256,
                    read_spacing_um=0.5,
                    tile_size_lv0=256,
                    read_level=0,
                    use_padding=True,
                    is_within_tolerance=True,
                    sample_id=f"s{i}",
                    coordinates_npz_path=Path(f"/tmp/s{i}.npz"),
                    coordinates_meta_path=Path(f"/tmp/s{i}.meta.json"),
                ),
            )
            for i in range(NUM_SAMPLES)
        ]

        slide_config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'slide_run',
            features=FeaturesConfig(encoder=test_slide, extraction=extraction(None, None), cache=CacheOptions(root_dir=shared_cache)),
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )
        tile_config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'tile_run',
            features=FeaturesConfig(encoder=test_tile, extraction=extraction(None, None), cache=CacheOptions(root_dir=shared_cache)),
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )

        def _fake_preprocess(self_, tiling_dir, *, skip_existing=True, backend="auto"):
            tiling_dir = Path(tiling_dir)
            tiling_dir.mkdir(parents=True, exist_ok=True)
            (tiling_dir / "process_list.csv").write_text("sample_id,tiling_status\n", encoding="utf-8")

        def _fake_populate_tile_cache(
            self_,
            *,
            cache_resolution,
            loaded_tilings,
            prepared_tilings,
            tiling_dir,
            preprocessing,
            encoder_name,
            output_variant,
            num_gpus,
        ):
            cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                _save_committable(torch.ones(2, D), cache_resolution.feature_path_for_id(f"s{i}"))
            record_feature_dim(cache_resolution, D)
            record_sample_identity_signatures(cache_resolution, [f"s{i}" for i in range(NUM_SAMPLES)])

        def _fake_populate_slide_cache(
            self_,
            *,
            tile_cache,
            slide_cache,
            loaded_tilings,
            preprocessing,
            model_name,
            output_variant,
            num_gpus,
        ):
            tile_cache.features_dir.mkdir(parents=True, exist_ok=True)
            slide_cache.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                _save_committable(torch.ones(2, D), tile_cache.feature_path_for_id(f"s{i}"))
                _save_committable(torch.ones(D), slide_cache.feature_path_for_id(f"s{i}"))
            record_feature_dim(tile_cache, D)
            record_feature_dim(slide_cache, D)
            record_sample_identity_signatures(tile_cache, [f"s{i}" for i in range(NUM_SAMPLES)])
            record_sample_identity_signatures(slide_cache, [f"s{i}" for i in range(NUM_SAMPLES)])

        with patch("soma.extraction.extractor.torch.cuda.is_available", return_value=False), patch(
            "soma.extraction.extractor.torch.cuda.device_count", return_value=1
        ), patch.object(
            _PooledFeatureExtractor,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.extractor.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction.extractor._validate_runtime"), patch.object(
            _PooledFeatureExtractor,
            "_populate_tile_cache",
            autospec=True,
            side_effect=_fake_populate_tile_cache,
        ) as populate_tile_cache, patch.object(
            _PooledFeatureExtractor,
            "_populate_slide_cache",
            autospec=True,
            side_effect=_fake_populate_slide_cache,
        ) as populate_slide_cache:
            with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(slide_config).run()
            assert populate_slide_cache.called

        with patch("soma.extraction.extractor.torch.cuda.is_available", return_value=False), patch(
            "soma.extraction.extractor.torch.cuda.device_count", return_value=1
        ), patch.object(
            _PooledFeatureExtractor,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.extractor.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction.extractor._validate_runtime"), patch.object(
            _PooledFeatureExtractor,
            "_populate_tile_cache",
            side_effect=AssertionError("tile extraction should be reused"),
        ):
            with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
                Pipeline(tile_config).run()

    def test_fold_bundle_records_the_feature_source_identity_not_its_path(self, tmp_path: Path):
        import soma

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        (feature_dir / "cache_metadata.json").write_text(
            json.dumps({"cache_key": "abc123", "artifact_kind": "tile", "encoder_name": "uni2"})
        )
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            aggregator='mean_pool',
            loaders={"batch_size": 2, "pin_memory": False},
            trainer={"max_epochs": 1},
            scheduler='cosine',
        )
        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        bundle = soma.load(result.run_dir / "bundle")
        source = bundle.provenance["feature_source"]
        assert source["cache_key"] == "abc123" and source["encoder_name"] == "uni2"
        assert str(tmp_path) not in json.dumps(bundle.provenance)
        assert (bundle.provenance["fold"], bundle.provenance["run_id"]) == (0, FIXED_RUN_ID)
        assert bundle.preprocessing["classes"] == ["normal", "tumor"]

    def test_dataset_and_splits_properties(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            scheduler='cosine',
        )
        pipeline = Pipeline(config, source=CachedSetSource(feature_dir))

        from soma.tasks.classification import categorical_label_map

        assert len(categorical_label_map(pipeline.dataset.records)) == 2
        assert pipeline.splits.num_folds == 1

    def test_pipeline_writes_experiment_metadata_and_indexes(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = experiment(
            dataset_csv,
            splits_csv,
            output_dir=tmp_path / 'output',
            aggregator='mean_pool',
            loaders={"batch_size": 2},
            trainer={"max_epochs": 2},
            scheduler='cosine',
        )

        with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
            result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

        experiment_dir = result.run_dir.parents[1]
        experiment_yaml = experiment_dir / "experiment.yaml"
        run_yaml = result.run_dir / "run.yaml"
        latest = experiment_dir / "latest"
        runs_index = Path(config.run.output_dir) / "indexes" / "runs.csv"

        assert experiment_yaml.exists()
        assert run_yaml.exists()
        assert latest.is_symlink()
        assert latest.resolve() == result.run_dir.resolve()

        with runs_index.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        # Append-only index: one line per status change (running -> completed) …
        assert [row["status"] for row in raw_rows] == ["running", "completed"]
        # … and one row per run_id for readers.
        from soma.pipeline.layout import read_run_index

        rows = read_run_index(runs_index)
        assert rows == [
            {
                "run_id": FIXED_RUN_ID,
                "experiment_id": rows[0]["experiment_id"],
                "status": "completed",
                "started_at": rows[0]["started_at"],
                "finished_at": rows[0]["finished_at"],
                "seed": "0",
                "wandb_id": "",
                "git_sha": rows[0]["git_sha"],
                "run_dir": str(result.run_dir.resolve()),
                "error": "",
            }
        ]


# ---------------------------------------------------------------------------
# Patient pipeline
# ---------------------------------------------------------------------------

D_PATIENT = 768
NUM_PATIENTS = 4  # 2 slides each → 8 slides total


def _setup_patient_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create dataset.csv (with patient_id), splits.csv, and patient feature .pt files.

    4 patients × 2 slides each = 8 slides.
    Splits are patient-clean: all slides for a patient go to the same split.
    """
    sample_ids = [f"s{i}" for i in range(8)]
    patient_ids = [f"p{i // 2}" for i in range(8)]  # p0 p0 p1 p1 p2 p2 p3 p3
    labels = ["tumor" if (i // 2) % 2 == 0 else "normal" for i in range(8)]

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": [f"/slides/{s}.svs" for s in sample_ids],
            "label": labels,
            "patient_id": patient_ids,
        }
    ).to_csv(dataset_csv, index=False)

    # Splits: p0+p1 in train, p2 in tune, p3 in test
    splits_rows = []
    for sid, pid in zip(sample_ids, patient_ids):
        if pid in ("p0", "p1"):
            split = "train"
        elif pid == "p2":
            split = "tune"
        else:
            split = "test"
        splits_rows.append({"fold": 0, "sample_id": sid, "split": split})
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(splits_rows).to_csv(splits_csv, index=False)

    # One patient feature per patient_id (1-D tensors)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.manual_seed(0)
    for pid in [f"p{i}" for i in range(NUM_PATIENTS)]:
        torch.save(torch.randn(D_PATIENT), feature_dir / f"{pid}.pt")

    return dataset_csv, splits_csv, feature_dir


def _write_patient_feature_manifest(feature_dir: Path, statuses: dict[str, str]) -> None:
    rows = []
    for patient_id, status in statuses.items():
        rows.append(
            {
                "sample_id": patient_id,
                "feature_status": status,
                "feature_path": str((feature_dir / f"{patient_id}.pt").resolve())
                if status == "success"
                else "",
            }
        )
    pd.DataFrame(rows).to_csv(feature_dir / "process_list.csv", index=False)


def test_patient_run_scores_patients_and_fills_a_tileless_tune_patient(tmp_path: Path):
    """``dataset_type: patient`` through ``Pipeline``: predictions are keyed by patient id,
    and a tune patient the feature manifest records as empty is scored by the baseline."""
    dataset_csv, splits_csv, feature_dir = _setup_patient_data(tmp_path)
    (feature_dir / "p2.pt").unlink()
    _write_patient_feature_manifest(feature_dir, {"p0": "success", "p1": "success", "p2": "empty", "p3": "success"})
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / 'output',
        cohort={"collapse": "patient_id"},
        loaders={"batch_size": 2, "pin_memory": False},
        trainer={"max_epochs": 2},
        scheduler='cosine',
    )
    with patch("soma.pipeline.layout.make_run_id", return_value=FIXED_RUN_ID):
        result = Pipeline(config, source=CachedSetSource(feature_dir)).run()

    run_dir = _expected_run_dir(config)
    assert Predictions.from_parquet(run_dir / "predictions_test.parquet").sample_ids == ["p3"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["tune"]["coverage"] == pytest.approx(0.0)
    assert metrics["tune"]["num_placeholder_samples"] == 1
    assert (run_dir / "bundle" / "weights.safetensors").is_file()
    assert result.summary["test/num_samples"] == 1


def test_patient_leakage_is_refused_when_the_folds_are_built(tmp_path: Path):
    dataset_csv, _, feature_dir = _setup_patient_data(tmp_path)
    dataset = legacy_samples_from_csv(dataset_csv)

    # s0 belongs to p0 but is put in test; the other p0 slide (s1) stays in train
    bad_splits = []
    for sid in dataset.sample_ids:
        pid = dataset.samples[sid].patient_id
        if sid == "s0":
            split = "test"
        elif pid in ("p0", "p1"):
            split = "train"
        elif pid == "p2":
            split = "tune"
        else:
            split = "test"
        bad_splits.append({"fold": 0, "sample_id": sid, "split": split})

    splits_csv = tmp_path / "bad_splits.csv"
    pd.DataFrame(bad_splits).to_csv(splits_csv, index=False)

    # Leakage is a data-contract violation: the folds cannot even be built.
    with pytest.raises(ValueError, match="p0"):
        legacy_folds_from_csv(splits_csv, dataset, unit="patient_id")
