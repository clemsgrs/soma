"""Tests for soma.pipeline — train_one_fold, train, and Pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import pytest

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, FoldSplit, Splits
from soma.features import FeatureStore
from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_SAMPLES = 8


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


# ---------------------------------------------------------------------------
# train_one_fold (Layer 1 — standalone)
# ---------------------------------------------------------------------------


class TestTrainOneFold:
    def test_returns_fold_result(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, learning_rate=1e-3, patience=10, batch_size=2),
            output_dir=tmp_path / "fold_0",
        )

        assert isinstance(result, FoldResult)
        assert result.fold == 0
        assert result.train_result is not None
        assert result.tune_report.split == "tune"
        assert result.test_report.split == "test"

    def test_saves_checkpoint(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        assert (fold_dir / "best_model.pt").exists()

    def test_saves_metrics_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        metrics_path = fold_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "test" in metrics
        assert "accuracy" in metrics["test"]

    def test_saves_predictions_csv(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        fold_dir = tmp_path / "fold_0"

        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=fold_dir,
        )

        preds_path = fold_dir / "predictions.csv"
        assert preds_path.exists()
        preds_df = pd.read_csv(preds_path)
        assert "sample_id" in preds_df.columns
        assert "true_label" in preds_df.columns
        assert "predicted_label" in preds_df.columns

    def test_num_classes_auto_inferred(self, tmp_path: Path):
        """num_classes should be auto-inferred from dataset labels."""
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),  # no num_classes specified
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=tmp_path / "fold_0",
        )

        # Should have 2-class probabilities (tumor/normal)
        assert len(result.test_report.predictions[0].probabilities) == 2


    def test_slide_level_features_no_aggregator(self, tmp_path: Path):
        """train_one_fold with 1-D features and aggregator=None uses SlideModel."""
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)

        # Slide-level: one (D,) tensor per sample
        slide_dir = tmp_path / "slide_feats"
        slide_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_dir / f"s{i}.pt")
        store = FeatureStore(slide_dir)

        result = train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=None,
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=tmp_path / "fold_slide",
        )

        assert isinstance(result, FoldResult)
        assert result.test_report is not None
        assert "accuracy" in result.test_report.metrics

    def test_slide_level_features_with_aggregator_raises(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        slide_dir = tmp_path / "slide_feats"
        slide_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_dir / f"s{i}.pt")
        store = FeatureStore(slide_dir)

        with pytest.raises(ValueError, match="aggregator must be None"):
            train_one_fold(
                feature_store=store,
                dataset=dataset,
                fold_split=splits.folds[0],
                aggregator=AggregatorConfig(name="mean_pool"),
                task=TaskConfig(name="classification"),
                training=TrainingConfig(epochs=2, patience=10, batch_size=2),
                output_dir=tmp_path / "fold_slide",
            )


# ---------------------------------------------------------------------------
# train (Layer 1 — all folds)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_single_fold_returns_pipeline_result(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)

        result = train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].fold == 0

    def test_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        result = train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        assert len(result.fold_results) == 2
        assert result.fold_results[0].fold == 0
        assert result.fold_results[1].fold == 1
        assert "accuracy_mean" in result.summary
        assert "accuracy_std" in result.summary

    def test_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        summary = json.loads((output_dir / "summary.json").read_text())
        assert "accuracy_mean" in summary

    def test_fold_subdirectories(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        store = FeatureStore(feature_dir)
        output_dir = tmp_path / "output"

        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            aggregator=AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="classification"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            output_dir=output_dir,
        )

        assert (output_dir / "fold_0" / "best_model.pt").exists()
        assert (output_dir / "fold_1" / "best_model.pt").exists()


# ---------------------------------------------------------------------------
# Pipeline (Layer 2 — orchestrator)
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_run_single_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)
        result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.output_dir == output_dir

    def test_run_saves_config_yaml(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        Pipeline(config, feature_dir=feature_dir).run()

        assert (output_dir / "config.yaml").exists()

    def test_run_saves_summary_json(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        Pipeline(config, feature_dir=feature_dir).run()

        summary_path = output_dir / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "accuracy_mean" in summary

    def test_run_multi_fold(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        result = Pipeline(config, feature_dir=feature_dir).run()

        assert len(result.fold_results) == 2
        assert (output_dir / "fold_0" / "best_model.pt").exists()
        assert (output_dir / "fold_1" / "best_model.pt").exists()

        summary = json.loads((output_dir / "summary.json").read_text())
        assert "accuracy_mean" in summary
        assert "accuracy_std" in summary

    def test_run_slide_level_features(self, tmp_path: Path):
        """Pipeline with aggregator=None should work with slide-level (1-D) features."""
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)

        # Create slide-level features (1-D per sample)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        torch.manual_seed(7)
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        output_dir = tmp_path / "output_slide"
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            aggregator=None,
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        result = Pipeline(config, feature_dir=slide_feature_dir).run()

        assert isinstance(result, PipelineResult)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].test_report is not None
        assert "accuracy" in result.fold_results[0].test_report.metrics

    def test_run_auto_extracts_slide_features_without_feature_dir(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        output_dir = tmp_path / "output_prism"

        def _fake_run(self, output_dir_arg, **kwargs):
            out = Path(output_dir_arg)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.randn(D), out / f"s{i}.pt")
            return FeatureStore(out)

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=output_dir,
            encoder=EncoderConfig(name="prism"),
            aggregator=None,
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )

        with patch("soma.extraction.FeatureExtractor.run", new=_fake_run):
            result = Pipeline(config).run()

        assert isinstance(result, PipelineResult)
        assert (output_dir / "features" / "s0.pt").exists()

    def test_pipeline_rejects_aggregator_for_slide_features(self, tmp_path: Path):
        dataset_csv, splits_csv, _ = _setup_synthetic_data(tmp_path)
        slide_feature_dir = tmp_path / "slide_features"
        slide_feature_dir.mkdir()
        for i in range(NUM_SAMPLES):
            torch.save(torch.randn(D), slide_feature_dir / f"s{i}.pt")

        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "output_slide_error",
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        with pytest.raises(ValueError, match="aggregator must be None"):
            Pipeline(config, feature_dir=slide_feature_dir).run()

    def test_pipeline_reuses_shared_cache_for_sibling_runs(self, tmp_path: Path):
        from types import SimpleNamespace

        from hs2p import SlideSpec
        from soma.encoders.base import SlideEncoder, TileEncoder
        from soma.encoders.registry import encoder_registry
        from soma.slide2vec_adapter import LoadedTiling, Slide2VecRuntime
        from soma.cache import record_feature_dim

        test_tile = "_test_pipeline_cache_tile"
        test_slide = "_test_pipeline_cache_slide"

        class _PipelineTileEncoder(TileEncoder):
            def __init__(self, **kwargs):
                self._device = torch.device("cpu")
                self._output_variant = kwargs.get("output_variant") or "default"

            def get_transform(self):
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
                    effective_tile_size_px=256,
                    effective_spacing_um=0.5,
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

        class _FakeModel:
            def __init__(self, *, output_variant: str | None = None):
                self.output_variant = output_variant

            def embed_tiles(self, slides, tiling_results, *, preprocessing, execution):
                output_dir = Path(execution.output_dir) / "tile_embeddings"
                output_dir.mkdir(parents=True, exist_ok=True)
                artifacts = []
                for slide, tiling in zip(slides, tiling_results):
                    tensor = torch.ones(len(tiling.x), D)
                    path = output_dir / f"{slide.sample_id}.pt"
                    meta_path = output_dir / f"{slide.sample_id}.meta.json"
                    torch.save(tensor, path)
                    meta_path.write_text("{}", encoding="utf-8")
                    artifacts.append(
                        SimpleNamespace(
                            sample_id=slide.sample_id,
                            path=path,
                            metadata_path=meta_path,
                            format="pt",
                            feature_dim=D,
                            num_tiles=len(tiling.x),
                        )
                    )
                return artifacts

            def aggregate_tiles(self, tile_artifacts, *, preprocessing=None, execution):
                output_dir = Path(execution.output_dir) / "slide_embeddings"
                output_dir.mkdir(parents=True, exist_ok=True)
                artifacts = []
                for artifact in tile_artifacts:
                    path = output_dir / f"{artifact.sample_id}.pt"
                    meta_path = output_dir / f"{artifact.sample_id}.meta.json"
                    torch.save(torch.ones(D), path)
                    meta_path.write_text("{}", encoding="utf-8")
                    artifacts.append(
                        SimpleNamespace(
                            sample_id=artifact.sample_id,
                            path=path,
                            metadata_path=meta_path,
                            format="pt",
                            feature_dim=D,
                        )
                    )
                return artifacts

        slide_config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "slide_run",
            cache=CacheConfig(root_dir=shared_cache),
            encoder=EncoderConfig(name=test_slide),
            aggregator=None,
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )
        tile_config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "tile_run",
            cache=CacheConfig(root_dir=shared_cache),
            encoder=EncoderConfig(name=test_tile),
            aggregator=AggregatorConfig(name="mean_pool"),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        )

        def _fake_preprocess(_self, *, model_name, slides, preprocessing, output_dir):
            del _self, model_name, slides, preprocessing
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "process_list.csv").write_text("sample_id,tiling_status\n", encoding="utf-8")

        def _fake_populate_tile_cache(
            _self,
            *,
            cache_resolution,
            loaded_tilings,
            prepared_tilings,
            tiling_dir,
            encoder,
            preprocessing,
            encoder_name,
            output_variant,
            num_gpus,
        ):
            del _self, loaded_tilings, prepared_tilings, tiling_dir, encoder, preprocessing, encoder_name, output_variant, num_gpus
            cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.ones(2, D), cache_resolution.features_dir / f"s{i}.pt")
            record_feature_dim(cache_resolution, D)

        def _fake_populate_slide_cache(
            _self,
            *,
            slide_cache,
            tile_cache,
            loaded_tilings,
            encoder,
            model_name,
            output_variant,
            num_gpus,
        ):
            del _self, tile_cache, loaded_tilings, encoder, model_name, output_variant, num_gpus
            slide_cache.features_dir.mkdir(parents=True, exist_ok=True)
            for i in range(NUM_SAMPLES):
                torch.save(torch.ones(D), slide_cache.features_dir / f"s{i}.pt")
            record_feature_dim(slide_cache, D)

        with patch.object(
            Slide2VecRuntime,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction.validate_runtime"), patch.object(
            Slide2VecRuntime,
            "populate_tile_cache",
            autospec=True,
            side_effect=_fake_populate_tile_cache,
        ) as populate_tile_cache, patch.object(
            Slide2VecRuntime,
            "populate_slide_cache",
            autospec=True,
            side_effect=_fake_populate_slide_cache,
        ) as populate_slide_cache:
            Pipeline(slide_config).run()
            assert populate_tile_cache.called
            assert populate_slide_cache.called

        with patch.object(
            Slide2VecRuntime,
            "preprocess",
            autospec=True,
            side_effect=_fake_preprocess,
        ), patch(
            "soma.extraction.load_tilings", return_value=loaded_tilings
        ), patch("soma.extraction.validate_runtime"), patch.object(
            Slide2VecRuntime,
            "populate_tile_cache",
            side_effect=AssertionError("tile extraction should be reused"),
        ):
            Pipeline(tile_config).run()

    def test_dataset_and_splits_properties(self, tmp_path: Path):
        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        config = PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "output",
        )
        pipeline = Pipeline(config, feature_dir=feature_dir)

        assert pipeline.dataset.num_classes == 2
        assert pipeline.splits.num_folds == 1
