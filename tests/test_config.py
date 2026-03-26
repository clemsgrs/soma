"""Tests for soma.config — frozen dataclass configurations."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
    save_config,
)


# --- Frozen immutability ---


def test_preprocessing_config_is_frozen():
    cfg = PreprocessingConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.requested_tile_size_px = 512


def test_pipeline_config_is_frozen():
    cfg = _make_pipeline_config()
    with pytest.raises(FrozenInstanceError):
        cfg.output_dir = "other"


# --- Default values ---


def test_preprocessing_config_defaults():
    cfg = PreprocessingConfig()
    assert cfg.requested_tile_size_px is None
    assert cfg.requested_spacing_um is None
    assert cfg.tissue_method == "hsv"
    assert cfg.min_tissue_fraction == 0.1
    assert cfg.overlap == 0.0
    assert cfg.seg_downsample == 64
    assert cfg.tolerance == 0.05
    assert cfg.ref_tile_size_px is None
    assert cfg.a_t == 4


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seed == 0
    assert cfg.epochs == 50
    assert cfg.learning_rate == 1e-4
    assert cfg.weight_decay == 1e-5
    assert cfg.optimizer == "adam"
    assert cfg.scheduler == "cosine"
    assert cfg.patience == 10
    assert cfg.batch_size == 1


def test_aggregator_config_defaults():
    cfg = AggregatorConfig()
    assert cfg.name == "abmil"
    assert cfg.params == {}


def test_task_config_defaults():
    cfg = TaskConfig()
    assert cfg.name == "classification"
    assert cfg.params == {}


def test_encoder_config_defaults():
    cfg = EncoderConfig()
    assert cfg.name == "uni2"
    assert cfg.precision == "fp16"
    assert cfg.batch_size == 32
    assert cfg.num_workers == 4
    assert cfg.output_variant is None
    assert cfg.save_tile_features is False


def test_encoder_config_roundtrip_with_output_variant(tmp_path: Path):
    cfg = _make_pipeline_config(encoder=EncoderConfig(name="h0-mini", output_variant="cls"))
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.encoder.output_variant == "cls"


def test_cache_config_defaults():
    cfg = CacheConfig()
    assert cfg.enabled is True
    assert cfg.root_dir is None
    assert cfg.reuse_policy == "strict"
    assert cfg.save_tile_features_for_slide is True


def test_aggregator_config_with_params():
    cfg = AggregatorConfig(name="abmil", params={"hidden_dim": 256, "dropout": 0.25})
    assert cfg.params["hidden_dim"] == 256
    assert cfg.params["dropout"] == 0.25


def test_task_config_with_params():
    cfg = TaskConfig(name="classification", params={"num_classes": 5})
    assert cfg.params["num_classes"] == 5


# --- YAML roundtrip ---


def test_save_and_load_config_roundtrip(tmp_path: Path):
    original = _make_pipeline_config()
    yaml_path = tmp_path / "config.yaml"

    save_config(original, yaml_path)
    assert yaml_path.exists()

    loaded = load_config(yaml_path)

    assert loaded.dataset_csv == original.dataset_csv
    assert loaded.splits_csv == original.splits_csv
    assert loaded.output_dir == original.output_dir
    assert loaded.cache.enabled == original.cache.enabled
    assert loaded.encoder.name == original.encoder.name
    assert loaded.aggregator.name == original.aggregator.name
    assert loaded.aggregator.params == original.aggregator.params
    assert loaded.task.name == original.task.name
    assert loaded.task.params == original.task.params
    assert loaded.training.epochs == original.training.epochs
    assert loaded.training.learning_rate == original.training.learning_rate
    assert loaded.tags == original.tags


def test_save_config_produces_valid_yaml(tmp_path: Path):
    cfg = _make_pipeline_config()
    yaml_path = tmp_path / "test.yaml"
    save_config(cfg, yaml_path)

    raw = yaml.safe_load(yaml_path.read_text())
    assert raw["encoder"]["name"] == "uni2"
    assert raw["cache"]["enabled"] is True
    assert raw["training"]["learning_rate"] == 2e-4
    assert raw["aggregator"]["params"]["hidden_dim"] == 128


def test_load_config_with_tags(tmp_path: Path):
    cfg = _make_pipeline_config(tags=["baseline", "uni2"])
    yaml_path = tmp_path / "test.yaml"
    save_config(cfg, yaml_path)

    loaded = load_config(yaml_path)
    assert loaded.tags == ["baseline", "uni2"]


def test_aggregator_none_roundtrip(tmp_path: Path):
    """PipelineConfig with aggregator=None should serialize and deserialize correctly."""
    cfg = _make_pipeline_config(aggregator=None)
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.aggregator is None


def test_aggregator_none_yaml_output(tmp_path: Path):
    cfg = _make_pipeline_config(aggregator=None)
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)

    raw = yaml.safe_load(yaml_path.read_text())
    assert raw["aggregator"] is None


# --- Helpers ---


def _make_pipeline_config(**overrides) -> PipelineConfig:
    defaults = dict(
        dataset_csv="data/dataset.csv",
        splits_csv="data/splits.csv",
        output_dir="runs/exp1",
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="classification", params={"num_classes": 3}),
        training=TrainingConfig(epochs=100, learning_rate=2e-4),
        tags=["test"],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)
