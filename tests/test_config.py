"""Tests for soma.config — frozen dataclass configurations."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import yaml

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    DecoderConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    PreprocessingConfig,
    SubgroupConfig,
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
        cfg.output_root = "other"


# --- Default values ---


def test_preprocessing_config_defaults():
    cfg = PreprocessingConfig()
    assert cfg.backend == "auto"
    assert cfg.requested_backend == "auto"
    assert cfg.requested_tile_size_px is None
    assert cfg.requested_spacing_um is None
    assert cfg.requested_region_size_px is None
    assert cfg.region_tile_multiple is None
    assert cfg.read_tile_size_px is None
    assert cfg.read_region_size_px is None
    assert cfg.has_hierarchical_geometry is False
    field_names = {field.name for field in fields(PreprocessingConfig)}
    assert "hierarchical" not in field_names
    assert "npatch" not in field_names
    assert "hierarchical_patch_size_px" not in field_names
    assert cfg.tissue_method is None
    assert cfg.tissue_threshold == 0.1
    assert cfg.overlap == 0.0
    assert cfg.seg_downsample == 64
    assert cfg.sam2_device == "cpu"
    assert cfg.sam2_num_workers is None
    assert cfg.tolerance == 0.05
    assert cfg.ref_tile_size_px is None
    assert cfg.a_t == 4
    assert cfg.tissue_mask_tissue_value == 1
    assert cfg.preview.save_mask_preview is True
    assert cfg.preview.save_tiling_preview is True
    assert cfg.preview.downsample == 32
    assert cfg.preview.tissue_contour_color == (37, 94, 59)
    assert cfg.preview.mask_overlay_alpha == pytest.approx(0.5)


def test_preprocessing_dense_window_defaults_to_whole():
    cfg = PreprocessingConfig()
    assert cfg.dense_window_size is None
    assert cfg.dense_window_overlap == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dense_window_size": 0}, "dense_window_size"),
        ({"dense_window_size": -64}, "dense_window_size"),
        ({"dense_window_overlap": 1.0}, "dense_window_overlap"),
        ({"dense_window_overlap": -0.1}, "dense_window_overlap"),
        ({"dense_window_overlap": 0.5}, "dense_window_overlap requires dense_window_size"),
    ],
)
def test_preprocessing_dense_window_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PreprocessingConfig(**kwargs)


def test_preprocessing_dense_window_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        preprocessing=PreprocessingConfig(dense_window_size=512, dense_window_overlap=0.5)
    )
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.preprocessing.dense_window_size == 512
    assert loaded.preprocessing.dense_window_overlap == 0.5


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seed == 0
    assert cfg.epochs == 50
    assert cfg.learning_rate == 1e-4
    assert cfg.weight_decay == 1e-5
    assert cfg.optimizer == "adam"
    assert cfg.scheduler == "cosine"
    assert cfg.patience == 10
    assert cfg.monitor == "tune_loss"
    assert cfg.monitor_mode == "min"
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation == 1
    assert cfg.tune_is_test is False
    # DataLoader knobs (Phase 2.2). num_workers defaults to 0 so the suite stays
    # fast on test fixtures; users should raise it for real WSI runs.
    assert cfg.num_workers == 0
    assert cfg.pin_memory is True
    assert cfg.persistent_workers is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"gradient_accumulation": 0}, "gradient_accumulation"),
        ({"patience": 0}, "patience"),
        ({"monitor": ""}, "monitor"),
        ({"monitor_mode": "largest"}, "monitor_mode"),
        ({"num_workers": -1}, "num_workers"),
    ],
)
def test_training_config_rejects_non_positive_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_aggregator_config_explicit_name():
    cfg = AggregatorConfig(name="abmil")
    assert cfg.name == "abmil"
    assert cfg.params == {}


def test_aggregator_config_requires_name():
    with pytest.raises(TypeError):
        AggregatorConfig()


def test_task_config_requires_name():
    with pytest.raises(TypeError):
        TaskConfig()  # name is required


def test_task_config_params_default_empty():
    cfg = TaskConfig(name="binary_classification")
    assert cfg.params == {}


def test_evaluation_config_defaults():
    cfg = EvalConfig()
    assert cfg.metrics == []
    assert cfg.subgroups.columns == []
    assert cfg.save_probabilities is False


def test_pipeline_config_rejects_unknown_encoder():
    """Phase 3.2: unknown encoder name must fail at config construction.

    Catching this at __post_init__ saves hours of preprocessing before the
    pipeline would otherwise crash at encoder build time.
    """
    with pytest.raises(ValueError, match="encoder"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            encoder=EncoderConfig(name="not_a_real_encoder"),
            task=TaskConfig(name="binary_classification"),
        )


def test_pipeline_config_rejects_unknown_aggregator():
    """Phase 3.2: unknown aggregator name must fail at config construction."""
    with pytest.raises(ValueError, match="aggregator"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="not_a_real_aggregator"),
            task=TaskConfig(name="binary_classification"),
        )


def test_pipeline_config_defaults_to_no_aggregator():
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="slide",
        task=TaskConfig(name="binary_classification"),
    )
    assert cfg.aggregator is None


def test_evaluation_config_metrics_explicit():
    cfg = EvalConfig(metrics=["auroc", "f1"])
    assert cfg.metrics == ["auroc", "f1"]


# --- Segmentation decoder config plumbing ---


def _seg_config(**overrides):
    kwargs = dict(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation"),
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def test_segmentation_config_valid():
    cfg = _seg_config()
    assert cfg.dataset_type == "segmentation"
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.aggregator is None


def test_segmentation_requires_decoder():
    with pytest.raises(ValueError, match="decoder is required"):
        _seg_config(decoder=None)


def test_segmentation_rejects_aggregator():
    with pytest.raises(ValueError, match="aggregator must be None"):
        _seg_config(aggregator=AggregatorConfig(name="mean_pool"))


def test_segmentation_requires_segmentation_task():
    with pytest.raises(ValueError, match="task.name='segmentation'"):
        _seg_config(task=TaskConfig(name="binary_classification"))


def test_decoder_rejected_for_non_segmentation():
    with pytest.raises(ValueError, match="decoder must be None"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="tile",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="binary_classification"),
        )


def test_segmentation_rejects_unknown_decoder():
    with pytest.raises(ValueError, match="Unknown decoder name"):
        _seg_config(decoder=DecoderConfig(name="not_a_real_decoder"))


def test_decoder_config_round_trips_through_yaml(tmp_path: Path):
    cfg = _seg_config(decoder=DecoderConfig(name="lightweight_conv", params={"num_upsample_blocks": 2}))
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.dataset_type == "segmentation"
    assert loaded.decoder.name == "lightweight_conv"
    assert loaded.decoder.params == {"num_upsample_blocks": 2}


def test_subgroup_config_defaults():
    cfg = SubgroupConfig()
    assert cfg.columns == []


def test_subgroup_config_explicit():
    cfg = SubgroupConfig(columns=["sex", "grade"])
    assert cfg.columns == ["sex", "grade"]


def test_encoder_config_requires_name():
    with pytest.raises(TypeError):
        EncoderConfig()


def test_encoder_config_defaults():
    cfg = EncoderConfig(name="uni2")
    assert cfg.name == "uni2"
    assert cfg.precision is None
    assert cfg.batch_size == 32
    assert cfg.output_variant is None
    assert cfg.allow_non_recommended_settings is False
    assert cfg.save_tile_features is False


def test_encoder_config_public_fields_are_geometry_free():
    field_names = {field.name for field in fields(EncoderConfig)}
    assert "input_size" not in field_names
    assert "spacing_um" not in field_names


def test_execution_config_defaults():
    cfg = ExecutionConfig()
    field_names = {field.name for field in fields(ExecutionConfig)}
    assert cfg.num_gpus is None
    assert cfg.num_workers_per_gpu is None
    assert cfg.num_preprocessing_workers is None
    assert cfg.prefetch_factor is None
    assert cfg.precision is None
    assert "num_workers" not in field_names
    assert "persistent_workers" not in field_names


def test_encoder_config_roundtrip_with_output_variant(tmp_path: Path):
    cfg = _make_pipeline_config(encoder=EncoderConfig(name="h0-mini", output_variant="cls"))
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.encoder.output_variant == "cls"


def test_execution_config_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        execution=ExecutionConfig(
            num_gpus=2,
            num_workers_per_gpu=6,
            num_preprocessing_workers=0,
            prefetch_factor=8,
            precision="fp16",
        )
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.execution.num_gpus == 2
    assert loaded.execution.num_workers_per_gpu == 6
    assert loaded.execution.num_preprocessing_workers == 0
    assert loaded.execution.prefetch_factor == 8
    assert loaded.execution.precision == "fp16"


def test_encoder_config_roundtrip_with_allow_non_recommended_settings(tmp_path: Path):
    cfg = _make_pipeline_config(
        encoder=EncoderConfig(name="h0-mini", allow_non_recommended_settings=True)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.encoder.allow_non_recommended_settings is True


def test_cache_config_defaults():
    cfg = CacheConfig()
    assert cfg.enabled is True
    assert cfg.root_dir is None
    assert cfg.reuse_policy == "strict"
    assert cfg.fingerprint_files is False
    assert cfg.validate_payloads is False


def test_aggregator_config_with_params():
    cfg = AggregatorConfig(name="abmil", params={"hidden_dim": 256, "dropout": 0.25})
    assert cfg.params["hidden_dim"] == 256
    assert cfg.params["dropout"] == 0.25


def test_task_config_with_params():
    cfg = TaskConfig(name="multiclass_classification", params={"num_classes": 5})
    assert cfg.params["num_classes"] == 5


# --- YAML roundtrip ---


def test_save_and_load_config_roundtrip(tmp_path: Path):
    original = _make_pipeline_config(
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=7, backend="openslide")
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(original, yaml_path)
    assert yaml_path.exists()

    loaded = load_config(yaml_path)

    assert loaded.dataset_csv == original.dataset_csv
    assert loaded.splits_csv == original.splits_csv
    assert loaded.output_root == original.output_root
    assert loaded.preprocessing.backend == "openslide"
    assert loaded.preprocessing.tissue_mask_tissue_value == 7
    assert loaded.preprocessing.requested_tile_size_px == original.preprocessing.requested_tile_size_px

    assert loaded.cache.enabled == original.cache.enabled
    assert loaded.encoder.name == original.encoder.name
    assert loaded.execution == original.execution
    assert loaded.aggregator.name == original.aggregator.name
    assert loaded.aggregator.params == {"hidden_dim": 128, "dropout": 0.25}
    assert loaded.task.name == original.task.name
    assert loaded.task.params == original.task.params
    assert loaded.evaluation.metrics == original.evaluation.metrics
    assert loaded.training.epochs == original.training.epochs
    assert loaded.training.learning_rate == original.training.learning_rate
    assert loaded.tags == original.tags


def test_load_config_merges_bundled_defaults_for_new_layout(tmp_path: Path):
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "slide",
        }
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(raw, f)

    loaded = load_config(yaml_path)

    assert loaded.dataset_csv == "dataset.csv"
    assert loaded.splits_csv == "splits.csv"
    assert loaded.output_root == "runs"
    assert loaded.dataset_type == "slide"
    assert loaded.preprocessing.sam2_device == "cpu"
    assert loaded.encoder.name == "uni2"
    assert loaded.aggregator.name == "abmil"
    assert loaded.task.name == "binary_classification"
    assert loaded.evaluation.metrics == ["auroc", "balanced_accuracy"]
    assert loaded.training.epochs == 50


def test_load_config_with_target_fields(tmp_path: Path):
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "slide",
        },
        "preprocessing": {
            "backend": "cucim",
            "requested_tile_size_px": 256,
            "requested_spacing_um": 0.5,
        },
        "cache": {},
        "aggregation": None,
        "task": {"name": "binary_classification"},
        "training": {},
        "run": {
            "output_root": "out",
            "tags": [],
        },
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as handle:
        yaml.safe_dump(raw, handle)

    loaded = load_config(yaml_path)

    assert loaded.preprocessing.backend == "cucim"
    assert loaded.preprocessing.requested_tile_size_px == 256
    assert loaded.preprocessing.requested_spacing_um == 0.5
    assert loaded.output_root == "out"


def test_save_config_produces_valid_yaml(tmp_path: Path):
    cfg = _make_pipeline_config()
    yaml_path = tmp_path / "test.yaml"
    save_config(cfg, yaml_path)

    raw = yaml.safe_load(yaml_path.read_text())
    assert raw["data"]["dataset_csv"] == "data/dataset.csv"
    assert raw["run"]["output_root"] == "runs"
    assert raw["preprocessing"]["sam2_device"] == "cpu"
    assert raw["encoder"]["name"] == "uni2"
    assert "spacing_um" not in raw["encoder"]
    assert "input_size" not in raw["encoder"]
    assert raw["cache"]["enabled"] is True
    assert raw["training"]["learning_rate"] == 2e-4
    assert raw["aggregation"]["params"]["hidden_dim"] == 128
    assert "dataset_csv" not in raw


def test_preview_color_roundtrip_preserves_tuple(tmp_path: Path):
    cfg = _make_pipeline_config()
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.preprocessing.preview.tissue_contour_color == (37, 94, 59)


def test_preprocessing_sam2_worker_limit_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        preprocessing=PreprocessingConfig(sam2_device="cuda", sam2_num_workers=3)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.preprocessing.sam2_device == "cuda"
    assert loaded.preprocessing.sam2_num_workers == 3


def test_evaluation_metrics_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(evaluation=EvalConfig(metrics=["auroc_macro", "f1_macro"]))
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.metrics == ["auroc_macro", "f1_macro"]


def test_evaluation_metrics_empty_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config()
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.metrics == []


def test_evaluation_save_probabilities_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(evaluation=EvalConfig(save_probabilities=True))
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.save_probabilities is True


def test_evaluation_subgroups_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        evaluation=EvalConfig(
            metrics=["auroc_macro"],
            subgroups=SubgroupConfig(columns=["sex", "grade"]),
        )
    )
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.subgroups.columns == ["sex", "grade"]


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
    assert raw["aggregation"] is None


def test_pipeline_config_requires_task():
    with pytest.raises(TypeError, match="task"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
        )


def test_load_config_blank_sections_inherit_defaults(tmp_path: Path):
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "slide",
        },
        "task": {},
        "encoder": {},
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(raw, f)

    loaded = load_config(yaml_path)

    assert loaded.task.name == "binary_classification"
    assert loaded.encoder.name == "uni2"
    assert loaded.output_root == "runs"


def test_load_config_rejects_legacy_flat_layout(tmp_path: Path):
    raw = {
        "dataset_csv": "dataset.csv",
        "splits_csv": "splits.csv",
        "output_root": "out",
        "dataset_type": "slide",
        "task": {"name": "binary_classification"},
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(raw, f)

    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(yaml_path)


# --- dataset_type validation ---


def test_patient_dataset_type_is_valid():
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="patient",
        task=TaskConfig(name="binary_classification"),
    )
    assert cfg.dataset_type == "patient"
    assert cfg.aggregator is None


def test_patient_dataset_type_with_aggregator_raises():
    with pytest.raises(ValueError, match="aggregator"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="patient",
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="binary_classification"),
        )


def test_invalid_dataset_type_raises():
    with pytest.raises(ValueError, match="dataset_type"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="case",
            task=TaskConfig(name="binary_classification"),
        )


# --- feature_mode / augmentation (live segmentation path) ---


def _seg_kwargs(**overrides):
    from soma.config import DecoderConfig

    base = dict(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
    )
    base.update(overrides)
    return base


def test_feature_mode_defaults_to_cached():
    config = PipelineConfig(**_seg_kwargs())
    assert config.feature_mode == "cached"
    assert not config.augmentation.is_enabled()


def test_invalid_feature_mode_raises():
    with pytest.raises(ValueError, match="feature_mode"):
        PipelineConfig(**_seg_kwargs(feature_mode="streaming"))


def test_live_feature_mode_requires_segmentation():
    with pytest.raises(ValueError, match="feature_mode='live'"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="tile",
            feature_mode="live",
            task=TaskConfig(name="binary_classification"),
        )


def test_augmentation_requires_live_feature_mode():
    from soma.config import AugmentationConfig

    with pytest.raises(ValueError, match="augmentation requires feature_mode='live'"):
        PipelineConfig(**_seg_kwargs(augmentation=AugmentationConfig(horizontal_flip=0.5)))


def test_live_segmentation_with_augmentation_is_valid():
    from soma.config import AugmentationConfig

    config = PipelineConfig(
        **_seg_kwargs(
            feature_mode="live",
            augmentation=AugmentationConfig(horizontal_flip=0.5, rotation_degrees=10.0),
        )
    )
    assert config.feature_mode == "live"
    assert config.augmentation.is_enabled()


def test_live_no_aug_is_valid():
    config = PipelineConfig(**_seg_kwargs(feature_mode="live"))
    assert config.feature_mode == "live" and not config.augmentation.is_enabled()


def test_feature_mode_and_augmentation_roundtrip(tmp_path: Path):
    from soma.config import AugmentationConfig

    config = PipelineConfig(
        **_seg_kwargs(
            encoder=EncoderConfig(name="uni2"),
            feature_mode="live",
            augmentation=AugmentationConfig(horizontal_flip=0.5, brightness=0.2),
        )
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.feature_mode == "live"
    assert loaded.augmentation == config.augmentation


def test_augmentation_rejects_out_of_range():
    from soma.config import AugmentationConfig

    with pytest.raises(ValueError, match="horizontal_flip"):
        AugmentationConfig(horizontal_flip=1.5)
    with pytest.raises(ValueError, match="scale"):
        AugmentationConfig(scale=1.0)
    with pytest.raises(ValueError, match="hue"):
        AugmentationConfig(hue=0.9)


# --- Helpers ---


def _make_pipeline_config(**overrides) -> PipelineConfig:
    defaults = dict(
        dataset_csv="data/dataset.csv",
        splits_csv="data/splits.csv",
        output_root="runs",
        dataset_type="slide",
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="multiclass_classification", params={"num_classes": 3}),
        evaluation=EvalConfig(),
        training=TrainingConfig(epochs=100, learning_rate=2e-4),
        tags=["test"],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)
