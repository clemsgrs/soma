"""Tests for soma.config — frozen dataclass configurations."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import slide2vec
import yaml
from hs2p import (
    DEFAULT_MASK_SAMPLING,
    FilterConfig,
    MaskSamplingConfig,
    SegmentationConfig,
    TilingConfig,
)

import soma
import soma.config
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    DecoderConfig,
    EncoderLeaf,
    EvalConfig,
    FeaturesConfig,
    NormalizationConfig,
    PipelineConfig,
    ProjectionConfig,
    SubgroupConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from soma.spacing import resolve_effective_spacing_um


def _write_yaml(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- The extraction surface: slide2vec / hs2p dataclasses held as given (ADR 0019) ---

_MIRRORED_CLASSES = (
    "PreprocessingConfig",
    "EncoderConfig",
    "EncoderMemberConfig",
    "CompositeConfig",
    "ExecutionConfig",
    "MasksConfig",
    "SamplingConfig",
    "AttentionConfig",
    "PreviewConfig",
)

_ANNOTATED_SEGMENTATION_YAML = {
    "data": {
        "dataset_csv": "slides.csv",
        "splits_csv": "splits.csv",
        "dataset_type": "segmentation",
        "feature_mode": "cached",
    },
    "preprocessing": {
        "mask_backend": "openslide",
        "requested_tile_size_px": 512,
        "requested_spacing_um": 0.5,
        "spacing_policy": "native_if_coarser",
        "tolerance": 0.1,
        "dense_window_size": 224,
        "dense_window_overlap": 0.5,
        "masks": {
            "pixel_mapping": {"background": 0, "tumor": 1, "necrosis": 2},
            "min_coverage": {"tumor": 0.05, "necrosis": 0.05},
        },
        "sampling": {"strategy": "joint", "output_mode": "merged"},
    },
    "execution": {"precision": "fp16"},
    "cache": {"dtype": "fp16"},
    "encoder": {
        "name": "virchow2",
        "batch_size": 8,
        "adaptive_batching": True,
        "output_variant": "cls",
    },
    "decoder": {"name": "lightweight_conv"},
    "task": {"name": "segmentation", "params": {"classes": {"tumor": [1], "necrosis": [2]}, "ignore": [0]}},
}


def test_the_mirrored_extraction_classes_are_gone():
    for name in _MIRRORED_CLASSES:
        assert not hasattr(soma.config, name), name
        assert not hasattr(soma, name), name


def test_yaml_extraction_blocks_load_into_the_upstream_dataclasses(tmp_path: Path):
    """The YAML blocks are read into slide2vec's PreprocessingConfig (holding hs2p's
    tiling, segmentation, filter and mask-sampling configs), ExecutionOptions and
    DenseOptions; soma keeps only the kind, the encoder leaf and the cache."""
    features = load_config(_write_yaml(tmp_path, _ANNOTATED_SEGMENTATION_YAML)).features

    assert features.kind == "cached"
    assert features.encoder == EncoderLeaf(name="virchow2", params={"output_variant": "cls"})
    extraction = features.extraction
    assert isinstance(extraction, slide2vec.PreprocessingConfig)
    assert extraction.tiling == TilingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=512,
        tolerance=0.1,
        overlap=0.0,
        mask_backend="openslide",
    )
    assert extraction.segmentation == SegmentationConfig(method="hsv", downsample=64)
    # No filter stated: slide2vec's default (hs2p's filter at the tile size) applies.
    assert extraction.filter is None
    assert extraction.masks == MaskSamplingConfig(
        pixel_mapping={"background": 0, "tumor": 1, "necrosis": 2},
        min_coverage={"tumor": 0.05, "necrosis": 0.05},
        output_mode="merged",
        strategy="joint",
    )
    assert extraction.adaptive_batching is True
    assert features.dense == slide2vec.DenseOptions(
        spacing_um=0.5, target_size=512, tolerance=0.1, window_size=224, overlap=0.5
    )
    assert isinstance(features.execution, slide2vec.ExecutionOptions)
    assert (features.execution.precision, features.execution.batch_size) == ("fp16", 8)
    assert features.spacing_policy == "native_if_coarser"
    assert features.cache.dtype == "fp16"


def test_a_tissue_run_holds_hs2p_binary_tissue_masks_at_its_threshold(tmp_path: Path):
    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "slide"},
        "preprocessing": {"requested_spacing_um": 0.5, "requested_tile_size_px": 224, "min_coverage": {"tissue": 0.3}},
        "encoder": {"name": "uni2"},
    }
    masks = load_config(_write_yaml(tmp_path, data)).features.extraction.masks

    assert masks == replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.3})


def test_a_missing_spacing_or_tile_size_is_completed_from_the_encoder(tmp_path: Path):
    """hs2p's TilingConfig is all or nothing, so the encoder fills what the YAML leaves
    out and the YAML's own tiling values are kept."""
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "slide"},
        "preprocessing": {"overlap": 0.25, "backend": "openslide"},
        "encoder": {"name": "uni2"},
    }
    tiling = load_config(_write_yaml(tmp_path, data)).features.extraction.tiling

    defaults = resolve_preprocessing_defaults("uni2")
    assert tiling == TilingConfig(
        requested_spacing_um=defaults["spacing_um"],
        requested_tile_size_px=defaults["tile_size_px"],
        tolerance=0.05,
        overlap=0.25,
        backend="openslide",
    )


def test_a_stated_filter_is_held_at_the_tile_size(tmp_path: Path):
    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "slide"},
        "preprocessing": {"requested_spacing_um": 0.5, "requested_tile_size_px": 256, "a_t": 6},
        "encoder": {"name": "uni2"},
    }
    assert load_config(_write_yaml(tmp_path, data)).features.extraction.filter == FilterConfig(
        ref_tile_size=256, a_t=6
    )


def test_a_tile_run_declares_no_tiling(tmp_path: Path):
    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "tile"},
        "encoder": {"name": "uni", "batch_size": 16},
    }
    features = load_config(_write_yaml(tmp_path, data)).features

    assert features.extraction.tiling is None
    assert features.dense is None
    assert features.execution.batch_size == 16


def test_the_composite_block_loads_as_the_composite_encoder_leaf(tmp_path: Path):
    """A composite is an encoder registry entry whose params list the members; each
    member overrides the run's DenseOptions / ExecutionOptions fields for itself."""
    from soma.encoders import CompositeMember, CompositeParams

    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "segmentation"},
        "preprocessing": {"requested_spacing_um": 0.5, "requested_tile_size_px": 448},
        "composite": {
            "concat_resolution": "target",
            "concat_grid_size": [37, 37],
            "encoders": [
                {"name": "uni", "feature_kind": "cls_attention", "attention": {"blocks": [-2, -1]}},
                {
                    "name": "phikon",
                    "member_norm": "layernorm",
                    "batch_size": 4,
                    "allow_non_recommended_settings": True,
                    "dense_window_size": 224,
                },
            ],
        },
        "decoder": {"name": "lightweight_conv"},
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
    }
    features = load_config(_write_yaml(tmp_path, data)).features

    assert features.is_composite
    composite = CompositeParams(**features.encoder.params)
    assert composite.concat_resolution == "target"
    assert composite.concat_grid_size == (37, 37)
    assert composite.members == (
        CompositeMember(name="uni", dense={"feature_kind": "cls_attention", "attention_blocks": [-2, -1]}),
        CompositeMember(
            name="phikon",
            params={"allow_non_recommended_settings": True},
            dense={"window_size": 224},
            execution={"batch_size": 4},
            member_norm="layernorm",
        ),
    )
    member_dense = composite.members[1].dense_options(features.dense)
    assert (member_dense.window_size, member_dense.spacing_um, member_dense.target_size) == (224, 0.5, 448)


@pytest.mark.parametrize(
    "data",
    [
        _ANNOTATED_SEGMENTATION_YAML,
        {
            "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "detection"},
            "preprocessing": {"requested_spacing_um": 0.25, "requested_tile_size_px": 1024, "dense_window_size": 448},
            "composite": {"encoders": [{"name": "uni", "member_norm": "l2", "batch_size": 8}, {"name": "phikon"}]},
            "decoder": {"name": "lightweight_conv"},
            "task": {"name": "detection", "params": {"num_classes": 2}},
        },
        {
            "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "slide"},
            "preprocessing": {"tissue_method": "otsu", "seg_downsample": 32, "sam2_num_workers": 2, "a_t": 6},
            "encoder": {"name": "uni2", "precision": "fp32", "save_tile_features": True},
            "execution": {"num_gpus": 1, "num_workers_per_gpu": 3, "prefetch_factor": 2},
        },
    ],
    ids=["annotated-segmentation", "composite-detection", "slide"],
)
def test_a_saved_config_loads_back_into_the_same_features(tmp_path: Path, data: dict):
    config = load_config(_write_yaml(tmp_path, data, "given.yaml"))
    saved = tmp_path / "saved.yaml"

    save_config(config, saved)

    assert load_config(saved) == config


def test_the_encoder_leaf_is_validated_by_the_encoders_registry():
    with pytest.raises(ValueError, match="Unknown encoder name 'not_a_real_encoder'"):
        FeaturesConfig(encoder="not_a_real_encoder")
    with pytest.raises(ValueError, match="has no param.*'precision'"):
        FeaturesConfig(encoder=EncoderLeaf(name="uni2", params={"precision": "fp16"}))
    with pytest.raises(ValueError, match="Unknown encoder name 'nope' in a composite"):
        FeaturesConfig(encoder=EncoderLeaf(name="composite", params={"members": [{"name": "nope"}]}))
    assert FeaturesConfig(encoder="uni2").encoder == EncoderLeaf(name="uni2")


def test_features_validate_the_kind_and_the_spacing_policy():
    with pytest.raises(ValueError, match="features.kind"):
        FeaturesConfig(kind="streaming")
    with pytest.raises(ValueError, match="spacing_policy"):
        FeaturesConfig(spacing_policy="silently_upsample")
    with pytest.raises(TypeError, match="features.extraction"):
        FeaturesConfig(extraction={"tiling": None})


def test_removed_preprocessing_keys_are_named(tmp_path: Path):
    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "slide"},
        "preprocessing": {"read_tile_size_px": 448},
    }
    with pytest.raises(ValueError, match="read_tile_size_px was removed"):
        load_config(_write_yaml(tmp_path, data))
    data["preprocessing"] = {"tissue_mask_tissue_value": 7}
    with pytest.raises(ValueError, match="tissue_mask_tissue_value must be 1"):
        load_config(_write_yaml(tmp_path, data))


def test_native_if_coarser_resolves_one_effective_spacing_rule():
    def effective(policy: str, native: float | None) -> float:
        return resolve_effective_spacing_um(
            requested_spacing_um=0.5, spacing_at_level_0=native, tolerance=0.05, policy=policy
        )

    assert effective("native_if_coarser", None) == 0.5
    assert effective("native_if_coarser", 0.51) == 0.5
    assert effective("native_if_coarser", 0.657476464) == 0.657476464
    assert effective("strict", 0.657476464) == 0.5


# --- Frozen immutability ---


def test_pipeline_config_is_frozen():
    cfg = _make_pipeline_config()
    assert cfg.mirror_root is None
    with pytest.raises(FrozenInstanceError):
        cfg.output_root = "other"


def test_mirror_root_roundtrips_as_an_optional_run_setting(tmp_path: Path):
    cfg = _make_pipeline_config(mirror_root=tmp_path / "shared-mirror")
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["run"]["mirror_root"] == str(tmp_path / "shared-mirror")
    assert loaded.mirror_root == tmp_path / "shared-mirror"


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seed == 0
    assert cfg.epochs == 50
    assert cfg.max_steps is None
    assert cfg.learning_rate == 1e-4
    assert cfg.weight_decay == 1e-5
    assert cfg.optimizer == "adam"
    assert cfg.scheduler == "cosine"
    assert cfg.patience == 10
    assert cfg.monitor == "loss"
    assert cfg.monitor_mode == "min"
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation == 1
    assert cfg.tune_is_test is False
    assert cfg.checkpoint_selection == "best"
    # DataLoader knobs (Phase 2.2). num_workers defaults to 0 so the suite stays
    # fast on test fixtures; users should raise it for real WSI runs.
    assert cfg.num_workers == 0
    assert cfg.pin_memory is True
    assert cfg.persistent_workers is True


def test_training_config_accepts_an_optimizer_step_budget():
    cfg = TrainingConfig(epochs=None, max_steps=12500)

    assert cfg.epochs is None
    assert cfg.max_steps == 12500


def test_pipeline_config_rejects_both_epoch_and_step_budgets():
    with pytest.raises(ValueError, match="exactly one.*epochs.*max_steps"):
        _make_pipeline_config(training=TrainingConfig(epochs=5, max_steps=10))


def test_pipeline_config_rejects_a_missing_training_budget():
    with pytest.raises(ValueError, match="exactly one.*epochs.*max_steps"):
        _make_pipeline_config(training=TrainingConfig(epochs=None, max_steps=None))


def test_pipeline_config_rejects_a_non_positive_step_budget():
    with pytest.raises(ValueError, match="max_steps.*integer >= 1"):
        _make_pipeline_config(training=TrainingConfig(epochs=None, max_steps=0))


def test_step_budget_survives_config_roundtrip(tmp_path: Path):
    config = _make_pipeline_config(
        training=TrainingConfig(epochs=None, max_steps=12500)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(config, yaml_path)
    loaded = load_config(yaml_path)

    saved = yaml.safe_load(yaml_path.read_text())
    assert saved["training"]["epochs"] is None
    assert saved["training"]["max_steps"] == 12500
    assert loaded.training.epochs is None
    assert loaded.training.max_steps == 12500


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
        ({"checkpoint_selection": "final"}, "checkpoint_selection"),
    ],
)
def test_training_config_rejects_non_positive_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_checkpoint_selection_last_rejects_finite_patience():
    """`last` disables early stopping, so a finite patience is a silent lie."""
    with pytest.raises(ValueError, match="patience"):
        TrainingConfig(checkpoint_selection="last", patience=10)


def test_checkpoint_selection_last_accepts_monitor_as_validated_noop():
    """Under `last` the monitor still governs the logged diagnostics, so it is
    validated as usual — it simply no longer selects the checkpoint."""
    cfg = TrainingConfig(
        checkpoint_selection="last", patience=None, monitor="auroc", monitor_mode="max"
    )

    assert cfg.monitor == "auroc"
    assert cfg.monitor_mode == "max"
    with pytest.raises(ValueError, match="monitor_mode"):
        TrainingConfig(checkpoint_selection="last", patience=None, monitor_mode="largest")


def test_pipeline_rejects_monitor_outside_effective_task_metrics():
    """Monitor validity is protocol validation even when it does not select a checkpoint."""
    with pytest.raises(ValueError, match="monitor.*accuracy.*multiclass_classification"):
        _make_pipeline_config(
            training=TrainingConfig(
                checkpoint_selection="last",
                patience=None,
                monitor="accuracy",
                monitor_mode="max",
            )
        )


def test_pipeline_validates_the_monitor_against_the_heads_own_metrics():
    """A multi-output regression reports ``mean_pearson``, not the family's ``mae``/``r2``."""
    task = TaskConfig(name="regression", params={"num_outputs": 2})
    cfg = _make_pipeline_config(
        task=task, training=TrainingConfig(monitor="mean_pearson", monitor_mode="max")
    )
    assert cfg.training.monitor == "mean_pearson"
    with pytest.raises(ValueError, match="monitor 'mae'.*Available metrics: mean_pearson"):
        _make_pipeline_config(task=task, training=TrainingConfig(monitor="mae", monitor_mode="min"))
    # A one-output regression keeps the family defaults.
    single = TaskConfig(name="regression", params={})
    assert _make_pipeline_config(
        task=single, training=TrainingConfig(monitor="mae", monitor_mode="min")
    ).training.monitor == "mae"


def test_pipeline_defers_monitor_validation_to_a_custom_learner():
    """A configured learner owns its ``validation_summary`` keys; the Checkpoint callback
    validates the monitor against the real summary at the first evaluation."""
    cfg = _make_pipeline_config(
        training=TrainingConfig(
            learner={"name": "my_pkg.learners.EntropyLearner"},
            monitor="entropy",
            monitor_mode="min",
        )
    )
    assert cfg.training.monitor == "entropy"


def test_checkpoint_selection_survives_config_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        training=TrainingConfig(checkpoint_selection="last", patience=None)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["training"]["checkpoint_selection"] == "last"
    assert saved["training"]["patience"] is None
    assert loaded.training.checkpoint_selection == "last"
    assert loaded.training.patience is None


def test_checkpoint_selection_composes_with_allow_missing_tune():
    """Orthogonal knobs: which checkpoint is evaluated vs where tune comes from."""
    cfg = TrainingConfig(
        checkpoint_selection="last", patience=None, allow_missing_tune=True
    )

    assert cfg.checkpoint_selection == "last"
    assert cfg.allow_missing_tune is True
    assert TrainingConfig(allow_missing_tune=True).checkpoint_selection == "best"
    assert (
        TrainingConfig(checkpoint_selection="last", patience=None).allow_missing_tune
        is False
    )


def test_normalization_config_defaults():
    """Feature normalization is off by default (issue #283)."""
    cfg = NormalizationConfig()

    assert cfg.method == "none"
    assert cfg.eps == 1e-6


@pytest.mark.parametrize("method", ["none", "zscore", "l2", "layernorm"])
def test_normalization_config_accepts_vocabulary(method):
    assert NormalizationConfig(method=method).method == method


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "standardize"}, "method"),
        ({"method": "zscore", "eps": 0.0}, "eps"),
        ({"method": "zscore", "eps": -1e-6}, "eps"),
    ],
)
def test_normalization_config_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NormalizationConfig(**kwargs)


@pytest.mark.parametrize("eps", [True, "1e-6", float("nan"), float("inf"), float("-inf")])
def test_normalization_eps_requires_a_finite_positive_real(eps):
    with pytest.raises(ValueError, match="eps.*finite positive real"):
        NormalizationConfig(method="zscore", eps=eps)


def test_normalization_none_rejects_non_default_eps():
    with pytest.raises(ValueError, match="method='none'.*eps"):
        NormalizationConfig(method="none", eps=1e-4)


def test_pipeline_config_normalization_defaults_to_off():
    assert _make_pipeline_config().normalization == NormalizationConfig()


def test_saved_config_always_serializes_normalization_block(tmp_path: Path):
    """Guard the hash, not the record: the block is written even when off."""
    yaml_path = tmp_path / "config.yaml"

    save_config(_make_pipeline_config(), yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["normalization"] == {"method": "none", "eps": 1e-6}
    assert load_config(yaml_path).normalization == NormalizationConfig()


def test_normalization_survives_config_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        normalization=NormalizationConfig(method="zscore", eps=1e-5)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["normalization"] == {"method": "zscore", "eps": 1e-5}
    assert loaded.normalization == NormalizationConfig(method="zscore", eps=1e-5)


def test_load_config_rejects_unknown_normalization_method(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path)},
                "data": {
                    "dataset_csv": "d.csv",
                    "splits_csv": "s.csv",
                    "dataset_type": "slide",
                },
                "task": {"name": "binary_classification"},
                "normalization": {"method": "standardize"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="method"):
        load_config(yaml_path)


def test_projection_config_defaults():
    """Label-free projection is off by default (issue #284)."""
    cfg = ProjectionConfig()

    assert cfg.method == "none"
    assert cfg.target_dim is None
    assert cfg.seed == 0


@pytest.mark.parametrize("method", ["none", "pca", "random"])
def test_projection_config_accepts_vocabulary(method):
    target_dim = None if method == "none" else 64
    assert ProjectionConfig(method=method, target_dim=target_dim).method == method


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "whiten", "target_dim": 8}, "method"),
        ({"method": "pca"}, "target_dim"),
        ({"method": "random"}, "target_dim"),
        ({"method": "pca", "target_dim": 0}, "target_dim"),
        ({"method": "pca", "target_dim": -4}, "target_dim"),
    ],
)
def test_projection_config_rejects_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ProjectionConfig(**kwargs)


@pytest.mark.parametrize("target_dim", [2.9, 2.0, True, "2"])
def test_projection_target_dim_requires_a_positive_integer(target_dim):
    with pytest.raises(ValueError, match="target_dim.*positive integer"):
        ProjectionConfig(method="pca", target_dim=target_dim)


@pytest.mark.parametrize("seed", [0.9, 0.0, True, "7"])
def test_projection_seed_requires_an_integer(seed):
    with pytest.raises(ValueError, match="seed.*integer"):
        ProjectionConfig(method="random", target_dim=2, seed=seed)


def test_projection_seed_accepts_negative_integer():
    assert ProjectionConfig(method="random", target_dim=2, seed=-1).seed == -1


@pytest.mark.parametrize("fields", [{"target_dim": 8}, {"seed": 1}])
def test_projection_none_rejects_non_default_fields(fields):
    with pytest.raises(ValueError, match="method='none'.*(target_dim|seed)"):
        ProjectionConfig(method="none", **fields)


def test_pca_rejects_non_default_seed():
    with pytest.raises(ValueError, match="method='pca'.*seed"):
        ProjectionConfig(method="pca", target_dim=2, seed=1)


def test_pipeline_config_projection_defaults_to_off():
    assert _make_pipeline_config().projection == ProjectionConfig()


def test_saved_config_always_serializes_projection_block(tmp_path: Path):
    """Guard the hash, not the record: the block is written even when off."""
    yaml_path = tmp_path / "config.yaml"

    save_config(_make_pipeline_config(), yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["projection"] == {"method": "none", "target_dim": None, "seed": 0}
    assert load_config(yaml_path).projection == ProjectionConfig()


def test_projection_survives_config_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(
        projection=ProjectionConfig(method="random", target_dim=128, seed=7)
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["projection"] == {"method": "random", "target_dim": 128, "seed": 7}
    assert loaded.projection == ProjectionConfig(method="random", target_dim=128, seed=7)


def test_normalization_and_projection_are_independently_configurable(tmp_path: Path):
    """Standardize without projecting, project without standardizing, or both."""
    yaml_path = tmp_path / "config.yaml"
    cfg = _make_pipeline_config(
        projection=ProjectionConfig(method="random", target_dim=32)
    )

    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded.normalization == NormalizationConfig()
    assert loaded.projection == ProjectionConfig(method="random", target_dim=32)


def test_load_config_rejects_unknown_projection_method(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path)},
                "data": {
                    "dataset_csv": "d.csv",
                    "splits_csv": "s.csv",
                    "dataset_type": "slide",
                },
                "task": {"name": "binary_classification"},
                "projection": {"method": "whiten", "target_dim": 8},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="method"):
        load_config(yaml_path)


def test_aggregator_config_explicit_name():
    cfg = AggregatorConfig(name="transmil")
    assert cfg.name == "transmil"
    assert cfg.params == {}


def test_aggregator_config_requires_name():
    with pytest.raises(TypeError):
        AggregatorConfig()


def test_task_config_requires_name():
    with pytest.raises(TypeError):
        TaskConfig()


def test_task_config_params_default_empty():
    cfg = TaskConfig(name="binary_classification")
    assert cfg.params == {}


def test_evaluation_config_defaults():
    cfg = EvalConfig()
    assert cfg.metrics == []
    assert cfg.subgroups.columns == []
    # Symmetric 2x2 of dense-artifact toggles: cheap visual overlays default on,
    # heavy raw outputs default off.
    assert cfg.save_segmentation_overlays is True
    assert cfg.save_segmentation_probabilities is False
    assert cfg.save_detection_overlays is True
    assert cfg.save_detection_heatmaps is False
    assert cfg.holdout_test is False


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
            features=FeaturesConfig(encoder=EncoderLeaf(name="not_a_real_encoder")),
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


_CLASS_SCHEME = {"classes": {"tumor": [1], "stroma": [2], "necrosis": [3]}, "ignore": [0]}
_PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2, "necrosis": 3}


def _annotation_masks(**overrides) -> MaskSamplingConfig:
    values = dict(pixel_mapping=_PIXEL_MAPPING, min_coverage={"tumor": 0.1}, output_mode="merged")
    values.update(overrides)
    return MaskSamplingConfig(**values)


def _features(masks: MaskSamplingConfig | None = None, **overrides) -> FeaturesConfig:
    if masks is not None:
        overrides["extraction"] = slide2vec.PreprocessingConfig(masks=masks)
    return FeaturesConfig(**overrides)


def _seg_config(**overrides):
    masks = overrides.pop("masks", None)
    kwargs = dict(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation"),
    )
    if masks is not None:
        kwargs["features"] = _features(masks)
        # Annotation rasters need an explicit training class scheme.
        kwargs["task"] = TaskConfig(name="segmentation", params=dict(_CLASS_SCHEME))
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def test_segmentation_config_valid():
    cfg = _seg_config()
    assert cfg.dataset_type == "segmentation"
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.aggregator is None


def test_segmentation_requires_decoder():
    with pytest.raises(ValueError, match="requires a decoder"):
        _seg_config(decoder=None)


def test_segmentation_rejects_aggregator():
    with pytest.raises(ValueError, match="aggregator must be None"):
        _seg_config(aggregator=AggregatorConfig(name="abmil"))


def test_segmentation_requires_segmentation_task():
    with pytest.raises(ValueError, match="'segmentation' family"):
        _seg_config(task=TaskConfig(name="binary_classification"))


def test_decoder_rejected_for_non_segmentation():
    with pytest.raises(ValueError, match="decoder must be None"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="binary_classification"),
        )


def test_segmentation_rejects_unknown_decoder():
    with pytest.raises(ValueError, match="Unknown decoder"):
        _seg_config(decoder=DecoderConfig(name="not_a_decoder"))


def test_decoder_config_round_trips_through_yaml(tmp_path: Path):
    cfg = _seg_config(decoder=DecoderConfig(name="lightweight_conv", params={"hidden_dim": 64}))
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.decoder.name == "lightweight_conv"
    assert loaded.decoder.params == {"hidden_dim": 64}


# --- Annotation sampling: hs2p's MaskSamplingConfig in the extraction ---


def test_segmentation_slide_manifest_config_valid():
    cfg = _seg_config(
        masks=_annotation_masks(),
        task=TaskConfig(
            name="segmentation",
            params={"classes": {"tissue": [1, 2], "necrosis": [3]}, "ignore": [0]},
        ),
    )
    assert cfg.features.extraction.masks.pixel_mapping["necrosis"] == (3,)
    assert cfg.features.extraction.masks.output_mode == "merged"


def test_segmentation_slide_manifest_requires_task_classes():
    """Annotation rasters carry the dataset's own values: the training class scheme
    must be stated (task.params.classes), it is never inferred from pixel_mapping."""
    with pytest.raises(ValueError, match=r"requires task\.params\.classes"):
        _seg_config(
            masks=_annotation_masks(),
            task=TaskConfig(name="segmentation", params={"num_classes": 3}),
        )


def test_segmentation_slide_manifest_rejects_class_value_missing_from_pixel_mapping():
    """hs2p 5 reads annotation rasters against the closed pixel_mapping vocabulary, so a
    raw value the training scheme names must be declared there too."""
    with pytest.raises(
        ValueError, match=r"task\.params\.classes 'tumor' lists raw value 7.*pixel_mapping"
    ):
        _seg_config(
            masks=_annotation_masks(),
            task=TaskConfig(name="segmentation", params={"classes": {"tumor": [1, 7]}}),
        )


def test_segmentation_slide_manifest_rejects_ignored_value_missing_from_pixel_mapping():
    with pytest.raises(
        ValueError, match=r"task\.params\.ignore lists raw value 9.*pixel_mapping"
    ):
        _seg_config(
            masks=_annotation_masks(),
            task=TaskConfig(
                name="segmentation", params={"classes": {"tumor": [1]}, "ignore": [0, 9]}
            ),
        )


def test_segmentation_pre_cropped_classes_need_no_pixel_mapping():
    # hs2p's binary tissue vocabulary is no sampling vocabulary to check against.
    cfg = _seg_config(
        task=TaskConfig(name="segmentation", params={"classes": {"tumor": [7]}, "ignore": [9]})
    )
    assert cfg.features.extraction.masks.pixel_mapping == DEFAULT_MASK_SAMPLING.pixel_mapping


def test_segmentation_rejects_raw_value_in_two_classes():
    with pytest.raises(ValueError, match="'tumor' and 'stroma' both list raw value 2"):
        _seg_config(
            task=TaskConfig(
                name="segmentation", params={"classes": {"tumor": [1, 2], "stroma": [2]}}
            )
        )


def test_top_level_masks_sampling_yaml_rejected(tmp_path: Path):
    """A top-level masks:/sampling: block in YAML is not accepted (#109)."""
    path = tmp_path / "cfg.yaml"
    save_config(_seg_config(), path)
    raw = yaml.safe_load(path.read_text())
    raw["masks"] = {"pixel_mapping": {"background": 0, "tumor": 1}}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(path)


def test_masks_valid_for_slide_dataset():
    """An annotation vocabulary is valid on a dataset_type='slide' dataset (the
    annotation-restricted merged bag, #110): it restricts tile selection only; the MIL
    aggregator and task head are the ordinary slide-level pair."""
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="slide",
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification"),
        features=_features(
            MaskSamplingConfig(
                pixel_mapping={"background": 0, "tumor": 1},
                min_coverage={"tumor": 0.5},
                output_mode="merged",
            )
        ),
    )
    assert cfg.features.extraction.masks.pixel_mapping == {"background": (0,), "tumor": (1,)}
    assert cfg.aggregator.name == "abmil"
    assert cfg.task.name == "binary_classification"


def test_masks_rejected_for_tile_dataset():
    """An annotation vocabulary on a dataset_type='tile' dataset raises — patch manifests
    have no annotation-sampling step."""
    with pytest.raises(ValueError, match="samples annotations.*dataset_type"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="tile",
            task=TaskConfig(name="binary_classification"),
            features=_features(
                MaskSamplingConfig(
                    pixel_mapping={"background": 0, "tumor": 1},
                    min_coverage={"tumor": 0.5},
                    output_mode="merged",
                )
            ),
        )


@pytest.mark.parametrize("dataset_type", ["slide", "patient", "segmentation"])
def test_per_annotation_output_mode_is_deferred(dataset_type: str):
    """hs2p's per_annotation output mode stays rejected, pointing at #86."""
    kwargs = dict(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type=dataset_type,
        task=TaskConfig(name="binary_classification"),
        features=_features(_annotation_masks(output_mode="per_annotation")),
    )
    if dataset_type == "segmentation":
        kwargs.update(decoder=DecoderConfig(name="lightweight_conv"), task=TaskConfig(name="segmentation", params=dict(_CLASS_SCHEME)))
    with pytest.raises(ValueError, match="per_annotation.*#86"):
        PipelineConfig(**kwargs)


def test_masks_valid_for_patient_dataset():
    """An annotation vocabulary is valid on a dataset_type='patient' dataset (#111):
    every slide is tiled to its restricted merged bag; the patient encoder and task head
    are untouched."""
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="patient",
        task=TaskConfig(name="binary_classification"),
        features=_features(
            MaskSamplingConfig(
                pixel_mapping={"background": 0, "tumor": 1},
                min_coverage={"tumor": 0.5},
                output_mode="merged",
            )
        ),
    )
    assert cfg.features.extraction.masks.min_coverage == {"tumor": 0.5}
    assert cfg.aggregator is None


def test_sampling_requires_masks(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        {
            "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "segmentation"},
            "decoder": {"name": "lightweight_conv"},
            "task": {"name": "segmentation", "params": {"num_classes": 2}},
            "preprocessing": {"sampling": {"strategy": "joint"}},
        },
    )
    with pytest.raises(ValueError, match="sampling: requires a masks: block"):
        load_config(path)


def test_masks_sampling_round_trip_through_yaml(tmp_path: Path):
    cfg = _seg_config(
        masks=_annotation_masks(
            min_coverage={"tumor": 0.1, "stroma": 0.5},
            colors={"background": None, "tumor": [255, 0, 0], "stroma": [0, 255, 0], "necrosis": [0, 0, 255]},
            strategy="independent",
        ),
    )
    path = tmp_path / "seg-slide-manifest.yaml"
    save_config(cfg, path)
    raw = yaml.safe_load(path.read_text())
    assert "masks" not in raw and "sampling" not in raw
    assert raw["preprocessing"]["sampling"] == {"strategy": "independent", "output_mode": "merged"}
    loaded = load_config(path)
    assert loaded.features.extraction.masks == cfg.features.extraction.masks


def test_masks_accepts_hs2p_list_of_single_entry_mappings(tmp_path: Path):
    """An hs2p-style masks block (list of single-entry mappings) pastes in unchanged: it
    is read by hs2p's own resolver."""
    path = tmp_path / "cfg.yaml"
    save_config(_seg_config(), path)
    raw = yaml.safe_load(path.read_text())
    raw["preprocessing"]["masks"] = {
        "pixel_mapping": [{"background": 0}, {"tumor": 1}, {"stroma": 2}, {"necrosis": 3}],
        "min_coverage": [{"tumor": 0.1}],
    }
    raw["task"]["params"] = dict(_CLASS_SCHEME)
    path.write_text(yaml.safe_dump(raw))
    loaded = load_config(path)
    assert loaded.features.extraction.masks.pixel_mapping == {
        "background": (0,), "tumor": (1,), "stroma": (2,), "necrosis": (3,)
    }
    assert loaded.features.extraction.masks.min_coverage == {"tumor": 0.1}


# --- The composite encoder ---


def _composite(*members: dict, **params) -> EncoderLeaf:
    return EncoderLeaf(name="composite", params={"members": list(members), **params})


def test_composite_rejects_top_level_feature_adaptor():
    """Composite streams own per-member normalization, not a concatenated adaptor."""
    with pytest.raises(ValueError, match="not yet supported.*composite"):
        _seg_config(
            features=FeaturesConfig(encoder=_composite({"name": "uni"})),
            normalization=NormalizationConfig(method="zscore"),
        )


def test_composite_yaml_rejects_unknown_keys(tmp_path: Path):
    raw = {
        "data": {"dataset_csv": "dataset.csv", "splits_csv": "splits.csv", "dataset_type": "segmentation"},
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
        "decoder": {"name": "lightweight_conv"},
        "composite": {"encoders": [{"name": "uni"}], "concat_resoluton": "target"},
    }
    with pytest.raises(ValueError, match="unsupported keys.*concat_resoluton"):
        load_config(_write_yaml(tmp_path, raw))


def test_composite_member_rejects_shared_read_geometry():
    with pytest.raises(ValueError, match="cannot set dense field.*spacing_um"):
        FeaturesConfig(encoder=_composite({"name": "uni", "dense": {"spacing_um": 1.0}}))


def test_composite_enabled_for_detection():
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="detection",
        decoder=DecoderConfig(name="lightweight_conv"),
        features=FeaturesConfig(encoder=_composite({"name": "uni"})),
        task=TaskConfig(name="detection", params={"num_classes": 2}),
    )
    assert cfg.features.is_composite


def test_detection_rejects_point_id_in_a_class_and_in_drop():
    with pytest.raises(ValueError, match="'mnl' and 'drop' both list raw value 1"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="detection",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="detection", params={"classes": {"mnl": [0, 1]}, "drop": [1]}),
        )


def test_encoder_xor_composite(tmp_path: Path):
    raw = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "segmentation"},
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
        "decoder": {"name": "lightweight_conv"},
        "encoder": {"name": "uni"},
        "composite": {"encoders": [{"name": "phikon"}]},
    }
    with pytest.raises(ValueError, match="XOR"):
        load_config(_write_yaml(tmp_path, raw))


def test_composite_rejected_for_non_dense_dataset():
    with pytest.raises(ValueError, match="only supported for dataset_type"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            features=FeaturesConfig(encoder=_composite({"name": "uni"})),
            task=TaskConfig(name="binary_classification"),
        )


def test_composite_live_is_hard_error():
    with pytest.raises(ValueError, match="cached-only"):
        _seg_config(features=FeaturesConfig(kind="live", encoder=_composite({"name": "uni"})))


def test_composite_member_norm_defaults_by_feature_kind():
    from soma.encoders import CompositeParams
    from soma.encoders.composite import member_norm_for

    run = slide2vec.DenseOptions(spacing_um=0.5, target_size=224)
    composite = CompositeParams(
        members=[
            {"name": "uni", "dense": {"feature_kind": "cls_attention"}},
            {"name": "phikon", "dense": {"feature_kind": "patch_features_prenorm"}},
            {"name": "hibou-b"},
            {"name": "phikon", "member_norm": "layernorm"},
        ]
    )
    norms = [member_norm_for(member, member.dense_options(run)) for member in composite.members]
    assert norms == ["none", "l2", "l2", "layernorm"]
    assert composite.concat_resolution == "grid"


def test_subgroup_config_defaults():
    cfg = SubgroupConfig()
    assert cfg.columns == []


def test_subgroup_config_explicit():
    cfg = SubgroupConfig(columns=["sex", "grade"])
    assert cfg.columns == ["sex", "grade"]


def test_encoder_params_round_trip(tmp_path: Path):
    cfg = _make_pipeline_config(
        features=FeaturesConfig(
            encoder=EncoderLeaf(
                name="h0-mini", params={"output_variant": "cls", "allow_non_recommended_settings": True}
            )
        )
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)

    assert load_config(yaml_path).features.encoder == cfg.features.encoder


def test_execution_options_round_trip(tmp_path: Path):
    cfg = _make_pipeline_config(
        features=FeaturesConfig(
            encoder="uni2",
            execution=slide2vec.ExecutionOptions(
                num_gpus=2,
                num_workers_per_gpu=6,
                num_preprocessing_workers=0,
                prefetch_factor=8,
                precision="fp16",
                batch_size=12,
            ),
        )
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(cfg, yaml_path)

    assert load_config(yaml_path).features.execution == cfg.features.execution


def test_cache_config_defaults():
    cfg = CacheConfig()
    assert cfg.enabled is True
    assert cfg.root_dir is None
    assert cfg.reuse_policy == "strict"
    assert cfg.validate_payloads is False


def test_cache_config_rejects_noncanonical_fields():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        CacheConfig(fingerprint_files=True)


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
        features=FeaturesConfig(
            encoder="uni2",
            extraction=slide2vec.PreprocessingConfig(
                tiling=TilingConfig(
                    requested_spacing_um=0.5,
                    requested_tile_size_px=224,
                    tolerance=0.05,
                    overlap=0.0,
                    backend="openslide",
                ),
                segmentation=SegmentationConfig(method="otsu"),
                filter=FilterConfig(ref_tile_size=224),
            ),
        )
    )
    yaml_path = tmp_path / "config.yaml"

    save_config(original, yaml_path)
    loaded = load_config(yaml_path)

    assert loaded == original


def test_load_config_merges_bundled_defaults_for_new_layout(tmp_path: Path):
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "slide",
        }
    }
    loaded = load_config(_write_yaml(tmp_path, raw))

    assert loaded.dataset_csv == "dataset.csv"
    assert loaded.splits_csv == "splits.csv"
    assert loaded.output_root == "runs"
    assert loaded.dataset_type == "slide"
    assert loaded.features.extraction.segmentation.sam2_device == "cpu"
    # Neutral defaults: no baked-in encoder / aggregator / metrics (set those per run).
    assert loaded.features.encoder is None
    assert loaded.aggregator is None
    assert loaded.task.name == "binary_classification"
    assert loaded.evaluation.metrics == []
    assert loaded.training.epochs == 50


def test_load_config_with_target_fields(tmp_path: Path):
    raw = {
        "data": {"dataset_csv": "dataset.csv", "splits_csv": "splits.csv", "dataset_type": "slide"},
        "preprocessing": {"backend": "cucim", "requested_tile_size_px": 256, "requested_spacing_um": 0.5},
        "cache": {},
        "aggregation": None,
        "task": {"name": "binary_classification"},
        "training": {},
        "run": {"output_root": "out", "tags": []},
    }
    loaded = load_config(_write_yaml(tmp_path, raw))

    tiling = loaded.features.extraction.tiling
    assert (tiling.backend, tiling.requested_tile_size_px, tiling.requested_spacing_um) == ("cucim", 256, 0.5)
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
    assert raw["cache"]["enabled"] is True
    assert raw["training"]["learning_rate"] == 2e-4
    assert raw["aggregation"]["params"]["hidden_dim"] == 128
    assert "dataset_csv" not in raw


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


def test_evaluation_dense_artifact_flags_roundtrip(tmp_path: Path):
    # Flip every flag away from its default so the round-trip has to carry all four.
    cfg = _make_pipeline_config(
        evaluation=EvalConfig(
            save_segmentation_overlays=False,
            save_segmentation_probabilities=True,
            save_detection_overlays=False,
            save_detection_heatmaps=True,
        )
    )
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.save_segmentation_overlays is False
    assert loaded.evaluation.save_segmentation_probabilities is True
    assert loaded.evaluation.save_detection_overlays is False
    assert loaded.evaluation.save_detection_heatmaps is True


def test_evaluation_holdout_test_roundtrip(tmp_path: Path):
    cfg = _make_pipeline_config(evaluation=EvalConfig(holdout_test=True))
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, yaml_path)
    loaded = load_config(yaml_path)
    assert loaded.evaluation.holdout_test is True


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
        "data": {"dataset_csv": "dataset.csv", "splits_csv": "splits.csv", "dataset_type": "slide"},
        "task": {},
        "encoder": {},
    }
    loaded = load_config(_write_yaml(tmp_path, raw))

    assert loaded.task.name == "binary_classification"
    assert loaded.features.encoder is None  # blank encoder section + neutral default → no encoder
    assert loaded.output_root == "runs"


def test_load_config_rejects_legacy_flat_layout(tmp_path: Path):
    raw = {
        "dataset_csv": "dataset.csv",
        "splits_csv": "splits.csv",
        "output_root": "out",
        "dataset_type": "slide",
        "task": {"name": "binary_classification"},
    }
    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(_write_yaml(tmp_path, raw))


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


# --- features.kind / augmentation (live segmentation path) ---


def _seg_kwargs(**overrides):
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


_LIVE = FeaturesConfig(kind="live")


def test_feature_kind_defaults_to_cached():
    config = PipelineConfig(**_seg_kwargs())
    assert config.features.kind == "cached"
    assert config.augmentation is None


def test_live_feature_mode_requires_segmentation():
    with pytest.raises(ValueError, match="feature_mode='live'"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="tile",
            features=_LIVE,
            task=TaskConfig(name="binary_classification"),
        )


_FLIP = {"name": "segmentation", "params": {"horizontal_flip": 0.5, "rotation_degrees": 10.0}}


def test_augmentation_requires_live_feature_mode():
    with pytest.raises(ValueError, match="augmentation requires feature_mode='live'"):
        PipelineConfig(**_seg_kwargs(augmentation=_FLIP))


def test_live_segmentation_with_augmentation_is_valid():
    config = PipelineConfig(**_seg_kwargs(features=_LIVE, augmentation=_FLIP))
    assert config.features.kind == "live"
    assert config.augmentation == _FLIP


def test_live_detection_is_valid_but_its_points_cannot_be_augmented():
    detection = dict(
        dataset_type="detection",
        task=TaskConfig(name="detection", params={"num_classes": 1, "match_distance": 3.0}),
    )
    assert PipelineConfig(**_seg_kwargs(features=_LIVE, **detection)).features.kind == "live"
    with pytest.raises(ValueError, match="point targets cannot be augmented yet"):
        PipelineConfig(**_seg_kwargs(features=_LIVE, augmentation=_FLIP, **detection))


def test_detection_score_threshold_is_refused_with_its_replacements_named():
    with pytest.raises(ValueError, match="min_score.*validation_threshold"):
        PipelineConfig(
            **_seg_kwargs(
                dataset_type="detection",
                task=TaskConfig(
                    name="detection", params={"num_classes": 1, "match_distance": 3.0, "score_threshold": 0.5}
                ),
            )
        )


def test_augmentation_is_a_name_params_leaf(tmp_path: Path):
    with pytest.raises(ValueError, match="name, params"):
        load_config_from_dict({"augmentation": {"horizontal_flip": 0.5}}, tmp_path)
    with pytest.raises(ValueError, match="Unknown augmentation"):
        PipelineConfig(**_seg_kwargs(features=_LIVE, augmentation={"name": "nope", "params": {}}))


def test_live_no_aug_is_valid():
    config = PipelineConfig(**_seg_kwargs(features=_LIVE))
    assert config.features.kind == "live" and config.augmentation is None


def test_feature_mode_and_augmentation_roundtrip(tmp_path: Path):
    config = PipelineConfig(
        **_seg_kwargs(
            features=FeaturesConfig(kind="live", encoder="uni2"),
            augmentation={"name": "segmentation", "params": {"horizontal_flip": 0.5, "brightness": 0.2}},
        )
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.features.kind == "live"
    assert loaded.augmentation == config.augmentation


# --- Helpers ---


def _make_pipeline_config(**overrides) -> PipelineConfig:
    defaults = dict(
        dataset_csv="data/dataset.csv",
        splits_csv="data/splits.csv",
        output_root="runs",
        dataset_type="slide",
        features=FeaturesConfig(encoder="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="multiclass_classification", params={"num_classes": 3}),
        evaluation=EvalConfig(),
        training=TrainingConfig(epochs=100, learning_rate=2e-4),
        tags=["test"],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_training_config_optimizer_and_scheduler_are_component_leaves():
    """A string is the leaf's name; a ``{name, params}`` mapping is the full leaf; a
    scheduler may be a list of leaves (warm-up is an ordinary ``constant`` leaf)."""
    cfg = TrainingConfig()
    assert (cfg.optimizer, cfg.scheduler, cfg.learner) == ("adam", "cosine", None)
    cfg = TrainingConfig(
        optimizer={"name": "sgd", "params": {"momentum": 0.9}},
        scheduler=[{"name": "constant", "params": {"factor": 0.25, "total_iters": 3}}, "cosine"],
        learner={"name": "my_pkg.learners.EntropyLearner", "params": {"weight": 0.1}},
    )
    assert cfg.optimizer["name"] == "sgd" and cfg.scheduler[1] == "cosine"
    assert cfg.learner["params"] == {"weight": 0.1}
    assert TrainingConfig(scheduler=None).scheduler is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"optimizer": {"params": {"lr": 1.0}}}, "optimizer"),
        ({"optimizer": 3}, "optimizer"),
        ({"scheduler": {"factor": 0.5}}, "scheduler"),
        ({"scheduler": [None, 7]}, "scheduler"),
        ({"learner": "mil"}, "learner"),
        ({"learner": {"params": {}}}, "learner"),
    ],
)
def test_training_config_rejects_malformed_component_leaves(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TrainingConfig(**kwargs)


def test_training_config_has_no_dedicated_warmup_knobs():
    with pytest.raises(TypeError):
        TrainingConfig(lr_warmup_epochs=5)


@pytest.mark.parametrize(
    "retired", [{"commit_every": 1024}, {"on_unrecorded_identity": "warn"}]
)
def test_load_config_rejects_retired_cache_keys_as_unknown(tmp_path: Path, retired):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path)},
                "data": {
                    "dataset_csv": "d.csv",
                    "splits_csv": "s.csv",
                    "dataset_type": "slide",
                },
                "task": {"name": "binary_classification"},
                "cache": {"enabled": True, **retired},
            }
        ),
        encoding="utf-8",
    )

    (key,) = retired
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{key}'"):
        load_config(yaml_path)


def load_config_from_dict(overrides: dict, tmp_path: Path) -> PipelineConfig:
    """Write a minimal segmentation YAML with ``overrides`` and load it."""
    import yaml

    data = {
        "data": {"dataset_csv": "d.csv", "splits_csv": "s.csv", "dataset_type": "segmentation"},
        "decoder": {"name": "lightweight_conv"},
        "task": {"name": "segmentation", "params": {"num_classes": 2}},
        **overrides,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)
