"""``soma.config``: seven sections mirroring the public callables, hydrated from YAML."""

from dataclasses import replace
from pathlib import Path

import pytest
import slide2vec
import yaml
from hs2p import MaskSamplingConfig, SegmentationConfig, TilingConfig

import soma
import soma.config
from soma.config import (
    CacheOptions,
    CohortConfig,
    Component,
    CRoMaOptions,
    EncoderLeaf,
    EvaluationConfig,
    ExperimentConfig,
    FeaturesConfig,
    LearnerConfig,
    LoadersConfig,
    RunConfig,
    TrainerConfig,
)
from tests.experiment_configs import experiment

SECTIONS = ["run", "cohort", "features", "loaders", "learner", "trainer", "evaluation"]

#: The YAML of a slide-manifest segmentation run, every section spelled.
ANNOTATED_SEGMENTATION = {
    "run": {"output_dir": "runs/seg", "seed": 3},
    "cohort": {"records": "slides.csv", "splits": "splits.csv", "tune_is_test": True},
    "features": {
        "encoder": {"name": "virchow2", "params": {"output_variant": "cls"}},
        "extraction": {
            "tiling": {
                "requested_spacing_um": 0.5,
                "requested_tile_size_px": 512,
                "tolerance": 0.1,
                "overlap": 0.0,
                "mask_backend": "openslide",
            },
            "masks": {
                "pixel_mapping": {"background": 0, "tumor": 1, "necrosis": 2},
                "min_coverage": {"tumor": 0.05, "necrosis": 0.05},
                "output_mode": "merged",
            },
            "adaptive_batching": True,
        },
        "execution": {"precision": "fp16", "batch_size": 8},
        "dense": {"spacing_um": 0.5, "target_size": 512, "tolerance": 0.1, "window_size": 224, "overlap": 0.5},
        "cache": {"dtype": "fp16"},
        "spacing_policy": "native_if_coarser",
    },
    "loaders": {"batch_size": 4, "num_workers": 2, "sampler": {"name": "class_conditioned"}},
    "learner": {
        "name": "dense",
        "params": {
            "decoder": {"name": "lightweight_conv"},
            "head": {"name": "segmentation", "params": {"classes": {"tumor": [1], "necrosis": [2]}, "ignore": [0]}},
            "optimizer": {"name": "adamw", "params": {"lr": 3.0e-4}},
            "scheduler": "cosine",
        },
    },
    "trainer": {"max_epochs": 30, "callbacks": [{"name": "checkpoint", "params": {"monitor": "mean_dice", "mode": "max"}}]},
    "evaluation": {"metrics": ["mean_dice"], "save_overlays": False},
}


def _write_yaml(tmp_path: Path, data: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- the surface ---------------------------------------------------------------------


def test_the_former_config_surface_is_gone():
    removed = (
        "PipelineConfig",
        "TaskConfig",
        "TrainingConfig",
        "EvalConfig",
        "RepresentationConfig",
        "SubgroupConfig",
        "HeatmapConfig",
        "CacheConfig",
        "AggregatorConfig",
        "DecoderConfig",
        "NormalizationConfig",
        "ProjectionConfig",
        "load_config",
        "save_config",
        "config_yaml_dict",
        "task_config_dict",
    )
    for name in removed:
        assert not hasattr(soma.config, name), name
        assert not hasattr(soma, name), name


def test_the_root_namespace_exports_the_experiment_and_its_plan():
    assert soma.ExperimentConfig is ExperimentConfig
    assert soma.Plan.__name__ == "Plan"
    assert callable(soma.resolve)


def test_an_experiment_has_seven_sections_and_only_needs_a_cohort():
    config = ExperimentConfig(cohort=CohortConfig(records="d.csv", splits="s.csv"))
    assert list(config.to_dict()) == SECTIONS
    assert config.learner is None
    assert config.run == RunConfig() and config.loaders == LoadersConfig()
    assert config.trainer == TrainerConfig() and config.evaluation == EvaluationConfig()


def test_a_config_takes_field_writes(tmp_path: Path):
    # Python users (and the benchmarks) change a config by assigning its fields.
    config = experiment("d.csv", "s.csv", aggregator="abmil")
    config.run = RunConfig(output_dir="elsewhere", seed=4)
    config.trainer.max_steps = 100
    config.loaders.batch_size = 8
    config.learner.params["optimizer"] = Component("adamw", {"lr": 3e-4})
    config.to_yaml(tmp_path / "config.yaml")
    saved = ExperimentConfig.from_yaml(tmp_path / "config.yaml")
    assert saved.run == RunConfig(output_dir="elsewhere", seed=4)
    # A step budget written over the default epoch budget replaces it, as at construction.
    assert (saved.trainer.max_steps, saved.trainer.max_epochs) == (100, None)
    assert saved.loaders.batch_size == 8
    assert saved.learner.params["optimizer"] == Component("adamw", {"lr": 3e-4})


# --- hydration -------------------------------------------------------------------------


def test_yaml_hydrates_every_section_into_its_dataclass(tmp_path: Path):
    config = ExperimentConfig.from_yaml(_write_yaml(tmp_path, ANNOTATED_SEGMENTATION))

    assert config.run == RunConfig(output_dir="runs/seg", seed=3)
    assert config.cohort == CohortConfig(records="slides.csv", splits="splits.csv", tune_is_test=True)
    features = config.features
    assert features.encoder == EncoderLeaf("virchow2", {"output_variant": "cls"})
    assert isinstance(features.extraction, slide2vec.PreprocessingConfig)
    assert features.extraction.tiling == TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=512, tolerance=0.1, overlap=0.0, mask_backend="openslide"
    )
    assert features.extraction.masks == MaskSamplingConfig(
        pixel_mapping={"background": 0, "tumor": 1, "necrosis": 2},
        min_coverage={"tumor": 0.05, "necrosis": 0.05},
        output_mode="merged",
    )
    # A block left out keeps slide2vec's own default.
    assert features.extraction.segmentation == slide2vec.PreprocessingConfig().segmentation
    assert features.extraction.adaptive_batching is True
    assert (features.execution.precision, features.execution.batch_size) == ("fp16", 8)
    assert features.dense == slide2vec.DenseOptions(
        spacing_um=0.5, target_size=512, tolerance=0.1, window_size=224, overlap=0.5
    )
    assert features.cache == CacheOptions(dtype="fp16")
    assert features.spacing_policy == "native_if_coarser"
    assert config.loaders == LoadersConfig(batch_size=4, num_workers=2, sampler=Component("class_conditioned"))
    learner = config.learner
    assert isinstance(learner, LearnerConfig) and learner.name == "dense"
    assert learner.params["decoder"] == Component("lightweight_conv")
    assert learner.params["head"] == Component(
        "segmentation", {"classes": {"tumor": [1], "necrosis": [2]}, "ignore": [0]}
    )
    assert learner.params["optimizer"] == Component("adamw", {"lr": 3.0e-4})
    assert learner.params["scheduler"] == [Component("cosine")]
    assert config.trainer.callbacks == [Component("checkpoint", {"monitor": "mean_dice", "mode": "max"})]
    assert config.evaluation == EvaluationConfig(metrics=["mean_dice"], save_overlays=False)


def test_a_config_round_trips_through_its_yaml(tmp_path: Path):
    config = ExperimentConfig.from_yaml(_write_yaml(tmp_path, ANNOTATED_SEGMENTATION))
    path = tmp_path / "saved.yaml"
    config.to_yaml(path)
    assert ExperimentConfig.from_yaml(path) == config
    assert yaml.safe_load(path.read_text()) == config.to_dict()


def test_a_python_config_and_its_yaml_are_one_config(tmp_path: Path):
    from tests.extraction_configs import extraction

    config = experiment(
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
        aggregator={"name": "abmil", "params": {"hidden_dim": 64}},
        trainer={"max_steps": 100},
        run={"folds": (0, 1), "run_id": "r"},
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    assert ExperimentConfig.from_yaml(path) == config


def test_an_unknown_key_names_the_section_and_its_keys(tmp_path: Path):
    with pytest.raises(ValueError, match=r"ExperimentConfig has no key\(s\) \['training'\]"):
        ExperimentConfig.from_dict({"cohort": {"records": "d", "splits": "s"}, "training": {}})
    with pytest.raises(ValueError, match=r"ExperimentConfig.trainer has no key\(s\) \['epochs'\]"):
        ExperimentConfig.from_dict({"cohort": {"records": "d", "splits": "s"}, "trainer": {"epochs": 3}})
    with pytest.raises(ValueError, match=r"features.extraction.tiling has no key"):
        ExperimentConfig.from_dict(
            {"cohort": {"records": "d", "splits": "s"}, "features": {"extraction": {"tiling": {"spacing": 1}}}}
        )


def test_overrides_merge_into_the_yaml_sections(tmp_path: Path):
    path = _write_yaml(tmp_path, ANNOTATED_SEGMENTATION)
    config = ExperimentConfig.from_yaml(
        path, overrides={"run": {"output_dir": "elsewhere"}, "features": {"cache": {"root_dir": "/cache"}}}
    )
    assert config.run.output_dir == "elsewhere" and config.run.seed == 3
    assert config.features.cache == CacheOptions(root_dir="/cache", dtype="fp16")


# --- {name, params} leaves -------------------------------------------------------------


def test_a_leaf_is_a_name_or_a_name_params_mapping():
    assert Component.of("abmil") == Component("abmil")
    assert Component.of({"name": "abmil", "params": {"hidden_dim": 8}}) == Component("abmil", {"hidden_dim": 8})
    assert Component.of({"name": "abmil", "params": None}) == Component("abmil")
    with pytest.raises(ValueError, match="name, params"):
        Component.of({"name": "abmil", "hidden_dim": 8})
    with pytest.raises(ValueError, match="non-empty name"):
        Component("")
    with pytest.raises(TypeError, match="must be a mapping"):
        Component("abmil", params=[1])  # type: ignore[arg-type]


def test_the_learner_normalizes_its_leaves():
    learner = LearnerConfig(
        "mil",
        {"aggregator": "abmil", "head": {"name": "regression"}, "optimizer": "sgd", "scheduler": "none", "loss": None},
    )
    assert learner.params["aggregator"] == Component("abmil")
    assert learner.params["head"] == Component("regression")
    assert learner.params["optimizer"] == Component("sgd")
    assert learner.params["scheduler"] == []
    learner = LearnerConfig(
        "mil", {"scheduler": ["cosine", {"name": "constant", "params": {"factor": 0.5}, "interval": "step"}]}
    )
    assert learner.params["scheduler"] == [
        Component("cosine"),
        {"name": "constant", "params": {"factor": 0.5}, "interval": "step"},
    ]
    assert LearnerConfig().name == "mil"


def test_the_encoder_leaf_is_validated_by_the_encoders_registry():
    assert FeaturesConfig(encoder="uni2").encoder == EncoderLeaf("uni2")
    with pytest.raises(ValueError, match="Unknown encoder name 'nope'"):
        FeaturesConfig(encoder="nope")
    with pytest.raises(ValueError, match="no param"):
        FeaturesConfig(encoder=EncoderLeaf("uni2", {"variant": "cls"}))
    leaf = FeaturesConfig(encoder={"name": "uni2", "params": {"output_variant": "cls"}}).encoder
    assert leaf.output_variant == "cls" and leaf.allow_non_recommended_settings is False


# --- section-local rules ---------------------------------------------------------------


def test_features_validate_their_kind_policy_and_upstream_types():
    with pytest.raises(ValueError, match="features.kind"):
        FeaturesConfig(kind="streamed")
    with pytest.raises(ValueError, match="features.spacing_policy"):
        FeaturesConfig(spacing_policy="nearest")
    with pytest.raises(TypeError, match="features.extraction must be a slide2vec"):
        FeaturesConfig(extraction={"tiling": None})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_identity_mismatch"):
        CacheOptions(on_identity_mismatch="ignore")


def test_the_trainer_takes_one_budget():
    assert TrainerConfig().max_epochs == 50
    # A step budget replaces the epoch budget, as it does in the Trainer.
    assert TrainerConfig(max_steps=100) == TrainerConfig(max_epochs=None, max_steps=100)
    with pytest.raises(ValueError, match="needs a budget"):
        TrainerConfig(max_epochs=None)
    with pytest.raises(ValueError, match="max_steps"):
        TrainerConfig(max_steps=0)
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        TrainerConfig(accumulate_grad_batches=0)
    with pytest.raises(ValueError, match="precision"):
        TrainerConfig(precision="fp16")
    assert TrainerConfig().callbacks == [Component("checkpoint"), Component("early_stopping")]


def test_run_folds_need_a_shared_run_dir():
    assert RunConfig(folds=[1, 0], run_id="r").folds == (1, 0)
    with pytest.raises(ValueError, match="run.folds needs run.run_id"):
        RunConfig(folds=(0,))
    with pytest.raises(ValueError, match="distinct non-negative"):
        RunConfig(folds=(0, 0), resume=True)


def test_the_cohort_collapses_only_onto_patients():
    assert CohortConfig(records="d", splits="s", collapse="patient_id").collapse == "patient_id"
    with pytest.raises(ValueError, match="cohort.collapse"):
        CohortConfig(records="d", splits="s", collapse="site")


def test_loaders_validate_their_counts():
    with pytest.raises(ValueError, match="loaders.batch_size"):
        LoadersConfig(batch_size=0)
    with pytest.raises(ValueError, match="loaders.num_workers"):
        LoadersConfig(num_workers=-1)
    assert LoadersConfig(augment="segmentation").augment == Component("segmentation")


def test_evaluation_holds_crom_a_and_attention_heatmap_options():
    options = CRoMaOptions(confounder_column="center", split="test")
    assert EvaluationConfig(representation=options).representation == options
    with pytest.raises(ValueError, match="representation.m must be exactly 5"):
        CRoMaOptions(confounder_column="center", split="test", m=3)
    with pytest.raises(ValueError, match="representation.split"):
        CRoMaOptions(confounder_column="center", split="")
    assert EvaluationConfig(attention_heatmaps={"alpha": 0.3}).attention_heatmaps == {"alpha": 0.3}
    with pytest.raises(ValueError, match="attention_heatmaps has no key"):
        EvaluationConfig(attention_heatmaps={"colormap": "jet"})


def test_sections_reject_the_wrong_type():
    with pytest.raises(TypeError, match="run must be a RunConfig"):
        ExperimentConfig(cohort=CohortConfig(records="d", splits="s"), run={"seed": 1})  # type: ignore[arg-type]


def test_a_saved_config_keeps_slide2vec_s_dense_and_execution_options(tmp_path: Path):
    config = experiment(
        decoder="lightweight_conv",
        head="segmentation",
        head_params={"num_classes": 2},
        features=FeaturesConfig(
            encoder="uni2",
            dense=slide2vec.DenseOptions(spacing_um=0.5, target_size=448, window_size=224, feature_kind="cls_attention"),
            execution=slide2vec.ExecutionOptions(batch_size=4, precision="bf16", prefetch_factor=2),
            extraction=slide2vec.PreprocessingConfig(segmentation=SegmentationConfig(method="otsu", downsample=16)),
        ),
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.features.dense == config.features.dense
    assert loaded.features.execution == config.features.execution
    assert loaded.features.extraction.segmentation == SegmentationConfig(method="otsu", downsample=16)
    assert replace(loaded, run=config.run) == config
