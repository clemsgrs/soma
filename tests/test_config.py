"""Tests for soma.config — frozen dataclass configurations."""

from dataclasses import FrozenInstanceError, fields, replace
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
    MasksConfig,
    PipelineConfig,
    PreprocessingConfig,
    SamplingConfig,
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
    assert cfg.min_coverage == {"tissue": 0.1}
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
    # masks/sampling now live under preprocessing (#109); accept them as top-level
    # kwargs here for test convenience and fold them into a PreprocessingConfig.
    masks = overrides.pop("masks", None)
    sampling = overrides.pop("sampling", None)
    preprocessing = overrides.pop("preprocessing", None)
    if masks is not None or sampling is not None:
        preprocessing = replace(
            preprocessing if preprocessing is not None else PreprocessingConfig(),
            masks=masks,
            sampling=sampling,
        )
    kwargs = dict(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation"),
    )
    if preprocessing is not None:
        kwargs["preprocessing"] = preprocessing
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def test_segmentation_config_valid():
    cfg = _seg_config()
    assert cfg.dataset_type == "segmentation"
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.aggregator is None


def test_segmentation_requires_decoder():
    with pytest.raises(ValueError, match="requires either a decoder .* or a pixel_classifier"):
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


# --- A2: segmentation slide-manifest ingestion (masks:/sampling:) ---

_PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2, "necrosis": 3}


def test_masks_config_valid_defaults():
    masks = MasksConfig(pixel_mapping=_PIXEL_MAPPING, min_coverage={"tumor": 0.1, "stroma": 0.5})
    assert masks.colors is None
    sampling = SamplingConfig()
    assert (sampling.strategy, sampling.output_mode) == ("joint", "merged")


def test_masks_config_accepts_background_free_vocabulary():
    # No reserved-name rule: a background-free vocabulary like {tumor: 2} is accepted,
    # with structural validation still applied.
    masks = MasksConfig(pixel_mapping={"tumor": 2}, min_coverage={"tumor": 0.5})
    assert masks.pixel_mapping == {"tumor": 2}
    assert masks.min_coverage == {"tumor": 0.5}


def test_masks_config_rejects_empty_pixel_mapping():
    with pytest.raises(ValueError, match="non-empty"):
        MasksConfig(pixel_mapping={})


def test_masks_config_rejects_duplicate_pixel_values():
    with pytest.raises(ValueError, match="unique pixel values"):
        MasksConfig(pixel_mapping={"tumor": 1, "stroma": 1})


def test_masks_config_min_coverage_must_be_subset():
    with pytest.raises(ValueError, match="min_coverage references labels absent"):
        MasksConfig(pixel_mapping=_PIXEL_MAPPING, min_coverage={"glands": 0.1})


def test_masks_config_min_coverage_range():
    with pytest.raises(ValueError, match=r"min_coverage\['tumor'\] must be in \[0, 1\]"):
        MasksConfig(pixel_mapping=_PIXEL_MAPPING, min_coverage={"tumor": 1.5})


def test_masks_config_colors_must_be_subset_and_rgb():
    with pytest.raises(ValueError, match="colors references labels absent"):
        MasksConfig(pixel_mapping=_PIXEL_MAPPING, colors={"glands": [1, 2, 3]})
    with pytest.raises(ValueError, match=r"colors\['tumor'\] must be None or a length-3 RGB"):
        MasksConfig(pixel_mapping=_PIXEL_MAPPING, colors={"tumor": [1, 2]})


def test_sampling_config_rejects_unknown_strategy_and_mode():
    with pytest.raises(ValueError, match="sampling.strategy must be 'joint' or 'independent'"):
        SamplingConfig(strategy="greedy")
    with pytest.raises(ValueError, match="sampling.output_mode must be 'merged' or 'per_annotation'"):
        SamplingConfig(output_mode="single")


def test_segmentation_slide_manifest_config_valid():
    cfg = _seg_config(
        masks=MasksConfig(pixel_mapping=_PIXEL_MAPPING, min_coverage={"tumor": 0.1}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
    )
    # masks/sampling now live under preprocessing (#109).
    assert cfg.preprocessing.masks.pixel_mapping["necrosis"] == 3
    assert cfg.preprocessing.sampling.output_mode == "merged"


def test_masks_and_sampling_live_under_preprocessing():
    """masks/sampling are PreprocessingConfig fields, not top-level PipelineConfig fields (#109)."""
    pipeline_field_names = {f.name for f in fields(PipelineConfig)}
    assert "masks" not in pipeline_field_names
    assert "sampling" not in pipeline_field_names
    preprocessing_field_names = {f.name for f in fields(PreprocessingConfig)}
    assert "masks" in preprocessing_field_names
    assert "sampling" in preprocessing_field_names


def test_pipeline_config_rejects_top_level_masks_kwarg():
    """No backward-compat top-level masks= on PipelineConfig (clean break, #109)."""
    with pytest.raises(TypeError):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="segmentation",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="segmentation"),
            masks=MasksConfig(pixel_mapping=_PIXEL_MAPPING),
        )


def test_top_level_masks_sampling_yaml_rejected(tmp_path: Path):
    """A top-level masks:/sampling: block in YAML is no longer accepted (#109)."""
    path = tmp_path / "cfg.yaml"
    save_config(_seg_config(), path)
    raw = yaml.safe_load(path.read_text())
    raw["masks"] = {"pixel_mapping": {"background": 0, "tumor": 1}}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(path)


def test_masks_valid_for_slide_dataset():
    """AC1/AC2: a masks block is valid on a dataset_type='slide' dataset (the
    annotation-restricted merged bag, #110), wired through the ordinary
    featurizer→aggregator→predictor with an ordinary slide-level classification task — the
    masks block restricts tile selection only; the MIL aggregator + task head are untouched."""
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="slide",
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification"),
        preprocessing=PreprocessingConfig(
            masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
        ),
    )
    assert cfg.preprocessing.masks.pixel_mapping == {"background": 0, "tumor": 1}
    # The aggregator + task head are the ordinary slide-level pair, not special-cased by masks.
    assert cfg.aggregator.name == "abmil"
    assert cfg.task.name == "binary_classification"


def test_masks_rejected_for_tile_dataset():
    """AC5: a masks block on a dataset_type='tile' dataset raises — patch manifests have no
    annotation-sampling step."""
    with pytest.raises(ValueError, match="masks: .* dataset_type"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="tile",
            task=TaskConfig(name="binary_classification"),
            preprocessing=PreprocessingConfig(
                masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
            ),
        )


def test_per_annotation_output_mode_rejected_for_slide():
    """AC5: output_mode='per_annotation' on a slide dataset raises, pointing at #86."""
    with pytest.raises(ValueError, match="per_annotation.*#86"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            task=TaskConfig(name="binary_classification"),
            preprocessing=PreprocessingConfig(
                masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
                sampling=SamplingConfig(output_mode="per_annotation"),
            ),
        )


def test_masks_valid_for_patient_dataset():
    """AC1/AC2 (#111): a masks block is valid on a dataset_type='patient' dataset — the
    annotation-restricted merged bag extends from 'slide' to 'patient'. Every slide is tiled
    to its restricted merged bag (tiling/selection identical to 'slide'); patient-level
    aggregation then consumes those restricted slide bags. A patient pipeline carries no
    trainable aggregator (it uses a pretrained patient encoder), so the masks block restricts
    tile selection only; the patient-encoder + task head are untouched."""
    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="patient",
        task=TaskConfig(name="binary_classification"),
        preprocessing=PreprocessingConfig(
            masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
        ),
    )
    assert cfg.preprocessing.masks.pixel_mapping == {"background": 0, "tumor": 1}
    assert cfg.dataset_type == "patient"
    # Patient pipelines never carry a trainable aggregator; the masks block does not change that.
    assert cfg.aggregator is None


def test_per_annotation_output_mode_rejected_for_patient():
    """AC4 (#111): output_mode='per_annotation' stays rejected on a patient dataset, pointing
    at #86 (the bag path only supports the merged compartment bag)."""
    with pytest.raises(ValueError, match="per_annotation.*#86"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="patient",
            task=TaskConfig(name="binary_classification"),
            preprocessing=PreprocessingConfig(
                masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
                sampling=SamplingConfig(output_mode="per_annotation"),
            ),
        )


def test_sampling_requires_masks():
    with pytest.raises(ValueError, match="sampling: requires a masks: block"):
        _seg_config(sampling=SamplingConfig())


def test_per_annotation_output_mode_deferred():
    with pytest.raises(ValueError, match="per_annotation.*deferred.*#86"):
        _seg_config(
            masks=MasksConfig(pixel_mapping=_PIXEL_MAPPING),
            sampling=SamplingConfig(output_mode="per_annotation"),
        )


def test_masks_sampling_round_trip_through_yaml(tmp_path: Path):
    cfg = _seg_config(
        masks=MasksConfig(
            pixel_mapping=_PIXEL_MAPPING,
            min_coverage={"tumor": 0.1, "stroma": 0.5},
            colors={"background": None, "tumor": [255, 0, 0], "stroma": [0, 255, 0], "necrosis": [0, 0, 255]},
        ),
        sampling=SamplingConfig(strategy="independent", output_mode="merged"),
    )
    path = tmp_path / "seg-slide-manifest.yaml"
    save_config(cfg, path)
    # masks/sampling are serialized under preprocessing (#109).
    raw = yaml.safe_load(path.read_text())
    assert "masks" not in raw and "sampling" not in raw
    assert "masks" in raw["preprocessing"] and "sampling" in raw["preprocessing"]
    loaded = load_config(path)
    assert loaded.preprocessing.masks.pixel_mapping == _PIXEL_MAPPING
    assert loaded.preprocessing.masks.min_coverage == {"tumor": 0.1, "stroma": 0.5}
    assert loaded.preprocessing.masks.colors["tumor"] == [255, 0, 0]
    assert loaded.preprocessing.masks.colors["background"] is None
    assert (loaded.preprocessing.sampling.strategy, loaded.preprocessing.sampling.output_mode) == (
        "independent",
        "merged",
    )


def test_masks_accepts_hs2p_list_of_single_entry_mappings(tmp_path: Path):
    """An hs2p-style masks block (list of single-entry mappings) pastes in unchanged."""
    path = tmp_path / "cfg.yaml"
    base = _seg_config()
    save_config(base, path)
    raw = yaml.safe_load(path.read_text())
    raw["preprocessing"]["masks"] = {
        "pixel_mapping": [{"background": 0}, {"tumor": 1}, {"stroma": 2}],
        "min_coverage": [{"tumor": 0.1}],
    }
    path.write_text(yaml.safe_dump(raw))
    loaded = load_config(path)
    assert loaded.preprocessing.masks.pixel_mapping == {"background": 0, "tumor": 1, "stroma": 2}
    assert loaded.preprocessing.masks.min_coverage == {"tumor": 0.1}


def test_pixel_classifier_config_round_trips_through_yaml(tmp_path: Path):
    from soma.config import AttentionConfig, PixelClassifierConfig, PreprocessingConfig

    cfg = _seg_config(
        decoder=None,
        pixel_classifier=PixelClassifierConfig(name="xgboost", params={"n_estimators": 100}),
        preprocessing=PreprocessingConfig(
            attention=AttentionConfig(blocks=[-1, -2], include_registers=True)
        ),
    )
    # cross-defaulted feature_kind survives the roundtrip.
    assert cfg.preprocessing.feature_kind == "cls_attention"
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.pixel_classifier.name == "xgboost"
    assert loaded.pixel_classifier.params == {"n_estimators": 100}
    assert loaded.decoder is None
    assert loaded.preprocessing.feature_kind == "cls_attention"
    assert loaded.preprocessing.attention.blocks == (-1, -2)
    assert loaded.preprocessing.attention.include_registers is True


def test_composite_round_trips_through_yaml(tmp_path: Path):
    from soma.config import CompositeConfig, EncoderMemberConfig, PixelClassifierConfig

    cfg = _seg_config(
        decoder=None,
        pixel_classifier=PixelClassifierConfig(name="xgboost"),
        composite=CompositeConfig(
            encoders=[
                EncoderMemberConfig(name="uni", feature_kind="cls_attention"),
                EncoderMemberConfig(name="phikon", feature_kind="patch_features"),
            ],
        ),
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.composite is not None and len(loaded.composite.encoders) == 2
    assert [m.name for m in loaded.composite.encoders] == ["uni", "phikon"]
    assert loaded.composite.encoders[1].feature_kind == "patch_features"
    # Pixel-classifier consumer → target mode; member_norm cross-defaults by feature_kind.
    assert loaded.composite.concat_resolution == "target"
    assert loaded.composite.encoders[0].member_norm == "none"  # cls_attention
    assert loaded.composite.encoders[1].member_norm == "l2"  # patch_features


def test_composite_decoder_path_cross_defaults_to_grid(tmp_path: Path):
    from soma.config import CompositeConfig, EncoderMemberConfig

    cfg = _seg_config(
        composite=CompositeConfig(
            encoders=[EncoderMemberConfig(name="uni"), EncoderMemberConfig(name="phikon")]
        ),
    )
    assert cfg.composite.concat_resolution == "grid"
    assert all(m.feature_kind == "patch_features" for m in cfg.composite.encoders)
    assert all(m.member_norm == "l2" for m in cfg.composite.encoders)


def test_composite_grid_size_and_explicit_resolution_round_trip(tmp_path: Path):
    from soma.config import CompositeConfig, EncoderMemberConfig

    cfg = _seg_config(
        composite=CompositeConfig(
            encoders=[EncoderMemberConfig(name="uni", member_norm="layernorm")],
            concat_resolution="target",
            concat_grid_size=(37, 37),
        ),
    )
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.composite.concat_resolution == "target"
    assert loaded.composite.concat_grid_size == (37, 37)
    assert loaded.composite.encoders[0].member_norm == "layernorm"


def test_composite_only_yaml_loads_without_tripping_xor(tmp_path: Path):
    # The real regression: a documented `composite:` config (no `encoder:` key) must load.
    # Previously the bundled default `encoder: uni2` merged in and tripped the XOR check.
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "segmentation",
        },
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
        "pixel_classifier": {"name": "xgboost"},
        "composite": {
            "encoders": [
                {"name": "uni", "feature_kind": "cls_attention"},
                {"name": "phikon", "feature_kind": "patch_features"},
            ],
        },
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(raw, f)
    loaded = load_config(yaml_path)
    assert loaded.encoder is None
    assert [m.name for m in loaded.composite.encoders] == ["uni", "phikon"]


def test_composite_yaml_rejects_unknown_keys(tmp_path: Path):
    raw = {
        "data": {
            "dataset_csv": "dataset.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "segmentation",
        },
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
        "pixel_classifier": {"name": "xgboost"},
        "composite": {
            "encoders": [{"name": "uni"}],
            "concat_resoluton": "target",
        },
    }
    yaml_path = tmp_path / "config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(raw, f)

    with pytest.raises(ValueError, match="unsupported keys.*concat_resoluton"):
        load_config(yaml_path)


def test_composite_enabled_for_detection():
    from soma.config import CompositeConfig, EncoderMemberConfig

    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="detection",
        decoder=DecoderConfig(name="lightweight_conv"),
        composite=CompositeConfig(encoders=[EncoderMemberConfig(name="uni")]),
        task=TaskConfig(name="detection", params={"num_classes": 2}),
    )
    assert cfg.composite.concat_resolution == "grid"
    assert cfg.composite.encoders[0].feature_kind == "patch_features"


def test_encoder_xor_composite():
    from soma.config import CompositeConfig, EncoderConfig, EncoderMemberConfig, PixelClassifierConfig

    with pytest.raises(ValueError, match="XOR"):
        _seg_config(
            decoder=None,
            pixel_classifier=PixelClassifierConfig(name="xgboost"),
            encoder=EncoderConfig(name="uni"),
            composite=CompositeConfig(encoders=[EncoderMemberConfig(name="phikon")]),
        )


def test_composite_rejected_for_non_dense_dataset():
    from soma.config import CompositeConfig, EncoderMemberConfig

    with pytest.raises(ValueError, match="only supported for dataset_type"):
        PipelineConfig(
            dataset_csv="data.csv",
            splits_csv="splits.csv",
            output_root="out",
            dataset_type="slide",
            composite=CompositeConfig(encoders=[EncoderMemberConfig(name="uni")]),
            task=TaskConfig(name="binary_classification"),
        )


def test_composite_live_is_hard_error():
    from soma.config import CompositeConfig, EncoderMemberConfig

    with pytest.raises(ValueError, match="cached-only"):
        _seg_config(
            feature_mode="live",
            composite=CompositeConfig(encoders=[EncoderMemberConfig(name="uni")]),
        )


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
    # Clean roundtrip: params are exactly what was set (no default-merge bleed now that
    # the bundled aggregation default is neutral).
    assert loaded.aggregator.params == original.aggregator.params
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
    # Neutral defaults: no baked-in encoder / aggregator / metrics (set those per run).
    assert loaded.encoder is None
    assert loaded.aggregator is None
    assert loaded.task.name == "binary_classification"
    assert loaded.evaluation.metrics == []
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
    assert loaded.encoder is None  # blank encoder section + neutral default → no encoder
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
