from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from soma.config import (
    AggregatorConfig,
    AugmentationConfig,
    CacheConfig,
    CompositeConfig,
    DecoderConfig,
    EncoderConfig,
    EvalConfig,
    EncoderMemberConfig,
    HeatmapConfig,
    MasksConfig,
    NormalizationConfig,
    PipelineConfig,
    PixelClassifierConfig,
    PreprocessingConfig,
    ProjectionConfig,
    SamplingConfig,
    TaskConfig,
    TrainingConfig,
    config_yaml_dict,
)
from soma.output_layout import (
    ExperimentSpec,
    build_experiment_spec,
    canonical_experiment_payload,
    capture_environment,
    create_run_metadata,
    read_test_results,
    register_test_result,
    resolve_managed_output_paths,
    update_run_index,
    write_run_metadata,
)
from soma.output_layout import test_identity_digest as compute_test_digest


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_pipeline_config(tmp_path: Path, **overrides) -> PipelineConfig:
    dataset_csv = _write_csv(
        tmp_path / "dataset.csv",
        "sample_id,image_path,label\ns0,/slides/s0.svs,tumor\n",
    )
    splits_csv = _write_csv(
        tmp_path / "splits.csv",
        "fold,sample_id,split\n0,s0,train\n",
    )
    defaults = dict(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "outputs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(seed=7, epochs=10, learning_rate=1e-4),
        tags=["baseline"],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_segmentation_config(tmp_path: Path, **overrides) -> PipelineConfig:
    dataset_csv = _write_csv(
        tmp_path / "segmentation_dataset.csv",
        "sample_id,image_path,label_mask_path\ns0,/slides/s0.png,/masks/s0.png\n",
    )
    splits_csv = _write_csv(
        tmp_path / "segmentation_splits.csv",
        "fold,sample_id,split\n0,s0,train\n",
    )
    defaults = dict(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "outputs",
        dataset_type="segmentation",
        preprocessing=PreprocessingConfig(requested_tile_size_px=256, requested_spacing_um=0.5),
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=None,
        decoder=DecoderConfig(name="lightweight_conv", params={"hidden_dim": 64}),
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
        training=TrainingConfig(seed=7, epochs=10, learning_rate=1e-4),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_capture_environment_stamps_exactly_three_fields():
    # Bounded provenance (issue #213): EXACTLY {soma, torch, cuda} — nothing further.
    env = capture_environment()
    assert set(env) == {"soma", "torch", "cuda"}
    assert env["soma"]  # soma __version__ is always resolvable


def test_run_yaml_stamps_bounded_environment(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    experiment = build_experiment_spec(config)
    run_dir = tmp_path / "run"
    metadata = create_run_metadata(
        config=config,
        experiment=experiment,
        run_dir=run_dir,
        run_id="r0",
        status="running",
    )
    assert set(metadata.environment) == {"soma", "torch", "cuda"}

    write_run_metadata(run_dir, metadata)
    payload = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert set(payload["environment"]) == {"soma", "torch", "cuda"}
    # No deeper env / GPU-model / clean-tree fields leaked into the stamp.
    assert "gpu" not in payload["environment"]


def test_canonical_experiment_payload_omits_seed(tmp_path: Path):
    cfg_a = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=1, epochs=10, learning_rate=1e-4))
    cfg_b = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=999, epochs=10, learning_rate=1e-4))

    payload_a = canonical_experiment_payload(cfg_a)
    payload_b = canonical_experiment_payload(cfg_b)

    assert payload_a == payload_b
    assert payload_a["training"]["epochs"] == 10
    assert "seed" not in payload_a["training"]


def test_mirror_root_does_not_change_experiment_identity(tmp_path: Path):
    local_only = _make_pipeline_config(tmp_path)
    mirrored = _make_pipeline_config(tmp_path, mirror_root=tmp_path / "shared")

    assert canonical_experiment_payload(mirrored) == canonical_experiment_payload(local_only)
    assert build_experiment_spec(mirrored).experiment_id == build_experiment_spec(
        local_only
    ).experiment_id


def test_canonical_experiment_payload_omits_default_checkpoint_selection(tmp_path: Path):
    """Guarded identity: the default `best` must not perturb legacy experiment ids."""
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert "checkpoint_selection" not in payload["training"]


def test_canonical_experiment_payload_omits_default_roi_batch_sampling(tmp_path: Path):
    """An opt-out run keeps its legacy experiment identity payload."""
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert "roi_batch_sampling" not in payload["training"]
    assert "class_request_ratios" not in payload["training"]
    assert "roi_draws_per_epoch" not in payload["training"]


def test_null_class_request_ratios_are_explicit_in_sampling_identity(tmp_path: Path):
    config = _make_segmentation_config(
        tmp_path,
        training=TrainingConfig(
            batch_size=4,
            roi_batch_sampling="class_conditioned",
            roi_draws_per_epoch=8,
        ),
    )

    assert canonical_experiment_payload(config)["training"][
        "class_request_ratios"
    ] is None


def test_explicit_class_request_ratios_define_experiment_identity(tmp_path: Path):
    equal = _make_segmentation_config(
        tmp_path,
        training=TrainingConfig(
            batch_size=4,
            roi_batch_sampling="class_conditioned",
            class_request_ratios=[1, 1],
            roi_draws_per_epoch=8,
        ),
    )
    weighted = _make_segmentation_config(
        tmp_path,
        training=TrainingConfig(
            batch_size=4,
            roi_batch_sampling="class_conditioned",
            class_request_ratios=[1, 2],
            roi_draws_per_epoch=8,
        ),
    )

    assert canonical_experiment_payload(equal)["training"][
        "class_request_ratios"
    ] == [1, 1]
    assert build_experiment_spec(equal).experiment_id != build_experiment_spec(
        weighted
    ).experiment_id


def test_segmentation_confusion_evidence_is_identity_neutral_but_persisted(tmp_path: Path):
    ordinary = _make_pipeline_config(tmp_path)
    with_evidence = _make_pipeline_config(
        tmp_path,
        evaluation=EvalConfig(
            save_segmentation_confusion_evidence=True,
        ),
    )

    assert "save_segmentation_confusion_evidence" not in canonical_experiment_payload(
        ordinary
    )["evaluation"]
    assert build_experiment_spec(with_evidence).experiment_id == build_experiment_spec(
        ordinary
    ).experiment_id
    assert config_yaml_dict(with_evidence)["evaluation"][
        "save_segmentation_confusion_evidence"
    ] is True


def test_pipeline_config_rejects_roi_batch_sampling_outside_cached_segmentation(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="cached neural segmentation"):
        _make_pipeline_config(
            tmp_path,
            training=TrainingConfig(
                batch_size=4,
                roi_batch_sampling="class_conditioned",
                roi_draws_per_epoch=8,
            ),
        )


def test_build_experiment_spec_distinguishes_checkpoint_selection(tmp_path: Path):
    best = _make_pipeline_config(tmp_path)
    last = _make_pipeline_config(
        tmp_path,
        training=TrainingConfig(
            seed=7, epochs=10, learning_rate=1e-4, checkpoint_selection="last", patience=None
        ),
    )

    last_payload = canonical_experiment_payload(last)
    assert last_payload["training"]["checkpoint_selection"] == "last"
    assert build_experiment_spec(best).experiment_id != build_experiment_spec(last).experiment_id


def test_canonical_experiment_payload_omits_default_normalization(tmp_path: Path):
    """Guarded identity: `normalization: none` must not re-mint legacy experiment ids."""
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert "normalization" not in payload


def test_build_experiment_spec_distinguishes_normalization(tmp_path: Path):
    off = _make_pipeline_config(tmp_path)
    zscore = _make_pipeline_config(
        tmp_path, normalization=NormalizationConfig(method="zscore")
    )
    l2 = _make_pipeline_config(tmp_path, normalization=NormalizationConfig(method="l2"))

    assert canonical_experiment_payload(zscore)["normalization"] == {
        "method": "zscore",
        "eps": 1e-6,
    }
    ids = {
        build_experiment_spec(cfg).experiment_id for cfg in (off, zscore, l2)
    }
    assert len(ids) == 3


def test_canonical_experiment_payload_omits_default_projection(tmp_path: Path):
    """Guarded identity: `projection: none` must not re-mint legacy experiment ids."""
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert "projection" not in payload


def test_build_experiment_spec_distinguishes_projection(tmp_path: Path):
    off = _make_pipeline_config(tmp_path)
    pca = _make_pipeline_config(
        tmp_path, projection=ProjectionConfig(method="pca", target_dim=64)
    )
    random = _make_pipeline_config(
        tmp_path, projection=ProjectionConfig(method="random", target_dim=64)
    )
    narrower = _make_pipeline_config(
        tmp_path, projection=ProjectionConfig(method="pca", target_dim=32)
    )
    reseeded = _make_pipeline_config(
        tmp_path, projection=ProjectionConfig(method="random", target_dim=64, seed=1)
    )

    assert canonical_experiment_payload(pca)["projection"] == {
        "method": "pca",
        "target_dim": 64,
    }
    assert canonical_experiment_payload(random)["projection"] == {
        "method": "random",
        "target_dim": 64,
        "seed": 0,
    }
    ids = {
        build_experiment_spec(cfg).experiment_id
        for cfg in (off, pca, random, narrower, reseeded)
    }
    assert len(ids) == 5


def test_build_experiment_spec_uses_slug_and_short_hash(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)

    spec = build_experiment_spec(config)

    assert isinstance(spec, ExperimentSpec)
    assert spec.slug.startswith("dataset-uni2-abmil-binary-classification_")
    assert spec.experiment_dirname == spec.slug
    assert len(spec.short_hash) == 12
    assert spec.dataset_checksum
    assert spec.splits_checksum


def test_build_experiment_spec_distinguishes_dataset_type(tmp_path: Path):
    slide_config = _make_pipeline_config(tmp_path, aggregator=None, dataset_type="slide")
    tile_config = _make_pipeline_config(tmp_path, aggregator=None, dataset_type="tile")

    slide_spec = build_experiment_spec(slide_config)
    tile_spec = build_experiment_spec(tile_config)

    assert slide_spec.experiment_id != tile_spec.experiment_id


def test_build_experiment_spec_distinguishes_dense_decoder_settings(tmp_path: Path):
    shallow_decoder = _make_segmentation_config(
        tmp_path,
        decoder=DecoderConfig(name="lightweight_conv", params={"hidden_dim": 64}),
    )
    wider_decoder = _make_segmentation_config(
        tmp_path,
        decoder=DecoderConfig(name="lightweight_conv", params={"hidden_dim": 128}),
    )

    shallow_spec = build_experiment_spec(shallow_decoder)
    wider_spec = build_experiment_spec(wider_decoder)

    assert shallow_spec.experiment_id != wider_spec.experiment_id


def test_build_experiment_spec_distinguishes_dense_pixel_classifier_choices(tmp_path: Path):
    logistic = _make_segmentation_config(
        tmp_path,
        decoder=None,
        pixel_classifier=PixelClassifierConfig(name="logistic", params={"C": 1.0}),
    )
    random_forest = _make_segmentation_config(
        tmp_path,
        decoder=None,
        pixel_classifier=PixelClassifierConfig(name="random_forest", params={"n_estimators": 25}),
    )

    logistic_spec = build_experiment_spec(logistic)
    forest_spec = build_experiment_spec(random_forest)

    assert logistic_spec.experiment_id != forest_spec.experiment_id


def test_build_experiment_spec_distinguishes_dense_composite_model_choices(tmp_path: Path):
    l2_composite = _make_segmentation_config(
        tmp_path,
        encoder=None,
        composite=CompositeConfig(
            encoders=[EncoderMemberConfig(name="uni2", member_norm="l2")]
        ),
    )
    layernorm_composite = _make_segmentation_config(
        tmp_path,
        encoder=None,
        composite=CompositeConfig(
            encoders=[EncoderMemberConfig(name="uni2", member_norm="layernorm")]
        ),
    )

    l2_spec = build_experiment_spec(l2_composite)
    layernorm_spec = build_experiment_spec(layernorm_composite)

    assert l2_spec.experiment_id != layernorm_spec.experiment_id


def test_build_experiment_spec_distinguishes_evaluation_metrics(tmp_path: Path):
    auroc_config = _make_pipeline_config(tmp_path, evaluation=EvalConfig(metrics=["auroc"]))
    f1_config = _make_pipeline_config(tmp_path, evaluation=EvalConfig(metrics=["f1"]))

    auroc_spec = build_experiment_spec(auroc_config)
    f1_spec = build_experiment_spec(f1_config)

    assert auroc_spec.experiment_id != f1_spec.experiment_id


def test_build_experiment_spec_distinguishes_evaluation_probability_artifacts(tmp_path: Path):
    without_probabilities = _make_segmentation_config(
        tmp_path,
        evaluation=EvalConfig(metrics=["mean_dice"], save_segmentation_probabilities=False),
    )
    with_probabilities = _make_segmentation_config(
        tmp_path,
        evaluation=EvalConfig(metrics=["mean_dice"], save_segmentation_probabilities=True),
    )

    without_spec = build_experiment_spec(without_probabilities)
    with_spec = build_experiment_spec(with_probabilities)

    assert without_spec.experiment_id != with_spec.experiment_id


def test_build_experiment_spec_distinguishes_heatmap_artifact_settings(tmp_path: Path):
    disabled = _make_pipeline_config(tmp_path, heatmaps=HeatmapConfig(enabled=False))
    enabled = _make_pipeline_config(
        tmp_path,
        heatmaps=HeatmapConfig(enabled=True, cmap="viridis", alpha=0.75, blur_sigma=1.5),
    )

    disabled_spec = build_experiment_spec(disabled)
    enabled_spec = build_experiment_spec(enabled)

    assert disabled_spec.experiment_id != enabled_spec.experiment_id


def test_build_experiment_spec_distinguishes_mask_settings_via_preprocessing(tmp_path: Path):
    tumor_mask = _make_pipeline_config(
        tmp_path,
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            masks=MasksConfig(pixel_mapping={"tumor": 1}, min_coverage={"tumor": 0.2}),
        ),
    )
    stroma_mask = _make_pipeline_config(
        tmp_path,
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            masks=MasksConfig(pixel_mapping={"stroma": 2}, min_coverage={"stroma": 0.2}),
        ),
    )

    tumor_spec = build_experiment_spec(tumor_mask)
    stroma_spec = build_experiment_spec(stroma_mask)

    assert tumor_spec.experiment_id != stroma_spec.experiment_id


def test_build_experiment_spec_distinguishes_sampling_settings_via_preprocessing(tmp_path: Path):
    mask = MasksConfig(pixel_mapping={"tumor": 1}, min_coverage={"tumor": 0.2})
    joint_sampling = _make_pipeline_config(
        tmp_path,
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            masks=mask,
            sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        ),
    )
    independent_sampling = _make_pipeline_config(
        tmp_path,
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            masks=mask,
            sampling=SamplingConfig(strategy="independent", output_mode="merged"),
        ),
    )

    joint_spec = build_experiment_spec(joint_sampling)
    independent_spec = build_experiment_spec(independent_sampling)

    assert joint_spec.experiment_id != independent_spec.experiment_id


def test_build_experiment_spec_distinguishes_feature_mode(tmp_path: Path):
    cached = _make_segmentation_config(tmp_path, feature_mode="cached")
    live = _make_segmentation_config(tmp_path, feature_mode="live")

    cached_spec = build_experiment_spec(cached)
    live_spec = build_experiment_spec(live)

    assert cached_spec.experiment_id != live_spec.experiment_id


def test_build_experiment_spec_distinguishes_live_augmentation_choices(tmp_path: Path):
    no_augmentation = _make_segmentation_config(tmp_path, feature_mode="live")
    horizontal_flip = _make_segmentation_config(
        tmp_path,
        feature_mode="live",
        augmentation=AugmentationConfig(horizontal_flip=0.5),
    )

    no_aug_spec = build_experiment_spec(no_augmentation)
    flip_spec = build_experiment_spec(horizontal_flip)

    assert no_aug_spec.experiment_id != flip_spec.experiment_id


def test_build_experiment_spec_is_stable_for_equivalent_configurations(tmp_path: Path):
    first = _make_segmentation_config(
        tmp_path,
        decoder=DecoderConfig(
            name="lightweight_conv",
            params={"hidden_dim": 64, "num_groups": 8},
        ),
    )
    second = _make_segmentation_config(
        tmp_path,
        decoder=DecoderConfig(
            name="lightweight_conv",
            params={"num_groups": 8, "hidden_dim": 64},
        ),
    )

    first_spec = build_experiment_spec(first)
    repeated_first_spec = build_experiment_spec(first)
    second_spec = build_experiment_spec(second)

    assert first_spec.experiment_id == repeated_first_spec.experiment_id
    assert first_spec.experiment_id == second_spec.experiment_id


def test_build_experiment_spec_ignores_inactive_heatmap_rendering_settings(tmp_path: Path):
    default_disabled = _make_pipeline_config(tmp_path, heatmaps=HeatmapConfig(enabled=False))
    restyled_disabled = _make_pipeline_config(
        tmp_path,
        heatmaps=HeatmapConfig(enabled=False, cmap="viridis", alpha=0.75, blur_sigma=1.5),
    )

    default_spec = build_experiment_spec(default_disabled)
    restyled_spec = build_experiment_spec(restyled_disabled)

    assert default_spec.experiment_id == restyled_spec.experiment_id


def test_resolve_managed_output_paths_groups_same_experiment_and_changes_run_dir(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)

    first = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    second = resolve_managed_output_paths(config, run_id="2026-04-09_16-23-10__local")

    assert first.experiment_dir == second.experiment_dir
    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == first.experiment_dir / "runs"
    assert first.index_dir == Path(config.output_root) / "indexes"


def test_create_run_metadata_records_status_and_seed(tmp_path: Path):
    config = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=123, epochs=10))
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")

    metadata = create_run_metadata(
        config=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=layout.run_id,
        status="running",
    )

    assert metadata.seed == 123
    assert metadata.status == "running"
    assert metadata.experiment_id == layout.experiment.experiment_id
    assert metadata.resolved_output_dir == layout.run_dir.resolve()


def test_run_index_upserts_rows(tmp_path: Path):
    # The per-run index is append/upsert-safe and remains; the racy experiment-level
    # index writer was removed (ADR 0003) — the leaderboard rebuilds that projection by
    # scanning run dirs instead.
    config = _make_pipeline_config(tmp_path)
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    run = create_run_metadata(
        config=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=layout.run_id,
        status="running",
    )

    update_run_index(layout.index_dir / "runs.csv", run)
    completed = run.with_updates(status="completed")
    update_run_index(layout.index_dir / "runs.csv", completed)

    with (layout.index_dir / "runs.csv").open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))

    assert len(run_rows) == 1
    assert run_rows[0]["status"] == "completed"


def test_experiment_index_writer_is_removed():
    # ADR 0003 decision 3b: the racy unlocked read-modify-rewrite writer is gone for good.
    import soma.output_layout as output_layout

    assert not hasattr(output_layout, "update_experiment_index")


def test_read_csv_rows_handles_large_fields(tmp_path: Path):
    from soma.output_layout import _read_csv_rows

    path = tmp_path / "rows.csv"
    path.write_text("name,notes\nrow1,\"" + ("x" * 200_000) + "\"\n", encoding="utf-8")

    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(1024)
        rows = _read_csv_rows(path)
    finally:
        csv.field_size_limit(old_limit)

    assert rows == [{"name": "row1", "notes": "x" * 200_000}]


# --- test-invariant experiment identity (issue #247) --------------------------------


def _make_config_with_manifests(
    tmp_path: Path, *, dataset_csv_text: str, splits_csv_text: str, **overrides
) -> PipelineConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_csv = _write_csv(tmp_path / "dataset.csv", dataset_csv_text)
    splits_csv = _write_csv(tmp_path / "splits.csv", splits_csv_text)
    defaults = dict(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "outputs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(seed=7, epochs=10, learning_rate=1e-4),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


_TRAIN_TUNE_DATASET = (
    "sample_id,image_path,label\n"
    "s0,/slides/s0.svs,tumor\n"
    "s1,/slides/s1.svs,normal\n"
    "s2,/slides/s2.svs,tumor\n"
)
_TRAIN_TUNE_SPLITS = "fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,tune\n"


def test_experiment_id_is_invariant_to_added_test_rows(tmp_path: Path):
    # The headline acceptance criterion (#247): dropping in a new/official test set — extra
    # dataset rows + extra `test` split rows — must NOT change experiment_id.
    base = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
    )
    with_test = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET
        + "t0,/slides/t0.svs,tumor\nt1,/slides/t1.svs,normal\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,t0,test\n0,t1,test_official\n",
    )

    assert (
        build_experiment_spec(base).experiment_id
        == build_experiment_spec(with_test).experiment_id
    )


def test_experiment_id_is_invariant_to_altered_test_rows(tmp_path: Path):
    # Swapping which samples make up the test set (same train/tune) leaves the id fixed.
    local_test = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET + "t0,/slides/t0.svs,tumor\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,t0,test\n",
    )
    official_test = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET + "u9,/slides/u9.svs,normal\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,u9,test\n",
    )

    assert (
        build_experiment_spec(local_test).experiment_id
        == build_experiment_spec(official_test).experiment_id
    )
    # ...but the two test sets are distinguishable by their test-identity digest.
    assert compute_test_digest(local_test) != compute_test_digest(official_test)


def test_experiment_id_changes_when_a_train_row_changes(tmp_path: Path):
    base = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
    )
    altered_label = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET.replace(
            "s0,/slides/s0.svs,tumor", "s0,/slides/s0.svs,normal"
        ),
        splits_csv_text=_TRAIN_TUNE_SPLITS,
    )

    assert (
        build_experiment_spec(base).experiment_id
        != build_experiment_spec(altered_label).experiment_id
    )


def test_experiment_id_changes_when_a_train_tune_assignment_changes(tmp_path: Path):
    base = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
    )
    moved = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        # s2 moves from tune to train.
        splits_csv_text="fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,train\n",
    )

    assert (
        build_experiment_spec(base).experiment_id
        != build_experiment_spec(moved).experiment_id
    )


def test_overwrite_test_flag_is_not_part_of_identity(tmp_path: Path):
    # overwrite_test is an operational clobber-guard, never part of what the experiment is.
    off = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
        evaluation=EvalConfig(overwrite_test=False),
    )
    on = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
        evaluation=EvalConfig(overwrite_test=True),
    )

    assert build_experiment_spec(off).experiment_id == build_experiment_spec(on).experiment_id


def test_leaderboard_triple_is_test_invariant(tmp_path: Path):
    # The stored checksums that drive the leaderboard triple must be test-invariant too, so
    # the same model scored on two test sets stays in ONE comparable leaderboard group.
    local_test = _make_config_with_manifests(
        tmp_path / "a",
        dataset_csv_text=_TRAIN_TUNE_DATASET + "t0,/slides/t0.svs,tumor\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,t0,test\n",
    )
    official_test = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET + "u9,/slides/u9.svs,normal\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,u9,test\n",
    )

    local_spec = build_experiment_spec(local_test)
    official_spec = build_experiment_spec(official_test)
    assert local_spec.dataset_checksum == official_spec.dataset_checksum
    assert local_spec.splits_checksum == official_spec.splits_checksum


def test_run_metadata_records_test_and_file_provenance(tmp_path: Path):
    config = _make_config_with_manifests(
        tmp_path,
        dataset_csv_text=_TRAIN_TUNE_DATASET + "t0,/slides/t0.svs,tumor\n",
        splits_csv_text=_TRAIN_TUNE_SPLITS + "0,t0,test\n",
    )
    experiment = build_experiment_spec(config)
    metadata = create_run_metadata(
        config=config,
        experiment=experiment,
        run_dir=tmp_path / "run",
        run_id="r0",
        status="running",
    )

    assert metadata.test_checksum == compute_test_digest(config)
    # Whole-file digests (which DO see test rows) are kept as provenance, distinct from the
    # test-invariant training digests on the ExperimentSpec.
    assert metadata.dataset_file_checksum
    assert metadata.splits_file_checksum
    assert metadata.dataset_file_checksum != experiment.dataset_checksum

    write_run_metadata(tmp_path / "run", metadata)
    payload = yaml.safe_load((tmp_path / "run" / "run.yaml").read_text(encoding="utf-8"))
    assert payload["test_checksum"] == metadata.test_checksum


# --- test-results registry / overwrite guard (issue #247) ---------------------------


def test_register_test_result_records_fresh_and_coexists(tmp_path: Path):
    run_dir = tmp_path / "run"

    first = register_test_result(run_dir, "digest-a", split_names=["test"], run_id="r0")
    assert first.skipped is False
    second = register_test_result(
        run_dir, "digest-b", split_names=["test_official"], run_id="r0"
    )
    assert second.skipped is False

    registry = read_test_results(run_dir)
    assert set(registry) == {"digest-a", "digest-b"}
    assert registry["digest-b"]["split_names"] == ["test_official"]


def test_register_test_result_refuses_rescore_unless_overwrite(tmp_path: Path):
    run_dir = tmp_path / "run"
    register_test_result(run_dir, "digest-a", split_names=["test"], run_id="r0")
    recorded_at = read_test_results(run_dir)["digest-a"]["recorded_at"]

    refused = register_test_result(run_dir, "digest-a", split_names=["test"], run_id="r0")
    assert refused.skipped is True
    assert refused.prior is not None
    # The prior result is untouched by a refused re-score.
    assert read_test_results(run_dir)["digest-a"]["recorded_at"] == recorded_at

    overwritten = register_test_result(
        run_dir, "digest-a", split_names=["test"], run_id="r0", overwrite=True
    )
    assert overwritten.skipped is False


def test_experiment_spec_roundtrips_through_yaml(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    spec = build_experiment_spec(config)
    path = tmp_path / "experiment.yaml"

    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["experiment_id"] == spec.experiment_id
    assert loaded["slug"] == spec.slug
    assert loaded["canonical_spec"]["task"]["name"] == "binary_classification"
