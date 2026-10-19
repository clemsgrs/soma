from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest
import yaml
from slide2vec import DenseOptions

from tests.experiment_configs import experiment
from tests.extraction_configs import annotation_masks, extraction

from soma.config import EncoderLeaf, ExperimentConfig, FeaturesConfig
from soma.models.encoders import CompositeMember
from soma.pipeline.layout import (
    ExperimentSpec,
    build_experiment_spec,
    canonical_experiment_payload,
    capture_environment,
    component_record,
    create_run_metadata,
    external_fingerprints,
    read_test_results,
    register_test_result,
    resolve_managed_output_paths,
    update_run_index,
    read_run_index,
    compact_run_index,
    write_run_metadata,
)
from soma.pipeline.layout import test_identity_digest as compute_test_digest
from soma.plan import Plan, check, load_cohort, resolve


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


#: One cohort with a train, tune and test split, so a config over it resolves to a plan.
_COHORT_SPLITS = "fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,tune\n0,s3,test\n"


def _resolved(config: ExperimentConfig) -> Plan:
    """The plan a run of ``config`` resolves to over its cohort."""
    return resolve(config, load_cohort(config))


def _make_pipeline_config(tmp_path: Path, **overrides) -> Plan:
    """A slide-level AB-MIL classifier's plan; ``overrides`` are :func:`experiment` fields."""
    dataset_csv = _write_csv(
        tmp_path / "dataset.csv",
        "sample_id,image_path,label\n"
        + "".join(f"s{i},/slides/s{i}.svs,{label}\n" for i, label in enumerate(("tumor", "normal") * 2)),
    )
    splits_csv = _write_csv(tmp_path / "splits.csv", _COHORT_SPLITS)
    fields = dict(
        output_dir=tmp_path / "outputs",
        seed=7,
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
        aggregator={"name": "abmil", "params": {"hidden_dim": 128}},
        optimizer={"name": "adam", "params": {"lr": 1e-4}},
        trainer={"max_epochs": 10},
    )
    fields.update(overrides)
    return _resolved(experiment(dataset_csv, splits_csv, **fields))


def _segmentation_features(**overrides) -> FeaturesConfig:
    values = dict(
        encoder="uni2",
        extraction=extraction(0.5, 256),
        dense=DenseOptions(spacing_um=0.5, target_size=256),
    )
    values.update(overrides)
    return FeaturesConfig(**values)


def _make_segmentation_config(tmp_path: Path, **overrides) -> Plan:
    """A cached segmentation run's plan; ``overrides`` are :func:`experiment` fields."""
    dataset_csv = _write_csv(
        tmp_path / "segmentation_dataset.csv",
        "sample_id,image_path,label_mask_path\n"
        + "".join(f"s{i},/slides/s{i}.png,/masks/s{i}.png\n" for i in range(4)),
    )
    splits_csv = _write_csv(tmp_path / "segmentation_splits.csv", _COHORT_SPLITS)
    fields = dict(
        output_dir=tmp_path / "outputs",
        seed=7,
        features=_segmentation_features(),
        decoder={"name": "lightweight_conv", "params": {"hidden_dim": 64}},
        head="segmentation",
        head_params={"num_classes": 2},
        optimizer={"name": "adam", "params": {"lr": 1e-4}},
        trainer={"max_epochs": 10},
    )
    fields.update(overrides)
    return _resolved(experiment(dataset_csv, splits_csv, **fields))


def _callback(payload: dict, name: str) -> dict | None:
    for leaf in payload["trainer"]["callbacks"]:
        if leaf["name"] == name:
            return leaf["params"]
    return None


def test_capture_environment_stamps_exactly_three_fields():
    # Bounded provenance (issue #213): EXACTLY {soma, torch, cuda} — nothing further.
    env = capture_environment()
    assert set(env) == {"soma", "torch", "cuda"}
    assert env["soma"]  # soma __version__ is always resolvable


def test_run_yaml_stamps_bounded_environment(tmp_path: Path):
    plan = _make_pipeline_config(tmp_path)
    experiment_spec = build_experiment_spec(plan)
    run_dir = tmp_path / "run"
    metadata = create_run_metadata(
        plan=plan,
        experiment=experiment_spec,
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


#: A user's metric module, written to a file the component record digests.
_USER_METRICS = "soma_test_user_metrics"
_USER_METRICS_SOURCE = (
    "def positive_rate(predictions, targets):\n"
    "    return float((predictions['predicted_label'] == 1).mean())\n"
)


@pytest.fixture
def user_metric(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "user_code"
    directory.mkdir()
    path = directory / f"{_USER_METRICS}.py"
    path.write_text(_USER_METRICS_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(directory))
    yield path
    sys.modules.pop(_USER_METRICS, None)


def test_the_component_record_lists_a_custom_metric_by_the_digest_of_its_file(tmp_path: Path, user_metric: Path):
    name = f"{_USER_METRICS}.positive_rate"
    plan = Plan(
        experiment(
            tmp_path / "dataset.csv",
            tmp_path / "splits.csv",
            features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
            aggregator="abmil",
            evaluation={"metrics": ["auroc", name]},
        )
    )

    record = component_record(plan)

    digest = hashlib.sha256(_USER_METRICS_SOURCE.encode("utf-8")).hexdigest()
    (entry,) = [entry for entry in record if entry["kind"] == "metric" and entry["name"] == name]
    assert entry["object"] == name
    assert entry["source_sha256"] == entry["fingerprint"] == digest
    assert "distribution" not in entry
    # The metric is the record's only component from outside soma, slide2vec, hs2p and torch.
    assert external_fingerprints(record) == {f"metric '{name}'": digest}


def test_canonical_experiment_payload_omits_the_run_section(tmp_path: Path):
    plan_a = _make_pipeline_config(tmp_path, seed=1)
    plan_b = _make_pipeline_config(tmp_path, seed=999, output_dir=tmp_path / "elsewhere")

    payload_a = canonical_experiment_payload(plan_a)
    payload_b = canonical_experiment_payload(plan_b)

    assert payload_a == payload_b
    assert payload_a["trainer"]["max_epochs"] == 10
    assert "run" not in payload_a


def test_mirror_root_does_not_change_experiment_identity(tmp_path: Path):
    local_only = _make_pipeline_config(tmp_path)
    mirrored = _make_pipeline_config(tmp_path, run={"mirror_root": tmp_path / "shared"})

    assert canonical_experiment_payload(mirrored) == canonical_experiment_payload(local_only)
    assert build_experiment_spec(mirrored).experiment_id == build_experiment_spec(
        local_only
    ).experiment_id


def test_canonical_experiment_payload_names_checkpoint_selection_unconditionally(tmp_path: Path):
    """Every resolved knob is emitted, defaults included."""
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert _callback(payload, "checkpoint")["selection"] == "best"
    assert payload["identity_version"] == 9


def test_canonical_epoch_experiment_payload_names_null_step_budget(tmp_path: Path):
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert payload["trainer"]["max_steps"] is None


def test_canonical_step_experiment_payload_names_the_step_budget(tmp_path: Path):
    payload = canonical_experiment_payload(
        _make_pipeline_config(tmp_path, trainer={"max_steps": 12500})
    )

    assert payload["trainer"]["max_epochs"] is None
    assert payload["trainer"]["max_steps"] == 12500


def test_canonical_experiment_payload_names_a_null_sampler(tmp_path: Path):
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert payload["loaders"]["sampler"] is None


def test_loader_plumbing_is_not_part_of_identity(tmp_path: Path):
    """num_workers / pin_memory / persistent_workers change throughput, not the model."""
    base = _make_pipeline_config(tmp_path, loaders={"num_workers": 0})
    tuned = _make_pipeline_config(
        tmp_path, loaders={"num_workers": 8, "pin_memory": True, "persistent_workers": True}
    )

    assert build_experiment_spec(base).experiment_id == build_experiment_spec(tuned).experiment_id
    payload = canonical_experiment_payload(base)
    for key in ("num_workers", "pin_memory", "persistent_workers"):
        assert key not in payload["loaders"]
    assert payload["loaders"]["batch_size"] == 1


def test_spacing_policy_is_always_part_of_identity(tmp_path: Path):
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert payload["features"]["spacing_policy"] == "strict"


def test_null_class_request_ratios_are_explicit_in_sampling_identity(tmp_path: Path):
    plan = _make_segmentation_config(
        tmp_path,
        loaders={"batch_size": 4, "sampler": {"name": "class_conditioned", "params": {"draws_per_epoch": 8}}},
    )

    sampler = canonical_experiment_payload(plan)["loaders"]["sampler"]
    assert sampler["params"]["class_request_ratios"] is None


def test_explicit_class_request_ratios_define_experiment_identity(tmp_path: Path):
    def sampler(ratios):
        params = {"class_request_ratios": ratios, "draws_per_epoch": 8}
        return {"batch_size": 4, "sampler": {"name": "class_conditioned", "params": params}}

    equal = _make_segmentation_config(tmp_path, loaders=sampler([1, 1]))
    weighted = _make_segmentation_config(tmp_path, loaders=sampler([1, 2]))

    assert canonical_experiment_payload(equal)["loaders"]["sampler"]["params"][
        "class_request_ratios"
    ] == [1, 1]
    assert build_experiment_spec(equal).experiment_id != build_experiment_spec(
        weighted
    ).experiment_id


def test_check_refuses_a_roi_sampler_outside_cached_segmentation(tmp_path: Path):
    config = experiment(
        tmp_path / "dataset.csv",
        tmp_path / "splits.csv",
        aggregator="abmil",
        loaders={"batch_size": 4, "sampler": {"name": "class_conditioned", "params": {"draws_per_epoch": 8}}},
    )
    with pytest.raises(ValueError, match="cached neural segmentation"):
        check(config)


def test_build_experiment_spec_distinguishes_checkpoint_selection(tmp_path: Path):
    best = _make_pipeline_config(tmp_path)
    last = _make_pipeline_config(
        tmp_path,
        trainer={"max_epochs": 10, "callbacks": [{"name": "checkpoint", "params": {"selection": "last"}}]},
    )

    assert _callback(canonical_experiment_payload(last), "checkpoint")["selection"] == "last"
    assert build_experiment_spec(best).experiment_id != build_experiment_spec(last).experiment_id


def test_canonical_experiment_payload_names_default_normalization(tmp_path: Path):
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert payload["learner"]["params"]["normalization"]["method"] == "none"


def test_build_experiment_spec_distinguishes_normalization(tmp_path: Path):
    off = _make_pipeline_config(tmp_path)
    zscore = _make_pipeline_config(tmp_path, normalization={"method": "zscore"})
    l2 = _make_pipeline_config(tmp_path, normalization={"method": "l2"})

    assert canonical_experiment_payload(zscore)["learner"]["params"]["normalization"] == {
        "method": "zscore",
        "eps": 1e-6,
    }
    ids = {
        build_experiment_spec(plan).experiment_id for plan in (off, zscore, l2)
    }
    assert len(ids) == 3


def test_canonical_experiment_payload_names_default_projection(tmp_path: Path):
    payload = canonical_experiment_payload(_make_pipeline_config(tmp_path))

    assert payload["learner"]["params"]["projection"]["method"] == "none"


def test_build_experiment_spec_distinguishes_projection(tmp_path: Path):
    off = _make_pipeline_config(tmp_path)
    pca = _make_pipeline_config(tmp_path, projection={"method": "pca", "target_dim": 64})
    random = _make_pipeline_config(tmp_path, projection={"method": "random", "target_dim": 64})
    narrower = _make_pipeline_config(tmp_path, projection={"method": "pca", "target_dim": 32})
    reseeded = _make_pipeline_config(
        tmp_path, projection={"method": "random", "target_dim": 64, "seed": 1}
    )

    def projection(plan):
        return canonical_experiment_payload(plan)["learner"]["params"]["projection"]

    assert projection(pca) == {"method": "pca", "target_dim": 64, "seed": 0}
    assert projection(random) == {"method": "random", "target_dim": 64, "seed": 0}
    ids = {
        build_experiment_spec(plan).experiment_id
        for plan in (off, pca, random, narrower, reseeded)
    }
    assert len(ids) == 5


def test_build_experiment_spec_uses_slug_and_short_hash(tmp_path: Path):
    plan = _make_pipeline_config(tmp_path)

    spec = build_experiment_spec(plan)

    assert isinstance(spec, ExperimentSpec)
    assert spec.slug.startswith("dataset-uni2-abmil-binary-classification_")
    assert spec.experiment_dirname == spec.slug
    assert len(spec.short_hash) == 12
    assert spec.dataset_checksum
    assert spec.splits_checksum


def test_build_experiment_spec_distinguishes_how_the_cohort_is_split(tmp_path: Path):
    def plan(**cohort):
        return _make_config_with_manifests(
            tmp_path,
            dataset_csv_text=_TRAIN_TUNE_DATASET,
            splits_csv_text=_TRAIN_TUNE_SPLITS,
            cohort=cohort,
        )

    assert build_experiment_spec(plan()).experiment_id != build_experiment_spec(
        plan(tune_is_test=True)
    ).experiment_id


def test_build_experiment_spec_distinguishes_dense_decoder_settings(tmp_path: Path):
    shallow_decoder = _make_segmentation_config(
        tmp_path, decoder={"name": "lightweight_conv", "params": {"hidden_dim": 64}}
    )
    wider_decoder = _make_segmentation_config(
        tmp_path, decoder={"name": "lightweight_conv", "params": {"hidden_dim": 128}}
    )

    shallow_spec = build_experiment_spec(shallow_decoder)
    wider_spec = build_experiment_spec(wider_decoder)

    assert shallow_spec.experiment_id != wider_spec.experiment_id


def test_build_experiment_spec_distinguishes_dense_composite_model_choices(tmp_path: Path):
    l2_composite = _make_segmentation_config(
        tmp_path,
        features=_segmentation_features(
            encoder=EncoderLeaf("composite", {"members": [{"name": "uni2", "member_norm": "l2"}]})
        ),
    )
    layernorm_composite = _make_segmentation_config(
        tmp_path,
        features=_segmentation_features(
            encoder=EncoderLeaf(
                "composite", {"members": [{"name": "uni2", "member_norm": "layernorm"}]}
            )
        ),
    )

    l2_spec = build_experiment_spec(l2_composite)
    layernorm_spec = build_experiment_spec(layernorm_composite)

    assert l2_spec.experiment_id != layernorm_spec.experiment_id


def test_build_experiment_spec_distinguishes_evaluation_metrics(tmp_path: Path):
    auroc_plan = _make_pipeline_config(tmp_path, evaluation={"metrics": ["auroc"]})
    f1_plan = _make_pipeline_config(tmp_path, evaluation={"metrics": ["f1"]})

    auroc_spec = build_experiment_spec(auroc_plan)
    f1_spec = build_experiment_spec(f1_plan)

    assert auroc_spec.experiment_id != f1_spec.experiment_id


def test_evaluation_artifact_toggles_are_identity_neutral(tmp_path: Path):
    """What is *saved* alongside the evaluation is not what the experiment is."""
    plain = _make_segmentation_config(tmp_path, evaluation={"metrics": ["mean_dice"]})
    with_artifacts = _make_segmentation_config(
        tmp_path,
        evaluation={
            "metrics": ["mean_dice"],
            "save_probabilities": True,
            "save_overlays": False,
            "save_heatmaps": True,
            "overwrite_test": True,
        },
    )

    assert build_experiment_spec(plain).experiment_id == build_experiment_spec(with_artifacts).experiment_id
    payload = canonical_experiment_payload(plain)
    assert "save_probabilities" not in payload["evaluation"]
    assert payload["evaluation"]["metrics"] == ["mean_dice"]


def test_build_experiment_spec_distinguishes_heatmap_artifact_settings(tmp_path: Path):
    disabled = _make_pipeline_config(tmp_path)
    enabled = _make_pipeline_config(
        tmp_path,
        evaluation={"attention_heatmaps": {"cmap": "viridis", "alpha": 0.75, "blur_sigma": 1.5}},
    )

    disabled_spec = build_experiment_spec(disabled)
    enabled_spec = build_experiment_spec(enabled)

    assert disabled_spec.experiment_id != enabled_spec.experiment_id


def _masked_features(masks) -> FeaturesConfig:
    return FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224, masks=masks))


def test_build_experiment_spec_distinguishes_mask_settings_via_extraction(tmp_path: Path):
    tumor_mask = _make_pipeline_config(
        tmp_path,
        features=_masked_features(annotation_masks({"tumor": 1}, {"tumor": 0.2})),
    )
    stroma_mask = _make_pipeline_config(
        tmp_path,
        features=_masked_features(annotation_masks({"stroma": 2}, {"stroma": 0.2})),
    )

    tumor_spec = build_experiment_spec(tumor_mask)
    stroma_spec = build_experiment_spec(stroma_mask)

    assert tumor_spec.experiment_id != stroma_spec.experiment_id


def test_build_experiment_spec_distinguishes_sampling_settings_via_extraction(tmp_path: Path):
    joint_sampling = _make_pipeline_config(
        tmp_path,
        features=_masked_features(
            annotation_masks({"tumor": 1}, {"tumor": 0.2}, strategy="joint")
        ),
    )
    independent_sampling = _make_pipeline_config(
        tmp_path,
        features=_masked_features(
            annotation_masks({"tumor": 1}, {"tumor": 0.2}, strategy="independent")
        ),
    )

    joint_spec = build_experiment_spec(joint_sampling)
    independent_spec = build_experiment_spec(independent_sampling)

    assert joint_spec.experiment_id != independent_spec.experiment_id


def test_build_experiment_spec_distinguishes_features_kind(tmp_path: Path):
    cached = _make_segmentation_config(tmp_path, features=_segmentation_features(kind="cached"))
    live = _make_segmentation_config(tmp_path, features=_segmentation_features(kind="live"))

    cached_spec = build_experiment_spec(cached)
    live_spec = build_experiment_spec(live)

    assert cached_spec.experiment_id != live_spec.experiment_id


def test_build_experiment_spec_distinguishes_live_augmentation_choices(tmp_path: Path):
    no_augmentation = _make_segmentation_config(
        tmp_path, features=_segmentation_features(kind="live")
    )
    horizontal_flip = _make_segmentation_config(
        tmp_path,
        features=_segmentation_features(kind="live"),
        loaders={"augment": {"name": "segmentation", "params": {"horizontal_flip": 0.5}}},
    )

    no_aug_spec = build_experiment_spec(no_augmentation)
    flip_spec = build_experiment_spec(horizontal_flip)

    assert no_aug_spec.experiment_id != flip_spec.experiment_id


def test_build_experiment_spec_is_stable_for_equivalent_configurations(tmp_path: Path):
    first = _make_segmentation_config(
        tmp_path,
        decoder={"name": "lightweight_conv", "params": {"hidden_dim": 64, "num_groups": 8}},
    )
    second = _make_segmentation_config(
        tmp_path,
        decoder={"name": "lightweight_conv", "params": {"num_groups": 8, "hidden_dim": 64}},
    )

    first_spec = build_experiment_spec(first)
    repeated_first_spec = build_experiment_spec(first)
    second_spec = build_experiment_spec(second)

    assert first_spec.experiment_id == repeated_first_spec.experiment_id
    assert first_spec.experiment_id == second_spec.experiment_id


@pytest.mark.parametrize(
    ("make", "features"),
    [
        (_make_pipeline_config, FeaturesConfig(encoder="phikon")),
        (_make_segmentation_config, FeaturesConfig(encoder="phikon")),
        (_make_segmentation_config, FeaturesConfig(encoder="phikon", dense=DenseOptions(spacing_um=1.0, target_size=512))),
    ],
    ids=["slide-preset", "segmentation-preset", "segmentation-dense"],
)
def test_a_config_and_its_saved_yaml_share_one_experiment_identity(tmp_path: Path, make, features):
    """The geometry left to the encoder preset or to ``dense`` is hashed as resolved, so
    resuming from the ``config.yaml`` a Python config saved (which spells it out) finds the
    same experiment."""
    plan = make(tmp_path, features=features)
    plan.config.to_yaml(tmp_path / "config.yaml")

    reloaded = _resolved(ExperimentConfig.from_yaml(tmp_path / "config.yaml"))
    assert canonical_experiment_payload(reloaded) == canonical_experiment_payload(plan)


@pytest.mark.parametrize(
    ("make", "features"),
    [
        (
            _make_pipeline_config,
            FeaturesConfig(
                encoder=EncoderLeaf("uni2", {"allow_non_recommended_settings": False}),
                extraction=extraction(0.5, 224),
            ),
        ),
        (
            _make_segmentation_config,
            _segmentation_features(encoder=EncoderLeaf("composite", {"members": [{"name": "uni2"}, {"name": "phikon"}]})),
        ),
        (
            _make_segmentation_config,
            _segmentation_features(
                encoder=EncoderLeaf("composite", {"members": [CompositeMember("uni2"), CompositeMember("phikon")]})
            ),
        ),
        (
            _make_segmentation_config,
            _segmentation_features(
                encoder=EncoderLeaf(
                    "composite",
                    {"members": [{"name": "uni2", "params": {"allow_non_recommended_settings": False}}]},
                )
            ),
        ),
    ],
    ids=["leaf-default-param", "composite-default-resolution", "composite-member-dataclasses", "composite-member-default-param"],
)
def test_encoder_params_are_hashed_as_their_params_dataclass_holds_them(tmp_path: Path, make, features):
    """Encoder params left to their defaults, spelling them, or given as dataclasses rather
    than mappings define the same experiment, so a Python config and the ``config.yaml``
    it saves share one experiment id."""
    plan = make(tmp_path, features=features)
    plan.config.to_yaml(tmp_path / "config.yaml")

    reloaded = _resolved(ExperimentConfig.from_yaml(tmp_path / "config.yaml"))

    assert build_experiment_spec(reloaded).experiment_id == build_experiment_spec(plan).experiment_id


def test_resolve_managed_output_paths_groups_same_experiment_and_changes_run_dir(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)

    first = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    second = resolve_managed_output_paths(config, run_id="2026-04-09_16-23-10__local")

    assert first.experiment_dir == second.experiment_dir
    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == first.experiment_dir / "runs"
    assert first.index_dir == Path(config.run.output_dir) / "indexes"


def test_create_run_metadata_records_status_and_seed(tmp_path: Path):
    config = _make_pipeline_config(tmp_path, seed=123)
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")

    metadata = create_run_metadata(
        plan=config,
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
        plan=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=layout.run_id,
        status="running",
    )

    index_path = layout.index_dir / "runs.csv"
    update_run_index(index_path, run)
    completed = run.with_updates(status="completed")
    update_run_index(index_path, completed)

    with index_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    # Append-only on disk: both status changes are kept as separate lines…
    assert [row["status"] for row in raw_rows] == ["running", "completed"]
    # …and readers see one row per run_id, the last one appended.
    rows = read_run_index(index_path)
    assert len(rows) == 1 and rows[0]["status"] == "completed"
    # Compaction rewrites the file to the deduplicated view.
    assert compact_run_index(index_path) == 1
    with index_path.open(newline="", encoding="utf-8") as handle:
        assert [row["status"] for row in csv.DictReader(handle)] == ["completed"]


def test_run_index_rewrites_when_header_schema_changes(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    run = create_run_metadata(
        plan=config, experiment=layout.experiment, run_dir=layout.run_dir,
        run_id=layout.run_id, status="running",
    )
    index_path = layout.index_dir / "runs.csv"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("run_id,status\nold-run,completed\n")

    update_run_index(index_path, run)

    rows = read_run_index(index_path)
    assert [row["run_id"] for row in rows] == ["2026-04-09_16-22-10__local", "old-run"]
    assert rows[1]["status"] == "completed" and rows[1]["experiment_id"] == ""


def test_run_manifest_records_identity_version(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    layout = resolve_managed_output_paths(config, run_id="r1")
    run = create_run_metadata(
        plan=config, experiment=layout.experiment, run_dir=layout.run_dir,
        run_id=layout.run_id, status="running",
    )
    assert run.to_dict()["identity_version"] == 9
    assert layout.experiment.to_dict()["identity_version"] == 9


def test_git_provenance_comes_from_the_soma_checkout_not_cwd(tmp_path: Path, monkeypatch):
    import subprocess

    from soma.pipeline import layout as provenance

    repo = tmp_path / "repo"
    pkg = repo / "soma"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "init")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(provenance, "_soma_package_dir", lambda: pkg)
    # cwd is a different, dirty repo: it must not leak into the run's provenance.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = _make_pipeline_config(tmp_path)
    layout = resolve_managed_output_paths(config, run_id="r1")
    run = create_run_metadata(
        plan=config, experiment=layout.experiment, run_dir=layout.run_dir,
        run_id=layout.run_id, status="running",
    )
    assert run.git_sha == sha and run.git_dirty is False

    (pkg / "__init__.py").write_text("# edited\n")
    assert provenance.soma_git_state().dirty is True


def test_git_provenance_is_null_for_a_wheel_install(tmp_path: Path, monkeypatch):
    from soma.pipeline import layout as provenance

    site = tmp_path / "site-packages" / "soma"
    site.mkdir(parents=True)
    monkeypatch.setattr(provenance, "_soma_package_dir", lambda: site)
    monkeypatch.chdir(tmp_path)  # tmp_path is not a git checkout either
    state = provenance.soma_git_state()
    assert state.sha is None and state.dirty is None


def test_experiment_index_writer_is_removed():
    # ADR 0003 decision 3b: the racy unlocked read-modify-rewrite writer is gone for good.
    import soma.pipeline.layout as output_layout

    assert not hasattr(output_layout, "update_experiment_index")


# --- test-invariant experiment identity (issue #247) --------------------------------


def _make_config_with_manifests(
    tmp_path: Path, *, dataset_csv_text: str, splits_csv_text: str, **overrides
) -> Plan:
    """A plan over the given cohort files, unresolved: experiment identity digests the
    files' train/tune slice, which these fixtures vary (a fold need not hold a test split)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_csv = _write_csv(tmp_path / "dataset.csv", dataset_csv_text)
    splits_csv = _write_csv(tmp_path / "splits.csv", splits_csv_text)
    fields = dict(
        output_dir=tmp_path / "outputs",
        seed=7,
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
        aggregator={"name": "abmil", "params": {"hidden_dim": 128}},
        optimizer={"name": "adam", "params": {"lr": 1e-4}},
        trainer={"max_epochs": 10},
    )
    fields.update(overrides)
    return Plan(experiment(dataset_csv, splits_csv, **fields))


_TRAIN_TUNE_DATASET = (
    "sample_id,image_path,label\n"
    "s0,/slides/s0.svs,tumor\n"
    "s1,/slides/s1.svs,normal\n"
    "s2,/slides/s2.svs,tumor\n"
)
_TRAIN_TUNE_SPLITS = "fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,tune\n"


def test_classification_manifest_identity_is_portable_across_storage_roots(
    tmp_path: Path,
):
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=(
            "sample_id,image_path,mask_path,coordinates_path,label,site\n"
            "s0,/local/images/s0.svs,/local/masks/s0.png,"
            "/local/coords/s0.coordinates.npz,tumor,north\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=(
            "sample_id,image_path,mask_path,coordinates_path,label,site\n"
            "s0,/archive/images/s0.svs,/archive/masks/s0.png,"
            "/archive/coords/s0.coordinates.npz,tumor,north\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )

    local_spec = build_experiment_spec(local)
    relocated_spec = build_experiment_spec(relocated)

    assert relocated_spec.dataset_checksum == local_spec.dataset_checksum
    assert relocated_spec.splits_checksum == local_spec.splits_checksum
    assert relocated_spec.experiment_id == local_spec.experiment_id


def test_adding_explicit_path_checksum_column_changes_dataset_identity(tmp_path: Path):
    without_checksum = _make_config_with_manifests(
        tmp_path / "without",
        dataset_csv_text=(
            "sample_id,image_path,label\n"
            "s0,/local/images/s0.svs,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    checksum_a = _make_config_with_manifests(
        tmp_path / "checksum-a",
        dataset_csv_text=(
            "sample_id,image_path,image_path_sha256,label\n"
            "s0,/local/images/s0.svs,aaaaaaaa,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )

    without_digest = build_experiment_spec(without_checksum).dataset_checksum
    checksum_digest = build_experiment_spec(checksum_a).dataset_checksum

    assert checksum_digest != without_digest


def test_changing_explicit_path_checksum_column_changes_dataset_identity(
    tmp_path: Path,
):
    checksum_a = _make_config_with_manifests(
        tmp_path / "checksum-a",
        dataset_csv_text=(
            "sample_id,image_path,image_path_sha256,label\n"
            "s0,/local/images/s0.svs,aaaaaaaa,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    checksum_b = _make_config_with_manifests(
        tmp_path / "checksum-b",
        dataset_csv_text=(
            "sample_id,image_path,image_path_sha256,label\n"
            "s0,/archive/images/s0.svs,bbbbbbbb,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )

    checksum_a_digest = build_experiment_spec(checksum_a).dataset_checksum
    checksum_b_digest = build_experiment_spec(checksum_b).dataset_checksum

    assert checksum_b_digest != checksum_a_digest


def test_unrecognized_path_column_defines_dataset_identity(tmp_path: Path):
    source_a = _make_config_with_manifests(
        tmp_path / "source-a",
        dataset_csv_text=(
            "sample_id,image_path,source_path,label\n"
            "s0,/images/s0.svs,/source/a.json,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    source_b = _make_config_with_manifests(
        tmp_path / "source-b",
        dataset_csv_text=(
            "sample_id,image_path,source_path,label\n"
            "s0,/images/s0.svs,/source/b.json,tumor\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )

    assert (
        build_experiment_spec(source_b).dataset_checksum
        != build_experiment_spec(source_a).dataset_checksum
    )


def test_segmentation_manifest_identity_ignores_label_mask_storage_root(
    tmp_path: Path,
):
    common = {
        "splits_csv_text": "fold,sample_id,split\n0,s0,train\n",
        "aggregator": None,
        "decoder": {"name": "lightweight_conv", "params": {"hidden_dim": 64}},
        "head": "segmentation",
        "head_params": {"num_classes": 2},
    }
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=(
            "sample_id,image_path,label_mask_path,class_names\n"
            "s0,/local/images/s0.png,/local/labels/s0.png,background|tumor\n"
        ),
        **common,
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=(
            "sample_id,image_path,label_mask_path,class_names\n"
            "s0,/archive/images/s0.png,/archive/labels/s0.png,background|tumor\n"
        ),
        **common,
    )

    local_spec = build_experiment_spec(local)
    relocated_spec = build_experiment_spec(relocated)

    assert relocated_spec.dataset_checksum == local_spec.dataset_checksum
    assert relocated_spec.experiment_id == local_spec.experiment_id


def test_segmentation_roi_manifest_identity_ignores_mask_crop_storage_root(
    tmp_path: Path,
):
    """Relocating an explicit ROI manifest's stored mask crops keeps its identity."""
    common = {
        "splits_csv_text": "fold,sample_id,split\n0,s0_0_0,train\n",
        "aggregator": None,
        "decoder": {"name": "lightweight_conv", "params": {"hidden_dim": 64}},
        "head": "segmentation",
        "head_params": {"num_classes": 2},
    }
    header = (
        "sample_id,image_path,label_mask_path,slide_id,region_x,region_y,"
        "label_mask_crop_path,class_names\n"
    )
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=header
        + "s0_0_0,/slides/s0.tif,/labels/s0.tif,s0,0,0,"
        "/local/roi_sampling/k/masks/s0/0_0.png,background|tumor\n",
        **common,
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=header
        + "s0_0_0,/slides/s0.tif,/labels/s0.tif,s0,0,0,"
        "/archive/roi_sampling/k/masks/s0/0_0.png,background|tumor\n",
        **common,
    )

    local_spec = build_experiment_spec(local)
    relocated_spec = build_experiment_spec(relocated)

    assert relocated_spec.dataset_checksum == local_spec.dataset_checksum
    assert relocated_spec.experiment_id == local_spec.experiment_id


def test_detection_manifest_identity_ignores_points_and_ignore_mask_storage_roots(tmp_path: Path):
    common = {
        "splits_csv_text": "fold,sample_id,split\n0,s0,train\n",
        "aggregator": None,
        "decoder": {"name": "lightweight_conv", "params": {"hidden_dim": 64}},
        "head": "detection",
        "head_params": {"num_classes": 2},
    }
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=(
            "sample_id,image_path,points_path,ignore_mask_path,spacing_at_level_0\n"
            "s0,/local/images/s0.png,/local/points/s0.csv,/local/ignore/s0.png,0.5\n"
        ),
        **common,
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=(
            "sample_id,image_path,points_path,ignore_mask_path,spacing_at_level_0\n"
            "s0,/archive/images/s0.png,/archive/points/s0.csv,/archive/ignore/s0.png,0.5\n"
        ),
        **common,
    )

    local_spec = build_experiment_spec(local)
    relocated_spec = build_experiment_spec(relocated)

    assert relocated_spec.dataset_checksum == local_spec.dataset_checksum
    assert relocated_spec.experiment_id == local_spec.experiment_id


def test_semantic_manifest_metadata_changes_dataset_identity(tmp_path: Path):
    north = _make_config_with_manifests(
        tmp_path / "north",
        dataset_csv_text=(
            "sample_id,image_path,label,site\n"
            "s0,/images/s0.svs,tumor,north\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    south = _make_config_with_manifests(
        tmp_path / "south",
        dataset_csv_text=(
            "sample_id,image_path,label,site\n"
            "s0,/images/s0.svs,tumor,south\n"
        ),
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )

    north_spec = build_experiment_spec(north)
    south_spec = build_experiment_spec(south)

    assert south_spec.dataset_checksum != north_spec.dataset_checksum
    assert south_spec.splits_checksum == north_spec.splits_checksum
    assert south_spec.experiment_id != north_spec.experiment_id


def test_training_sample_membership_changes_dataset_and_split_identity(tmp_path: Path):
    dataset = (
        "sample_id,image_path,label\n"
        "s0,/images/s0.svs,tumor\n"
        "s1,/images/s1.svs,normal\n"
    )
    sample_zero = _make_config_with_manifests(
        tmp_path / "sample-zero",
        dataset_csv_text=dataset,
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    sample_one = _make_config_with_manifests(
        tmp_path / "sample-one",
        dataset_csv_text=dataset,
        splits_csv_text="fold,sample_id,split\n0,s1,train\n",
    )

    sample_zero_spec = build_experiment_spec(sample_zero)
    sample_one_spec = build_experiment_spec(sample_one)

    assert sample_one_spec.dataset_checksum != sample_zero_spec.dataset_checksum
    assert sample_one_spec.splits_checksum != sample_zero_spec.splits_checksum
    assert sample_one_spec.experiment_id != sample_zero_spec.experiment_id


def test_fold_assignment_changes_only_split_identity(tmp_path: Path):
    dataset = "sample_id,image_path,label\ns0,/images/s0.svs,tumor\n"
    fold_zero = _make_config_with_manifests(
        tmp_path / "fold-zero",
        dataset_csv_text=dataset,
        splits_csv_text="fold,sample_id,split\n0,s0,train\n",
    )
    fold_one = _make_config_with_manifests(
        tmp_path / "fold-one",
        dataset_csv_text=dataset,
        splits_csv_text="fold,sample_id,split\n1,s0,train\n",
    )

    fold_zero_spec = build_experiment_spec(fold_zero)
    fold_one_spec = build_experiment_spec(fold_one)

    assert fold_one_spec.dataset_checksum == fold_zero_spec.dataset_checksum
    assert fold_one_spec.splits_checksum != fold_zero_spec.splits_checksum
    assert fold_one_spec.experiment_id != fold_zero_spec.experiment_id


def test_test_identity_is_portable_across_storage_roots(tmp_path: Path):
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=(
            "sample_id,image_path,label\n"
            "s0,/local/images/s0.svs,tumor\n"
            "t0,/local/images/t0.svs,normal\n"
        ),
        splits_csv_text=(
            "fold,sample_id,split\n"
            "0,s0,train\n"
            "0,t0,test\n"
        ),
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=(
            "sample_id,image_path,label\n"
            "s0,/archive/images/s0.svs,tumor\n"
            "t0,/archive/images/t0.svs,normal\n"
        ),
        splits_csv_text=(
            "fold,sample_id,split\n"
            "0,s0,train\n"
            "0,t0,test\n"
        ),
    )

    assert compute_test_digest(relocated) == compute_test_digest(local)


def test_run_metadata_keeps_semantic_identity_and_physical_file_provenance(
    tmp_path: Path,
):
    splits = "fold,sample_id,split\n0,s0,train\n0,t0,test\n"
    local = _make_config_with_manifests(
        tmp_path / "local",
        dataset_csv_text=(
            "sample_id,image_path,label\n"
            "s0,/local/images/s0.svs,tumor\n"
            "t0,/local/images/t0.svs,normal\n"
        ),
        splits_csv_text=splits,
    )
    relocated = _make_config_with_manifests(
        tmp_path / "relocated",
        dataset_csv_text=(
            "sample_id,image_path,label\n"
            "s0,/archive/images/s0.svs,tumor\n"
            "t0,/archive/images/t0.svs,normal\n"
        ),
        splits_csv_text=splits,
    )
    local_experiment = build_experiment_spec(local)
    relocated_experiment = build_experiment_spec(relocated)

    local_metadata = create_run_metadata(
        plan=local,
        experiment=local_experiment,
        run_dir=tmp_path / "local-run",
        run_id="local",
        status="running",
    )
    relocated_metadata = create_run_metadata(
        plan=relocated,
        experiment=relocated_experiment,
        run_dir=tmp_path / "relocated-run",
        run_id="relocated",
        status="running",
    )

    assert relocated_metadata.test_checksum == local_metadata.test_checksum
    assert relocated_metadata.comparison_key == local_metadata.comparison_key
    assert relocated_metadata.dataset_file_checksum != local_metadata.dataset_file_checksum
    assert relocated_metadata.splits_file_checksum == local_metadata.splits_file_checksum


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
        evaluation={"overwrite_test": False},
    )
    on = _make_config_with_manifests(
        tmp_path / "b",
        dataset_csv_text=_TRAIN_TUNE_DATASET,
        splits_csv_text=_TRAIN_TUNE_SPLITS,
        evaluation={"overwrite_test": True},
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
        plan=config,
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
    assert loaded["canonical_spec"]["learner"]["params"]["head"]["name"] == "binary_classification"
