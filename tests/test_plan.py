"""``soma.resolve(config, cohort) -> Plan``: every default, what the cohort decides, and
the cross-section rules a config must satisfy (``soma.plan.check``)."""

from __future__ import annotations

import dataclasses
import sys
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
import slide2vec
import yaml
from hs2p import TilingConfig

from soma import ExperimentConfig, Plan, presets, resolve
from soma.config import Component, CRoMaOptions, EncoderLeaf, FeaturesConfig
from soma.plan import check, load_cohort
from tests.experiment_configs import experiment
from tests.extraction_configs import annotation_masks, extraction


def _cohort_files(root: Path, **columns) -> tuple[Path, Path]:
    """Eight samples over two classes, five train / one tune / two test."""
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"s{i}" for i in range(8)]
    frame = pd.DataFrame(
        {"sample_id": ids, "image_path": [f"/slides/{s}.tif" for s in ids], "label": ["a", "b"] * 4, **columns}
    )
    records = root / "dataset.csv"
    frame.to_csv(records, index=False)
    splits = root / "splits.csv"
    pd.DataFrame({"sample_id": ids, "split": ["train"] * 5 + ["tune"] + ["test"] * 2}).to_csv(splits, index=False)
    return records, splits


def _resolve(config: ExperimentConfig) -> Plan:
    return resolve(config, load_cohort(config))


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    return _cohort_files(tmp_path / "data")


# --- defaults and inference ------------------------------------------------------------


def test_every_leaf_takes_its_registered_defaults(files):
    plan = _resolve(experiment(*files, aggregator="abmil", scheduler="cosine"))
    params = plan.learner.params
    assert params["aggregator"] == Component(
        "abmil", {"hidden_dim": 128, "activation": "tanh", "gated": True, "dropout": 0.25}
    )
    # No optimizer leaf: the Learner's own default, spelled out.
    optimizer = params["optimizer"]
    assert optimizer.name == "adam" and optimizer.params["lr"] == 1e-4 and optimizer.params["weight_decay"] == 1e-5
    assert optimizer.params["betas"] == [0.9, 0.999]
    assert params["scheduler"] == [Component("cosine", {"T_max": None, "eta_min": 0.0, "last_epoch": -1})]
    assert params["normalization"] == {"method": "none", "eps": 1e-6}
    assert params["projection"] == {"method": "none", "target_dim": None, "seed": 0}
    assert plan.trainer.callbacks == [
        Component("checkpoint", {"monitor": "loss", "mode": "min", "selection": "best", "dirpath": None}),
        Component("early_stopping", {"monitor": "loss", "mode": "min", "patience": 10, "hold_until_moves": True}),
    ]
    assert plan.features.encoder is None


def test_the_cohort_decides_what_the_config_leaves_unset(files):
    plan = _resolve(experiment(*files, aggregator="abmil"))
    assert plan.learner.params["head"] == Component(
        "binary_classification",
        {"num_classes": 2, "label_map": {"a": 0, "b": 1}, "dropout": 0.0, "targets": {"label": "label"}},
    )


def test_explicit_params_win_and_a_contradiction_raises(files):
    explicit = _resolve(experiment(*files, aggregator="abmil", head_params={"label_map": {"b": 0, "a": 1}}))
    # An explicit label map decides the classes and their count.
    assert explicit.learner.params["head"].params["label_map"] == {"b": 0, "a": 1}
    assert explicit.learner.params["head"].params["num_classes"] == 2
    wider = _resolve(
        experiment(*files, aggregator="abmil", head="multiclass_classification", head_params={"label_map": {"a": 0, "b": 1, "c": 2}})
    )
    assert wider.learner.params["head"].params["num_classes"] == 3
    with pytest.raises(ValueError, match=r"num_classes=3 contradicts what the cohort shows \(num_classes=2\)"):
        _resolve(experiment(*files, aggregator="abmil", head="multiclass_classification", head_params={"num_classes": 3}))
    with pytest.raises(ValueError, match=r"num_classes=3 contradicts .*\(num_classes=2\)"):
        _resolve(
            experiment(
                *files,
                aggregator="abmil",
                head="multiclass_classification",
                head_params={"num_classes": 3, "label_map": {"b": 0, "a": 1}},
            )
        )


def test_an_ordinal_head_keeps_an_explicit_num_classes_and_else_spans_its_label_map(files):
    def head(**params):
        plan = _resolve(experiment(*files, aggregator="abmil", head="ordinal_classification", head_params=params))
        return plan.learner.params["head"].params

    # num_classes bounds the rounded prediction, so it reaches the map's top index.
    assert head(label_map={"a": 0, "b": 2})["num_classes"] == 3
    # An explicit num_classes may be wider than the map or the grades the cohort shows.
    assert head(num_classes=5, label_map={"a": 0, "b": 1})["num_classes"] == 5
    assert {key: head(num_classes=5)[key] for key in ("num_classes", "label_map")} == {
        "num_classes": 5,
        "label_map": {"a": 0, "b": 1},
    }


def test_a_null_head_param_resolves_like_the_omitted_one(tmp_path: Path):
    # Null is unset, as the params defaults and the infer hooks read it: the cohort
    # decides the field (null is no "keep the raw labels" request), in Python and
    # through a saved config.yaml.
    def plans(config: ExperimentConfig, name: str) -> list[Plan]:
        path = tmp_path / f"{name}.yaml"
        config.to_yaml(path)
        return [_resolve(config), _resolve(ExperimentConfig.from_yaml(path))]

    files = _cohort_files(tmp_path / "data")
    for head in ("binary_classification", "ordinal_classification"):
        omitted = _resolve(experiment(*files, aggregator="abmil", head=head))
        for nulls in ({"label_map": None}, {"num_classes": None}, {"label_map": None, "num_classes": None}):
            config = experiment(*files, aggregator="abmil", head=head, head_params=nulls)
            assert plans(config, head) == [omitted, omitted]
    survival = _cohort_files(tmp_path / "survival", time=[1.0, 2.0] * 4, event=[1, 0] * 4, bin=[0, 1, 2, 3] * 2)
    omitted = _resolve(experiment(*survival, head="survival", aggregator="abmil"))
    config = experiment(*survival, head="survival", aggregator="abmil", head_params={"num_bins": None})
    assert plans(config, "survival") == [omitted, omitted]
    assert omitted.learner.params["head"].params["num_bins"] == 4


def test_an_ordinal_head_keeps_raw_grades_through_an_explicit_identity_map(tmp_path: Path):
    files = _cohort_files(tmp_path / "data", label=[0, 2, 4, 0, 2, 4, 0, 2])
    config = experiment(
        *files,
        aggregator="abmil",
        head="ordinal_classification",
        head_params={"label_map": {0: 0, 2: 2, 4: 4}, "num_classes": 5},
    )
    params = _resolve(config).learner.params["head"].params
    assert (params["label_map"], params["num_classes"]) == ({0: 0, 2: 2, 4: 4}, 5)
    # Left unset, the cohort's grades are ranked instead.
    ranked = _resolve(experiment(*files, aggregator="abmil", head="ordinal_classification"))
    assert ranked.learner.params["head"].params["label_map"] == {0: 0, 2: 1, 4: 2}


def test_head_targets_map_keys_to_cohort_columns(tmp_path: Path):
    records, splits = _cohort_files(tmp_path, diagnosis=["x", "y"] * 4)
    config = experiment(records, splits, aggregator="abmil", head_params={"targets": {"label": "diagnosis"}})
    plan = _resolve(config)
    assert plan.learner.params["head"].params["targets"] == {"label": "diagnosis"}
    assert plan.learner.params["head"].params["label_map"] == {"x": 0, "y": 1}
    with pytest.raises(ValueError, match=r"head targets name key\(s\) \['time'\]"):
        check(experiment(records, splits, aggregator="abmil", head_params={"targets": {"time": "t"}}))


def test_survival_bins_are_inferred_after_the_columns_are_checked(tmp_path: Path):
    records, splits = _cohort_files(tmp_path, time=[1.0, 2.0] * 4, event=[1, 0] * 4, bin=[0, 1, 2, 3] * 2)
    plan = _resolve(experiment(records, splits, head="survival", aggregator="abmil"))
    assert plan.learner.params["head"].params["num_bins"] == 4
    # An explicit width may exceed the bins the cohort uses.
    wide = _resolve(experiment(records, splits, head="survival", aggregator="abmil", head_params={"num_bins": 6}))
    assert wide.learner.params["head"].params["num_bins"] == 6


def test_a_plan_round_trips_through_plan_yaml(files, tmp_path: Path):
    plan = _resolve(experiment(*files, aggregator="abmil", trainer={"max_steps": 20}))
    path = tmp_path / "plan.yaml"
    plan.to_yaml(path)
    assert Plan.from_yaml(path) == plan
    assert set(plan) == {"run", "cohort", "features", "loaders", "learner", "trainer", "evaluation"}
    assert plan["trainer"] is plan.trainer and plan.trainer.max_epochs is None


def test_a_python_config_and_its_saved_yaml_resolve_to_one_plan(files, tmp_path: Path):
    config = experiment(
        *files,
        aggregator="abmil",
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
        optimizer={"name": "adamw", "params": {"lr": 3e-4}},
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    assert _resolve(ExperimentConfig.from_yaml(path)) == _resolve(config)


def test_resolving_a_plan_again_changes_nothing(files):
    plan = _resolve(experiment(*files, aggregator="abmil", scheduler="cosine"))
    assert resolve(plan.config, load_cohort(plan.config)) == plan


def test_comparable_leaves_out_where_and_which_folds_a_launch_runs_but_keeps_its_seed(files):
    base = _resolve(experiment(*files, aggregator="abmil", seed=1, run={"run_id": "r", "folds": (0,)}))
    moved = _resolve(
        experiment(*files, aggregator="abmil", seed=1, output_dir="elsewhere", run={"run_id": "x", "folds": (0,), "mirror_root": "/m"})
    )
    assert moved != base and moved.comparable() == base.comparable()
    assert base.comparable()["features"] == base.to_dict()["features"]
    reseeded = _resolve(experiment(*files, aggregator="abmil", seed=2, run={"run_id": "r", "folds": (0,)}))
    assert reseeded.comparable() != base.comparable()
    # Launches pinned to one run_id split its folds: each takes on its own.
    refolded = _resolve(experiment(*files, aggregator="abmil", seed=1, run={"run_id": "r", "folds": (1,)}))
    assert refolded.comparable() == base.comparable()


def test_derived_shape_family_and_unit(files):
    mil = _resolve(experiment(*files, aggregator="abmil", features=FeaturesConfig(encoder="uni2")))
    assert (mil.shape, mil.task_family, mil.unit) == ("set", "binary_classification", "slide")
    tile = _resolve(experiment(*files, features=FeaturesConfig(encoder="uni2")))
    assert tile.unit == "tile"
    grid = experiment(*files, decoder="lightweight_conv", head="segmentation", head_params={"num_classes": 2})
    assert (Plan(grid).shape, Plan(grid).task_family) == ("grid", "segmentation")


# --- a dotted-path class ---------------------------------------------------------------


@pytest.fixture
def user_package(tmp_path: Path, monkeypatch):
    package = tmp_path / "pkgs" / "mylab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "learners.py").write_text(
        textwrap.dedent(
            """
            from soma.learn import MILLearner

            class ScaledMIL(MILLearner):
                def __init__(self, model, *, scale=1.0, **kwargs):
                    super().__init__(model, **kwargs)
                    self.scale = scale
            """
        )
    )
    monkeypatch.syspath_prepend(str(package.parent))
    yield "mylab.learners.ScaledMIL"
    for name in [m for m in sys.modules if m == "mylab" or m.startswith("mylab.")]:
        del sys.modules[name]


def test_a_dotted_path_class_keeps_its_params_verbatim(files, user_package):
    config = experiment(*files, aggregator="abmil", learner=user_package, scale=0.5)
    with pytest.warns(UserWarning, match="is not registered"):
        plan = _resolve(config)
    assert plan.learner.name == user_package
    own = {k: v for k, v in plan.learner.params.items() if k not in ("aggregator", "head", "normalization", "projection", "optimizer", "scheduler")}
    assert own == {"scale": 0.5}


# --- check: the cross-section rules ----------------------------------------------------


def _grid(files, **kwargs) -> ExperimentConfig:
    kwargs.setdefault("decoder", "lightweight_conv")
    kwargs.setdefault("head", "segmentation")
    kwargs.setdefault("head_params", {"num_classes": 2})
    return experiment(*files, **kwargs)


def test_a_plan_refuses_field_writes(files):
    # The config is the user's to edit; the plan is what a run does, frozen.
    plan = _resolve(experiment(*files, aggregator="abmil"))
    with pytest.raises(FrozenInstanceError):
        plan.trainer.max_epochs = 3
    with pytest.raises(FrozenInstanceError):
        plan.config.run = plan.run
    assert plan.trainer.max_epochs == 50
    # Wrapping a config as a plan freezes the plan's copy, not the caller's config.
    config = experiment(*files, aggregator="abmil")
    wrapped = Plan(config)
    config.trainer.max_epochs = 3
    assert wrapped.trainer.max_epochs == 50


def test_check_refuses_a_field_write_its_section_would_refuse(files):
    config = experiment(*files, aggregator="abmil")
    config.trainer.max_steps = 0
    with pytest.raises(ValueError, match="trainer.max_steps"):
        check(config)
    config = experiment(*files, aggregator="abmil")
    config.cohort.collapse = "site"
    with pytest.raises(ValueError, match="cohort.collapse"):
        check(config)


def test_a_run_needs_a_head_or_a_representation(files):
    with pytest.raises(ValueError, match="needs a learner with a head"):
        check(experiment(*files, head=None))
    with pytest.raises(ValueError, match="needs a head"):
        check(ExperimentConfig.from_dict({"cohort": {"records": "d", "splits": "s"}, "learner": {"name": "mil"}}))


def test_the_grid_shape_takes_a_decoder_and_a_grid_head(files):
    with pytest.raises(ValueError, match="needs a decoder"):
        check(experiment(*files, learner="dense", head="segmentation", head_params={"num_classes": 2}))
    with pytest.raises(ValueError, match="takes a decoder, not an aggregator"):
        check(_grid(files, aggregator="abmil"))
    with pytest.raises(ValueError, match="trains a segmentation or detection head"):
        check(_grid(files, head="binary_classification", head_params=None))


def test_the_set_shape_refuses_a_grid_head_and_a_decoder(files):
    with pytest.raises(ValueError, match="trains the grid shape"):
        check(experiment(*files, head="segmentation", head_params={"num_classes": 2}, learner="mil"))
    with pytest.raises(ValueError, match="decoder is for the grid shape"):
        check(experiment(*files, decoder="lightweight_conv", learner="mil"))


def test_unknown_components_and_params_fail_before_any_data(files):
    with pytest.raises(ValueError, match="Unknown aggregator 'nope'"):
        check(experiment(*files, aggregator="nope"))
    with pytest.raises(ValueError, match="has no param"):
        check(experiment(*files, aggregator={"name": "abmil", "params": {"width": 3}}))
    with pytest.raises(ValueError, match="has no param"):
        check(experiment(*files, aggregator="abmil", head_params={"loss": "cox"}))
    with pytest.raises(ValueError, match="dropout"):
        check(experiment(*files, aggregator="abmil", head_params={"dropout": 1.5}))
    with pytest.raises(ValueError, match="spells the scheduler"):
        check(experiment(*files, aggregator="abmil", schedulers=["cosine"]))


def test_live_features_and_augmentation_serve_the_grid_shape(files):
    with pytest.raises(ValueError, match="features.kind='live'"):
        check(experiment(*files, aggregator="abmil", features=FeaturesConfig(kind="live")))
    with pytest.raises(ValueError, match="requires features.kind='live'"):
        check(_grid(files, loaders={"augment": "segmentation"}))
    check(_grid(files, features=FeaturesConfig(kind="live"), loaders={"augment": "segmentation"}))
    check(
        _grid(
            files,
            head="detection",
            head_params={"num_classes": 1, "match_distance": 3.0},
            features=FeaturesConfig(kind="live"),
            loaders={"augment": "segmentation"},
        )
    )


def test_the_roi_sampler_is_a_cached_segmentation_contract(files):
    with pytest.raises(ValueError, match="cached neural segmentation"):
        check(experiment(*files, aggregator="abmil", loaders={"sampler": "uniform"}))
    with pytest.raises(ValueError, match="whole batches"):
        check(_grid(files, loaders={"batch_size": 4, "sampler": {"name": "uniform", "params": {"draws_per_epoch": 10}}}))
    with pytest.raises(ValueError, match="class_request_ratios"):
        check(
            _grid(
                files,
                loaders={"sampler": {"name": "class_conditioned", "params": {"class_request_ratios": [0, 0]}}},
            )
        )
    check(_grid(files, loaders={"sampler": {"name": "class_conditioned", "params": {"class_request_ratios": [1, 3]}}}))


def test_monitors_name_a_metric_the_head_reports(files):
    monitor = {"monitor": "auroc", "mode": "max"}
    config = experiment(
        *files,
        aggregator="abmil",
        trainer={"callbacks": [{"name": "checkpoint", "params": monitor}, {"name": "early_stopping", "params": monitor}]},
    )
    check(config)
    with pytest.raises(ValueError, match="checkpoint monitor 'mae' is not an effective metric"):
        check(
            experiment(*files, aggregator="abmil", trainer={"callbacks": [{"name": "checkpoint", "params": {"monitor": "mae"}}]})
        )


def test_last_checkpoint_selection_never_early_stops(files):
    last = {"name": "checkpoint", "params": {"selection": "last"}}
    check(experiment(*files, aggregator="abmil", trainer={"callbacks": [last]}))
    with pytest.raises(ValueError, match="remove the early_stopping callback"):
        check(experiment(*files, aggregator="abmil", trainer={"callbacks": [last, "early_stopping"]}))
    with pytest.raises(ValueError, match="val_every_n_epochs must be 1"):
        check(experiment(*files, aggregator="abmil", trainer={"callbacks": [last], "val_every_n_epochs": 2}))


def test_survival_refuses_tiles_label_aware_aggregators_and_bad_cox_risk_sets(files):
    with pytest.raises(ValueError, match="belong to a slide or patient"):
        check(experiment(*files, head="survival", features=FeaturesConfig(encoder="uni2")))
    with pytest.raises(ValueError, match="'clam_sb' is not supported for survival"):
        check(experiment(*files, head="survival", aggregator="clam_sb"))
    with pytest.raises(ValueError, match="batch_size >= 2"):
        check(experiment(*files, head="cox_survival", aggregator="abmil"))
    with pytest.raises(ValueError, match="accumulate_grad_batches = 1"):
        check(
            experiment(
                *files, head="cox_survival", aggregator="abmil", loaders={"batch_size": 4}, trainer={"accumulate_grad_batches": 2}
            )
        )
    with pytest.raises(ValueError, match="batch_size = 1"):
        check(experiment(*files, head="cox_survival", head_params={"cox_window": 4}, aggregator="abmil", loaders={"batch_size": 2}))
    check(experiment(*files, head="cox_survival", head_params={"cox_window": 4}, aggregator="abmil"))


def test_annotation_masks_serve_bags_and_slide_manifest_segmentation(files):
    masks = extraction(0.5, 224, masks=annotation_masks())
    check(experiment(*files, aggregator="abmil", features=FeaturesConfig(extraction=masks)))
    with pytest.raises(ValueError, match="samples annotations"):
        check(experiment(*files, features=FeaturesConfig(encoder="uni2", extraction=masks)))
    per_annotation = extraction(0.5, 224, masks=annotation_masks(output_mode="per_annotation"))
    with pytest.raises(ValueError, match="per_annotation"):
        check(experiment(*files, aggregator="abmil", features=FeaturesConfig(extraction=per_annotation)))


def test_a_segmentation_class_scheme_is_checked_against_the_mask_vocabulary(files):
    masks = extraction(0.5, 64, masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.5}))
    features = FeaturesConfig(extraction=masks)
    check(_grid(files, features=features, head_params={"classes": {"tumor": [1]}, "ignore": [0]}))
    with pytest.raises(ValueError, match="requires its 'classes' param"):
        check(_grid(files, features=features))
    with pytest.raises(ValueError, match="does not declare"):
        check(_grid(files, features=features, head_params={"classes": {"tumor": [1], "stroma": [2]}}))


def test_a_null_class_scheme_is_checked_like_an_omitted_one(files):
    # ``soma init`` spells the scheme's fields as null for the user to fill in.
    unset = {"classes": None, "ignore": None, "num_classes": None}
    check(_grid(files, head_params={}))
    check(_grid(files, head_params=unset))
    check(_grid(files, head="detection", head_params={"classes": None, "drop": None}))


def test_patient_runs_take_no_aggregator(files):
    with pytest.raises(ValueError, match="take no aggregator"):
        check(experiment(*files, aggregator="abmil", cohort={"collapse": "patient_id"}))


def test_the_feature_adaptor_is_refused_where_it_cannot_fit(files):
    with pytest.raises(ValueError, match="tile-level features"):
        check(experiment(*files, features=FeaturesConfig(encoder="uni2"), normalization={"method": "zscore"}))
    with pytest.raises(ValueError, match="requires features.kind='cached'"):
        check(_grid(files, features=FeaturesConfig(kind="live"), normalization={"method": "zscore"}))
    with pytest.raises(ValueError, match="normalization.method"):
        check(experiment(*files, aggregator="abmil", normalization={"method": "whiten"}))


def test_the_composite_encoder_serves_cached_grid_runs_only(files):
    members = {"members": [{"name": "uni2"}, {"name": "virchow2"}]}
    composite = FeaturesConfig(encoder={"name": "composite", "params": members})
    check(_grid(files, features=composite))
    with pytest.raises(ValueError, match="composite encoder serves cached grid runs only"):
        check(experiment(*files, aggregator="abmil", features=composite))


def _pooled(encoder: str, spacing_um: float, tile_size_px: int, **params) -> ExperimentConfig:
    """A MIL preset over ``encoder`` that tiles slides at ``spacing_um`` and ``tile_size_px``."""
    config = presets.mil_classification(encoder)
    config.features.encoder = EncoderLeaf(encoder, params)
    config.features.extraction = dataclasses.replace(
        config.features.extraction,
        tiling=TilingConfig(
            requested_spacing_um=spacing_um, requested_tile_size_px=tile_size_px, tolerance=0.05, overlap=0.0
        ),
    )
    return config


def test_a_pooled_tiling_off_the_encoders_supported_spacing_needs_the_opt_in():
    # uni2 supports 0.5 µm only: refused before anything runs, not when preprocessing starts.
    with pytest.raises(ValueError) as refused:
        check(_pooled("uni2", 1.0, 224))
    message = str(refused.value)
    assert "spacing 0.5 µm" in message
    assert "features.encoder.params.allow_non_recommended_settings" in message
    check(_pooled("uni2", 1.0, 224, allow_non_recommended_settings=True))


def test_an_unset_tiling_is_left_to_the_encoder():
    # ``soma init`` prints these with the tiling (and hipt's region geometry) null, for the
    # user to fill in; virchow2 supports several spacings and names no default.
    check(presets.mil_classification("virchow2"))
    check(presets.mil_classification("uni2", aggregator="hipt"))


def test_a_precision_off_the_encoders_recommendation_needs_the_opt_in():
    config = presets.mil_classification("uni2")
    config.features.execution = slide2vec.ExecutionOptions(precision="fp32")
    with pytest.raises(ValueError, match="supports precision fp16; this run extracts at precision fp32"):
        check(config)
    config.features.encoder = EncoderLeaf("uni2", {"allow_non_recommended_settings": True})
    check(config)


@pytest.mark.parametrize("allowed", [False, True])
def test_an_encoder_of_fixed_input_size_refuses_another_pooled_tile_size(allowed):
    # musk takes 384 px tiles and no other size; the opt-in does not change that.
    check(_pooled("musk", 0.5, 384))
    with pytest.raises(ValueError, match="cannot take 224 px tiles"):
        check(_pooled("musk", 0.5, 224, allow_non_recommended_settings=allowed))


def test_a_representation_run_is_task_free_cached_and_seed_zero(files):
    options = CRoMaOptions(confounder_column="center", split="test")
    config = experiment(*files, head=None, evaluation={"representation": options})
    check(config)
    with pytest.raises(ValueError, match="drop the learner section"):
        check(experiment(*files, evaluation={"representation": options}))
    with pytest.raises(ValueError, match="run.seed=0"):
        check(experiment(*files, head=None, seed=1, evaluation={"representation": options}))


def test_a_plan_holds_the_upstream_dataclasses_and_writes_primitives(files):
    plan = _resolve(experiment(*files, aggregator="abmil", features=FeaturesConfig(extraction=extraction(0.5, 224))))
    assert isinstance(plan.features.execution, slide2vec.ExecutionOptions)
    assert isinstance(plan.features.extraction, slide2vec.PreprocessingConfig)
    assert yaml.safe_load(plan.to_yaml()) == plan.to_dict()
    assert plan.to_dict()["features"]["extraction"]["tiling"]["requested_spacing_um"] == 0.5
