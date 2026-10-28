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
    omitted = _resolve(experiment(*survival, head="discrete_survival", aggregator="abmil"))
    config = experiment(*survival, head="discrete_survival", aggregator="abmil", head_params={"num_bins": None})
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
    plan = _resolve(experiment(records, splits, head="discrete_survival", aggregator="abmil"))
    assert plan.learner.params["head"].params["num_bins"] == 4
    # An explicit width may exceed the bins the cohort uses.
    wide = _resolve(experiment(records, splits, head="discrete_survival", aggregator="abmil", head_params={"num_bins": 6}))
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
        check(experiment(*files, head="discrete_survival", features=FeaturesConfig(encoder="uni2")))
    with pytest.raises(ValueError, match="'clam_sb' is not supported for survival"):
        check(experiment(*files, head="discrete_survival", aggregator="clam_sb"))
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


def test_only_a_cox_window_is_checkpointed(files):
    window = {"head": "cox_survival", "head_params": {"cox_window": 4}, "aggregator": "abmil"}
    check(experiment(*files, **window, checkpoint_cox_window=True))
    with pytest.raises(ValueError, match="checkpoint_cox_window.*cox_window >= 2"):
        check(experiment(*files, head="cox_survival", aggregator="abmil", loaders={"batch_size": 4}, checkpoint_cox_window=True))
    with pytest.raises(ValueError, match="checkpoint_cox_window.*cox_window >= 2"):
        check(experiment(*files, aggregator="abmil", checkpoint_cox_window=True))


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


def test_a_slide_encoder_takes_no_aggregator():
    # titan gives one vector per slide: there is no bag to pool, and check says so before
    # any slide is tiled or embedded.
    config = presets.slide_classification("titan")
    config.learner.params["aggregator"] = Component("abmil")
    with pytest.raises(
        ValueError, match=r"encoder 'titan' gives one vector per slide.*drop learner\.params\.aggregator"
    ):
        check(config)
    del config.learner.params["aggregator"]
    check(config)


@pytest.mark.parametrize(
    ("preset", "encoder", "num_classes", "level"),
    [("segmentation", "titan", 2, "slide"), ("detection", "titan", 1, "slide"), ("segmentation", "moozy", 2, "patient")],
)
def test_a_dense_run_needs_a_tile_level_encoder(preset, encoder, num_classes, level):
    # A grid run encodes each tile into a token grid, which a slide or patient encoder cannot.
    # The dense presets refuse such an encoder when built; a written config meets check.
    config = getattr(presets, preset)("uni2", "linear", num_classes)
    config.features = FeaturesConfig(encoder=encoder)
    with pytest.raises(
        ValueError, match=rf"encoder '{encoder}' gives one vector per {level}.*a dense run needs a tile-level encoder"
    ):
        check(config)


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
    config = presets.mil_classification(encoder, "abmil")
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
    check(presets.mil_classification("virchow2", "abmil"))
    check(presets.mil_classification("uni2", aggregator="hipt"))


def test_a_precision_off_the_encoders_recommendation_needs_the_opt_in():
    config = presets.mil_classification("uni2", "abmil")
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


# --- survival bins derived from the training split (binning: train_quantiles) ---------

QUANTILES = {"binning": "train_quantiles", "num_bins": 2}


def _quantile_survival(files, **kwargs) -> ExperimentConfig:
    params = {**QUANTILES, **kwargs.pop("head_params", {})}
    return experiment(*files, head="discrete_survival", head_params=params, **kwargs)


def _survival_folds(root: Path, rows: list[tuple], folds: list[dict[str, str]]) -> tuple[Path, Path]:
    """``rows`` of ``(sample_id, patient_id, time, event)`` and one ``{sample_id: split}`` per fold."""
    root.mkdir(parents=True, exist_ok=True)
    records, splits = root / "dataset.csv", root / "splits.csv"
    pd.DataFrame(
        [{"sample_id": s, "patient_id": p, "image_path": f"/slides/{s}.tif", "time": t, "event": e} for s, p, t, e in rows]
    ).to_csv(records, index=False)
    pd.DataFrame(
        [{"fold": fold, "sample_id": s, "split": split} for fold, roles in enumerate(folds) for s, split in roles.items()]
    ).to_csv(splits, index=False)
    return records, splits


def test_train_quantile_bins_need_num_bins_and_take_no_bin_edges_or_bin_column(files):
    with pytest.raises(ValueError, match="binning 'train_quantiles' needs num_bins"):
        check(experiment(*files, head="discrete_survival", aggregator="abmil", head_params={"binning": "train_quantiles"}))
    with pytest.raises(
        ValueError, match=r"binning 'train_quantiles' takes no bin_edges: its fit hook fits them per fold"
    ):
        check(_quantile_survival(files, aggregator="abmil", head_params={"bin_edges": [1.5]}))
    # The head reads no bin, so a targets mapping cannot name one.
    with pytest.raises(ValueError, match=r"head targets name key\(s\) \['bin'\]"):
        check(_quantile_survival(files, aggregator="abmil", head_params={"targets": {"bin": "b"}}))


def test_train_quantile_bins_refuse_a_bin_column_at_cohort_load(tmp_path: Path):
    files = _cohort_files(tmp_path, time=[float(i) for i in range(1, 9)], event=[1] * 8, bin=[0, 1] * 4)
    message = (
        r"cohort.records has a 'bin' column, but the head derives bins from the training split "
        r"\(binning: train_quantiles\); drop the column or set binning: column"
    )
    with pytest.raises(ValueError, match=message):
        load_cohort(_quantile_survival(files, aggregator="abmil"))
    # binning: column reads the same file.
    load_cohort(experiment(*files, head="discrete_survival", aggregator="abmil"))


def test_the_plan_records_the_binning_and_bin_count_never_the_fitted_edges(tmp_path: Path):
    files = _cohort_files(tmp_path, time=[float(i) for i in range(1, 9)], event=[1] * 8)
    plan = _resolve(_quantile_survival(files, aggregator="abmil"))
    assert plan.learner.params["head"].params == {
        "num_bins": 2,
        "binning": "train_quantiles",
        "bin_edges": None,
        "alpha": 0.15,
        "dropout": 0.0,
        "targets": {"time": "time", "event": "event"},
    }


def test_resolve_names_the_fold_whose_training_split_cannot_form_the_bins(tmp_path: Path):
    rows = [(f"s{i}", f"p{i}", float(i + 1), event) for i, event in enumerate([1, 1, 1, 0, 0, 0, 1, 1])]
    roles = ["train"] * 4 + ["tune"] * 2 + ["test"] * 2
    fold_0 = dict(zip([f"s{i}" for i in range(8)], roles))
    # Fold 1 trains on s3-s6: one uncensored time (7.0) for two bins.
    fold_1 = dict(zip([f"s{i}" for i in (3, 4, 5, 6, 0, 1, 2, 7)], roles))
    files = _survival_folds(tmp_path / "few", rows, [fold_0, fold_1])
    with pytest.raises(ValueError, match=r"^Fold 1: .*num_bins=2 .* 1 uncensored training time\(s\)"):
        _resolve(_quantile_survival(files, aggregator="abmil"))
    _resolve(_quantile_survival(_survival_folds(tmp_path / "one", rows, [fold_0]), aggregator="abmil"))

    # Five training times [1, 1, 1, 1, 2]: the 1/3 and 2/3 quantiles are both 1.
    rows = [(f"s{i}", f"p{i}", time, 1) for i, time in enumerate([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])]
    files = _survival_folds(tmp_path / "tied", rows, [dict(zip([f"s{i}" for i in range(8)], ["train"] * 5 + ["tune"] + ["test"] * 2))])
    with pytest.raises(ValueError, match=r"^Fold 0: .*num_bins=3 .*repeat the cut point 1 "):
        _resolve(_quantile_survival(files, aggregator="abmil", head_params={"num_bins": 3}))


def test_a_patient_run_counts_each_training_patient_once(tmp_path: Path):
    # p0's two slides share one uncensored time; p1 is censored. Two slides, one patient.
    rows = [("a", "p0", 1.0, 1), ("b", "p0", 1.0, 1), ("c", "p1", 2.0, 0), ("d", "p2", 3.0, 1), ("e", "p3", 4.0, 1)]
    files = _survival_folds(tmp_path, rows, [{"a": "train", "b": "train", "c": "train", "d": "tune", "e": "test"}])
    _resolve(_quantile_survival(files, aggregator="abmil"))  # slides: two uncensored times
    with pytest.raises(ValueError, match=r"^Fold 0: .*num_bins=2 .* 1 uncensored training time\(s\)"):
        _resolve(_quantile_survival(files, cohort={"collapse": "patient_id"}))


# --- survival bins fixed in the config (binning: fixed) --------------------------------


def _fixed_survival(files, **kwargs) -> ExperimentConfig:
    params = {"binning": "fixed", "bin_edges": [12, 24, 36], **kwargs.pop("head_params", {})}
    return experiment(*files, head="discrete_survival", aggregator="abmil", head_params=params, **kwargs)


def test_fixed_bins_need_their_bin_edges_and_only_they_take_them(files):
    with pytest.raises(ValueError, match=r"binning 'fixed' needs bin_edges"):
        check(_fixed_survival(files, head_params={"bin_edges": None}))
    # The mode is always explicit: bin_edges never switch the default binning to fixed.
    with pytest.raises(ValueError, match=r"binning 'column' takes no bin_edges: .* bin column"):
        check(experiment(*files, head="discrete_survival", aggregator="abmil", head_params={"bin_edges": [12, 24, 36]}))


@pytest.mark.parametrize(
    ("bin_edges", "message"),
    [
        ([], r"at least one cut point"),
        ("12, 24", r"at least one cut point"),
        ([12, "24"], r"bin_edges must be finite numbers, got '24'"),
        ([True, 24], r"bin_edges must be finite numbers, got True"),  # YAML's true
        ([12, float("inf")], r"bin_edges must be finite numbers, got inf"),
        ([12, float("nan")], r"bin_edges must be finite numbers, got nan"),
        # Times are >= 0 and bins left-closed: an edge at or below 0 leaves bin 0 empty.
        ([0, 12], r"bin_edges must be above 0, got 0 .* leaves the first bin empty"),
        ([-6, 12], r"bin_edges must be above 0, got -6 "),
        # Never sorted or deduplicated silently.
        ([24, 12, 36], r"bin_edges must be strictly increasing, got \[24.0, 12.0, 36.0\]"),
        ([12, 24, 24], r"bin_edges must be strictly increasing, got \[12.0, 24.0, 24.0\]"),
    ],
)
def test_fixed_bin_edges_are_increasing_positive_numbers(files, bin_edges, message):
    with pytest.raises(ValueError, match=message):
        check(_fixed_survival(files, head_params={"bin_edges": bin_edges}))


def test_a_stated_num_bins_must_match_the_fixed_bin_edges(files):
    with pytest.raises(ValueError, match=r"num_bins=3 disagrees with bin_edges \[12.0, 24.0, 36.0\], which make 4 bins"):
        check(_fixed_survival(files, head_params={"num_bins": 3}))
    check(_fixed_survival(files, head_params={"num_bins": 4}))


def test_fixed_bins_refuse_a_bin_column_at_cohort_load(tmp_path: Path):
    files = _cohort_files(tmp_path, time=[float(i) for i in range(1, 9)], event=[1] * 8, bin=[0, 1] * 4)
    message = (
        r"cohort.records has a 'bin' column, but the head derives bins from its bin_edges "
        r"\(binning: fixed\); drop the column or set binning: column"
    )
    with pytest.raises(ValueError, match=message):
        load_cohort(_fixed_survival(files))


def test_resolve_names_the_fold_whose_training_cases_all_fall_in_one_fixed_bin(tmp_path: Path):
    times = [5.0, 15.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]
    rows = [(f"s{i}", f"p{i}", time, 1) for i, time in enumerate(times)]
    roles = ["train"] * 4 + ["tune"] * 2 + ["test"] * 2
    fold_0 = dict(zip([f"s{i}" for i in range(8)], roles))
    # Fold 1 trains on s4-s7, all below the first cut point: likely years against month edges.
    fold_1 = dict(zip([f"s{i}" for i in (4, 5, 6, 7, 0, 1, 2, 3)], roles))
    files = _survival_folds(tmp_path, rows, [fold_0, fold_1])
    with pytest.raises(ValueError, match=r"^Fold 1: .*all 4 training cases fall in one bin, \[0, 12\).*units of the time column"):
        _resolve(_fixed_survival(files))
