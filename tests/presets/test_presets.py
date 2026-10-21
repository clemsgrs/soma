"""``soma.presets``: ten registered functions, each a complete experiment from a few
typed arguments, edited by assigning fields (design §8.4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from hs2p import MaskSamplingConfig, TilingConfig

import soma
from soma import ExperimentConfig, presets
from soma.config import Component, CohortConfig
from soma.pipeline.layout import build_experiment_spec
from soma.plan import check, head_family, learner_shape, load_cohort, resolve, sample_unit

#: ``(preset, arguments, shape, task family, sample unit, the model's aggregator or decoder)``.
PRESETS = [
    ("mil_classification", {"encoder": "uni2", "aggregator": "abmil"}, "set", "multiclass_classification", "slide", "abmil"),
    ("mil_regression", {"encoder": "uni2", "aggregator": "mean_pool"}, "set", "regression", "slide", "mean_pool"),
    ("tile_classification", {"encoder": "uni2"}, "set", "multiclass_classification", "tile", None),
    ("tile_regression", {"encoder": "uni2"}, "set", "regression", "tile", None),
    ("mil_survival", {"encoder": "uni2", "aggregator": "transmil", "loss": "nll", "num_bins": 4}, "set", "survival", "slide", "transmil"),
    ("mil_survival", {"encoder": "uni2", "aggregator": "abmil", "loss": "cox"}, "set", "survival", "slide", "abmil"),
    (
        "segmentation", {"encoder": "uni2", "decoder": "heavy_conv", "num_classes": 3},
        "grid", "segmentation", "tile", "heavy_conv",
    ),
    (
        "detection", {"encoder": "uni2", "decoder": "lightweight_conv", "num_classes": 1},
        "grid", "detection", "tile", "lightweight_conv",
    ),
]


def _cohort(root: Path, **columns) -> tuple[Path, Path]:
    ids = [f"s{index}" for index in range(8)]
    root.mkdir(parents=True, exist_ok=True)
    records, splits = root / "dataset.csv", root / "splits.csv"
    frame = {"sample_id": ids, "image_path": [f"/slides/{s}.tif" for s in ids], **columns}
    pd.DataFrame(frame).to_csv(records, index=False)
    pd.DataFrame({"sample_id": ids, "split": ["train"] * 5 + ["tune"] + ["test"] * 2}).to_csv(splits, index=False)
    return records, splits


def test_components_lists_the_ten_presets_by_what_their_head_trains_on():
    assert "presets" in soma.components()
    entries = {entry["name"]: entry for entry in soma.components("presets")}
    # Tile images, a slide's pooled tiles, a slide encoder's embedding, then the dense presets.
    assert list(entries) == [
        "tile_classification", "tile_regression",
        "mil_classification", "mil_regression", "mil_survival",
        "slide_classification", "slide_regression", "slide_survival",
        "segmentation", "detection",
    ]
    for name in ("tile_classification", "tile_regression", "slide_classification", "slide_regression"):
        assert (entries[name]["params"], entries[name]["required"]) == ({"encoder": None}, ["encoder"])
    # A slide encoder is tied to its tile encoder's tiling: no spacing or tile_size.
    assert entries["slide_survival"]["params"] == {"encoder": None, "loss": None, "num_bins": None}
    assert entries["slide_survival"]["required"] == ["encoder", "loss"]
    for name in ("mil_classification", "mil_regression"):
        assert entries[name]["params"] == {"encoder": None, "aggregator": None, "spacing": None, "tile_size": None}
        assert entries[name]["required"] == ["encoder", "aggregator"]
    assert entries["mil_survival"]["params"] == {
        "encoder": None, "aggregator": None, "loss": None, "num_bins": None, "spacing": None, "tile_size": None,
    }
    assert entries["mil_survival"]["required"] == ["encoder", "aggregator", "loss"]
    assert entries["segmentation"]["required"] == ["encoder", "decoder", "num_classes"]
    assert entries["segmentation"]["params"] == {"encoder": None, "decoder": None, "num_classes": None}
    assert entries["detection"]["required"] == ["encoder", "decoder", "num_classes"]
    assert entries["detection"]["params"] == {
        "encoder": None, "decoder": None, "num_classes": None, "match_distance": 3.0,
    }
    assert all(entry["description"] for entry in entries.values())
    # The survival preset takes the mil_ prefix of the other slide MIL presets, with no alias.
    assert not hasattr(presets, "survival")


@pytest.mark.parametrize(
    ("name", "arguments", "shape", "family", "unit", "pooling"),
    PRESETS,
    ids=[f"{p[0]}-{p[1]['loss']}" if "loss" in p[1] else p[0] for p in PRESETS],
)
def test_a_preset_builds_a_complete_config(name, arguments, shape, family, unit, pooling):
    config = getattr(presets, name)(**arguments)
    assert isinstance(config, ExperimentConfig)
    check(config)
    assert config.features.encoder.name == "uni2"
    assert (learner_shape(config), head_family(config), sample_unit(config)) == (shape, family, unit)
    leaf = config.learner.params.get("aggregator") or config.learner.params.get("decoder")
    assert (None if leaf is None else leaf.name) == pooling


#: The presets that train on a slide encoder's embedding: ``(preset, arguments, task family)``.
SLIDE_ENCODER_PRESETS = [
    ("slide_classification", {"encoder": "titan"}, "multiclass_classification"),
    ("slide_regression", {"encoder": "prism"}, "regression"),
    ("slide_survival", {"encoder": "titan", "loss": "nll", "num_bins": 4}, "survival"),
    ("slide_survival", {"encoder": "titan", "loss": "cox"}, "survival"),
]


@pytest.mark.parametrize(
    ("name", "arguments", "family"),
    SLIDE_ENCODER_PRESETS,
    ids=[f"{p[0]}-{p[1]['loss']}" if "loss" in p[1] else p[0] for p in SLIDE_ENCODER_PRESETS],
)
def test_a_slide_encoder_preset_trains_its_head_on_one_embedding_per_slide(name, arguments, family):
    config = getattr(presets, name)(**arguments)
    check(config)
    assert (learner_shape(config), head_family(config), sample_unit(config)) == ("set", family, "slide")
    assert "aggregator" not in config.learner.params
    # The MIL presets' tissue tiles, at the slide encoder's own tiling.
    mil = presets.mil_classification("uni2", "abmil").features.extraction
    assert (config.features.extraction.tiling, config.features.extraction.masks) == (None, mil.masks)
    # The probe recipe at a batch size for a cohort of a few hundred slides.
    assert config.loaders.batch_size == 32
    assert config.learner.params["optimizer"] == Component("adamw", {"lr": 3e-4})
    assert "scheduler" not in config.learner.params


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("mil_classification", ("uni2",)),
        ("mil_regression", ("uni2",)),
        ("mil_survival", ("uni2", "abmil")),
        ("segmentation", ("uni2", "linear")),
        ("detection", ("uni2", "linear")),
    ],
)
def test_a_preset_has_no_default_for_a_choice_that_changes_the_model(name, arguments):
    # The aggregator, the survival loss and the class count are the user's to name.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        getattr(presets, name)(*arguments)


def test_tile_regression_is_the_tile_probe_with_a_regression_head():
    config = presets.tile_regression("uni2")
    assert config.learner.params["head"] == Component("regression")
    # Batches of 256 embeddings, AdamW (lr 3e-4) without a schedule: tile_classification's recipe.
    assert (config.loaders.batch_size, config.learner.params["optimizer"]) == (256, Component("adamw", {"lr": 3e-4}))
    assert "scheduler" not in config.learner.params
    probe = presets.tile_classification("uni2")
    del config.learner.params["head"], probe.learner.params["head"]
    assert config == probe


def test_the_survival_loss_picks_the_head():
    # nll derives num_bins bins from each fold's training split: no bin column to compute.
    assert presets.mil_survival("uni2", "abmil", "nll", num_bins=4).learner.params["head"] == Component(
        "discrete_survival", {"binning": "train_quantiles", "num_bins": 4}
    )
    # The MIL loader forwards one bag at a time, so Cox accumulates a 32-bag risk set.
    config = presets.mil_survival("uni2", "abmil", "cox")
    assert config.learner.params["head"] == Component("cox_survival", {"cox_window": 32})
    assert config.loaders.batch_size == 1
    nll = presets.mil_survival("uni2", "abmil", "nll", num_bins=4)
    del config.learner.params["head"], nll.learner.params["head"]
    assert config == nll  # the loss changes the head only
    with pytest.raises(ValueError, match="loss must be one of 'nll', 'cox', got 'coxph'"):
        presets.mil_survival("uni2", "abmil", "coxph")


@pytest.mark.parametrize(
    ("name", "arguments"), [("mil_survival", ("uni2", "abmil")), ("slide_survival", ("titan",))]
)
def test_the_survival_bin_count_belongs_to_nll_alone(name, arguments):
    with pytest.raises(ValueError, match="loss 'nll' needs num_bins"):
        getattr(presets, name)(*arguments, "nll")
    with pytest.raises(ValueError, match="loss 'cox' takes no num_bins"):
        getattr(presets, name)(*arguments, "cox", num_bins=4)


def test_slide_survival_takes_its_cox_risk_set_from_the_batch():
    assert presets.slide_survival("titan", "nll", num_bins=4).learner.params["head"] == Component(
        "discrete_survival", {"binning": "train_quantiles", "num_bins": 4}
    )
    # One embedding per slide: Cox runs in padded mode, its risk set the batch of 32
    # slides (no cox_window, which accumulates variable-size MIL bags).
    config = presets.slide_survival("titan", "cox")
    assert (config.learner.params["head"], config.loaders.batch_size) == (Component("cox_survival"), 32)
    with pytest.raises(ValueError, match="loss must be one of 'nll', 'cox', got 'coxph'"):
        presets.slide_survival("titan", "coxph")


#: ``(preset, arguments after the encoder, what its refusal of a slide encoder says)``.
TILE_ENCODER_PRESETS = [
    ("tile_classification", (), "use slide_classification"),
    ("tile_regression", (), "use slide_regression"),
    ("mil_classification", ("abmil",), "use slide_classification"),
    ("mil_regression", ("abmil",), "use slide_regression"),
    ("mil_survival", ("abmil", "cox"), "use slide_survival"),
    ("segmentation", ("linear", 2), "decodes each tile's token grid"),
    ("detection", ("linear", 1), "decodes each tile's token grid"),
]


@pytest.mark.parametrize(
    ("name", "arguments", "instead"), TILE_ENCODER_PRESETS, ids=[p[0] for p in TILE_ENCODER_PRESETS]
)
def test_a_tile_encoder_preset_refuses_a_slide_encoder_when_built(name, arguments, instead):
    # Without an aggregator, a slide encoder would make each sample a slide, under a
    # preset whose name says tiles; with one, there is no bag to pool.
    message = rf"encoder 'titan' is slide-level, but this preset needs a tile-level encoder: .*{instead}"
    with pytest.raises(ValueError, match=message):
        getattr(presets, name)("titan", *arguments)


@pytest.mark.parametrize(
    ("name", "arguments", "instead"),
    [
        (
            "slide_classification", (),
            "use mil_classification to pool its tiles with an aggregator, or tile_classification",
        ),
        ("slide_regression", (), "use mil_regression to pool its tiles with an aggregator, or tile_regression"),
        # No preset trains survival on single tiles.
        ("slide_survival", ("cox",), r"use mil_survival to pool its tiles with an aggregator\.$"),
    ],
)
def test_a_slide_encoder_preset_refuses_a_tile_encoder_naming_the_presets_to_use(name, arguments, instead):
    message = rf"encoder 'uni2' is tile-level, but this preset needs a slide-level encoder: {instead}"
    with pytest.raises(ValueError, match=message):
        getattr(presets, name)("uni2", *arguments)


PATIENT_ENCODER_CALLS = [(name, arguments) for name, arguments, _ in TILE_ENCODER_PRESETS] + [
    ("slide_classification", ()),
    ("slide_regression", ()),
    ("slide_survival", ("cox",)),
]


@pytest.mark.parametrize(("name", "arguments"), PATIENT_ENCODER_CALLS, ids=[p[0] for p in PATIENT_ENCODER_CALLS])
def test_every_preset_refuses_a_patient_encoder(name, arguments):
    message = "encoder 'moozy' is patient-level: no preset covers patient-level encoders yet"
    with pytest.raises(ValueError, match=message):
        getattr(presets, name)("moozy", *arguments)


def test_each_call_returns_a_config_of_its_own():
    first = presets.mil_classification("uni2", "abmil")
    first.trainer.max_epochs = 3
    first.learner.params["head"].params["dropout"] = 0.5
    first.features.extraction.masks.min_coverage["tissue"] = 0.9
    second = presets.mil_classification("uni2", "abmil")
    assert second.trainer.max_epochs == 50
    assert second.learner.params["head"].params == {}
    assert second.features.extraction.masks.min_coverage == {"tissue": 0.1}


#: The presets that tile whole slides, each with its required arguments.
SLIDE_PRESETS = [
    ("mil_classification", {"aggregator": "abmil"}),
    ("mil_regression", {"aggregator": "abmil"}),
    ("mil_survival", {"aggregator": "abmil", "loss": "nll", "num_bins": 4}),
]


@pytest.mark.parametrize(("name", "arguments"), SLIDE_PRESETS, ids=[p[0] for p in SLIDE_PRESETS])
def test_a_slide_preset_tiles_at_the_given_spacing_and_the_encoder_s_tile_size(name, arguments):
    tissue = MaskSamplingConfig(pixel_mapping={"background": 0, "tissue": 1}, min_coverage={"tissue": 0.1})
    # Without the tiling arguments the tiling is unset, for the run to take the encoder's.
    extraction = getattr(presets, name)("uni2", **arguments).features.extraction
    assert (extraction.tiling, extraction.masks) == (None, tissue)

    extraction = getattr(presets, name)("uni2", **arguments, spacing=0.5).features.extraction
    # uni2 takes 224 px tiles; the tolerance and overlap are slide2vec's defaults.
    assert extraction.tiling == TilingConfig(
        requested_spacing_um=0.5, requested_tile_size_px=224, tolerance=0.05, overlap=0.0
    )
    assert extraction.masks == tissue


@pytest.mark.parametrize(("name", "arguments"), SLIDE_PRESETS, ids=[p[0] for p in SLIDE_PRESETS])
def test_check_refuses_a_preset_spacing_the_encoder_does_not_support(name, arguments):
    check(getattr(presets, name)("uni2", **arguments, spacing=0.5, tile_size=224))
    with pytest.raises(ValueError, match="encoder 'uni2' supports spacing 0.5 µm"):
        check(getattr(presets, name)("uni2", **arguments, spacing=1.0))


def test_the_encoder_s_own_tiling_given_as_arguments_keeps_the_experiment_id(tmp_path: Path):
    records, splits = _cohort(tmp_path, label=["a", "b"] * 4)

    def experiment_id(config: ExperimentConfig) -> str:
        config.cohort = CohortConfig(records=records, splits=splits)
        return build_experiment_spec(resolve(config, load_cohort(config))).experiment_id

    # The id digests the extraction as the run resolves it: uni2's preset is 0.5 µm, 224 px.
    explicit = presets.mil_classification("uni2", "abmil", spacing=0.5, tile_size=224)
    assert experiment_id(explicit) == experiment_id(presets.mil_classification("uni2", "abmil"))


def test_a_tile_size_alone_takes_the_encoder_s_spacing():
    tiling = presets.mil_classification("uni2", "abmil", tile_size=448).features.extraction.tiling
    assert (tiling.requested_spacing_um, tiling.requested_tile_size_px) == (0.5, 448)
    # virchow2 supports 0.25, 0.5, 1 and 2 µm and names no default: a spacing is needed,
    # while its tile size comes from the encoder.
    tiling = presets.mil_classification("virchow2", "abmil", spacing=1.0).features.extraction.tiling
    assert (tiling.requested_spacing_um, tiling.requested_tile_size_px) == (1.0, 224)
    with pytest.raises(ValueError, match="encoder 'virchow2' has no default spacing: pass spacing as well as tile_size"):
        presets.mil_survival("virchow2", "abmil", "nll", num_bins=4, tile_size=224)


def test_a_preset_s_arguments_are_checked():
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.tile_classification("not-an-encoder")
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.mil_classification("not-an-encoder", "abmil", spacing=0.5)
    # CLAM's label-aware auxiliary loss assumes classification.
    with pytest.raises(ValueError, match="not supported for survival"):
        check(presets.mil_survival("uni2", "clam_sb", "cox"))
    with pytest.raises(ValueError, match="Unknown decoder"):
        check(presets.segmentation("uni2", "not-a-decoder", 2))


def test_slide_and_tile_presets_resolve_against_a_cohort(tmp_path: Path):
    records, splits = _cohort(tmp_path / "labels", label=["a", "b", "c", "d"] * 2)
    for config in (presets.mil_classification("uni2", "abmil"), presets.tile_classification("uni2")):
        config.cohort = CohortConfig(records=records, splits=splits)
        head = resolve(config, load_cohort(config)).learner.params["head"]
        assert (head.params["num_classes"], head.params["label_map"]) == (4, {"a": 0, "b": 1, "c": 2, "d": 3})

    records, splits = _cohort(tmp_path / "survival", time=[1.0, 2.0, 3.0, 4.0] * 2, event=[1] * 8)
    config = presets.mil_survival("uni2", "abmil", "nll", num_bins=4)
    config.cohort = replace(config.cohort, records=records, splits=splits)
    params = resolve(config, load_cohort(config)).learner.params["head"].params
    assert (params["binning"], params["num_bins"]) == ("train_quantiles", 4)

    # Cox reads no bins.
    records, splits = _cohort(tmp_path / "cox", time=[1.0, 2.0] * 4, event=[1, 0] * 4)
    config = presets.mil_survival("uni2", "abmil", "cox")
    config.cohort = replace(config.cohort, records=records, splits=splits)
    assert resolve(config, load_cohort(config)).learner.params["head"].params["cox_window"] == 32

    records, splits = _cohort(tmp_path / "regression", value=[0.5, 1.5, 2.5, 3.5] * 2)
    config = presets.mil_regression("uni2", "abmil")
    config.cohort = replace(config.cohort, records=records, splits=splits)
    assert resolve(config, load_cohort(config)).learner.params["head"].params["num_outputs"] == 1
