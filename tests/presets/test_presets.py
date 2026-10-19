"""``soma.presets``: six registered functions, each a complete experiment from a few
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
    ("survival", {"encoder": "uni2", "aggregator": "transmil", "loss": "nll", "num_bins": 4}, "set", "survival", "slide", "transmil"),
    ("survival", {"encoder": "uni2", "aggregator": "abmil", "loss": "cox"}, "set", "survival", "slide", "abmil"),
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


def test_components_lists_the_six_presets_with_their_arguments():
    assert "presets" in soma.components()
    entries = {entry["name"]: entry for entry in soma.components("presets")}
    assert sorted(entries) == [
        "detection", "mil_classification", "mil_regression", "segmentation", "survival", "tile_classification",
    ]
    for name in ("mil_classification", "mil_regression"):
        assert entries[name]["params"] == {"encoder": None, "aggregator": None, "spacing": None, "tile_size": None}
        assert entries[name]["required"] == ["encoder", "aggregator"]
    assert entries["tile_classification"]["params"] == {"encoder": None}
    assert entries["survival"]["params"] == {
        "encoder": None, "aggregator": None, "loss": None, "num_bins": None, "spacing": None, "tile_size": None,
    }
    assert entries["survival"]["required"] == ["encoder", "aggregator", "loss"]
    assert entries["segmentation"]["required"] == ["encoder", "decoder", "num_classes"]
    assert entries["segmentation"]["params"] == {"encoder": None, "decoder": None, "num_classes": None}
    assert entries["detection"]["required"] == ["encoder", "decoder", "num_classes"]
    assert entries["detection"]["params"] == {
        "encoder": None, "decoder": None, "num_classes": None, "match_distance": 3.0,
    }
    assert all(entry["description"] for entry in entries.values())


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


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("mil_classification", ("uni2",)),
        ("mil_regression", ("uni2",)),
        ("survival", ("uni2", "abmil")),
        ("segmentation", ("uni2", "linear")),
        ("detection", ("uni2", "linear")),
    ],
)
def test_a_preset_has_no_default_for_a_choice_that_changes_the_model(name, arguments):
    # The aggregator, the survival loss and the class count are the user's to name.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        getattr(presets, name)(*arguments)


def test_the_survival_loss_picks_the_head():
    # nll derives num_bins bins from each fold's training split: no bin column to compute.
    assert presets.survival("uni2", "abmil", "nll", num_bins=4).learner.params["head"] == Component(
        "discrete_survival", {"binning": "train_quantiles", "num_bins": 4}
    )
    # The MIL loader forwards one bag at a time, so Cox accumulates a 32-bag risk set.
    config = presets.survival("uni2", "abmil", "cox")
    assert config.learner.params["head"] == Component("cox_survival", {"cox_window": 32})
    assert config.loaders.batch_size == 1
    nll = presets.survival("uni2", "abmil", "nll", num_bins=4)
    del config.learner.params["head"], nll.learner.params["head"]
    assert config == nll  # the loss changes the head only
    with pytest.raises(ValueError, match="loss must be one of 'nll', 'cox', got 'coxph'"):
        presets.survival("uni2", "abmil", "coxph")


def test_the_survival_bin_count_belongs_to_nll_alone():
    with pytest.raises(ValueError, match="loss 'nll' needs num_bins"):
        presets.survival("uni2", "abmil", "nll")
    with pytest.raises(ValueError, match="loss 'cox' takes no num_bins"):
        presets.survival("uni2", "abmil", "cox", num_bins=4)


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
    ("survival", {"aggregator": "abmil", "loss": "nll", "num_bins": 4}),
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
        presets.survival("virchow2", "abmil", "nll", num_bins=4, tile_size=224)


def test_a_preset_s_arguments_are_checked():
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.tile_classification("not-an-encoder")
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.mil_classification("not-an-encoder", "abmil", spacing=0.5)
    # CLAM's label-aware auxiliary loss assumes classification.
    with pytest.raises(ValueError, match="not supported for survival"):
        check(presets.survival("uni2", "clam_sb", "cox"))
    with pytest.raises(ValueError, match="Unknown decoder"):
        check(presets.segmentation("uni2", "not-a-decoder", 2))


def test_slide_and_tile_presets_resolve_against_a_cohort(tmp_path: Path):
    records, splits = _cohort(tmp_path / "labels", label=["a", "b", "c", "d"] * 2)
    for config in (presets.mil_classification("uni2", "abmil"), presets.tile_classification("uni2")):
        config.cohort = CohortConfig(records=records, splits=splits)
        head = resolve(config, load_cohort(config)).learner.params["head"]
        assert (head.params["num_classes"], head.params["label_map"]) == (4, {"a": 0, "b": 1, "c": 2, "d": 3})

    records, splits = _cohort(tmp_path / "survival", time=[1.0, 2.0, 3.0, 4.0] * 2, event=[1] * 8)
    config = presets.survival("uni2", "abmil", "nll", num_bins=4)
    config.cohort = replace(config.cohort, records=records, splits=splits)
    params = resolve(config, load_cohort(config)).learner.params["head"].params
    assert (params["binning"], params["num_bins"]) == ("train_quantiles", 4)

    # Cox reads no bins.
    records, splits = _cohort(tmp_path / "cox", time=[1.0, 2.0] * 4, event=[1, 0] * 4)
    config = presets.survival("uni2", "abmil", "cox")
    config.cohort = replace(config.cohort, records=records, splits=splits)
    assert resolve(config, load_cohort(config)).learner.params["head"].params["cox_window"] == 32

    records, splits = _cohort(tmp_path / "regression", value=[0.5, 1.5, 2.5, 3.5] * 2)
    config = presets.mil_regression("uni2", "abmil")
    config.cohort = replace(config.cohort, records=records, splits=splits)
    assert resolve(config, load_cohort(config)).learner.params["head"].params["num_outputs"] == 1
