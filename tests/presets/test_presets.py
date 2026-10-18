"""``soma.presets``: five registered functions, each a complete experiment from a few
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
    ("mil_classification", {"encoder": "uni2"}, "set", "multiclass_classification", "slide", "abmil"),
    ("tile_classification", {"encoder": "uni2"}, "set", "multiclass_classification", "tile", None),
    ("survival", {"encoder": "uni2", "aggregator": "transmil"}, "set", "survival", "slide", "transmil"),
    ("segmentation", {"encoder": "uni2", "decoder": "heavy_conv"}, "grid", "segmentation", "tile", "heavy_conv"),
    ("detection", {"encoder": "uni2", "decoder": "lightweight_conv"}, "grid", "detection", "tile", "lightweight_conv"),
]


def _cohort(root: Path, **columns) -> tuple[Path, Path]:
    ids = [f"s{index}" for index in range(8)]
    root.mkdir(parents=True, exist_ok=True)
    records, splits = root / "dataset.csv", root / "splits.csv"
    frame = {"sample_id": ids, "image_path": [f"/slides/{s}.tif" for s in ids], **columns}
    pd.DataFrame(frame).to_csv(records, index=False)
    pd.DataFrame({"sample_id": ids, "split": ["train"] * 5 + ["tune"] + ["test"] * 2}).to_csv(splits, index=False)
    return records, splits


def test_components_lists_the_five_presets_with_their_arguments():
    assert "presets" in soma.components()
    entries = {entry["name"]: entry for entry in soma.components("presets")}
    assert sorted(entries) == ["detection", "mil_classification", "segmentation", "survival", "tile_classification"]
    assert entries["mil_classification"]["params"] == {
        "encoder": None, "aggregator": "abmil", "spacing": None, "tile_size": None,
    }
    assert entries["mil_classification"]["required"] == ["encoder"]
    assert entries["tile_classification"]["params"] == {"encoder": None}
    assert entries["survival"]["params"] == {"encoder": None, "aggregator": None, "spacing": None, "tile_size": None}
    assert entries["survival"]["required"] == ["encoder", "aggregator"]
    assert entries["segmentation"]["required"] == ["encoder", "decoder"]
    assert entries["segmentation"]["params"] == {"encoder": None, "decoder": None, "num_classes": 2}
    assert entries["detection"]["required"] == ["encoder", "decoder"]
    assert entries["detection"]["params"] == {
        "encoder": None, "decoder": None, "num_classes": 1, "match_distance": 3.0,
    }
    assert all(entry["description"] for entry in entries.values())


@pytest.mark.parametrize(
    ("name", "arguments", "shape", "family", "unit", "pooling"), PRESETS, ids=[p[0] for p in PRESETS]
)
def test_a_preset_builds_a_complete_config(name, arguments, shape, family, unit, pooling):
    config = getattr(presets, name)(**arguments)
    assert isinstance(config, ExperimentConfig)
    check(config)
    assert config.features.encoder.name == "uni2"
    assert (learner_shape(config), head_family(config), sample_unit(config)) == (shape, family, unit)
    leaf = config.learner.params.get("aggregator") or config.learner.params.get("decoder")
    assert (None if leaf is None else leaf.name) == pooling


def test_the_mil_preset_defaults_to_abmil_and_takes_another_aggregator():
    assert presets.mil_classification("uni2").learner.params["aggregator"] == Component("abmil")
    assert presets.mil_classification("uni2", aggregator="mean_pool").learner.params["aggregator"] == Component("mean_pool")


def test_each_call_returns_a_config_of_its_own():
    first = presets.mil_classification("uni2")
    first.trainer.max_epochs = 3
    first.learner.params["head"].params["dropout"] = 0.5
    first.features.extraction.masks.min_coverage["tissue"] = 0.9
    second = presets.mil_classification("uni2")
    assert second.trainer.max_epochs == 50
    assert second.learner.params["head"].params == {}
    assert second.features.extraction.masks.min_coverage == {"tissue": 0.1}


#: The presets that tile whole slides, each with its required arguments.
SLIDE_PRESETS = [("mil_classification", {}), ("survival", {"aggregator": "abmil"})]


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
    explicit = presets.mil_classification("uni2", spacing=0.5, tile_size=224)
    assert experiment_id(explicit) == experiment_id(presets.mil_classification("uni2"))


def test_a_tile_size_alone_takes_the_encoder_s_spacing():
    tiling = presets.mil_classification("uni2", tile_size=448).features.extraction.tiling
    assert (tiling.requested_spacing_um, tiling.requested_tile_size_px) == (0.5, 448)
    # virchow2 supports 0.25, 0.5, 1 and 2 µm and names no default: a spacing is needed,
    # while its tile size comes from the encoder.
    tiling = presets.mil_classification("virchow2", spacing=1.0).features.extraction.tiling
    assert (tiling.requested_spacing_um, tiling.requested_tile_size_px) == (1.0, 224)
    with pytest.raises(ValueError, match="encoder 'virchow2' has no default spacing: pass spacing as well as tile_size"):
        presets.survival("virchow2", "abmil", tile_size=224)


def test_a_preset_s_arguments_are_checked():
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.tile_classification("not-an-encoder")
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.mil_classification("not-an-encoder", spacing=0.5)
    # CLAM's label-aware auxiliary loss assumes classification.
    with pytest.raises(ValueError, match="not supported for survival"):
        check(presets.survival("uni2", "clam_sb"))
    with pytest.raises(ValueError, match="Unknown decoder"):
        check(presets.segmentation("uni2", "not-a-decoder"))


def test_classification_and_survival_presets_resolve_against_a_cohort(tmp_path: Path):
    records, splits = _cohort(tmp_path / "labels", label=["a", "b", "c", "d"] * 2)
    for config in (presets.mil_classification("uni2"), presets.tile_classification("uni2")):
        config.cohort = CohortConfig(records=records, splits=splits)
        head = resolve(config, load_cohort(config)).learner.params["head"]
        assert (head.params["num_classes"], head.params["label_map"]) == (4, {"a": 0, "b": 1, "c": 2, "d": 3})

    records, splits = _cohort(tmp_path / "survival", time=[1.0, 2.0] * 4, event=[1, 0] * 4, bin=[0, 1, 2, 3] * 2)
    config = presets.survival("uni2", "abmil")
    config.cohort = replace(config.cohort, records=records, splits=splits)
    assert resolve(config, load_cohort(config)).learner.params["head"].params["num_bins"] == 4
