"""``soma.presets``: five registered functions, each a complete experiment from a few
typed arguments, edited by assigning fields (design §8.4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import soma
from soma import ExperimentConfig, presets
from soma.config import Component, CohortConfig
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
    assert entries["mil_classification"]["params"] == {"encoder": None, "aggregator": "abmil"}
    assert entries["mil_classification"]["required"] == ["encoder"]
    assert entries["tile_classification"]["params"] == {"encoder": None}
    assert entries["survival"]["required"] == ["encoder", "aggregator"]
    assert entries["segmentation"]["required"] == ["encoder", "decoder"]
    assert entries["detection"]["params"] == {"encoder": None, "decoder": None}
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


def test_a_preset_s_arguments_are_checked():
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'"):
        presets.tile_classification("not-an-encoder")
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
