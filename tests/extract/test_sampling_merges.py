"""Sampling-side merges: list-valued ``masks.pixel_mapping`` entries (#484).

A sampling label may group several raw mask values; hs2p sums their coverage. The
vocabulary is hs2p's ``MaskSamplingConfig`` held as given (ADR 0019), so the merge
reaches hs2p unchanged and keys the tiling and feature caches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from soma.extract.cache import roi_sampling_signature
from soma.extract.cache.keys import build_roi_sampling_cache_key, build_tiling_cache_key
from soma.config import ExperimentConfig, FeaturesConfig
from tests.experiment_configs import experiment
from tests.extraction_configs import annotation_masks, extraction


def _keys(pixel_mapping: dict, min_coverage: dict | None = None) -> tuple[str, str, str]:
    """(tiling key, ROI sampling key, dense sampling-signature digest) for a vocabulary."""
    preprocessing = extraction(
        0.5, 256, method="otsu", masks=annotation_masks(pixel_mapping, min_coverage or {"tumor": 0.5})
    )
    dense = json.dumps(roi_sampling_signature(preprocessing), sort_keys=True, separators=(",", ":"))
    return (
        build_tiling_cache_key(preprocessing=preprocessing),
        build_roi_sampling_cache_key(preprocessing=preprocessing),
        hashlib.sha256(dense.encode("utf-8")).hexdigest()[:16],
    )


def test_list_order_does_not_change_any_cache_key():
    assert _keys({"background": 0, "tumor": [1, 2]}) == _keys({"background": 0, "tumor": [2, 1]})


def test_a_merge_changes_the_cache_keys():
    keys = [
        _keys({"background": 0, "tumor": [1, 2]}),
        _keys({"background": 0, "tumor": 1, "stroma": 2}),
        _keys({"background": 0, "tumor": 1}),
    ]
    assert len(set(keys)) == len(keys)


def test_a_one_value_list_is_the_scalar():
    """hs2p holds every label's values as a tuple, so ``[1]`` and ``1`` are one label."""
    assert _keys({"background": 0, "tumor": [1]}) == _keys({"background": 0, "tumor": 1})


def test_list_entries_round_trip_through_yaml(tmp_path: Path):
    config = experiment(
        decoder="lightweight_conv",
        features=FeaturesConfig(
            extraction=extraction(
                0.5,
                256,
                method="otsu",
                masks=annotation_masks({"background": 0, "tumor": [1, 2], "stroma": 3}, {"tumor": 0.5}),
            )
        ),
        head="segmentation",
        head_params={"classes": {"tumor": [1, 2], "stroma": [3]}, "ignore": [0]},
    )
    path = tmp_path / "cfg.yaml"
    config.to_yaml(path)

    assert yaml.safe_load(path.read_text())["features"]["extraction"]["masks"]["pixel_mapping"] == {
        "background": [0],
        "tumor": [1, 2],
        "stroma": [3],
    }
    assert ExperimentConfig.from_yaml(path).features.extraction.masks.pixel_mapping == {
        "background": (0,),
        "tumor": (1, 2),
        "stroma": (3,),
    }


@pytest.mark.parametrize(
    "pixel_mapping, match",
    [
        ({"tumor": [1, 2], "stroma": [2, 3]}, r"'tumor' and 'stroma' both map to 2"),
        ({"tumor": [1, 1]}, r"lists 1 more than once"),
        ({"tumor": [1, 256]}, r"outside the supported label range"),
    ],
)
def test_an_invalid_vocabulary_is_refused_when_the_yaml_is_read(tmp_path: Path, pixel_mapping, match):
    data = {
        "cohort": {"records": "d.csv", "splits": "s.csv"},
        "features": {"extraction": {"masks": {"pixel_mapping": pixel_mapping, "min_coverage": {"tumor": 0.5}}}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.from_yaml(path)


def test_training_classes_may_split_a_merged_sampling_label():
    """The layers are independent: a merged sampling label may be two training classes,
    and every raw value of a list entry counts as declared."""
    from soma.extract.dense.reader import check_classes_declared

    pixel_mapping = annotation_masks({"background": 0, "tumor": [1, 2]}).pixel_mapping
    check_classes_declared(
        {"classes": {"low_grade": [1], "high_grade": [2]}, "ignore": [0]}, pixel_mapping
    )
    with pytest.raises(ValueError, match="lists raw value 3"):
        check_classes_declared({"classes": {"tumor": [1, 3]}}, pixel_mapping)
