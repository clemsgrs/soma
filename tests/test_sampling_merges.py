"""Sampling-side merges: list-valued ``preprocessing.masks.pixel_mapping`` (#484).

A sampling label may group several raw mask values; hs2p sums their coverage. The
merge lives in the sampling layer only, so it must reach hs2p unchanged and key the
tiling/feature caches, while scalar-only vocabularies keep today's cache keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from soma.cache.keys import _canonical_json, build_roi_sampling_cache_key, build_tiling_cache_key
from soma.config import (
    MasksConfig,
    PreprocessingConfig,
    SamplingConfig,
    load_config,
    save_config,
)
from soma.dense_slide_extraction import sampling_signature
from soma.slide2vec_adapter import _build_masks_block

_SAMPLING = SamplingConfig(strategy="joint", output_mode="merged")


def _preprocessing(masks: MasksConfig) -> PreprocessingConfig:
    return PreprocessingConfig(
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        tissue_method="otsu",
        masks=masks,
        sampling=_SAMPLING,
    )


def _keys(masks: MasksConfig) -> tuple[str, str, str]:
    """(tiling key, ROI sampling key, dense sampling-signature digest) for ``masks``."""
    preprocessing = _preprocessing(masks)
    dense = _canonical_json(sampling_signature(masks, _SAMPLING, preprocessing))
    return (
        build_tiling_cache_key(preprocessing=preprocessing),
        build_roi_sampling_cache_key(preprocessing=preprocessing),
        hashlib.sha256(dense.encode("utf-8")).hexdigest()[:16],
    )


def test_scalar_only_sampling_signatures_are_byte_stable():
    # Pinned from the scalar-only implementation: accepting lists must not invalidate
    # any existing tiling, ROI sampling or dense feature cache.
    masks = MasksConfig(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        min_coverage={"tumor": 0.5, "stroma": 0.1},
        colors={"tumor": [255, 0, 0]},
    )
    assert _keys(masks) == ("801e47f338b5b4de", "6007ed830f203477", "6bf09fe54a9e1a56")
    assert _canonical_json(
        sampling_signature(masks, _SAMPLING, _preprocessing(masks))
    ) == (
        '{"colors":{"tumor":[255,0,0]},"min_coverage":{"stroma":0.1,"tumor":0.5},'
        '"output_mode":"merged","overlap":0.0,"pixel_mapping":{"background":0,"stroma":2,'
        '"tumor":1},"spacing_um":0.5,"strategy":"joint","tile_size_px":[256,256]}'
    )


def test_list_order_does_not_change_any_cache_key():
    forward = MasksConfig(pixel_mapping={"background": 0, "tumor": [1, 2]})
    backward = MasksConfig(pixel_mapping={"background": 0, "tumor": [2, 1]})
    assert _keys(forward) == _keys(backward)


def test_a_merge_changes_the_cache_keys():
    merged = MasksConfig(pixel_mapping={"background": 0, "tumor": [1, 2]})
    single = MasksConfig(pixel_mapping={"background": 0, "tumor": 1, "stroma": 2})
    one_value_list = MasksConfig(pixel_mapping={"background": 0, "tumor": [1]})
    scalar = MasksConfig(pixel_mapping={"background": 0, "tumor": 1})
    keys = [_keys(masks) for masks in (merged, single, one_value_list, scalar)]
    assert len(set(keys)) == len(keys)


def test_mixed_scalar_and_list_entries_reach_hs2p_unchanged():
    masks = MasksConfig(
        pixel_mapping={"background": 0, "tumor": [2, 1], "stroma": 3},
        min_coverage={"tumor": 0.5},
    )
    assert masks.pixel_mapping == {"background": 0, "tumor": [2, 1], "stroma": 3}
    block = _build_masks_block(_preprocessing(masks))
    assert block["pixel_mapping"]["tumor"] == [2, 1]
    assert block["pixel_mapping"]["stroma"] == 3
    assert block["pixel_mapping"]["background"] == 0


def test_list_entries_round_trip_through_yaml(tmp_path: Path):
    from soma.config import DecoderConfig, PipelineConfig, TaskConfig

    config = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        preprocessing=_preprocessing(
            MasksConfig(
                pixel_mapping={"background": 0, "tumor": [1, 2], "stroma": 3},
                min_coverage={"tumor": 0.5},
            )
        ),
        task=TaskConfig(
            name="segmentation",
            params={"classes": {"tumor": [1, 2], "stroma": [3]}, "ignore": [0]},
        ),
    )
    path = tmp_path / "cfg.yaml"
    save_config(config, path)

    assert yaml.safe_load(path.read_text())["preprocessing"]["masks"]["pixel_mapping"] == {
        "background": 0,
        "tumor": [1, 2],
        "stroma": 3,
    }
    assert load_config(path).preprocessing.masks.pixel_mapping == {
        "background": 0,
        "tumor": [1, 2],
        "stroma": 3,
    }


@pytest.mark.parametrize(
    "pixel_mapping, match",
    [
        ({"tumor": [1, 2], "stroma": [2, 3]}, r"unique pixel values.*'tumor' and 'stroma' both list raw value 2"),
        ({"tumor": [1, 2], "stroma": 1}, r"unique pixel values.*'tumor' and 'stroma' both list raw value 1"),
        ({"tumor": 1, "stroma": 1}, r"unique pixel values.*'tumor' and 'stroma' both list raw value 1"),
        ({"tumor": [1, 1]}, r"masks\.pixel_mapping\['tumor'\] lists raw value 1 twice"),
        ({"tumor": []}, r"masks\.pixel_mapping\['tumor'\] lists no raw values"),
        ({"tumor": [1, 256]}, r"masks\.pixel_mapping\['tumor'\] raw values must be integers in \[0, 255\], got 256"),
        ({"tumor": -1}, r"masks\.pixel_mapping\['tumor'\] raw values must be integers in \[0, 255\], got -1"),
        ({"tumor": [1, 1.5]}, r"masks\.pixel_mapping\['tumor'\] raw values must be integers in \[0, 255\], got 1\.5"),
        ({"tumor": "1"}, r"masks\.pixel_mapping\['tumor'\] raw values must be integers in \[0, 255\], got '1'"),
        ({"tumor": True}, r"masks\.pixel_mapping\['tumor'\] raw values must be integers in \[0, 255\], got True"),
    ],
)
def test_masks_config_rejects_invalid_pixel_mapping(pixel_mapping, match):
    with pytest.raises(ValueError, match=match):
        MasksConfig(pixel_mapping=pixel_mapping)


def test_min_coverage_and_colors_still_validate_against_label_names():
    pixel_mapping = {"background": 0, "tumor": [1, 2]}
    with pytest.raises(ValueError, match="min_coverage references labels absent"):
        MasksConfig(pixel_mapping=pixel_mapping, min_coverage={"stroma": 0.1})
    with pytest.raises(ValueError, match="colors references labels absent"):
        MasksConfig(pixel_mapping=pixel_mapping, colors={"stroma": [1, 2, 3]})
    MasksConfig(pixel_mapping=pixel_mapping, min_coverage={"tumor": 0.5}, colors={"tumor": None})


def test_training_classes_may_split_a_merged_sampling_label():
    """The layers are independent: a merged sampling label may be two training classes,
    and every raw value of a list entry counts as declared."""
    from soma.dense.reader import check_classes_declared

    pixel_mapping = {"background": 0, "tumor": [1, 2]}
    check_classes_declared(
        {"classes": {"low_grade": [1], "high_grade": [2]}, "ignore": [0]}, pixel_mapping
    )
    with pytest.raises(ValueError, match="lists raw value 3"):
        check_classes_declared({"classes": {"tumor": [1, 3]}}, pixel_mapping)
