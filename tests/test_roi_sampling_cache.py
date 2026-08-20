"""Tests for the roi_sampling cache kind — cross-run reuse of slide-manifest ROI coords.

Fully offline — no hs2p, no slides. Everything is asserted through the module's public
resolve/write interface (prior art: tests/test_cache.py, tests/test_dense_cache.py);
no test reaches around it into the directory layout beyond corrupting one artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.cache import (
    build_roi_sampling_cache_key,
    resolve_roi_sampling_cache,
    write_roi_sampling_coords,
)
from soma.config import MasksConfig, PreprocessingConfig, SamplingConfig
from soma.dataset import SegmentationManifest


def _make_manifest(
    tmp_path: Path,
    *,
    label_mask_by_id: dict[str, str] | None = None,
) -> SegmentationManifest:
    label_mask_by_id = label_mask_by_id or {
        "s1": "/annotations/s1.tif",
        "s2": "/annotations/s2.tif",
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "image_path": f"/slides/{sample_id}.svs",
                "label_mask_path": label_mask,
            }
            for sample_id, label_mask in label_mask_by_id.items()
        ]
    ).to_csv(csv_path, index=False)
    return SegmentationManifest(csv_path)


def _preprocessing(
    *,
    tile_size: int = 512,
    colors: dict[str, list[int] | None] | None = None,
) -> PreprocessingConfig:
    return PreprocessingConfig(
        requested_tile_size_px=tile_size,
        masks=MasksConfig(
            pixel_mapping={"background": 0, "tumor": 1},
            min_coverage={"tumor": 0.1},
            colors=colors,
        ),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
    )


# --------------------------------------------------------------------------- #
# Roundtrip and hit semantics
# --------------------------------------------------------------------------- #


def test_roundtrip_written_coords_hit_and_load_back_identically(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert resolution.coords_by_id == {}
    assert resolution.miss_sample_ids == ["s1", "s2"]
    assert resolution.complete is False

    coords = {"s1": [(0, 0), (512, 1024)], "s2": [(2048, 0)]}
    write_roi_sampling_coords(cache_resolution=resolution, coords_by_sample_id=coords)

    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert reloaded.coords_by_id == coords
    assert reloaded.miss_sample_ids == []
    assert reloaded.complete is True


def test_zero_roi_slide_is_a_hit_loading_an_empty_list(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    write_roi_sampling_coords(
        cache_resolution=resolution,
        coords_by_sample_id={"s1": [], "s2": [(64, 128)]},
    )

    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert reloaded.coords_by_id == {"s1": [], "s2": [(64, 128)]}
    assert reloaded.miss_sample_ids == []
    assert reloaded.complete is True


def test_corrupt_artifact_is_a_single_slide_miss_siblings_still_hit(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    write_roi_sampling_coords(
        cache_resolution=resolution,
        coords_by_sample_id={"s1": [(0, 0)], "s2": [(512, 512)]},
    )
    coords_path = resolution.coords_dir / f"{resolution.cache_stem_by_id['s1']}.csv"
    coords_path.write_text("x,y\nnot-an-int,7\n", encoding="utf-8")

    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert reloaded.miss_sample_ids == ["s1"]
    assert reloaded.coords_by_id == {"s2": [(512, 512)]}
    assert reloaded.complete is False


def test_missing_artifact_is_a_single_slide_miss_siblings_still_hit(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    write_roi_sampling_coords(
        cache_resolution=resolution,
        coords_by_sample_id={"s1": [(0, 0)], "s2": [(512, 512)]},
    )
    (resolution.coords_dir / f"{resolution.cache_stem_by_id['s2']}.csv").unlink()

    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert reloaded.miss_sample_ids == ["s2"]
    assert reloaded.coords_by_id == {"s1": [(0, 0)]}


# --------------------------------------------------------------------------- #
# Metadata contract
# --------------------------------------------------------------------------- #


def test_fresh_directory_initializes_cleanly_and_resolves_again(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    assert resolution.metadata_path.is_file()
    assert resolution.manifest_path.is_file()
    # A second resolve against the untouched fresh directory must not raise.
    resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )


def test_metadata_mismatch_raises_hard_error(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    tampered = json.loads(resolution.metadata_path.read_text(encoding="utf-8"))
    tampered["preprocessing"]["requested_tile_size_px"] = 224
    resolution.metadata_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        resolve_roi_sampling_cache(
            cache_root=tmp_path / "cache",
            dataset=dataset,
            preprocessing=preprocessing,
        )


# --------------------------------------------------------------------------- #
# Identity: directory key and per-slide stems
# --------------------------------------------------------------------------- #


def test_changing_label_mask_path_changes_only_that_slides_stem(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    preprocessing = _preprocessing()
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=preprocessing,
    )
    write_roi_sampling_coords(
        cache_resolution=resolution,
        coords_by_sample_id={"s1": [(0, 0)], "s2": [(512, 512)]},
    )

    reannotated = _make_manifest(
        tmp_path / "reannotated",
        label_mask_by_id={
            "s1": "/annotations/s1.v2.tif",
            "s2": "/annotations/s2.tif",
        },
    )
    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=reannotated,
        preprocessing=preprocessing,
    )
    assert reloaded.key == resolution.key
    assert reloaded.cache_stem_by_id["s1"] != resolution.cache_stem_by_id["s1"]
    assert reloaded.cache_stem_by_id["s2"] == resolution.cache_stem_by_id["s2"]
    # The re-annotated slide misses (its coords live under the old stem); the sibling hits.
    assert reloaded.miss_sample_ids == ["s1"]
    assert reloaded.coords_by_id == {"s2": [(512, 512)]}


def test_sampling_determining_knob_changes_the_directory_key(tmp_path: Path):
    key_512 = build_roi_sampling_cache_key(preprocessing=_preprocessing(tile_size=512))
    key_256 = build_roi_sampling_cache_key(preprocessing=_preprocessing(tile_size=256))
    assert key_512 != key_256


def test_min_coverage_change_changes_the_directory_key(tmp_path: Path):
    base = _preprocessing()
    changed = PreprocessingConfig(
        requested_tile_size_px=512,
        masks=MasksConfig(
            pixel_mapping={"background": 0, "tumor": 1},
            min_coverage={"tumor": 0.5},
        ),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
    )
    assert build_roi_sampling_cache_key(preprocessing=base) != build_roi_sampling_cache_key(
        preprocessing=changed
    )


def test_mask_preview_colors_change_neither_key_nor_stems(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    plain = _preprocessing(colors=None)
    colored = _preprocessing(colors={"tumor": [255, 0, 0]})
    assert build_roi_sampling_cache_key(preprocessing=plain) == build_roi_sampling_cache_key(
        preprocessing=colored
    )

    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=plain,
    )
    write_roi_sampling_coords(
        cache_resolution=resolution,
        coords_by_sample_id={"s1": [(0, 0)], "s2": []},
    )
    # A colors-only change resolves to the same directory and hits everywhere.
    reloaded = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=colored,
    )
    assert reloaded.cache_dir == resolution.cache_dir
    assert reloaded.cache_stem_by_id == resolution.cache_stem_by_id
    assert reloaded.complete is True


def test_key_payload_is_exactly_kind_plus_preprocessing_no_schema_version(tmp_path: Path):
    """The roi_sampling key hashes {artifact_kind, preprocessing} and nothing else.

    In particular no SCHEMA_VERSION (the #364 scoping decision) and no engine version:
    the key must equal a hash of the two-field payload recomputed here from the public
    preprocessing signature.
    """
    import hashlib

    from soma.cache import preprocessing_signature

    preprocessing = _preprocessing()
    payload = {
        "artifact_kind": "roi_sampling",
        "preprocessing": preprocessing_signature(preprocessing),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert build_roi_sampling_cache_key(preprocessing=preprocessing) == expected


def test_write_rejects_sample_ids_outside_the_resolution(tmp_path: Path):
    dataset = _make_manifest(tmp_path)
    resolution = resolve_roi_sampling_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        preprocessing=_preprocessing(),
    )
    with pytest.raises(ValueError, match="unknown"):
        write_roi_sampling_coords(
            cache_resolution=resolution,
            coords_by_sample_id={"stranger": [(0, 0)]},
        )
