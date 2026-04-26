"""Tests for soma.cache — shared feature-cache utilities."""

from __future__ import annotations

import json
import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import torch

from soma.cache import (
    CACHE_METADATA_NAME,
    CacheValidationResult,
    build_slide_cache_key,
    build_tiling_cache_key,
    build_hierarchical_cache_key,
    build_tile_cache_key,
    manifest_digest,
    probe_resolved_backends,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_feature_payload_dir,
    resolve_tiling_cache,
    resolve_tiling_cache_root,
    resolve_hierarchical_cache,
    resolve_slide_cache,
    resolve_tile_cache,
    write_feature_payload,
    write_tiling_cache_payload,
    write_tiling_cache_stub,
    write_cache_payload,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset


def _make_dataset(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        rows
        or [
            {"sample_id": "s2", "image_path": "/slides/s2.svs", "label": "normal"},
            {"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def test_manifest_digest_stable_under_row_order(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    rows = [
        {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": None,
        }
        for sample in dataset.samples.values()
    ]
    assert manifest_digest(rows) == manifest_digest(list(reversed(rows)))


def test_tile_cache_key_changes_with_preprocessing(tmp_path: Path):
    key_a = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=256),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_tissue_mask_value(tmp_path: Path):
    key_a = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=1),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=2),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_precision(tmp_path: Path):
    key_a = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp32"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_output_variant(tmp_path: Path):
    key_a = build_tile_cache_key(
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls"),
    )
    key_b = build_tile_cache_key(
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls_patch_mean"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_backend(tmp_path: Path):
    key_a = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="auto"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="openslide"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_build_tiling_cache_key_changes_with_preprocessing(tmp_path: Path):
    key_a = build_tiling_cache_key(
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
    )
    key_b = build_tiling_cache_key(
        preprocessing=PreprocessingConfig(requested_tile_size_px=256, requested_spacing_um=0.5),
    )
    assert key_a != key_b


def test_resolve_tiling_cache_root_is_sibling_of_feature_cache(tmp_path: Path):
    root = resolve_tiling_cache_root(
        CacheConfig(root_dir=tmp_path / "shared" / "feature_cache"),
        tiling_dir=tmp_path / "run" / "tiling",
    )
    assert root == tmp_path / "shared" / "tiling_cache"


def test_resolve_cache_root_uses_output_root_when_provided(tmp_path: Path):
    root = resolve_cache_root(
        CacheConfig(),
        feature_dir=tmp_path / "run" / "features",
        output_root=tmp_path / "outputs",
    )
    assert root == tmp_path / "outputs" / "feature_cache"


def test_resolve_tiling_cache_root_uses_output_root_when_provided(tmp_path: Path):
    root = resolve_tiling_cache_root(
        CacheConfig(),
        tiling_dir=tmp_path / "run" / "tiling",
        output_root=tmp_path / "outputs",
    )
    assert root == tmp_path / "outputs" / "tiling_cache"


def test_probe_resolved_backends_uses_explicit_backend_without_probe(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    with patch("soma.cache.keys.resolve_backend") as resolve_backend:
        mapping = probe_resolved_backends(
            dataset=dataset,
            requested_backend="openslide",
        )
    assert mapping == {"s1": "openslide", "s2": "openslide"}
    resolve_backend.assert_not_called()


def test_probe_resolved_backends_uses_runtime_backend_probe_for_auto(tmp_path: Path):
    dataset = _make_dataset(tmp_path)

    def _fake_resolve_backend(requested_backend, *, wsi_path, mask_path=None):
        del requested_backend, mask_path
        return SimpleNamespace(backend="cucim" if Path(wsi_path).name == "s1.svs" else "openslide")

    with patch("soma.cache.keys.resolve_backend", side_effect=_fake_resolve_backend) as resolve_backend:
        mapping = probe_resolved_backends(
            dataset=dataset,
            requested_backend="auto",
        )

    assert mapping == {"s1": "cucim", "s2": "openslide"}
    assert resolve_backend.call_count == 2


def test_resolve_tiling_cache_records_backend_provenance(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    provenance = {
        "requested_backend": "auto",
        "backend": "openslide",
        "backend_by_sample_id": {
            "s1": "openslide",
            "s2": "openslide",
        },
    }
    resolution = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        backend_provenance=provenance,
        requested_preprocessing={"backend": "auto", "requested_tile_size_px": None},
    )

    metadata = json.loads(resolution.metadata_path.read_text())
    assert metadata["requested_backend"] == "auto"
    assert metadata["requested_preprocessing"] == {"backend": "auto", "requested_tile_size_px": None}
    assert "raw_preprocessing" not in metadata
    assert resolution.complete is False


def test_write_tiling_cache_stub_points_to_shared_cache_paths(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    resolution = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide", "s2": "openslide"},
        },
    )
    canonical_process_list = resolution.cache_dir / "process_list.csv"
    canonical_process_list.write_text(
        "sample_id,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,"
        "coordinates_npz_path,coordinates_meta_path,mask_preview_path,tiling_preview_path,annotation,error,traceback\n"
        f"s1,/slides/s1.svs,,openslide,openslide,success,1,{resolution.artifacts_dir / 's1.coordinates.npz'},{resolution.artifacts_dir / 's1.coordinates.meta.json'},{resolution.previews_dir / 'mask' / 's1.jpg'},{resolution.previews_dir / 'tiling' / 's1.jpg'},,,\n"
        f"s2,/slides/s2.svs,,openslide,openslide,success,1,{resolution.artifacts_dir / 's2.coordinates.npz'},{resolution.artifacts_dir / 's2.coordinates.meta.json'},{resolution.previews_dir / 'mask' / 's2.jpg'},{resolution.previews_dir / 'tiling' / 's2.jpg'},,,\n",
        encoding="utf-8",
    )
    for sample_id in dataset.sample_ids:
        (resolution.artifacts_dir / f"{sample_id}.coordinates.npz").write_bytes(b"npz")
        (resolution.artifacts_dir / f"{sample_id}.coordinates.meta.json").write_text("{}", encoding="utf-8")
        (resolution.previews_dir / "mask" / f"{sample_id}.jpg").parent.mkdir(parents=True, exist_ok=True)
        (resolution.previews_dir / "mask" / f"{sample_id}.jpg").write_bytes(b"mask")
        (resolution.previews_dir / "tiling" / f"{sample_id}.jpg").parent.mkdir(parents=True, exist_ok=True)
        (resolution.previews_dir / "tiling" / f"{sample_id}.jpg").write_bytes(b"tiling")

    write_tiling_cache_stub(tiling_dir=tmp_path / "run" / "tiling", cache_resolution=resolution)

    stub_process_list = pd.read_csv(tmp_path / "run" / "tiling" / "process_list.csv").set_index("sample_id")
    assert Path(stub_process_list.loc["s1", "coordinates_meta_path"]).is_absolute()
    assert Path(stub_process_list.loc["s1", "coordinates_meta_path"]).parent == resolution.artifacts_dir
    assert Path(stub_process_list.loc["s1", "mask_preview_path"]).parent == resolution.previews_dir / "mask"
    assert Path(stub_process_list.loc["s1", "tiling_preview_path"]).parent == resolution.previews_dir / "tiling"
    assert (tmp_path / "run" / "tiling" / "README.txt").is_file()


def test_write_tiling_cache_payload_rewrites_paths_into_cache(tmp_path: Path):
    dataset = _make_dataset(tmp_path, rows=[{"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"}])
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "s1.coordinates.npz").write_bytes(b"npz")
    (live_dir / "s1.coordinates.meta.json").write_text("{}", encoding="utf-8")
    (live_dir / "s1.mask.jpg").write_bytes(b"mask")
    (live_dir / "s1.tiling.jpg").write_bytes(b"tiling")
    (live_dir / "process_list.csv").write_text(
        "sample_id,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,mask_preview_path,tiling_preview_path,annotation,error,traceback\n"
        f"s1,/slides/s1.svs,,openslide,openslide,success,1,{live_dir / 's1.coordinates.npz'},{live_dir / 's1.coordinates.meta.json'},{live_dir / 's1.mask.jpg'},{live_dir / 's1.tiling.jpg'},,,\n",
        encoding="utf-8",
    )
    resolution = resolve_tiling_cache(
        cache_root=tmp_path / "tiling_cache",
        dataset=dataset,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide"},
        },
    )

    write_tiling_cache_payload(live_dir=live_dir, cache_resolution=resolution)

    recorded = pd.read_csv(resolution.process_list_path)
    assert Path(recorded.loc[0, "coordinates_npz_path"]).parent == resolution.artifacts_dir
    assert Path(recorded.loc[0, "coordinates_npz_path"]).name == f"{resolution.cache_stem_by_id['s1']}.coordinates.npz"
    assert Path(recorded.loc[0, "coordinates_meta_path"]).name == f"{resolution.cache_stem_by_id['s1']}.coordinates.meta.json"
    assert Path(recorded.loc[0, "mask_preview_path"]).parent == resolution.previews_dir / "mask"
    assert Path(recorded.loc[0, "tiling_preview_path"]).parent == resolution.previews_dir / "tiling"
    assert "sample_cache_stem" in set(recorded.columns)
    assert len(list(resolution.artifacts_dir.glob("*.npz"))) == 1
    assert len(list((resolution.previews_dir / "mask").glob("*.jpg"))) == 1


def test_write_feature_payload_writes_tensor_directly_to_features_dir(tmp_path: Path):
    feature_dir = tmp_path / "feature_cache" / "slide" / "abc123" / "slide_embeddings"
    tensor = torch.ones(3, 4)

    output_path = write_feature_payload(
        feature_dir=feature_dir,
        sample_id="s1",
        tensor=tensor,
        metadata={"feature_dim": 4, "artifact_type": "slide_embeddings"},
    )

    assert output_path == feature_dir / "s1.pt"
    assert output_path.is_file()
    assert torch.equal(torch.load(output_path, weights_only=True, map_location="cpu"), tensor)
    metadata = json.loads((feature_dir / "s1.meta.json").read_text(encoding="utf-8"))
    assert metadata["feature_dim"] == 4
    assert metadata["artifact_type"] == "slide_embeddings"


def test_resolve_tiling_cache_accepts_hipt_region_size_metadata(tmp_path: Path):
    dataset = _make_dataset(
        tmp_path,
        rows=[{"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"}],
    )
    cache_root = tmp_path / "tiling_cache"
    preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        requested_region_size_px=1792,
        region_tile_multiple=8,
        hierarchical=True,
        read_tile_size_px=224,
        read_region_size_px=1792,
    )
    provenance = {
        "requested_backend": "openslide",
        "backend": "openslide",
        "backend_by_sample_id": {"s1": "openslide"},
    }

    resolution = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset,
        preprocessing=preprocessing,
        backend_provenance=provenance,
    )
    (resolution.cache_dir / "process_list.csv").write_text(
        "sample_id,tiling_status,backend,requested_backend\n"
        "s1,success,openslide,openslide\n",
        encoding="utf-8",
    )

    with (
        patch(
            "soma.cache.tiling.load_tiling_process_df",
            return_value=pd.DataFrame(
                [
                    {
                        "sample_id": "s1",
                        "sample_cache_stem": resolution.cache_stem_by_id["s1"],
                        "tiling_status": "success",
                        "backend": "openslide",
                        "requested_backend": "openslide",
                    }
                ]
            ),
        ),
    ):
        refreshed = resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=preprocessing,
            backend_provenance=provenance,
        )

    assert refreshed.complete is True
    assert refreshed.reused is True


def test_resolve_tile_cache_records_backend_provenance(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    provenance = {
        "requested_backend": "auto",
        "backend": "openslide",
        "backend_by_sample_id": {
            "s1": "openslide",
            "s2": "openslide",
        },
    }
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="auto"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
        backend_provenance=provenance,
    )

    metadata = json.loads(resolution.metadata_path.read_text())
    assert metadata["requested_backend"] == "auto"
    assert metadata["backend"] == "openslide"
    assert metadata["backend_by_sample_id"] == {"s1": "openslide", "s2": "openslide"}


def test_resolve_feature_cache_treats_known_empty_samples_as_complete(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"

    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    metadata["empty_sample_ids"] = ["s2"]
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 16), resolution.feature_path_for_id("s1"))
    record_sample_identity_signatures(resolution, ["s1"])

    reused = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )

    assert reused.complete is True
    assert reused.reused is True


def test_hierarchical_cache_key_changes_with_region_geometry(tmp_path: Path):
    key_a = build_hierarchical_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_hierarchical_cache_key(
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_region_size_px=896,
            region_tile_multiple=4,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_slide_cache_key_changes_with_upstream_tile_cache_key(tmp_path: Path):
    tile_dependency_a = {
        "tile_encoder_name": "virchow",
        "tile_preprocessing": {"requested_tile_size_px": 224},
        "tile_execution": {"output_variant": "default"},
    }
    tile_dependency_b = {
        "tile_encoder_name": "virchow",
        "tile_preprocessing": {"requested_tile_size_px": 256},
        "tile_execution": {"output_variant": "default"},
    }
    key_a = build_slide_cache_key(
        slide_encoder_name="prism",
        tile_dependency_signature=tile_dependency_a,
        execution=EncoderConfig(name="prism", precision="fp16"),
    )
    key_b = build_slide_cache_key(
        slide_encoder_name="prism",
        tile_dependency_signature=tile_dependency_b,
        execution=EncoderConfig(name="prism", precision="fp16"),
    )
    assert key_a != key_b


def test_resolve_tile_cache_reuses_complete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for sample_id in dataset.sample_ids:
        torch.save(torch.randn(4, 16), resolution.feature_path_for_id(sample_id))
    record_sample_identity_signatures(resolution, list(dataset.sample_ids))

    reused = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert reused.complete is True
    assert reused.reused is True


def test_resolve_tile_cache_marks_incomplete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 16), resolution.feature_path_for_id("s1"))
    record_sample_identity_signatures(resolution, ["s1"])

    resumed = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert resumed.complete is False
    assert resumed.missing_sample_ids() == ["s2"]


def test_resolve_tile_cache_logs_partial_state_when_some_samples_exist(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 16), resolution.feature_path_for_id("s1"))
    record_sample_identity_signatures(resolution, ["s1"])

    with patch("soma.cache.io.slide2vec_progress.emit_progress_log") as emit_progress_log:
        resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )

    messages = [str(call.args[0]) for call in emit_progress_log.call_args_list]
    assert any("feature cache partial" in message for message in messages)
    assert any(
        "1/2 feature file already materialized on disk; embedding the 1 missing sample"
        in message
        for message in messages
    )


def test_resolve_tile_cache_backfills_legacy_identity_metadata_from_manifest(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata.pop("sample_identity_signature_by_id", None)
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 16), resolution.feature_path_for_id("s1"))

    reused = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )

    refreshed_metadata = json.loads(reused.metadata_path.read_text())
    assert refreshed_metadata["sample_identity_signature_by_id"] == {
        "s1": reused.cache_stem_by_id["s1"],
        "s2": reused.cache_stem_by_id["s2"],
    }
    assert reused.complete is False
    assert reused.reused is False
    assert reused.missing_sample_ids() == ["s2"]


def test_resolve_tile_cache_logs_resolving_state(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    with patch("soma.cache.io.slide2vec_progress.emit_progress_log") as emit_progress_log:
        resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )
    messages = [str(call.args[0]) for call in emit_progress_log.call_args_list]
    assert any("resolving feature cache" in message for message in messages)


def test_resolve_tiling_cache_logs_resolving_state(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    with patch("soma.cache.io.slide2vec_progress.emit_progress_log") as emit_progress_log:
        resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
            backend_provenance={
                "requested_backend": "openslide",
                "backend": "openslide",
                "backend_by_sample_id": {"s1": "openslide", "s2": "openslide"},
            },
        )
    messages = [str(call.args[0]) for call in emit_progress_log.call_args_list]
    assert any("resolving tiling cache" in message for message in messages)


def test_resolve_cache_fails_on_metadata_mismatch(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata.pop("artifact_kind")
    metadata["encoder_name"] = "other-encoder"
    metadata["unexpected_field"] = "boom"
    metadata["schema_version"] = "v2"
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError) as excinfo:
        resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )

    message = str(excinfo.value)
    assert f"Feature cache metadata mismatch for {resolution.cache_dir}" in message
    assert "missing=[" in message
    assert "artifact_kind=" in message
    assert "extra=[" in message
    assert "unexpected_field=" in message
    assert "changed=[" in message
    assert "encoder_name:" in message
    assert "schema_version:" in message


def test_resolve_tiling_cache_reports_metadata_mismatch_details(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    resolution = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide", "s2": "openslide"},
        },
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata.pop("preprocessing")
    metadata["schema_version"] = "v2"
    metadata["unexpected_field"] = "boom"
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError) as excinfo:
        resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
            backend_provenance={
                "requested_backend": "openslide",
                "backend": "openslide",
                "backend_by_sample_id": {"s1": "openslide", "s2": "openslide"},
            },
        )

    message = str(excinfo.value)
    assert f"Tiling cache metadata mismatch for {resolution.cache_dir}" in message
    assert "missing=[" in message
    assert "preprocessing=" in message
    assert "extra=[" in message
    assert "unexpected_field=" in message
    assert "changed=[" in message
    assert "schema_version:" in message


def test_resolve_tiling_cache_reuses_shared_sample_across_datasets(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    cache_root = tmp_path / "tiling_cache"
    dataset_a = _make_dataset(
        tmp_path / "a",
        rows=[{"sample_id": "s1", "image_path": "/slides/shared.svs", "label": "tumor"}],
    )
    dataset_b = _make_dataset(
        tmp_path / "b",
        rows=[
            {"sample_id": "s1", "image_path": "/slides/shared.svs", "label": "tumor"},
            {"sample_id": "s2", "image_path": "/slides/other.svs", "label": "normal"},
        ],
    )
    preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    provenance_a = {
        "requested_backend": "openslide",
        "backend": "openslide",
        "backend_by_sample_id": {"s1": "openslide"},
    }
    resolution_a = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset_a,
        preprocessing=preprocessing,
        backend_provenance=provenance_a,
    )
    stem_a = resolution_a.cache_stem_by_id["s1"]
    npz_a = resolution_a.artifacts_dir / f"{stem_a}.coordinates.npz"
    meta_a = resolution_a.artifacts_dir / f"{stem_a}.coordinates.meta.json"
    mask_preview_a = resolution_a.previews_dir / "mask" / f"{stem_a}.jpg"
    tiling_preview_a = resolution_a.previews_dir / "tiling" / f"{stem_a}.jpg"
    npz_a.write_bytes(b"npz")
    meta_a.write_text("{}", encoding="utf-8")
    mask_preview_a.parent.mkdir(parents=True, exist_ok=True)
    mask_preview_a.write_bytes(b"mask")
    tiling_preview_a.parent.mkdir(parents=True, exist_ok=True)
    tiling_preview_a.write_bytes(b"tiling")
    resolution_a.process_list_path.write_text(
        "sample_id,sample_cache_stem,tiling_status,backend,requested_backend,coordinates_npz_path,coordinates_meta_path,mask_preview_path,tiling_preview_path\n"
        f"s1,{stem_a},success,openslide,openslide,{npz_a},{meta_a},{mask_preview_a},{tiling_preview_a}\n",
        encoding="utf-8",
    )

    resolution_b = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset_b,
        preprocessing=preprocessing,
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide", "s2": "openslide"},
        },
    )
    assert resolution_b.complete is False
    assert resolution_b.reused is False
    assert resolution_b.cache_stem_by_id["s1"] == stem_a


def test_resolve_tiling_cache_treats_changed_image_path_as_distinct_sample(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    cache_root = tmp_path / "tiling_cache"
    dataset_a = _make_dataset(
        tmp_path / "a",
        rows=[{"sample_id": "s1", "image_path": "/slides/path-a.svs", "label": "tumor"}],
    )
    dataset_b = _make_dataset(
        tmp_path / "b",
        rows=[{"sample_id": "s1", "image_path": "/slides/path-b.svs", "label": "tumor"}],
    )
    preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    resolution_a = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset_a,
        preprocessing=preprocessing,
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide"},
        },
    )
    stem_a = resolution_a.cache_stem_by_id["s1"]
    npz_a = resolution_a.artifacts_dir / f"{stem_a}.coordinates.npz"
    meta_a = resolution_a.artifacts_dir / f"{stem_a}.coordinates.meta.json"
    npz_a.write_bytes(b"npz")
    meta_a.write_text("{}", encoding="utf-8")
    resolution_a.process_list_path.write_text(
        "sample_id,sample_cache_stem,tiling_status,backend,requested_backend,coordinates_npz_path,coordinates_meta_path\n"
        f"s1,{stem_a},success,openslide,openslide,{npz_a},{meta_a}\n",
        encoding="utf-8",
    )

    resolution_b = resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset_b,
        preprocessing=preprocessing,
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s1": "openslide"},
        },
    )
    assert resolution_b.complete is False
    assert resolution_b.cache_stem_by_id["s1"] != stem_a


def test_resolve_feature_payload_dir_understands_cache_dir(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "tile_embeddings"
    features_dir.mkdir(parents=True)
    (cache_dir / CACHE_METADATA_NAME).write_text("{}")
    assert resolve_feature_payload_dir(cache_dir) == features_dir
    plain_dir = tmp_path / "plain_features"
    plain_dir.mkdir()
    assert resolve_feature_payload_dir(plain_dir) == plain_dir


def test_resolve_feature_payload_dir_understands_slide2vec_root(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    slide_dir = artifact_root / "slide_embeddings"
    hier_dir = artifact_root / "hierarchical_embeddings"
    slide_dir.mkdir(parents=True)
    hier_dir.mkdir(parents=True)
    assert resolve_feature_payload_dir(artifact_root) == slide_dir


def test_resolve_feature_payload_dir_prefers_hierarchical_embeddings(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    hier_dir = artifact_root / "hierarchical_embeddings"
    tile_dir = artifact_root / "tile_embeddings"
    hier_dir.mkdir(parents=True)
    tile_dir.mkdir(parents=True)
    assert resolve_feature_payload_dir(artifact_root) == hier_dir


def test_write_cache_payload_reuses_pt_artifacts_without_reserializing(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "tile_embeddings"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "s1.pt"
    torch.save(torch.ones(3, 7), artifact_path)
    cache_dir = tmp_path / "cache" / "features"

    artifact = SimpleNamespace(sample_id="s1", path=artifact_path)

    with patch(
        "soma.cache.io.torch.save",
        side_effect=AssertionError("torch.save should not be used"),
    ):
        feature_dim = write_cache_payload([artifact], feature_dir=cache_dir)

    cached_path = cache_dir / "s1.pt"
    assert feature_dim == 7
    assert cached_path.is_file()
    assert torch.load(cached_path, weights_only=True, map_location="cpu").shape == (3, 7)


def test_write_cache_payload_falls_back_to_content_copy_across_devices(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "tile_embeddings"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "s1.pt"
    torch.save(torch.ones(3, 7), artifact_path)
    cache_dir = tmp_path / "cache" / "features"

    artifact = SimpleNamespace(sample_id="s1", path=artifact_path)

    with patch("soma.cache.io.os.link", side_effect=OSError(errno.EXDEV, "Invalid cross-device link")), patch(
        "soma.cache.io.shutil.copyfile",
        wraps=__import__("shutil").copyfile,
    ) as copyfile:
        feature_dim = write_cache_payload([artifact], feature_dir=cache_dir)

    cached_path = cache_dir / "s1.pt"
    assert feature_dim == 7
    assert copyfile.called
    assert cached_path.is_file()
    assert torch.load(cached_path, weights_only=True, map_location="cpu").shape == (3, 7)


def test_resolve_hierarchical_cache_reuses_complete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_hierarchical_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for sample_id in dataset.sample_ids:
        torch.save(torch.randn(4, 9, 16), resolution.feature_path_for_id(sample_id))
    record_sample_identity_signatures(resolution, list(dataset.sample_ids))

    reused = resolve_hierarchical_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert reused.complete is True
    assert reused.reused is True


def test_resolve_tile_cache_reuses_shared_sample_across_datasets(tmp_path: Path):
    cache_root = tmp_path / "feature_cache"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dataset_a = _make_dataset(
        tmp_path / "a",
        rows=[{"sample_id": "s1", "image_path": "/slides/shared.svs", "label": "tumor"}],
    )
    dataset_b = _make_dataset(
        tmp_path / "b",
        rows=[
            {"sample_id": "s1", "image_path": "/slides/shared.svs", "label": "tumor"},
            {"sample_id": "s2", "image_path": "/slides/other.svs", "label": "normal"},
        ],
    )
    resolution_a = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset_a,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata_a = json.loads(resolution_a.metadata_path.read_text())
    metadata_a["feature_dim"] = 8
    resolution_a.metadata_path.write_text(json.dumps(metadata_a, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 8), resolution_a.feature_path_for_id("s1"))
    record_sample_identity_signatures(resolution_a, ["s1"])

    resolution_b = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset_b,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert resolution_b.complete is False
    assert resolution_b.missing_sample_ids() == ["s2"]


def test_resolve_tile_cache_treats_changed_image_path_as_distinct_sample(tmp_path: Path):
    cache_root = tmp_path / "feature_cache"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dataset_a = _make_dataset(
        tmp_path / "a",
        rows=[{"sample_id": "s1", "image_path": "/slides/path-a.svs", "label": "tumor"}],
    )
    dataset_b = _make_dataset(
        tmp_path / "b",
        rows=[{"sample_id": "s1", "image_path": "/slides/path-b.svs", "label": "tumor"}],
    )
    resolution_a = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset_a,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata_a = json.loads(resolution_a.metadata_path.read_text())
    metadata_a["feature_dim"] = 8
    resolution_a.metadata_path.write_text(json.dumps(metadata_a, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 8), resolution_a.feature_path_for_id("s1"))
    record_sample_identity_signatures(resolution_a, ["s1"])

    resolution_b = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset_b,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert resolution_b.complete is False
    assert resolution_b.missing_sample_ids() == ["s1"]
