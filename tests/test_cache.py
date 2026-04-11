"""Tests for soma.cache — shared feature-cache utilities."""

from __future__ import annotations

import json
import errno
import re
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
    resolve_cache_root,
    resolve_feature_payload_dir,
    resolve_tiling_cache,
    resolve_tiling_cache_root,
    resolve_hierarchical_cache,
    resolve_slide_cache,
    resolve_tile_cache,
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


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=256),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_tissue_mask_value(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=1),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=2),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_precision(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp32"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_output_variant(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls_patch_mean"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_backend(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="auto"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="openslide"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_build_tiling_cache_key_changes_with_preprocessing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tiling_cache_key(
        dataset=dataset,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
    )
    key_b = build_tiling_cache_key(
        dataset=dataset,
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
    with patch("soma.cache.resolve_backend") as resolve_backend:
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

    with patch("soma.cache.resolve_backend", side_effect=_fake_resolve_backend) as resolve_backend:
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
    )

    metadata = json.loads(resolution.metadata_path.read_text())
    assert metadata["requested_backend"] == "auto"
    assert metadata["backend"] == "openslide"
    assert metadata["backend_by_sample_id"] == {"s1": "openslide", "s2": "openslide"}
    assert resolution.complete is False


def test_resolve_tiling_cache_emits_miss_then_hit_logs(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    provenance = {
        "requested_backend": "openslide",
        "backend": "openslide",
        "backend_by_sample_id": {
            "s1": "openslide",
            "s2": "openslide",
        },
    }

    rich_reporter = SimpleNamespace(console=object(), progress=object())

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache.slide2vec_progress.emit_progress_log"
    ) as emit_progress_log:
        resolution = resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
            backend_provenance=provenance,
        )

    assert resolution.complete is False
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✗ tiling cache miss:")

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache._validate_tiling_cache_contents",
        return_value=CacheValidationResult(complete=True),
    ), patch("soma.cache.slide2vec_progress.emit_progress_log") as emit_progress_log:
        reused = resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
            backend_provenance=provenance,
        )

    assert reused.complete is True
    assert reused.reused is True
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✓ tiling cache hit:")


def test_resolve_tiling_cache_can_log_populated_for_refresh(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "tiling_cache"
    provenance = {
        "requested_backend": "openslide",
        "backend": "openslide",
        "backend_by_sample_id": {
            "s1": "openslide",
            "s2": "openslide",
        },
    }
    preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)

    rich_reporter = SimpleNamespace(console=object(), progress=object())

    resolve_tiling_cache(
        cache_root=cache_root,
        dataset=dataset,
        preprocessing=preprocessing,
        backend_provenance=provenance,
    )

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache._validate_tiling_cache_contents",
        return_value=CacheValidationResult(complete=True),
    ), patch("soma.cache.slide2vec_progress.emit_progress_log") as emit_progress_log:
        reused = resolve_tiling_cache(
            cache_root=cache_root,
            dataset=dataset,
            preprocessing=preprocessing,
            backend_provenance=provenance,
            complete_state="populated",
        )

    assert reused.complete is True
    assert reused.reused is True
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✓ tiling cache populated:")


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
        "coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        f"s1,/slides/s1.svs,,openslide,openslide,success,1,{resolution.artifacts_dir / 's1.npz'},{resolution.artifacts_dir / 's1.meta.json'},,\n"
        f"s2,/slides/s2.svs,,openslide,openslide,success,1,{resolution.artifacts_dir / 's2.npz'},{resolution.artifacts_dir / 's2.meta.json'},,\n",
        encoding="utf-8",
    )
    for sample_id in dataset.sample_ids:
        (resolution.artifacts_dir / f"{sample_id}.npz").write_bytes(b"npz")
        (resolution.artifacts_dir / f"{sample_id}.meta.json").write_text("{}", encoding="utf-8")

    write_tiling_cache_stub(tiling_dir=tmp_path / "run" / "tiling", cache_resolution=resolution)

    stub_process_list = pd.read_csv(tmp_path / "run" / "tiling" / "process_list.csv").set_index("sample_id")
    assert Path(stub_process_list.loc["s1", "coordinates_meta_path"]).is_absolute()
    assert Path(stub_process_list.loc["s1", "coordinates_meta_path"]).parent == resolution.artifacts_dir
    assert (tmp_path / "run" / "tiling" / "README.txt").is_file()


def test_write_tiling_cache_payload_rewrites_paths_into_cache(tmp_path: Path):
    dataset = _make_dataset(tmp_path, rows=[{"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"}])
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "s1.npz").write_bytes(b"npz")
    (live_dir / "s1.meta.json").write_text("{}", encoding="utf-8")
    (live_dir / "process_list.csv").write_text(
        "sample_id,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        f"s1,/slides/s1.svs,,openslide,openslide,success,1,{live_dir / 's1.npz'},{live_dir / 's1.meta.json'},,\n",
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
    assert len(list(resolution.artifacts_dir.glob("*.npz"))) == 1


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


def test_resolve_feature_cache_emits_miss_then_hit_logs(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"

    rich_reporter = SimpleNamespace(console=object(), progress=object())

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache.slide2vec_progress.emit_progress_log"
    ) as emit_progress_log:
        resolution = resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )

    assert resolution.complete is False
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✗ feature cache miss:")

    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for sample_id in dataset.sample_ids:
        torch.save(torch.randn(4, 16), resolution.features_dir / f"{sample_id}.pt")

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache.slide2vec_progress.emit_progress_log"
    ) as emit_progress_log:
        reused = resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )

    assert reused.complete is True
    assert reused.reused is True
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✓ feature cache hit:")


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
    torch.save(torch.randn(4, 16), resolution.features_dir / "s1.pt")

    rich_reporter = SimpleNamespace(console=object(), progress=object())
    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache.slide2vec_progress.emit_progress_log"
    ) as emit_progress_log:
        reused = resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )

    assert reused.complete is True
    assert reused.reused is True
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✓ feature cache hit:")


def test_resolve_hierarchical_cache_emits_feature_cache_logs(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"

    rich_reporter = SimpleNamespace(console=object(), progress=object())

    with patch("soma.cache.slide2vec_progress.get_progress_reporter", return_value=rich_reporter), patch(
        "soma.cache.slide2vec_progress.emit_progress_log"
    ) as emit_progress_log:
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

    assert resolution.complete is False
    emit_progress_log.assert_called_once()
    assert _strip_ansi(emit_progress_log.call_args.args[0]).startswith("✗ feature cache miss:")


def test_hierarchical_cache_key_changes_with_region_geometry(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_hierarchical_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224,
            requested_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_hierarchical_cache_key(
        dataset=dataset,
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
    dataset = _make_dataset(tmp_path)
    key_a = build_slide_cache_key(
        dataset=dataset,
        slide_encoder_name="prism",
        tile_cache_key="aaa111",
        execution=EncoderConfig(name="prism", precision="fp16"),
    )
    key_b = build_slide_cache_key(
        dataset=dataset,
        slide_encoder_name="prism",
        tile_cache_key="bbb222",
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
        torch.save(torch.randn(4, 16), resolution.features_dir / f"{sample_id}.pt")

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
    torch.save(torch.randn(4, 16), resolution.features_dir / "s1.pt")

    resumed = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert resumed.complete is False
    assert resumed.missing_sample_ids() == ["s2"]


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
    metadata["encoder_name"] = "other-encoder"
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="metadata mismatch"):
        resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )


def test_resolve_feature_payload_dir_understands_cache_dir(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "features"
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
        "soma.cache.torch.save",
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

    with patch("soma.cache.os.link", side_effect=OSError(errno.EXDEV, "Invalid cross-device link")), patch(
        "soma.cache.shutil.copyfile",
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
        torch.save(torch.randn(4, 9, 16), resolution.features_dir / f"{sample_id}.pt")

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
