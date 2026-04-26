"""Tests for soma.extraction — slide2vec-backed extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import slide2vec.progress as slide2vec_progress
import torch
from hs2p import SlideSpec
from slide2vec import ExecutionOptions

import soma.cache as cache_mod
from soma.cache import record_empty_sample_ids, record_feature_dim, record_sample_identity_signatures
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig as _PreprocessingConfig
from soma.dataset import Dataset
from soma.features import FeatureStore
from slide2vec.encoders.registry import encoder_registry
from soma.extraction import FeatureExtractor, _load_model, _run_with_coordinates, _validate_runtime
from soma.slide2vec_adapter import LoadedTiling, build_preprocessing_config, load_tilings
from soma.tile_extraction import TileFeatureExtractor, _install_tile_embedding_summary_patch


_TEST_TILE = "_cutover_tile"
_TEST_SLIDE = "_cutover_slide"
_TEST_PATIENT = "_cutover_patient"
_TEST_MULTI = "_cutover_multi_spacing"
_TISSUE_METHOD_SENTINEL = object()


def PreprocessingConfig(*args, tissue_method=_TISSUE_METHOD_SENTINEL, **kwargs):
    if tissue_method is _TISSUE_METHOD_SENTINEL:
        kwargs.setdefault("tissue_method", "hsv")
    else:
        kwargs["tissue_method"] = tissue_method
    return _PreprocessingConfig(*args, **kwargs)


def _register_test_encoders() -> None:
    if _TEST_TILE not in encoder_registry:
        encoder_registry.register(
            _TEST_TILE,
            object,
            metadata={
                "level": "tile",
                "input_size": 224,
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-tile",
            },
        )
    if _TEST_SLIDE not in encoder_registry:
        encoder_registry.register(
            _TEST_SLIDE,
            object,
            metadata={
                "level": "slide",
                "tile_encoder": _TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-slide",
            },
        )
    if _TEST_PATIENT not in encoder_registry:
        encoder_registry.register(
            _TEST_PATIENT,
            object,
            metadata={
                "level": "patient",
                "tile_encoder": _TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-patient",
            },
        )
    if _TEST_MULTI not in encoder_registry:
        encoder_registry.register(
            _TEST_MULTI,
            object,
            metadata={
                "level": "tile",
                "input_size": 224,
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": [0.25, 0.5],
                "precision": "fp16",
                "source": "test/cutover-multi",
            },
        )


_register_test_encoders()


@pytest.fixture(autouse=True)
def _force_cpu_for_default_extraction_paths():
    with patch("soma.extraction.torch.cuda.is_available", return_value=False), patch(
        "soma.extraction.torch.cuda.device_count",
        return_value=1,
    ):
        yield


def _make_dataset(tmp_path: Path, *, with_mask: bool = False) -> Dataset:
    rows: dict[str, list[object]] = {
        "sample_id": ["s0"],
        "image_path": [str(tmp_path / "s0.svs")],
        "label": ["tumor"],
    }
    if with_mask:
        rows["mask_path"] = [str(tmp_path / "s0-mask.tif")]
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _make_patient_dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "patient-dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "patient_id": "p0",
                "image_path": str(tmp_path / "s0.svs"),
                "label": "tumor",
            }
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _make_two_slide_patient_dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "two-slide-patient-dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "patient_id": "p0",
                "image_path": str(tmp_path / "s0.svs"),
                "label": "tumor",
            },
            {
                "sample_id": "s1",
                "patient_id": "p0",
                "image_path": str(tmp_path / "s1.svs"),
                "label": "tumor",
            },
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _tiling(sample_id: str = "s0") -> object:
    return SimpleNamespace(
        x=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        y=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        tissue_fractions=torch.tensor([1.0, 1.0], dtype=torch.float32).numpy(),
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        read_tile_size_px=224,
        read_spacing_um=0.5,
        tile_size_lv0=224,
        read_level=0,
        use_padding=True,
        is_within_tolerance=True,
        sample_id=sample_id,
        coordinates_npz_path=Path("/tmp/coords.npz"),
        coordinates_meta_path=Path("/tmp/coords.meta.json"),
    )


def _artifact(
    *,
    sample_id: str,
    output_dir: Path,
    kind: str,
    tensor: torch.Tensor,
):
    kind_dir = output_dir / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    path = kind_dir / f"{sample_id}.pt"
    meta_path = kind_dir / f"{sample_id}.meta.json"
    torch.save(tensor, path)
    feature_dim_for_meta = int(tensor.shape[-1] if tensor.ndim >= 2 else tensor.shape[0])
    meta_path.write_text(
        json.dumps({"artifact_type": kind, "feature_dim": feature_dim_for_meta}),
        encoding="utf-8",
    )
    if kind == "tile_embeddings":
        return SimpleNamespace(
            sample_id=sample_id,
            path=path,
            metadata_path=meta_path,
            format="pt",
            feature_dim=int(tensor.shape[1]),
            num_tiles=int(tensor.shape[0]),
        )
    if kind == "hierarchical_embeddings":
        return SimpleNamespace(
            sample_id=sample_id,
            path=path,
            metadata_path=meta_path,
            format="pt",
            feature_dim=int(tensor.shape[-1]),
            num_regions=int(tensor.shape[0]),
            tiles_per_region=int(tensor.shape[1]),
        )
    return SimpleNamespace(
        sample_id=sample_id,
        path=path,
        metadata_path=meta_path,
        format="pt",
        feature_dim=int(tensor.shape[0]),
    )


def test_build_execution_options_uses_cpu_budget_for_tiling_workers(monkeypatch, tmp_path: Path):
    import slide2vec.api as slide2vec_api
    from soma import slide2vec_adapter as adapter

    monkeypatch.setattr(slide2vec_api, "cpu_worker_limit", lambda: 24)
    monkeypatch.setattr(slide2vec_api, "slurm_cpu_limit", lambda: 24)

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE),
        execution=ExecutionConfig(),
        output_dir=tmp_path,
        num_gpus=None,
        save_tile_embeddings=False,
    )

    assert execution.num_workers_per_gpu == 16  # capped at 16
    assert execution.num_preprocessing_workers == 24
    assert execution.prefetch_factor == 4


def test_build_execution_options_forwards_explicit_num_workers_per_gpu(tmp_path: Path):
    from soma import slide2vec_adapter as adapter

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE),
        execution=ExecutionConfig(num_workers_per_gpu=6),
        output_dir=tmp_path,
        num_gpus=None,
        save_tile_embeddings=False,
    )

    assert execution.num_workers_per_gpu == 6


def test_build_execution_options_forwards_worker_pipeline_tuning(tmp_path: Path):
    from soma import slide2vec_adapter as adapter

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(
                num_workers_per_gpu=6,
                prefetch_factor=8,
            ),
            output_dir=tmp_path,
            num_gpus=None,
            save_tile_embeddings=False,
        )

    assert execution.num_workers_per_gpu == 6
    assert execution.prefetch_factor == 8


def test_load_model_forwards_allow_non_recommended_settings():
    with patch("soma.extraction.orchestration.Model.from_preset") as from_preset:
        _load_model(
            _TEST_TILE,
            output_variant="default",
            allow_non_recommended_settings=True,
        )

    assert from_preset.call_args.kwargs["allow_non_recommended_settings"] is True


def test_validate_runtime_uses_resolved_tile_size_for_hierarchical_runs():
    captured: dict[str, object] = {}

    def _fake_validate(encoder_name: str, **kwargs):
        captured["encoder_name"] = encoder_name
        captured.update(kwargs)

    preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        requested_region_size_px=1792,
        region_tile_multiple=8,
    )
    hierarchical_tiling = SimpleNamespace(
        requested_tile_size_px=1792,
        requested_spacing_um=0.5,
    )

    with patch("soma.extraction.extractor.validate_slide2vec_encoder_config", side_effect=_fake_validate):
        _validate_runtime(
            encoder_name=_TEST_TILE,
            output_variant="default",
            encoder=EncoderConfig(name=_TEST_TILE),
            preprocessing=preprocessing,
            tiling_results=[hierarchical_tiling],
        )

    assert captured["encoder_name"] == _TEST_TILE
    assert captured["requested_tile_size_px"] == 224
    assert captured["requested_spacing_um"] == 0.5


def test_validate_runtime_forwards_allow_non_recommended_settings():
    captured: dict[str, object] = {}

    def _fake_validate(encoder_name: str, **kwargs):
        captured["encoder_name"] = encoder_name
        captured.update(kwargs)

    preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=1.0,
    )
    tiling_result = SimpleNamespace(requested_tile_size_px=224, requested_spacing_um=1.0)

    with patch("soma.extraction.extractor.validate_slide2vec_encoder_config", side_effect=_fake_validate):
        _validate_runtime(
            encoder_name=_TEST_TILE,
            output_variant="default",
            encoder=EncoderConfig(name=_TEST_TILE, allow_non_recommended_settings=True),
            preprocessing=preprocessing,
            tiling_results=[tiling_result],
        )

    assert captured["allow_non_recommended"] is True


def test_preprocess_delegates_to_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=ExecutionConfig(num_preprocessing_workers=0),
    )
    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=SimpleNamespace(complete=False, metadata={"backend_by_sample_id": {"s0": "openslide"}}),
    ), patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        mock_instance = MockPipeline.return_value
        extractor.preprocess(tiling_dir=tmp_path / "tiling")
    MockPipeline.assert_called_once()
    assert MockPipeline.call_args.kwargs["execution"].num_preprocessing_workers == 0
    mock_instance.run.assert_called_once()


def test_preprocess_validates_encoder_settings_before_tiling(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=1.0),
        execution=ExecutionConfig(num_preprocessing_workers=0),
    )

    with patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        with pytest.raises(ValueError, match="allow_non_recommended_settings"):
            extractor.preprocess(tiling_dir=tmp_path / "tiling")

    MockPipeline.assert_not_called()


def test_preprocess_forwards_mask_path_to_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path, with_mask=True)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=ExecutionConfig(num_preprocessing_workers=0),
    )

    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=SimpleNamespace(
            complete=False,
            metadata={"backend_by_sample_id": {"s0": "openslide"}, "requested_backend": "auto"},
        ),
    ), patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        mock_instance = MockPipeline.return_value
        extractor.preprocess(tiling_dir=tmp_path / "tiling")

    slides = mock_instance.run.call_args.kwargs["slides"]
    assert len(slides) == 1
    assert slides[0].sample_id == "s0"
    assert slides[0].mask_path == Path(tmp_path / "s0-mask.tif")


def test_preprocess_uses_precomputed_mask_method_when_every_slide_has_mask(tmp_path: Path):
    dataset = _make_dataset(tmp_path, with_mask=True)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=ExecutionConfig(num_preprocessing_workers=0),
    )

    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=SimpleNamespace(
            complete=False,
            metadata={"backend_by_sample_id": {"s0": "openslide"}, "requested_backend": "auto"},
        ),
    ), patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        extractor.preprocess(tiling_dir=tmp_path / "tiling")

    preprocessing = MockPipeline.call_args.args[1]
    assert preprocessing.segmentation["method"] == "precomputed_mask"


def test_preprocess_skips_live_tiling_on_complete_tiling_cache_hit(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
    )
    cache_dir = tmp_path / "tiling_cache" / "abc123"
    artifacts_dir = cache_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (cache_dir / "cache_metadata.json").write_text("{}", encoding="utf-8")
    (cache_dir / "process_list.csv").write_text(
        "sample_id,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        f"s0,/slides/s0.svs,,openslide,openslide,success,1,{artifacts_dir / 's0.npz'},{artifacts_dir / 's0.meta.json'},,\n",
        encoding="utf-8",
    )
    (artifacts_dir / "s0.npz").write_bytes(b"npz")
    (artifacts_dir / "s0.meta.json").write_text("{}", encoding="utf-8")

    fake_resolution = SimpleNamespace(
        cache_dir=cache_dir,
        process_list_path=cache_dir / "process_list.csv",
        artifacts_dir=artifacts_dir,
        complete=True,
        metadata={"cache_key": "abc123"},
    )

    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=fake_resolution,
    ), patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        extractor.preprocess(tiling_dir=tmp_path / "tiling")

    assert not MockPipeline.called
    assert (tmp_path / "tiling" / "README.txt").is_file()
    recorded = pd.read_csv(tmp_path / "tiling" / "process_list.csv").set_index("sample_id")
    assert Path(recorded.loc["s0", "coordinates_meta_path"]).parent == artifacts_dir


def test_preprocess_rewrites_stale_local_process_list_when_cache_hit(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
    )
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    (tiling_dir / "process_list.csv").write_text("sample_id,tiling_status\ns0,success\n", encoding="utf-8")

    cache_dir = tmp_path / "tiling_cache" / "abc123"
    artifacts_dir = cache_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (cache_dir / "cache_metadata.json").write_text("{}", encoding="utf-8")
    (cache_dir / "process_list.csv").write_text(
        "sample_id,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        f"s0,/slides/s0.svs,,openslide,openslide,success,1,{artifacts_dir / 's0.npz'},{artifacts_dir / 's0.meta.json'},,\n",
        encoding="utf-8",
    )
    (artifacts_dir / "s0.npz").write_bytes(b"npz")
    (artifacts_dir / "s0.meta.json").write_text("{}", encoding="utf-8")

    fake_resolution = SimpleNamespace(
        cache_dir=cache_dir,
        process_list_path=cache_dir / "process_list.csv",
        artifacts_dir=artifacts_dir,
        complete=True,
        metadata={"cache_key": "abc123"},
    )

    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=fake_resolution,
    ), patch("soma.extraction.extractor.Pipeline", autospec=True) as MockPipeline:
        extractor.preprocess(tiling_dir=tiling_dir, skip_existing=True)

    assert not MockPipeline.called
    recorded = pd.read_csv(tiling_dir / "process_list.csv").set_index("sample_id")
    assert Path(recorded.loc["s0", "coordinates_meta_path"]).parent == artifacts_dir


def test_preprocess_uses_output_root_for_tiling_cache_when_cache_root_omitted(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
        output_root=tmp_path / "outputs",
    )

    with patch("soma.extraction.extractor.probe_resolved_backends", return_value={"s0": "openslide"}), patch(
        "soma.extraction.extractor.resolve_tiling_cache",
        return_value=SimpleNamespace(complete=False, metadata={"backend_by_sample_id": {"s0": "openslide"}}),
    ) as resolve_tiling_cache, patch("soma.extraction.extractor.Pipeline", autospec=True):
        extractor.preprocess(tiling_dir=tmp_path / "run" / "tiling")

    assert resolve_tiling_cache.call_args.kwargs["cache_root"] == tmp_path / "outputs" / "tiling_cache"


def test_extract_uses_output_root_for_feature_cache_when_cache_root_omitted(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
        output_root=tmp_path / "outputs",
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(
                sample_id="s0",
                image_path=Path("/tmp/s0.svs"),
                mask_path=None,
                spacing_at_level_0=None,
            ),
            tiling_result=_tiling(),
        )
    ]
    fake_store_dir = tmp_path / "outputs" / "feature_cache" / "tile" / "abc123"
    (fake_store_dir / "tile_embeddings").mkdir(parents=True, exist_ok=True)
    (fake_store_dir / "cache_metadata.json").write_text(
        '{"feature_type": "bag", "feature_dim": 8, "sample_ids": ["s0"], "cache_key": "abc123", "encoder_name": "_cutover_tile", "execution": {"output_variant": "default"}}',
        encoding="utf-8",
    )
    torch.save(torch.ones(2, 8), fake_store_dir / "tile_embeddings" / "s0.pt")
    (fake_store_dir / "tile_embeddings" / "s0.meta.json").write_text(
        '{"artifact_type": "tile_embeddings", "feature_dim": 8, "num_tiles": 2}',
        encoding="utf-8",
    )

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_extract_tile_cached",
        autospec=True,
        return_value=FeatureStore(fake_store_dir),
    ) as extract_tile_cached:
        store = extractor.extract(feature_dir=tmp_path / "run" / "features", tiling_dir=tmp_path / "run" / "tiling")

    assert extract_tile_cached.call_args.kwargs["cache_root"] == tmp_path / "outputs" / "feature_cache"
    assert store.load("s0").shape == (2, 8)


def test_preprocess_requires_tissue_method_without_precomputed_masks(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        _PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5, backend="openslide"),
    )

    with pytest.raises(ValueError, match="tissue_method is required"):
        extractor.preprocess(tiling_dir=tmp_path / "tiling")


def test_build_preprocessing_config_uses_segmentation_method_not_use_hsv():
    from soma.slide2vec_adapter import build_preprocessing_config

    preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        tissue_method="otsu",
        sam2_device="cuda:0",
        sam2_num_workers=4,
    )

    config = build_preprocessing_config(preprocessing)

    assert config.segmentation["method"] == "otsu"
    assert "use_hsv" not in config.segmentation
    assert config.segmentation["sam2_num_workers"] == 4
    assert config.segmentation["sam2_device"] == "cuda:0"
    assert config.preview["save_mask_preview"] is True
    assert config.preview["save_tiling_preview"] is True
    assert config.preview["downsample"] == 32
    assert config.preview["tissue_contour_color"] == (37, 94, 59)
    assert config.preview["mask_overlay_alpha"] == pytest.approx(0.5)


def test_load_tilings_records_requested_and_actual_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _make_dataset(tmp_path, with_mask=True)
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()

    process_df = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "image_path": str(tmp_path / "s0.svs"),
                "mask_path": str(tmp_path / "s0-mask.tif"),
                "tiling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": str(tmp_path / "s0.npz"),
                "coordinates_meta_path": str(tmp_path / "s0.meta.json"),
                "error": "",
                "traceback": "",
            }
        ]
    )
    (tiling_dir / "process_list.csv").write_text("sample_id,tiling_status\ns0,success\n", encoding="utf-8")
    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_process_df", lambda path: process_df)
    monkeypatch.setattr(
        "soma.slide2vec_adapter.load_tiling_result_from_row",
        lambda row: SimpleNamespace(requested_backend="auto", backend="openslide"),
    )
    monkeypatch.setattr("soma.slide2vec_adapter.validate_tiling_result_provenance", lambda *args, **kwargs: None)

    loaded = load_tilings(
        dataset=dataset,
        tiling_dir=tiling_dir,
        tissue_mask_tissue_value=1,
    )

    assert loaded[0].requested_backend == "auto"
    assert loaded[0].backend == "openslide"


def test_extract_tile_features_returns_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ) as embed_tiles:
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert embed_tiles.called
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_tile_feature_extractor_does_not_require_eval_method(tmp_path: Path):
    image_path = tmp_path / "tile.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(image_path)],
            "label": ["tumor"],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    class _FakeEncoder:
        def encode_tiles(self, batch):
            return torch.ones(batch.shape[0], 4)

    fake_loaded = SimpleNamespace(
        model=_FakeEncoder(),
        transforms=lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    with patch("soma.tile_extraction.load_model", return_value=fake_loaded):
        store = TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=False),
        ).run(feature_dir=tmp_path / "features")

    assert store.available_samples == ["s0"]
    assert torch.equal(store.load("s0"), torch.ones(4))


def test_tile_feature_extractor_keeps_encoder_inputs_in_float32(tmp_path: Path):
    image_path = tmp_path / "tile.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(image_path)],
            "label": ["tumor"],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    class _FakeEncoder:
        def encode_tiles(self, batch):
            assert batch.dtype == torch.float32
            return torch.ones(batch.shape[0], 4)

    fake_loaded = SimpleNamespace(
        model=_FakeEncoder(),
        transforms=lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    with patch("soma.tile_extraction.load_model", return_value=fake_loaded):
        store = TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=False),
        ).run(feature_dir=tmp_path / "features")

    assert store.available_samples == ["s0"]
    assert torch.equal(store.load("s0"), torch.ones(4))


def test_tile_feature_extractor_uses_cpu_worker_budget_when_num_workers_per_gpu_is_unset(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "tile.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(image_path)],
            "label": ["tumor"],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    class _FakeEncoder:
        def encode_tiles(self, batch):
            return torch.ones(batch.shape[0], 4)

    fake_loaded = SimpleNamespace(
        model=_FakeEncoder(),
        transforms=lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    seen_num_workers: list[int] = []
    seen_logs: list[str] = []
    seen_loader_kwargs: list[dict[str, object]] = []

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            seen_loader_kwargs.append(dict(kwargs))
            num_workers = kwargs["num_workers"]
            seen_num_workers.append(num_workers)
            batch_images, sample_id = dataset[0]
            self._items = [(batch_images.unsqueeze(0), [sample_id])]

        def __iter__(self):
            return iter(self._items)

    class _FakeReporter:
        def emit(self, event) -> None:
            return None

        def write_log(self, message: str, *, stream=None) -> None:
            seen_logs.append(message)

        def close(self) -> None:
            return None

    monkeypatch.setattr("slide2vec.api.cpu_worker_limit", lambda: 24)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with patch("soma.tile_extraction.load_model", return_value=fake_loaded), patch(
        "soma.tile_extraction.DataLoader",
        _FakeDataLoader,
    ), patch(
        "soma.tile_extraction.slide2vec_progress.create_api_progress_reporter",
        return_value=_FakeReporter(),
    ):
        store = TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(),
            cache=CacheConfig(enabled=False),
        ).run(feature_dir=tmp_path / "features")

    assert store.available_samples == ["s0"]
    assert seen_num_workers == [16]
    assert seen_loader_kwargs == [
        {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 16,
            "pin_memory": False,
            "prefetch_factor": 4,
        }
    ]
    assert seen_logs == ["Tile DataLoader workers: 16 (slide2vec cpu_worker_limit())"]


def test_build_execution_options_splits_auto_workers_across_gpus(monkeypatch, tmp_path: Path):
    import slide2vec.api as slide2vec_api
    from soma import slide2vec_adapter as adapter

    monkeypatch.setattr(slide2vec_api, "cpu_worker_limit", lambda: 32)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE),
        execution=ExecutionConfig(),
        output_dir=tmp_path,
        num_gpus=2,
        save_tile_embeddings=False,
    )

    assert execution.num_workers_per_gpu == 16  # capped at 16


def test_tile_feature_extractor_uses_torch_default_loader_workers(tmp_path: Path):
    image_path = tmp_path / "tile.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": [str(image_path), str(image_path)],
            "label": ["tumor", "tumor"],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    class _FakeEncoder:
        def encode_tiles(self, batch):
            return torch.ones(batch.shape[0], 4)

    fake_loaded = SimpleNamespace(
        model=_FakeEncoder(),
        transforms=lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    seen_loader_kwargs: list[dict[str, object]] = []

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            seen_loader_kwargs.append(dict(kwargs))
            batch_images, sample_id = dataset[0]
            self._items = [(batch_images.unsqueeze(0), [sample_id])]

        def __iter__(self):
            return iter(self._items)

    with patch("soma.tile_extraction.load_model", return_value=fake_loaded), patch(
        "soma.tile_extraction.DataLoader",
        _FakeDataLoader,
    ):
        store = TileFeatureExtractor(
            dataset,
            EncoderConfig(
                name=_TEST_TILE,
            ),
            execution=ExecutionConfig(
                num_workers_per_gpu=12,
                prefetch_factor=6,
            ),
            cache=CacheConfig(enabled=False),
        ).run(feature_dir=tmp_path / "features")

    assert store.available_samples == ["s0"]
    assert seen_loader_kwargs == [
        {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 12,
            "pin_memory": False,
            "prefetch_factor": 6,
        }
    ]


def test_tile_feature_extractor_renders_rich_progress_for_model_loading_and_batches(tmp_path: Path):
    image_paths = []
    for sample_id in ("s0", "s1", "s2"):
        image_path = tmp_path / f"{sample_id}.png"
        from PIL import Image

        Image.new("RGB", (8, 8), color="white").save(image_path)
        image_paths.append(str(image_path))

    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "image_path": image_paths,
            "label": ["tumor", "tumor", "tumor"],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    class _FakeEncoder:
        def encode_tiles(self, batch):
            return torch.ones(batch.shape[0], 4)

    fake_loaded = SimpleNamespace(
        model=_FakeEncoder(),
        transforms=lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    class _FakeReporter:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def emit(self, event) -> None:
            self.events.append((event.kind, dict(event.payload)))

        def close(self) -> None:
            return None

    fake_reporter = _FakeReporter()

    with patch("soma.tile_extraction.load_model", return_value=fake_loaded), patch(
        "soma.tile_extraction.slide2vec_progress.create_api_progress_reporter",
        return_value=fake_reporter,
    ):
        store = TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE, batch_size=2),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=False),
        ).run(feature_dir=tmp_path / "features")

    assert store.available_samples == ["s0", "s1", "s2"]
    assert [kind for kind, _ in fake_reporter.events] == [
        "model.loading",
        "model.ready",
        "embedding.slide.started",
        "embedding.tile.progress",
        "embedding.tile.progress",
        "embedding.slide.finished",
        "embedding.finished",
    ]
    assert fake_reporter.events[0][1] == {"model_name": _TEST_TILE}
    assert fake_reporter.events[1][1] == {"model_name": _TEST_TILE, "device": "cpu"}
    assert fake_reporter.events[2][1] == {
        "sample_id": "Embedding tiles",
        "total_tiles": 3,
    }
    assert fake_reporter.events[3][1] == {
        "sample_id": "Embedding tiles",
        "processed": 2,
        "total": 3,
        "unit": "tile",
    }
    assert fake_reporter.events[4][1] == {
        "sample_id": "Embedding tiles",
        "processed": 3,
        "total": 3,
        "unit": "tile",
    }
    assert fake_reporter.events[5][1] == {
        "sample_id": "Embedding tiles",
        "num_tiles": 3,
    }
    assert fake_reporter.events[-1][1] == {
        "slide_count": 1,
        "slides_completed": 1,
        "summary_subject": "Tiles",
        "tile_count": 3,
        "tiles_completed": 3,
        "tile_artifacts": 3,
        "slide_artifacts": 0,
    }


def test_tile_embedding_finished_summary_rows_are_tile_oriented(monkeypatch: pytest.MonkeyPatch):
    def _base_embedding_summary_rows(payload: dict[str, object]) -> list[tuple[str, str]]:
        slide_count = int(payload["slide_count"])
        completed = int(payload["slides_completed"])
        failed = max(0, slide_count - completed)
        return [
            ("Slides w/ tiles", str(slide_count)),
            ("Completed", str(completed)),
            ("Failed", str(failed)),
        ]

    monkeypatch.setattr(
        slide2vec_progress,
        "_embedding_summary_rows",
        _base_embedding_summary_rows,
        raising=True,
    )
    monkeypatch.setattr(
        slide2vec_progress,
        "_soma_tile_embedding_summary_patch_installed",
        False,
        raising=False,
    )

    _install_tile_embedding_summary_patch()

    rows = slide2vec_progress._embedding_summary_rows(
        {
            "slide_count": 1,
            "slides_completed": 1,
            "summary_subject": "Tiles",
            "tile_count": 42,
            "tiles_completed": 40,
        }
    )
    assert rows == [("Tiles", "42"), ("Completed", "40"), ("Failed", "2")]


def test_extract_defaults_tiling_dir_to_visible_run_local_path(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(
                sample_id=slide.sample_id,
                output_dir=Path(execution.output_dir),
                kind="tile_embeddings",
                tensor=torch.ones(2, 8),
            )

    with patch.object(FeatureExtractor, "preprocess", autospec=True) as preprocess, patch(
        "soma.extraction.extractor.load_tilings", return_value=loaded
    ), patch("soma.extraction.extractor._validate_runtime"), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        extractor.extract(feature_dir=tmp_path / "features")

    preprocess.assert_called_once()
    assert preprocess.call_args.kwargs["tiling_dir"] == tmp_path / "features" / "tiling"


def test_run_defaults_tiling_dir_to_sibling_run_local_path(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(
                sample_id=slide.sample_id,
                output_dir=Path(execution.output_dir),
                kind="tile_embeddings",
                tensor=torch.ones(2, 8),
            )

    with patch.object(FeatureExtractor, "preprocess", autospec=True) as preprocess, patch(
        "soma.extraction.extractor.load_tilings", return_value=loaded
    ), patch("soma.extraction.extractor._validate_runtime"), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        extractor.run(feature_dir=tmp_path / "features")

    preprocess.assert_called_once()
    assert preprocess.call_args.kwargs["tiling_dir"] == tmp_path / "tiling"


def test_extract_returns_manifest_aware_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        ),
        LoadedTiling(
            slide=SlideSpec(sample_id="s1", image_path=Path("/tmp/s1.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                **{**_tiling("s1").__dict__, "num_tiles": 0}
            ),
        ),
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        _artifact(sample_id="s0", output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert store.has_feature_manifest is True
    assert store.empty_feature_samples == ["s1"]
    assert store.expected_feature_samples == ["s0"]
    assert store.available_samples == ["s0"]
    recorded = pd.read_csv(tmp_path / "features" / "process_list.csv").set_index("sample_id")
    assert recorded.loc["s0", "encoder_name"] == _TEST_TILE
    assert recorded.loc["s0", "output_variant"] == "default"
    assert recorded.loc["s0", "feature_kind"] == "bag"
    assert recorded.loc["s1", "feature_kind"] == "bag"


def test_write_cached_process_list_marks_empty_samples(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
    )
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "tile_embeddings"
    features_dir.mkdir(parents=True)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    resolution = SimpleNamespace(
        metadata={
            "sample_ids": ["s0", "s1"],
            "empty_sample_ids": ["s1"],
            "feature_type": "bag",
            "feature_dim": 8,
            "encoder_name": _TEST_TILE,
            "execution": {"output_variant": "default"},
            "cache_key": "abc123",
        },
        cache_dir=cache_dir,
        features_dir=features_dir,
        cache_kind="tile",
        cache_ids=("s0", "s1"),
        feature_path_for_id=lambda sample_id: features_dir / f"{sample_id}.pt",
        empty_sample_ids={"s1"},
    )
    torch.save(torch.ones(2, 8), features_dir / "s0.pt")

    extractor._write_cached_process_list(feature_dir, cache_resolution=resolution)

    recorded = pd.read_csv(feature_dir / "process_list.csv").set_index("sample_id")
    assert recorded.loc["s0", "annotation"] == "tissue"
    assert recorded.loc["s0", "feature_status"] == "success"
    assert recorded.loc["s0", "feature_path"].endswith("s0.pt")
    assert recorded.loc["s1", "annotation"] == "tissue"
    assert recorded.loc["s1", "feature_status"] == "empty"
    assert pd.isna(recorded.loc["s1", "feature_path"])


def test_materialize_feature_dir_copies_cache_payloads_without_hardlinks(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
    )
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "tile_embeddings"
    features_dir.mkdir(parents=True)
    source = features_dir / "s0.pt"
    torch.save(torch.ones(2, 8), source)
    feature_dir = tmp_path / "features"
    resolution = SimpleNamespace(
        metadata={"sample_ids": ["s0"]},
        cache_dir=cache_dir,
        features_dir=features_dir,
        cache_kind="tile",
        cache_ids=("s0",),
        feature_path_for_id=lambda sample_id: features_dir / f"{sample_id}.pt",
        empty_sample_ids=set(),
    )

    with patch("soma.extraction.os.link") as link:
        extractor._materialize_feature_dir_from_cache(
            feature_dir,
            cache_resolution=resolution,
        )

    target = feature_dir / "s0.pt"
    link.assert_not_called()
    assert target.is_file()
    assert source.stat().st_ino != target.stat().st_ino
    assert torch.equal(torch.load(target, weights_only=True, map_location="cpu"), torch.ones(2, 8))


def test_write_feature_manifest_uses_manifest_metadata_without_loading_tensor(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=True),
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.save(torch.randn(10, 32), feature_dir / "s0.pt")
    pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "feature_status": "success",
                "feature_path": str((feature_dir / "s0.pt").resolve()),
                "num_tiles": 10,
                "feature_rank": 2,
                "feature_dim": 32,
            }
        ]
    ).to_csv(feature_dir / "process_list.csv", index=False)
    store = FeatureStore(feature_dir)
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        )
    ]

    with patch("soma.features.load_array", side_effect=AssertionError("should not reload tensor")):
        extractor._write_feature_manifest(
            feature_dir=feature_dir,
            store=store,
            loaded_tilings=loaded,
            encoder_name=_TEST_TILE,
            output_variant="default",
        )

    recorded = pd.read_csv(feature_dir / "process_list.csv").set_index("sample_id")
    assert recorded.loc["s0", "feature_rank"] == 2
    assert recorded.loc["s0", "feature_dim"] == 32
    assert recorded.loc["s0", "feature_kind"] == "bag"


def test_extract_defaults_to_all_visible_gpus_for_multi_gpu_embedding(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        preprocessing,
        execution,
        tiling_dir,
        slides,
    ):
        tile_artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="tile_embeddings",
            tensor=torch.ones(2, 8),
        )
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[])

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch("soma.extraction.torch.cuda.is_available", return_value=True), patch(
        "soma.extraction.torch.cuda.device_count",
        return_value=2,
    ), patch(
        "soma.extraction.extractor._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert run_with_coords.called
    assert run_with_coords.call_args.kwargs["execution"].num_gpus == 2
    assert run_with_coords.call_args.kwargs["execution"].output_dir.is_absolute()
    assert Path(run_with_coords.call_args.kwargs["tiling_dir"]).is_absolute()
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_run_with_coordinates_stages_process_list_into_output_dir(tmp_path: Path):
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    source_process_list = tiling_dir / "process_list.csv"
    source_process_list.write_text(
        "sample_id,annotation,tiling_status\ns0,tissue,success\n",
        encoding="utf-8",
    )

    execution = ExecutionOptions(
        output_dir=tmp_path / "features",
        num_gpus=2,
        output_format="pt",
    )

    updated_process_list = (
        "sample_id,annotation,tiling_status,feature_status,feature_path\n"
        "s0,tissue,success,success,/tmp/features/s0.pt\n"
    )

    with patch("soma.extraction.orchestration.Pipeline", autospec=True) as MockPipeline, patch(
        "soma.extraction.orchestration.Model.from_preset",
        return_value=object(),
    ):
        instance = MockPipeline.return_value

        def _fake_run_with_coordinates(coordinates_dir, *, slides):
            Path(coordinates_dir, "process_list.csv").write_text(updated_process_list, encoding="utf-8")
            return SimpleNamespace(tile_artifacts=[], slide_artifacts=[])

        instance.run_with_coordinates.side_effect = _fake_run_with_coordinates
        _run_with_coordinates(
            model_name=_TEST_TILE,
            output_variant="default",
            preprocessing=SimpleNamespace(),
            execution=execution,
            tiling_dir=tiling_dir,
            slides=[],
        )

    staged_process_list = execution.output_dir / "process_list.csv"
    assert source_process_list.read_text(encoding="utf-8") == updated_process_list
    assert staged_process_list.read_text(encoding="utf-8") == updated_process_list
    instance.run_with_coordinates.assert_called_once_with(tiling_dir, slides=[])


def test_run_with_coordinates_normalizes_empty_feature_path_column_for_slide2vec(tmp_path: Path):
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    source_process_list = tiling_dir / "process_list.csv"
    source_process_list.write_text(
        "sample_id,annotation,tiling_status,feature_status,feature_path\n"
        "s0,tissue,success,tbp,\n",
        encoding="utf-8",
    )

    execution = ExecutionOptions(
        output_dir=tmp_path / "features",
        num_gpus=2,
        output_format="pt",
    )

    with patch("soma.extraction.orchestration.Pipeline", autospec=True) as MockPipeline, patch(
        "soma.extraction.orchestration.Model.from_preset",
        return_value=object(),
    ):
        instance = MockPipeline.return_value

        def _fake_run_with_coordinates(coordinates_dir, *, slides):
            process_list_path = Path(coordinates_dir, "process_list.csv")
            df = pd.read_csv(process_list_path)
            mask = df["sample_id"] == "s0"
            df.loc[mask, "feature_status"] = "success"
            df.loc[mask, "feature_path"] = "/tmp/features/s0.pt"
            df.to_csv(process_list_path, index=False)
            return SimpleNamespace(tile_artifacts=[], slide_artifacts=[])

        instance.run_with_coordinates.side_effect = _fake_run_with_coordinates
        _run_with_coordinates(
            model_name=_TEST_TILE,
            output_variant="default",
            preprocessing=SimpleNamespace(),
            execution=execution,
            tiling_dir=tiling_dir,
            slides=[],
        )

    staged_process_list = execution.output_dir / "process_list.csv"
    recorded = pd.read_csv(staged_process_list).set_index("sample_id")
    assert recorded.loc["s0", "feature_status"] == "success"
    assert recorded.loc["s0", "feature_path"] == "/tmp/features/s0.pt"
    instance.run_with_coordinates.assert_called_once_with(tiling_dir, slides=[])


def _run_with_coordinates_cuda_state_helper(tmp_path, num_gpus):
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    (tiling_dir / "process_list.csv").write_text(
        "sample_id,annotation,tiling_status\ns0,tissue,success\n",
        encoding="utf-8",
    )
    execution = ExecutionOptions(
        output_dir=tmp_path / "features",
        num_gpus=num_gpus,
        output_format="pt",
    )
    with patch("soma.extraction.orchestration.Pipeline", autospec=True) as MockPipeline, patch(
        "soma.extraction.orchestration.Model.from_preset",
        return_value=object(),
    ), patch(
        "soma.extraction.gc.collect"
    ) as collect, patch(
        "soma.extraction.torch.cuda.is_available",
        return_value=True,
    ), patch(
        "soma.extraction.torch.cuda.empty_cache"
    ) as empty_cache, patch(
        "soma.extraction.torch.cuda.ipc_collect"
    ) as ipc_collect:
        instance = MockPipeline.return_value
        instance.run_with_coordinates.return_value = SimpleNamespace(tile_artifacts=[], slide_artifacts=[])
        _run_with_coordinates(
            model_name=_TEST_TILE,
            output_variant="default",
            preprocessing=SimpleNamespace(),
            execution=execution,
            tiling_dir=tiling_dir,
            slides=[],
        )
    return collect, empty_cache, ipc_collect, instance


def test_run_with_coordinates_releases_parent_cuda_state_multigpu(tmp_path: Path):
    collect, empty_cache, ipc_collect, instance = _run_with_coordinates_cuda_state_helper(tmp_path, num_gpus=2)
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    ipc_collect.assert_called_once_with()
    instance.run_with_coordinates.assert_called_once_with(tmp_path / "tiling", slides=[])


def test_run_with_coordinates_releases_parent_cuda_state_single_gpu(tmp_path: Path):
    collect, empty_cache, ipc_collect, instance = _run_with_coordinates_cuda_state_helper(tmp_path, num_gpus=1)
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    ipc_collect.assert_called_once_with()
    instance.run_with_coordinates.assert_called_once_with(tmp_path / "tiling", slides=[])


def test_run_with_coordinates_releases_parent_cuda_state_no_gpu_count(tmp_path: Path):
    collect, empty_cache, ipc_collect, instance = _run_with_coordinates_cuda_state_helper(tmp_path, num_gpus=None)
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    ipc_collect.assert_called_once_with()
    instance.run_with_coordinates.assert_called_once_with(tmp_path / "tiling", slides=[])


def test_extract_slide_features_returns_slide_embedding_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        artifacts = []
        for slide in slides:
            artifacts.append(
                _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))
            )
        return artifacts

    def _fake_aggregate_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        tile_artifacts,
        preprocessing,
        execution,
    ):
        for artifact in tile_artifacts:
            _artifact(sample_id=artifact.sample_id, output_dir=Path(execution.output_dir), kind="slide_embeddings", tensor=torch.ones(8))

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ), patch(
        "soma.extraction.extractor._aggregate_tiles",
        side_effect=_fake_aggregate_tiles,
    ):
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_slide_encoder_runtime_does_not_forward_output_variant_override(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        assert output_variant is None
        artifacts = []
        for slide in slides:
            artifacts.append(
                _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))
            )
        return artifacts

    def _fake_aggregate_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        tile_artifacts,
        preprocessing,
        execution,
    ):
        assert output_variant is None
        slide_artifacts = []
        for artifact in tile_artifacts:
            slide_artifacts.append(
                _artifact(
                    sample_id=artifact.sample_id,
                    output_dir=Path(execution.output_dir),
                    kind="slide_embeddings",
                    tensor=torch.ones(8),
                )
            )
        return slide_artifacts

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ) as validate_runtime, patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ), patch(
        "soma.extraction.extractor._aggregate_tiles",
        side_effect=_fake_aggregate_tiles,
    ):
        extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert validate_runtime.call_args.kwargs["output_variant"] is None


def test_slide_cache_population_writes_tile_cache_directly(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        return SimpleNamespace(
            name=model_name,
            level="slide",
            allow_non_recommended_settings=allow_non_recommended_settings,
            _load_backend=lambda: SimpleNamespace(device="cpu", model=None),
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        assert on_embedded_slide is not None
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ):
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert any((cache_root / "tile").glob("*/tile_embeddings/s0.pt"))
    assert any((cache_root / "slide").glob("*/slide_embeddings/s0.pt"))
    assert not any((cache_root / "slide").glob("*/tile_embeddings/s0.pt"))
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_slide_cache_population_uses_torch_default_loader_workers(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        return SimpleNamespace(
            name=model_name,
            level="slide",
            allow_non_recommended_settings=allow_non_recommended_settings,
            _load_backend=lambda: SimpleNamespace(device="cpu", model=None),
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        assert on_embedded_slide is not None
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ):
        extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )


def test_tile_cache_population_uses_cache_dir_as_live_output_target(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]
    resolved_preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    cache_resolution = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_TILE,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    seen_output_dirs: list[Path] = []

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        del model_name, output_variant, allow_non_recommended_settings, slides, tiling_results, preprocessing
        seen_output_dirs.append(Path(execution.output_dir))
        return [
            _artifact(
                sample_id="s0",
                output_dir=Path(execution.output_dir),
                kind="tile_embeddings",
                tensor=torch.ones(2, 8),
            )
        ]

    with patch("soma.extraction.extractor._embed_tiles", side_effect=_fake_embed_tiles):
        extractor._populate_tile_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded,
            prepared_tilings=[loaded[0].tiling_result],
            tiling_dir=tmp_path / "tiling",
            preprocessing=build_preprocessing_config(resolved_preprocessing),
            encoder_name=_TEST_TILE,
            output_variant="default",
            num_gpus=None,
        )

    assert seen_output_dirs == [cache_resolution.cache_dir]
    assert (cache_resolution.cache_dir / "tile_embeddings" / "s0.pt").is_file()
    assert cache_resolution.feature_path_for_id("s0").is_file()


def test_patient_cache_population_uses_cache_dir_as_live_output_target(tmp_path: Path):
    dataset = _make_patient_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_PATIENT),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        )
    ]
    resolved_preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_TILE,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    patient_cache = cache_mod.resolve_patient_cache(
        cache_root=cache_root,
        dataset=dataset,
        patient_encoder_name=_TEST_PATIENT,
        tile_encoder_name=_TEST_TILE,
        tile_preprocessing=resolved_preprocessing,
        tile_execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_TILE,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        tile_output_variant="default",
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_PATIENT,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    tile_cache.features_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(2, 8), tile_cache.feature_path_for_id("s0"))
    record_sample_identity_signatures(tile_cache, ["s0"])
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_TILE,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
        complete_state="populated",
    )
    seen_output_dirs: list[Path] = []

    def _fake_aggregate_patients(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        tile_artifacts,
        patient_id_map,
        preprocessing,
        slide_execution,
        patient_execution,
    ):
        del model_name, output_variant, allow_non_recommended_settings, tile_artifacts, patient_id_map, preprocessing
        seen_output_dirs.extend([Path(slide_execution.output_dir), Path(patient_execution.output_dir)])
        artifact_dir = Path(patient_execution.output_dir) / "patient_embeddings"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "p0.pt"
        torch.save(torch.ones(8), path)
        return [SimpleNamespace(patient_id="p0", path=path)]

    with patch("soma.extraction.extractor._aggregate_patients", side_effect=_fake_aggregate_patients):
        extractor._populate_patient_cache(
            patient_cache=patient_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded,
            patient_id_map={"s0": "p0"},
            model_name=_TEST_PATIENT,
            output_variant="default",
            num_gpus=None,
        )

    assert seen_output_dirs == [patient_cache.cache_dir, patient_cache.cache_dir]
    assert (patient_cache.cache_dir / "patient_embeddings" / "p0.pt").is_file()
    assert patient_cache.feature_path_for_id("p0").is_file()


def test_patient_cache_population_skips_empty_tile_cache_samples(tmp_path: Path):
    dataset = _make_two_slide_patient_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    resolved_preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_PATIENT),
        resolved_preprocessing,
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        ),
        LoadedTiling(
            slide=SlideSpec(sample_id="s1", image_path=Path("/tmp/s1.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s1"),
        ),
    ]
    tile_execution = extractor._resolved_execution_for_cache(
        encoder_name=_TEST_TILE,
        resolved_preprocessing=resolved_preprocessing,
        output_variant="default",
    )
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=tile_execution,
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide", "s1": "openslide"},
        },
    )
    patient_cache = cache_mod.resolve_patient_cache(
        cache_root=cache_root,
        dataset=dataset,
        patient_encoder_name=_TEST_PATIENT,
        tile_encoder_name=_TEST_TILE,
        tile_preprocessing=resolved_preprocessing,
        tile_execution=tile_execution,
        tile_output_variant="default",
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_PATIENT,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide", "s1": "openslide"},
        },
    )
    tile_cache.features_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(2, 8), tile_cache.feature_path_for_id("s0"))
    record_sample_identity_signatures(tile_cache, ["s0"])
    record_empty_sample_ids(tile_cache, ["s1"])
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=tile_execution,
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide", "s1": "openslide"},
        },
        complete_state="populated",
    )

    def _fake_aggregate_patients(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        tile_artifacts,
        patient_id_map,
        preprocessing,
        slide_execution,
        patient_execution,
    ):
        del model_name, output_variant, allow_non_recommended_settings, patient_id_map, preprocessing
        del slide_execution
        assert [artifact.sample_id for artifact in tile_artifacts] == ["s0"]
        artifact_dir = Path(patient_execution.output_dir) / "patient_embeddings"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "p0.pt"
        torch.save(torch.ones(8), path)
        return [SimpleNamespace(patient_id="p0", path=path)]

    with patch("soma.extraction.extractor._aggregate_patients", side_effect=_fake_aggregate_patients):
        extractor._populate_patient_cache(
            patient_cache=patient_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded,
            patient_id_map={"s0": "p0", "s1": "p0"},
            model_name=_TEST_PATIENT,
            output_variant="default",
            num_gpus=None,
        )

    assert patient_cache.feature_path_for_id("p0").is_file()


def test_patient_cache_population_records_fully_empty_patients(tmp_path: Path):
    dataset = _make_patient_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    resolved_preprocessing = PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_PATIENT),
        resolved_preprocessing,
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        )
    ]
    tile_execution = extractor._resolved_execution_for_cache(
        encoder_name=_TEST_TILE,
        resolved_preprocessing=resolved_preprocessing,
        output_variant="default",
    )
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=tile_execution,
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    patient_cache = cache_mod.resolve_patient_cache(
        cache_root=cache_root,
        dataset=dataset,
        patient_encoder_name=_TEST_PATIENT,
        tile_encoder_name=_TEST_TILE,
        tile_preprocessing=resolved_preprocessing,
        tile_execution=tile_execution,
        tile_output_variant="default",
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_PATIENT,
            resolved_preprocessing=resolved_preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    record_empty_sample_ids(tile_cache, ["s0"])
    tile_cache = cache_mod.resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=resolved_preprocessing,
        execution=tile_execution,
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
        complete_state="populated",
    )

    with patch("soma.extraction.extractor._aggregate_patients") as aggregate_patients:
        extractor._populate_patient_cache(
            patient_cache=patient_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded,
            patient_id_map={"s0": "p0"},
            model_name=_TEST_PATIENT,
            output_variant="default",
            num_gpus=None,
        )

    aggregate_patients.assert_not_called()
    metadata = json.loads(patient_cache.metadata_path.read_text())
    assert metadata["empty_sample_ids"] == ["p0"]


def test_patient_encoder_requires_every_sample_to_have_patient_id(tmp_path: Path):
    csv_path = tmp_path / "partial-patient-dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s0", "patient_id": "p0", "image_path": str(tmp_path / "s0.svs"), "label": "tumor"},
            {"sample_id": "s1", "patient_id": None, "image_path": str(tmp_path / "s1.svs"), "label": "tumor"},
        ]
    ).to_csv(csv_path, index=False)
    dataset = Dataset(csv_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_PATIENT),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=tmp_path / "shared-cache"),
    )

    with pytest.raises(ValueError, match="every dataset row must have a patient_id"):
        extractor._patient_id_map_for_patient_encoder()


def test_hierarchical_cache_population_uses_cache_dir_as_live_output_target(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        requested_region_size_px=448,
        region_tile_multiple=2,
    )
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        preprocessing,
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]
    cache_resolution = cache_mod.resolve_hierarchical_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=_TEST_TILE,
        preprocessing=preprocessing,
        execution=extractor._resolved_execution_for_cache(
            encoder_name=_TEST_TILE,
            resolved_preprocessing=preprocessing,
            output_variant="default",
        ),
        output_variant="default",
        backend_provenance={
            "requested_backend": "openslide",
            "backend": "openslide",
            "backend_by_sample_id": {"s0": "openslide"},
        },
    )
    seen_output_dirs: list[Path] = []

    def _fake_embed_tiles(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        slides,
        tiling_results,
        preprocessing,
        execution,
    ):
        del model_name, output_variant, allow_non_recommended_settings, slides, tiling_results, preprocessing
        seen_output_dirs.append(Path(execution.output_dir))
        return SimpleNamespace(
            hierarchical_artifacts=[
                _artifact(
                    sample_id="s0",
                    output_dir=Path(execution.output_dir),
                    kind="hierarchical_embeddings",
                    tensor=torch.ones(1, 4, 8),
                )
            ]
        )

    with patch("soma.extraction.extractor._embed_tiles", side_effect=_fake_embed_tiles):
        extractor._populate_hierarchical_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded,
            prepared_tilings=[loaded[0].tiling_result],
            tiling_dir=tmp_path / "tiling",
            preprocessing=build_preprocessing_config(preprocessing),
            encoder_name=_TEST_TILE,
            output_variant="default",
            num_gpus=None,
        )

    assert seen_output_dirs == [cache_resolution.cache_dir]
    assert (cache_resolution.cache_dir / "hierarchical_embeddings" / "s0.pt").is_file()
    assert cache_resolution.feature_path_for_id("s0").is_file()


def test_tile_cache_metadata_records_resolved_execution_fields(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(**kwargs):
        execution = kwargs["execution"]
        return [
            _artifact(
                sample_id="s0",
                output_dir=Path(execution.output_dir),
                kind="tile_embeddings",
                tensor=torch.ones(2, 8),
            )
        ]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    metadata_path = next((cache_root / "tile").glob("*/cache_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution"]["input_size"] == 224
    assert metadata["execution"]["output_variant"] == "default"
    assert metadata["execution"]["spacing_um"] == 0.5


def test_multi_gpu_uncached_extraction_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        preprocessing,
        execution,
        tiling_dir,
        slides,
    ):
        tile_artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="tile_embeddings",
            tensor=torch.ones(2, 8),
        )
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[])

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_multi_gpu_slide_cache_population_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    captured_output_variants: list[str | None] = []

    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        captured_output_variants.append(output_variant)
        return SimpleNamespace(
            name=model_name,
            level="slide",
            allow_non_recommended_settings=allow_non_recommended_settings,
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        assert on_embedded_slide is not None
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ) as compute_embedded_slides:
        store = extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert captured_output_variants == [None]
    assert compute_embedded_slides.called
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)
    assert any((cache_root / "tile").glob("*/tile_embeddings/*.pt"))
    assert any((cache_root / "slide").glob("*/slide_embeddings/*.pt"))


def test_multi_gpu_slide_cache_population_does_not_forward_output_variant_override(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    captured_output_variants: list[str | None] = []

    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        captured_output_variants.append(output_variant)
        assert output_variant is None
        return SimpleNamespace(
            name=model_name,
            level="slide",
            allow_non_recommended_settings=allow_non_recommended_settings,
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        assert on_embedded_slide is not None
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ):
        store = extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert captured_output_variants == [None]

    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_multi_gpu_slide_cache_refresh_keeps_resolved_output_variant_stable(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    captured_output_variants: list[str | None] = []

    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        captured_output_variants.append(output_variant)
        assert output_variant is None
        return SimpleNamespace(
            name=model_name,
            level="slide",
            allow_non_recommended_settings=allow_non_recommended_settings,
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        assert on_embedded_slide is not None
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ), patch.object(FeatureExtractor, "preprocess", autospec=True, return_value=None), patch(
        "soma.extraction.extractor.resolve_slide_cache",
        wraps=cache_mod.resolve_slide_cache,
    ) as resolve_slide_cache:
        store = extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )

    assert captured_output_variants == [None]
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)
    assert all(call.kwargs["output_variant"] == "default" for call in resolve_slide_cache.call_args_list)


def test_hierarchical_tile_extraction_writes_native_embeddings(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                read_tile_size_px=224,
                read_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]
    seen_hierarchical: list[bool] = []

    original_extract_uncached = FeatureExtractor._extract_uncached

    def _spy_extract_uncached(self, *, hierarchical=False, **kwargs):
        seen_hierarchical.append(hierarchical)
        feature_dir = kwargs["feature_dir"]
        _artifact(
            sample_id="s0",
            output_dir=Path(feature_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_extract_uncached",
        autospec=True,
        side_effect=_spy_extract_uncached,
    ) as extract_uncached:
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert extract_uncached.called
    assert seen_hierarchical == [True]
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_hierarchical_multi_gpu_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                read_tile_size_px=224,
                read_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]

    def _fake_run_with_coordinates(
        *,
        model_name,
        output_variant,
        allow_non_recommended_settings,
        preprocessing,
        execution,
        tiling_dir,
        slides,
    ):
        artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )
        return SimpleNamespace(hierarchical_artifacts=[artifact])

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            feature_dir=tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_hierarchical_cache_population_uses_native_cache(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(root_dir=cache_root),
    )
    feature_dir = tmp_path / "features"
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                read_tile_size_px=224,
                read_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]

    def _fake_populate_hierarchical_cache(
        self_,
        *,
        cache_resolution,
        loaded_tilings,
        prepared_tilings,
        tiling_dir,
        preprocessing,
        encoder_name,
        output_variant,
        num_gpus,
    ):
        cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 4, 8), cache_resolution.feature_path_for_id("s0"))
        record_sample_identity_signatures(cache_resolution, ["s0"])
        record_feature_dim(cache_resolution, 8)
        feature_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "sample_id": "s0",
                    "feature_status": "success",
                    "feature_path": str(cache_resolution.feature_path_for_id("s0").resolve()),
                    "num_tiles": 1,
                    "feature_rank": 3,
                    "feature_dim": 8,
                }
            ]
        ).to_csv(feature_dir / "process_list.csv", index=False)

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_populate_hierarchical_cache",
        autospec=True,
        side_effect=_fake_populate_hierarchical_cache,
    ) as populate_hierarchical_cache:
        store = extractor.extract(feature_dir=feature_dir, tiling_dir=tmp_path / "tiling")

    assert populate_hierarchical_cache.called
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_hierarchical_cache_extraction_accepts_allow_non_recommended_settings(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE, allow_non_recommended_settings=True),
        PreprocessingConfig(
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(root_dir=tmp_path / "shared-cache"),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(1, 4, 8), feature_dir / "s0.pt")
    (feature_dir / "s0.meta.json").write_text(
        json.dumps({"artifact_type": "hierarchical_embeddings", "feature_dim": 8}),
        encoding="utf-8",
    )

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_extract_hierarchical_cached",
        autospec=True,
        return_value=FeatureStore(feature_dir),
    ) as extract_hierarchical_cached:
        store = extractor.extract(feature_dir=feature_dir, tiling_dir=tmp_path / "tiling")

    assert extract_hierarchical_cached.called
    assert store.feature_dir == feature_dir
    assert store.feature_rank == 3
    assert store.feature_dim == 8


def test_slide_cache_population_writes_tile_cache_directly(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]
    resolved_preprocessing = PreprocessingConfig(
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
    )
    backend_provenance = {
        "requested_backend": "openslide",
        "backend": "openslide",
        "backend_by_sample_id": {"s0": "openslide"},
    }
    def _fake_load_model(model_name, *, output_variant, allow_non_recommended_settings):
        return SimpleNamespace(
            name=model_name,
            level="slide",
            _load_backend=lambda: SimpleNamespace(device="cpu", model=None),
            allow_non_recommended_settings=allow_non_recommended_settings,
        )

    def _fake_compute_embedded_slides(
        model,
        slide_records,
        tiling_results,
        *,
        preprocessing,
        execution,
        on_embedded_slide=None,
    ):
        assert on_embedded_slide is not None
        embedded_slide = SimpleNamespace(
            sample_id="s0",
            tile_embeddings=torch.ones(2, 8),
            slide_embedding=torch.ones(8),
            latents=None,
            image_path=Path("/tmp/s0.svs"),
            mask_path=None,
            tile_size_lv0=224,
        )
        on_embedded_slide(slide_records[0], tiling_results[0], embedded_slide)
        return [embedded_slide]

    with patch("soma.extraction.extractor.load_tilings", return_value=loaded), patch(
        "soma.extraction.extractor._validate_runtime"
    ), patch(
        "soma.extraction.extractor._load_model",
        side_effect=_fake_load_model,
    ), patch(
        "soma.extraction.extractor._compute_embedded_slides",
        side_effect=_fake_compute_embedded_slides,
    ):
        store = extractor.extract(feature_dir=tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert any((cache_root / "tile").glob("*/tile_embeddings/s0.pt"))
    assert any((cache_root / "slide").glob("*/slide_embeddings/s0.pt"))
    assert not any((cache_root / "slide").glob("*/tile_embeddings/s0.pt"))
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_multispacing_encoder_requires_explicit_spacing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(dataset, EncoderConfig(name=_TEST_MULTI))
    with pytest.raises(ValueError, match="supports multiple spacings"):
        extractor.preprocess(tiling_dir=tmp_path / "tiling")
