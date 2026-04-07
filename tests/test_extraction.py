"""Tests for soma.extraction — slide2vec-backed extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from hs2p import SlideSpec

from soma.cache import CacheConfig
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from slide2vec.encoders.registry import encoder_registry
from soma.extraction import FeatureExtractor
from soma.slide2vec_adapter import LoadedTiling


_TEST_TILE = "_cutover_tile"
_TEST_SLIDE = "_cutover_slide"
_TEST_MULTI = "_cutover_multi_spacing"


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


def _tiling(sample_id: str = "s0") -> object:
    return SimpleNamespace(
        x=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        y=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        tissue_fractions=torch.tensor([1.0, 1.0], dtype=torch.float32).numpy(),
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        effective_tile_size_px=224,
        effective_spacing_um=0.5,
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
    meta_path.write_text("{}", encoding="utf-8")
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


def test_preprocess_delegates_to_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
    )
    with patch("soma.extraction.Pipeline", autospec=True) as MockPipeline:
        mock_instance = MockPipeline.return_value
        extractor.preprocess(tmp_path / "tiling")
    MockPipeline.assert_called_once()
    mock_instance.run.assert_called_once()


def test_extract_tile_features_returns_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ) as embed_tiles:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert embed_tiles.called
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_extract_slide_features_returns_slide_embedding_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        artifacts = []
        for slide in slides:
            artifacts.append(
                _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))
            )
        return artifacts

    def _fake_aggregate_tiles(*, model_name, output_variant, tile_artifacts, preprocessing, execution):
        for artifact in tile_artifacts:
            _artifact(sample_id=artifact.sample_id, output_dir=Path(execution.output_dir), kind="slide_embeddings", tensor=torch.ones(8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ), patch(
        "soma.extraction._aggregate_tiles",
        side_effect=_fake_aggregate_tiles,
    ):
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_multi_gpu_uncached_extraction_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        tile_artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="tile_embeddings",
            tensor=torch.ones(2, 8),
        )
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
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
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        output_dir = Path(execution.output_dir)
        tile_artifact = _artifact(sample_id="s0", output_dir=output_dir, kind="tile_embeddings", tensor=torch.ones(2, 8))
        slide_artifact = _artifact(sample_id="s0", output_dir=output_dir, kind="slide_embeddings", tensor=torch.ones(8))
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[slide_artifact])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)
    assert any((cache_root / "tile").glob("*/features/s0.pt"))
    assert any((cache_root / "slide").glob("*/features/s0.pt"))


def test_hierarchical_tile_extraction_writes_native_embeddings(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
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
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
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
        output_dir = kwargs["output_dir"]
        _artifact(
            sample_id="s0",
            output_dir=Path(output_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_extract_uncached",
        autospec=True,
        side_effect=_spy_extract_uncached,
    ) as extract_uncached:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
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
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
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
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
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

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )
        return SimpleNamespace(hierarchical_artifacts=[artifact])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
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
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
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
        torch.save(torch.ones(1, 4, 8), cache_resolution.features_dir / "s0.pt")

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_populate_hierarchical_cache",
        autospec=True,
        side_effect=_fake_populate_hierarchical_cache,
    ) as populate_hierarchical_cache:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert populate_hierarchical_cache.called
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_slide_cache_population_delegates_to_cache_methods(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_populate_tile_cache(
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
        torch.save(torch.ones(2, 8), cache_resolution.features_dir / "s0.pt")

    def _fake_populate_slide_cache(
        self_,
        *,
        slide_cache,
        tile_cache,
        loaded_tilings,
        model_name,
        output_variant,
        num_gpus,
    ):
        slide_cache.features_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(8), slide_cache.features_dir / "s0.pt")

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_populate_tile_cache",
        autospec=True,
        side_effect=_fake_populate_tile_cache,
    ) as populate_tile_cache, patch.object(
        FeatureExtractor,
        "_populate_slide_cache",
        autospec=True,
        side_effect=_fake_populate_slide_cache,
    ) as populate_slide_cache:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert populate_tile_cache.called
    assert populate_slide_cache.called
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_multispacing_encoder_requires_explicit_spacing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(dataset, EncoderConfig(name=_TEST_MULTI))
    with pytest.raises(ValueError, match="supports multiple spacings"):
        extractor.preprocess(tmp_path / "tiling")
