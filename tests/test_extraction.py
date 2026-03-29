"""Tests for soma.extraction — FeatureExtractor class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image

from soma.cache import CacheConfig, CacheResolution
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.distributed import SlideTask
from soma.encoders.registry import encoder_registry
from soma.preprocessing.io import save_tiling_result
from soma.preprocessing.tiling import TilingResult
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_SAMPLES = 3
_TEST_CACHE_TILE = "_test_cache_tile"
_TEST_CACHE_SLIDE = "_test_cache_slide"
_TEST_CACHE_SLIDE_RECORDING = "_test_cache_slide_recording"


def _make_dataset(tmp_path: Path) -> Dataset:
    """Create a small dataset with synthetic WSI paths."""
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [str(tmp_path / f"s{i}.svs") for i in range(NUM_SAMPLES)],
            "label": ["tumor", "normal", "tumor"],
        }
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _make_mask_dataset(tmp_path: Path, mask_path: Path) -> Dataset:
    csv_path = tmp_path / "dataset_with_mask.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(tmp_path / "s0.svs")],
            "label": ["tumor"],
            "tissue_mask_path": [str(mask_path)],
        }
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _make_saved_tiling_result(
    *,
    image_path: Path,
    tissue_mask_path: Path | None,
    tissue_mask_tissue_value: int | None,
) -> TilingResult:
    return TilingResult(
        coordinates=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=224,
        effective_spacing_um=0.5,
        tile_size_lv0=224,
        is_within_tolerance=True,
        sample_id="s0",
        image_path=str(image_path),
        backend="openslide",
        requested_backend="auto",
        base_spacing_um=0.5,
        slide_dimensions=[1000, 800],
        level_downsamples=[1.0, 2.0, 4.0],
        overlap=0.0,
        min_tissue_fraction=0.1,
        step_px_lv0=224,
        tissue_method="precomputed_mask" if tissue_mask_path is not None else "hsv",
        seg_downsample=64,
        seg_level=0,
        seg_spacing_um=0.5,
        ref_tile_size_px=224,
        a_t=4,
        tissue_mask_path=None if tissue_mask_path is None else str(tissue_mask_path),
        tissue_mask_tissue_value=tissue_mask_tissue_value,
        mask_level=0 if tissue_mask_path is not None else None,
        mask_spacing_um=0.5 if tissue_mask_path is not None else None,
    )


def _make_synthetic_reader():
    """Create a SyntheticSlideReader-like mock."""
    reader = MagicMock()
    reader.backend_name = "synthetic"
    reader.dimensions = (1000, 800)
    reader.spacing = 0.5
    reader.level_count = 3
    reader.level_dimensions = [(1000, 800), (500, 400), (250, 200)]
    reader.level_downsamples = [1.0, 2.0, 4.0]
    reader.read_region = MagicMock(
        side_effect=lambda location, level, size, **_: np.full((size[1], size[0], 3), 180, dtype=np.uint8)
    )

    # Thumbnail: colored region (tissue) with white border (background)
    thumb = np.full((100, 125, 3), 255, dtype=np.uint8)
    thumb[10:90, 10:115] = np.array([150, 80, 100], dtype=np.uint8)  # tissue-like
    reader.get_thumbnail.return_value = thumb

    reader.close = MagicMock()
    reader.__enter__ = MagicMock(return_value=reader)
    reader.__exit__ = MagicMock(return_value=False)
    return reader


def _make_mask_reader(
    *,
    spacing: float,
    level_dimensions: list[tuple[int, int]],
    level_downsamples: list[float],
    read_region_side_effect,
    backend_name: str = "synthetic",
):
    reader = MagicMock()
    reader.backend_name = backend_name
    reader.dimensions = tuple(level_dimensions[0])
    reader.spacing = spacing
    reader.level_count = len(level_dimensions)
    reader.level_dimensions = level_dimensions
    reader.level_downsamples = level_downsamples
    reader.read_region = MagicMock(side_effect=read_region_side_effect)
    reader.close = MagicMock()
    reader.__enter__ = MagicMock(return_value=reader)
    reader.__exit__ = MagicMock(return_value=False)
    return reader


def _write_mask_tiff_pyramid(path: Path, pages: list[np.ndarray]) -> None:
    images = [Image.fromarray(page.astype(np.uint8)) for page in pages]
    images[0].save(path, save_all=True, append_images=images[1:])


def _mock_extract_dataset(encoder_name, slides, output_dir, **kwargs):
    """Mock extract_dataset that writes fake .pt files."""
    from soma.encoders.distributed import ExtractionSummary

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    skipped = []
    output_variant = kwargs.get("output_variant")
    dim = 8 if output_variant == "cls" else D
    for task in slides:
        pt_path = output_dir / f"{task.slide_id}.pt"
        if kwargs.get("skip_existing", True) and pt_path.exists():
            skipped.append(task.slide_id)
            continue
        n_tiles = len(task.tiling_result.coordinates)
        torch.save(torch.randn(n_tiles, dim), pt_path)
        completed.append(task.slide_id)
    return ExtractionSummary(
        completed=completed, skipped=skipped, failed=[], duration_s=0.1
    )


def _mock_extract_dataset_slide(encoder_name, slides, output_dir, **kwargs):
    from soma.encoders.distributed import ExtractionSummary

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in slides:
        torch.save(torch.randn(D), output_dir / f"{task.slide_id}.pt")
        if kwargs.get("save_tile_features"):
            tile_dir = output_dir / "tile_features"
            tile_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(len(task.tiling_result.coordinates), D), tile_dir / f"{task.slide_id}.pt")
    return ExtractionSummary(
        completed=[task.slide_id for task in slides],
        skipped=[],
        failed=[],
        duration_s=0.1,
    )


class _CacheTileEncoder(TileEncoder):
    def __init__(self, **kwargs):
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    def get_transform(self):
        return lambda x: x

    def encode_tiles(self, batch):
        return torch.ones(batch.shape[0], D)

    @property
    def encode_dim(self):
        return D

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class _CacheSlideEncoder(SlideEncoder):
    def __init__(self, **kwargs):
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0: int | None = None):
        return tile_features.mean(dim=0)

    @property
    def encode_dim(self):
        return D

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class _RecordingCacheSlideEncoder(_CacheSlideEncoder):
    prepare_calls: list[dict[str, object]] = []

    @classmethod
    def reset(cls) -> None:
        cls.prepare_calls = []

    def prepare_coordinates(
        self,
        coordinates,
        *,
        base_spacing_um: float,
        target_spacing_um: float,
    ):
        self.__class__.prepare_calls.append(
            {
                "coordinates": coordinates.clone(),
                "base_spacing_um": float(base_spacing_um),
                "target_spacing_um": float(target_spacing_um),
            }
        )
        return coordinates


if _TEST_CACHE_TILE not in encoder_registry:
    encoder_registry.register(
        _TEST_CACHE_TILE,
        _CacheTileEncoder,
        metadata={
            "level": "tile",
            "input_size": 256,
            "output_variants": {"default": {"encode_dim": D}},
            "default_output_variant": "default",
            "supported_spacing_um": 0.5,
            "precision": "fp16",
            "source": "test/cache-tile",
        },
    )

if _TEST_CACHE_SLIDE not in encoder_registry:
    encoder_registry.register(
        _TEST_CACHE_SLIDE,
        _CacheSlideEncoder,
        metadata={
            "level": "slide",
            "tile_encoder": _TEST_CACHE_TILE,
            "tile_encoder_output_variant": "default",
            "output_variants": {"default": {"encode_dim": D}},
            "default_output_variant": "default",
            "supported_spacing_um": 0.5,
            "precision": "fp16",
            "source": "test/cache-slide",
        },
    )

if _TEST_CACHE_SLIDE_RECORDING not in encoder_registry:
    encoder_registry.register(
        _TEST_CACHE_SLIDE_RECORDING,
        _RecordingCacheSlideEncoder,
        metadata={
            "level": "slide",
            "tile_encoder": _TEST_CACHE_TILE,
            "tile_encoder_output_variant": "default",
            "output_variants": {"default": {"encode_dim": D}},
            "default_output_variant": "default",
            "supported_spacing_um": 0.5,
            "precision": "fp16",
            "source": "test/cache-slide-recording",
        },
    )


# ---------------------------------------------------------------------------
# FeatureExtractor.preprocess()
# ---------------------------------------------------------------------------


class TestPreprocess:
    @patch("soma.extraction.open_slide")
    def test_saves_tiling_artifacts(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        for i in range(NUM_SAMPLES):
            assert (tiling_dir / f"s{i}.coordinates.npz").exists()
            assert (tiling_dir / f"s{i}.coordinates.meta.json").exists()

    @patch("soma.extraction.open_slide")
    def test_skip_existing(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        # Preprocess again — should skip existing
        mock_open.reset_mock()
        extractor.preprocess(tiling_dir, skip_existing=True)

        # open_slide should not have been called again
        mock_open.assert_not_called()

    @patch("soma.extraction.open_slide")
    def test_no_skip_existing(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        mock_open.reset_mock()
        mock_open.return_value = _make_synthetic_reader()
        extractor.preprocess(tiling_dir, skip_existing=False)

        assert mock_open.call_count == NUM_SAMPLES

    @staticmethod
    def _empty_tiling_result() -> TilingResult:
        return TilingResult(
            coordinates=np.empty((0, 2), dtype=np.int64),
            tissue_fractions=np.empty(0, dtype=np.float32),
            requested_tile_size_px=224,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=224,
            effective_spacing_um=0.5,
            tile_size_lv0=224,
            is_within_tolerance=True,
            tissue_mask=np.ones((10, 10), dtype=np.uint8),
        )

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_infers_preprocessing_defaults_from_tile_encoder(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        reader = _make_synthetic_reader()
        mock_open.return_value = reader
        mock_segment.return_value = np.ones((10, 10), dtype=np.uint8)
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_dataset(tmp_path)

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name="h0-mini"),
            PreprocessingConfig(
                requested_tile_size_px=None,
                requested_spacing_um=None,
            ),
        )
        extractor.preprocess(tmp_path / "tiling")

        detect_kwargs = mock_detect.call_args.kwargs
        generate_kwargs = mock_generate.call_args.kwargs
        assert detect_kwargs["ref_tile_size_px"] == 224
        assert detect_kwargs["requested_spacing_um"] == 0.5
        assert detect_kwargs["base_spacing_um"] == 0.5
        assert detect_kwargs["level_downsamples"] == [1.0, 2.0, 4.0]
        assert detect_kwargs["tolerance"] == 0.05
        assert generate_kwargs["requested_tile_size_px"] == 224
        assert generate_kwargs["requested_spacing_um"] == 0.5
        reader.read_region.assert_called_with((0, 0), 2, (250, 200))

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_infers_preprocessing_defaults_from_slide_encoder_tile_dependency(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        reader = _make_synthetic_reader()
        mock_open.return_value = reader
        mock_segment.return_value = np.ones((10, 10), dtype=np.uint8)
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_dataset(tmp_path)

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name="prism"),
            PreprocessingConfig(
                requested_tile_size_px=None,
                requested_spacing_um=None,
            ),
        )
        extractor.preprocess(tmp_path / "tiling")

        detect_kwargs = mock_detect.call_args.kwargs
        generate_kwargs = mock_generate.call_args.kwargs
        assert detect_kwargs["ref_tile_size_px"] == 224
        assert detect_kwargs["base_spacing_um"] == 0.5
        assert detect_kwargs["level_downsamples"] == [1.0, 2.0, 4.0]
        assert detect_kwargs["tolerance"] == 0.05
        assert generate_kwargs["requested_tile_size_px"] == 224
        assert generate_kwargs["requested_spacing_um"] == 0.5
        reader.read_region.assert_called_with((0, 0), 2, (250, 200))

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_preserves_explicit_ref_tile_size_override(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        reader = _make_synthetic_reader()
        mock_open.return_value = reader
        mock_segment.return_value = np.ones((10, 10), dtype=np.uint8)
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_dataset(tmp_path)

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name="h0-mini"),
            PreprocessingConfig(
                requested_tile_size_px=None,
                requested_spacing_um=None,
                ref_tile_size_px=32,
            ),
        )
        extractor.preprocess(tmp_path / "tiling")

        detect_kwargs = mock_detect.call_args.kwargs
        assert detect_kwargs["ref_tile_size_px"] == 32
        assert detect_kwargs["base_spacing_um"] == 0.5
        assert detect_kwargs["level_downsamples"] == [1.0, 2.0, 4.0]
        assert detect_kwargs["tolerance"] == 0.05
        reader.read_region.assert_called_with((0, 0), 2, (250, 200))

    @patch("soma.extraction.open_slide")
    def test_saves_provenance_metadata_in_tiling_artifact(self, mock_open, tmp_path: Path):
        reader = _make_synthetic_reader()
        reader.backend_name = "cucim"
        mock_open.return_value = reader
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(
                requested_tile_size_px=256,
                requested_spacing_um=0.5,
                min_tissue_fraction=0.2,
                overlap=0.25,
                seg_downsample=32,
                ref_tile_size_px=128,
                a_t=7,
                tissue_method="hsv",
            ),
        )
        extractor.preprocess(tiling_dir)

        meta = json.loads((tiling_dir / "s0.coordinates.meta.json").read_text())
        assert meta["sample_id"] == "s0"
        assert meta["image_path"] == str(tmp_path / "s0.svs")
        assert meta["backend"] == "cucim"
        assert meta["requested_backend"] == "auto"
        assert meta["base_spacing_um"] == 0.5
        assert meta["slide_dimensions"] == [1000, 800]
        assert meta["level_downsamples"] == [1.0, 2.0, 4.0]
        assert meta["overlap"] == 0.25
        assert meta["use_padding"] is True
        assert meta["min_tissue_fraction"] == 0.2
        assert meta["step_px_lv0"] == 192
        assert meta["tissue_method"] == "hsv"
        assert meta["seg_downsample"] == 32
        assert meta["seg_level"] == 2
        assert meta["seg_spacing_um"] == 2.0
        assert meta["ref_tile_size_px"] == 128
        assert meta["a_t"] == 7
        assert meta["provenance"]["backend"] == "cucim"
        assert meta["slide"]["base_spacing_um"] == 0.5
        assert meta["slide"]["seg_level"] == 2
        assert meta["slide"]["seg_spacing_um"] == 2.0
        assert meta["tiling"]["step_px_lv0"] == 192
        assert meta["tiling"]["use_padding"] is True
        assert meta["segmentation"]["a_t"] == 7
        assert meta["segmentation"]["seg_level"] == 2
        assert meta["segmentation"]["seg_spacing_um"] == 2.0

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_uses_precomputed_tissue_mask_when_present(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        slide_reader = _make_synthetic_reader()
        slide_reader.backend_name = "openslide"
        mask_raw = np.zeros((480, 600), dtype=np.uint8)
        mask_raw[60:420, 100:520] = 1
        mask_reader = _make_mask_reader(
            spacing=0.75,
            level_dimensions=[(600, 480), (300, 240)],
            level_downsamples=[1.0, 2.0],
            read_region_side_effect=lambda location, level, size, **_: np.repeat(
                mask_raw[:, :, None], 3, axis=2
            ),
            backend_name="openslide",
        )
        mask_path = tmp_path / "mask.tif"
        _write_mask_tiff_pyramid(mask_path, [mask_raw, mask_raw[::2, ::2]])
        mock_open.side_effect = lambda path, backend="auto", **_: (
            slide_reader if Path(path).name == "s0.svs" else mask_reader
        )
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_mask_dataset(tmp_path, mask_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(seg_downsample=3),
        )
        extractor.preprocess(tiling_dir)

        mock_segment.assert_not_called()
        expected = cv2.resize(mask_raw, (500, 400), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(
            mock_detect.call_args.args[0],
            np.where(expected == 1, 255, 0).astype(np.uint8),
        )
        slide_reader.read_region.assert_not_called()
        mask_reader.read_region.assert_called_once_with((0, 0), 0, (600, 480))
        meta = json.loads((tiling_dir / "s0.coordinates.meta.json").read_text())
        assert meta["seg_level"] == 1
        assert meta["seg_spacing_um"] == 1.0
        assert meta["tissue_method"] == "precomputed_mask"
        assert meta["tissue_mask_path"] == str(mask_path)
        assert meta["tissue_mask_tissue_value"] == 1
        assert meta["mask_level"] == 0
        assert meta["mask_spacing_um"] == 0.75
        assert mock_open.call_args_list[1].args == (mask_path, "openslide")

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_precomputed_tissue_mask_uses_channel_zero_and_configured_tissue_value(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        slide_reader = _make_synthetic_reader()
        slide_reader.backend_name = "openslide"
        mask_channel0 = np.zeros((400, 500), dtype=np.uint8)
        mask_channel0[40:360, 80:420] = 2
        mask_rgb = np.stack(
            [mask_channel0, np.full_like(mask_channel0, 9), np.full_like(mask_channel0, 13)],
            axis=2,
        )
        mask_reader = _make_mask_reader(
            spacing=1.0,
            level_dimensions=[(500, 400)],
            level_downsamples=[1.0],
            read_region_side_effect=lambda location, level, size, **_: mask_rgb,
            backend_name="openslide",
        )
        mask_path = tmp_path / "mask.tif"
        _write_mask_tiff_pyramid(mask_path, [mask_channel0])
        mock_open.side_effect = lambda path, backend="auto", **_: (
            slide_reader if Path(path).name == "s0.svs" else mask_reader
        )
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_mask_dataset(tmp_path, mask_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(seg_downsample=3, tissue_mask_tissue_value=2),
        )
        extractor.preprocess(tiling_dir)

        mock_segment.assert_not_called()
        np.testing.assert_array_equal(
            mock_detect.call_args.args[0],
            np.where(mask_channel0 == 2, 255, 0).astype(np.uint8),
        )
        meta = json.loads((tiling_dir / "s0.coordinates.meta.json").read_text())
        assert meta["tissue_mask_tissue_value"] == 2
        assert meta["mask_level"] == 0

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_precomputed_tissue_mask_falls_back_to_exact_tiff_page_on_interpolated_labels(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        slide_reader = _make_synthetic_reader()
        slide_reader.backend_name = "openslide"
        exact_mask = np.zeros((480, 600), dtype=np.uint8)
        exact_mask[50:430, 120:480] = 1
        mask_reader = _make_mask_reader(
            spacing=0.75,
            level_dimensions=[(600, 480), (300, 240)],
            level_downsamples=[1.0, 2.0],
            read_region_side_effect=lambda location, level, size, **_: np.dstack(
                [
                    np.where(exact_mask == 1, 255, 0).astype(np.uint8),
                    np.full_like(exact_mask, 127),
                    np.full_like(exact_mask, 255),
                ]
            ),
            backend_name="openslide",
        )
        mask_path = tmp_path / "mask.tif"
        _write_mask_tiff_pyramid(mask_path, [exact_mask, exact_mask[::2, ::2]])
        mock_open.side_effect = lambda path, backend="auto", **_: (
            slide_reader if Path(path).name == "s0.svs" else mask_reader
        )
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_mask_dataset(tmp_path, mask_path)

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(seg_downsample=3),
        )
        extractor.preprocess(tmp_path / "tiling")

        mock_segment.assert_not_called()
        expected = cv2.resize(exact_mask, (500, 400), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(
            mock_detect.call_args.args[0],
            np.where(expected == 1, 255, 0).astype(np.uint8),
        )

    @patch("soma.extraction.open_slide")
    def test_rejects_unrecoverable_non_discrete_precomputed_tissue_mask(
        self,
        mock_open,
        tmp_path: Path,
    ):
        slide_reader = _make_synthetic_reader()
        slide_reader.backend_name = "openslide"
        invalid_mask = np.zeros((480, 600), dtype=np.uint8)
        invalid_mask[50:150, 80:200] = 1
        invalid_mask[200:320, 250:420] = 2
        mask_reader = _make_mask_reader(
            spacing=0.75,
            level_dimensions=[(600, 480)],
            level_downsamples=[1.0],
            read_region_side_effect=lambda location, level, size, **_: np.repeat(
                invalid_mask[:, :, None], 3, axis=2
            ),
            backend_name="openslide",
        )
        mask_path = tmp_path / "bad-mask.tif"
        _write_mask_tiff_pyramid(mask_path, [invalid_mask])
        mock_open.side_effect = lambda path, backend="auto", **_: (
            slide_reader if Path(path).name == "s0.svs" else mask_reader
        )
        dataset = _make_mask_dataset(tmp_path, mask_path)

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(seg_downsample=3),
        )

        with pytest.raises(ValueError, match="non-discrete labels"):
            extractor.preprocess(tmp_path / "tiling")

        slide_reader.read_region.assert_not_called()

    @patch("soma.extraction.open_slide")
    def test_preprocess_reads_segmentation_from_resolved_seg_level(self, mock_open, tmp_path: Path):
        reader = _make_synthetic_reader()
        mock_open.return_value = reader
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(),
            PreprocessingConfig(seg_downsample=3),
        )
        extractor.preprocess(tiling_dir)

        reader.read_region.assert_any_call((0, 0), 1, (500, 400))
        assert reader.get_thumbnail.call_count == 0

    @patch("soma.extraction.generate_tiles")
    @patch("soma.extraction.detect_contours")
    @patch("soma.extraction.segment_tissue")
    @patch("soma.extraction.open_slide")
    def test_skip_existing_rejects_mask_provenance_mismatch(
        self,
        mock_open,
        mock_segment,
        mock_detect,
        mock_generate,
        tmp_path: Path,
    ):
        slide_reader = _make_synthetic_reader()
        slide_reader.backend_name = "openslide"
        mask_raw = np.zeros((800, 1000), dtype=np.uint8)
        mask_raw[80:720, 120:880] = 1
        mask_reader = _make_mask_reader(
            spacing=0.5,
            level_dimensions=[(1000, 800)],
            level_downsamples=[1.0],
            read_region_side_effect=lambda location, level, size, **_: np.repeat(
                mask_raw[:, :, None], 3, axis=2
            ),
            backend_name="openslide",
        )
        correct_mask_path = tmp_path / "mask.tif"
        wrong_mask_path = tmp_path / "wrong-mask.tif"
        _write_mask_tiff_pyramid(correct_mask_path, [mask_raw])
        _write_mask_tiff_pyramid(wrong_mask_path, [mask_raw])
        mock_open.side_effect = lambda path, backend="auto", **_: (
            slide_reader if Path(path).name == "s0.svs" else mask_reader
        )
        mock_detect.return_value = MagicMock(contours=[], mask=np.ones((10, 10), dtype=np.uint8))
        mock_generate.return_value = self._empty_tiling_result()
        dataset = _make_mask_dataset(tmp_path, correct_mask_path)
        tiling_dir = tmp_path / "tiling"
        save_tiling_result(
            _make_saved_tiling_result(
                image_path=tmp_path / "s0.svs",
                tissue_mask_path=wrong_mask_path,
                tissue_mask_tissue_value=1,
            ),
            tiling_dir,
            "s0",
        )

        extractor = FeatureExtractor(dataset, EncoderConfig())

        with pytest.raises(ValueError, match="tissue_mask_path mismatch"):
            extractor.preprocess(tiling_dir, skip_existing=True)

        mock_segment.assert_not_called()
        mock_detect.assert_not_called()
        mock_generate.assert_not_called()
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# FeatureExtractor.extract()
# ---------------------------------------------------------------------------


class TestExtract:
    def test_rejects_tiling_artifacts_with_mismatched_mask_provenance(self, tmp_path: Path):
        mask_path = tmp_path / "mask.tif"
        wrong_mask_path = tmp_path / "wrong-mask.tif"
        _write_mask_tiff_pyramid(mask_path, [np.zeros((8, 8), dtype=np.uint8)])
        _write_mask_tiff_pyramid(wrong_mask_path, [np.zeros((8, 8), dtype=np.uint8)])
        dataset = _make_mask_dataset(tmp_path, mask_path)
        tiling_dir = tmp_path / "tiling"
        save_tiling_result(
            _make_saved_tiling_result(
                image_path=tmp_path / "s0.svs",
                tissue_mask_path=wrong_mask_path,
                tissue_mask_tissue_value=1,
            ),
            tiling_dir,
            "s0",
        )

        extractor = FeatureExtractor(dataset, EncoderConfig())

        with pytest.raises(ValueError, match="tissue_mask_path mismatch"):
            extractor.extract(tmp_path / "features", tiling_dir=tiling_dir)

    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_returns_feature_store(self, mock_open, mock_extract, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        store = extractor.extract(feature_dir, tiling_dir=tiling_dir)

        assert isinstance(store, FeatureStore)
        assert len(store) == NUM_SAMPLES
        for i in range(NUM_SAMPLES):
            features = store.load(f"s{i}")
            assert features.shape[1] == D

    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_passes_output_variant_to_tile_extraction(
        self, mock_open, mock_extract, tmp_path: Path
    ):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(dataset, EncoderConfig(name="h0-mini", output_variant="cls"))
        extractor.preprocess(tiling_dir)
        store = extractor.extract(feature_dir, tiling_dir=tiling_dir)

        assert mock_extract.call_args.kwargs["output_variant"] == "cls"
        assert store.feature_dim == 8

    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset_slide)
    @patch("soma.extraction.open_slide")
    def test_passes_save_tile_features_for_slide_encoders(
        self, mock_open, mock_extract, tmp_path: Path
    ):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name="prism", save_tile_features=True),
            cache=CacheConfig(enabled=False),
        )
        extractor.preprocess(tiling_dir)
        store = extractor.extract(feature_dir, tiling_dir=tiling_dir)

        assert store.is_slide_level is True
        assert (feature_dir / "tile_features" / "s0.pt").exists()
        assert mock_extract.call_args.kwargs["save_tile_features"] is True

    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_shared_cache_reuses_tile_features_across_runs(
        self, mock_open, mock_extract, tmp_path: Path
    ):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        cache_root = tmp_path / "shared-cache"

        slide_extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_CACHE_SLIDE),
            cache=CacheConfig(root_dir=cache_root),
        )
        slide_extractor.preprocess(tiling_dir)
        slide_store = slide_extractor.extract(tmp_path / "run_slide" / "features", tiling_dir=tiling_dir)

        assert slide_store.is_slide_level is True
        tile_cache_dirs = list((cache_root / "tile").glob("*/features"))
        slide_cache_dirs = list((cache_root / "slide").glob("*/features"))
        assert len(tile_cache_dirs) == 1
        assert len(slide_cache_dirs) == 1
        assert (tile_cache_dirs[0] / "s0.pt").exists()
        assert (slide_cache_dirs[0] / "s0.pt").exists()

        mock_extract.reset_mock()
        tile_extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_CACHE_TILE),
            cache=CacheConfig(root_dir=cache_root),
        )
        tile_store = tile_extractor.extract(tmp_path / "run_tile" / "features", tiling_dir=tiling_dir)
        assert tile_store.is_slide_level is False
        mock_extract.assert_not_called()

    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_shared_cache_reuses_slide_features_without_repooling(
        self, mock_open, mock_extract, tmp_path: Path
    ):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        cache_root = tmp_path / "shared-cache"

        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_CACHE_SLIDE),
            cache=CacheConfig(root_dir=cache_root),
        )
        extractor.preprocess(tiling_dir)
        extractor.extract(tmp_path / "run1" / "features", tiling_dir=tiling_dir)
        mock_extract.reset_mock()
        extractor.extract(tmp_path / "run2" / "features", tiling_dir=tiling_dir)
        mock_extract.assert_not_called()

    def test_populate_slide_cache_uses_requested_spacing_for_coordinate_prep(
        self, tmp_path: Path
    ):
        dataset = _make_dataset(tmp_path)
        extractor = FeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_CACHE_SLIDE_RECORDING),
            cache=CacheConfig(enabled=False),
        )
        _RecordingCacheSlideEncoder.reset()

        tile_cache_dir = tmp_path / "tile-cache" / "features"
        tile_cache_dir.mkdir(parents=True, exist_ok=True)
        slide_cache_dir = tmp_path / "slide-cache" / "features"
        slide_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(1, D), tile_cache_dir / "s0.pt")

        tile_cache = CacheResolution(
            kind="tile",
            key="tile",
            cache_dir=tile_cache_dir.parent,
            metadata_path=tile_cache_dir.parent / "cache_metadata.json",
            manifest_path=tile_cache_dir.parent / "manifest.csv",
            features_dir=tile_cache_dir,
            reused=False,
            complete=True,
            metadata={"sample_ids": ["s0"], "execution": {"output_variant": None}},
        )
        slide_cache = CacheResolution(
            kind="slide",
            key="slide",
            cache_dir=slide_cache_dir.parent,
            metadata_path=slide_cache_dir.parent / "cache_metadata.json",
            manifest_path=slide_cache_dir.parent / "manifest.csv",
            features_dir=slide_cache_dir,
            reused=False,
            complete=False,
            metadata={"sample_ids": ["s0"], "execution": {"output_variant": None}},
        )
        slide_cache.metadata_path.write_text("{}", encoding="utf-8")

        tiling = TilingResult(
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tissue_fractions=np.array([1.0], dtype=np.float32),
            requested_tile_size_px=256,
            requested_spacing_um=0.75,
            read_level=0,
            effective_tile_size_px=384,
            effective_spacing_um=0.5,
            tile_size_lv0=512,
            is_within_tolerance=False,
            base_spacing_um=0.25,
        )
        task = SlideTask(
            slide_path=str(tmp_path / "s0.svs"),
            tiling_result=tiling,
            slide_id="s0",
        )

        extractor._populate_slide_cache(
            slide_cache,
            tile_cache,
            [task],
            tilings={"s0": tiling},
            backend="auto",
        )

        assert len(_RecordingCacheSlideEncoder.prepare_calls) == 1
        prepare_call = _RecordingCacheSlideEncoder.prepare_calls[0]
        assert torch.equal(
            prepare_call["coordinates"],
            torch.as_tensor(tiling.coordinates, dtype=torch.long),
        )
        assert prepare_call["base_spacing_um"] == 0.25
        assert prepare_call["target_spacing_um"] == 0.75


# ---------------------------------------------------------------------------
# FeatureExtractor.run()
# ---------------------------------------------------------------------------


class TestRun:
    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_end_to_end(self, mock_open, mock_extract, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        store = extractor.run(feature_dir)

        assert isinstance(store, FeatureStore)
        assert len(store) == NUM_SAMPLES
        # Tiling artifacts saved under .tiling/
        tiling_dir = feature_dir / ".tiling"
        assert (tiling_dir / "s0.coordinates.npz").exists()
