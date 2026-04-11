"""TileFeatureExtractor — encodes individual tile images into 1D feature vectors."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from slide2vec.inference import load_model

from soma.cache import (
    FeatureCacheResolution,
    record_feature_dim,
    resolve_cache_root,
    resolve_tile_dataset_cache,
)
from soma.config import CacheConfig, EncoderConfig
from soma.dataset import Dataset, SampleRecord
from soma.encoders.validation import resolve_encoder_precision
from soma.features import FeatureStore


logger = logging.getLogger(__name__)


class _TileImageDataset(TorchDataset):
    """Internal dataset that loads tile images from disk and applies a transform."""

    def __init__(self, records: list[SampleRecord], transform) -> None:
        self._records = records
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[object, str]:
        record = self._records[idx]
        image = Image.open(record.image_path).convert("RGB")
        return self._transform(image), record.sample_id


class TileFeatureExtractor:
    """Encode individual tile images into 1D feature vectors using a tile encoder.

    Loads each sample's tile image, applies the encoder's transform, runs the
    tile encoder, and saves a 1D ``.pt`` feature vector per sample. This is the
    entry point for ``dataset_type="tile"`` pipelines.

    Args:
        dataset: Dataset whose ``image_path`` fields point to tile images.
        encoder: Encoder configuration (name, precision, batch_size, etc.).
        cache: Optional cache configuration. When enabled, features are stored
            in a content-addressed cache directory and reused across runs.
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        *,
        cache: CacheConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._cache = cache or CacheConfig(enabled=False)

    def run(self, feature_dir: str | Path) -> FeatureStore:
        """Encode all tile images and return a FeatureStore over the results.

        Args:
            feature_dir: Directory to write ``.pt`` feature files into.
                Ignored when a complete cache hit is found.

        Returns:
            FeatureStore pointing at the directory containing 1D ``.pt`` files.
        """
        feature_dir = Path(feature_dir).resolve()

        cache_resolution: FeatureCacheResolution | None = None
        if self._cache.enabled:
            cache_root = resolve_cache_root(
                self._cache,
                feature_dir=feature_dir,
            )
            cache_resolution = resolve_tile_dataset_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                execution=self._encoder,
                output_variant=self._encoder.output_variant,
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached tile features from %s",
                    cache_resolution.features_dir,
                )
                return FeatureStore(cache_resolution.features_dir)

        out_dir = cache_resolution.features_dir if cache_resolution is not None else feature_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        precision = resolve_encoder_precision(self._encoder)
        dtype = _precision_to_dtype(precision)

        logger.info("Loading tile encoder '%s'...", self._encoder.name)
        loaded = load_model(
            name=self._encoder.name,
            output_variant=self._encoder.output_variant,
        )
        encoder = loaded.model
        transform = loaded.transforms
        encoder.eval()

        records = list(self._dataset.samples.values())
        image_dataset = _TileImageDataset(records, transform)
        loader = DataLoader(
            image_dataset,
            batch_size=self._encoder.batch_size,
            shuffle=False,
            num_workers=self._encoder.num_workers or 0,
            pin_memory=torch.cuda.is_available(),
        )

        device = loaded.device
        feature_dim: int | None = None

        logger.info(
            "Encoding %d tile images with '%s' (precision=%s, batch_size=%d)...",
            len(records),
            self._encoder.name,
            precision,
            self._encoder.batch_size,
        )
        with torch.inference_mode():
            for batch_images, batch_ids in loader:
                batch_images = batch_images.to(device)
                if dtype != torch.float32:
                    batch_images = batch_images.to(dtype)
                features = encoder.encode_tiles(batch_images)  # (B, D)
                features = features.float().cpu()
                if feature_dim is None:
                    feature_dim = features.shape[1]
                for feat, sample_id in zip(features, batch_ids):
                    torch.save(feat, out_dir / f"{sample_id}.pt")

        logger.info("Saved tile features to %s (dim=%s)", out_dir, feature_dim)

        if cache_resolution is not None and feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)

        return FeatureStore(out_dir)


def _precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32
