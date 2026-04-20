"""TileFeatureExtractor — encodes individual tile images into 1D feature vectors."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import slide2vec.progress as slide2vec_progress
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from slide2vec.inference import load_model

from soma.cache import (
    FeatureCacheResolution,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_tile_cache,
)
from soma.config import CacheConfig, EncoderConfig
from soma.dataset import Dataset, SampleRecord
from soma.features import FeatureStore
from soma.slide2vec_adapter import build_execution_options


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


@contextlib.contextmanager
def _make_tile_extraction_reporter_ctx(feature_dir: Path):
    active = slide2vec_progress.get_progress_reporter()
    if not isinstance(active, slide2vec_progress.NullProgressReporter):
        yield
        return

    reporter = slide2vec_progress.create_api_progress_reporter(output_dir=feature_dir)
    if isinstance(reporter, slide2vec_progress.NullProgressReporter):
        yield
        return

    with slide2vec_progress.activate_progress_reporter(reporter):
        yield


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
            cache_resolution = resolve_tile_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                preprocessing=None,
                execution=self._encoder,
                output_variant=self._encoder.output_variant,
                feature_type="tile",
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached tile features from %s",
                    cache_resolution.features_dir,
                )
                return FeatureStore(cache_resolution.features_dir)

        out_dir = cache_resolution.features_dir if cache_resolution is not None else feature_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._dataset.samples.values())
        total_samples = len(records)

        feature_dim: int | None = None

        with _make_tile_extraction_reporter_ctx(feature_dir):
            logger.info("Loading tile encoder '%s'...", self._encoder.name)
            slide2vec_progress.emit_progress("model.loading", model_name=self._encoder.name)
            loaded = load_model(
                name=self._encoder.name,
                output_variant=self._encoder.output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            )
            encoder = loaded.model
            transform = loaded.transforms
            device = loaded.device
            slide2vec_progress.emit_progress(
                "model.ready",
                model_name=self._encoder.name,
                device=str(device),
            )

            image_dataset = _TileImageDataset(records, transform)
            execution = build_execution_options(
                self._encoder,
                encoder_name=self._encoder.name,
                output_dir=out_dir,
                num_gpus=1,
                save_tile_embeddings=True,
            )
            resolved_num_workers = execution.resolved_num_workers()
            worker_source = (
                "explicit EncoderConfig.num_workers"
                if self._encoder.num_workers is not None
                else "slide2vec cpu_worker_limit()"
            )
            slide2vec_progress.emit_progress_log(
                f"Tile DataLoader workers: {resolved_num_workers} ({worker_source})"
            )
            loader = DataLoader(
                image_dataset,
                batch_size=self._encoder.batch_size,
                shuffle=False,
                num_workers=resolved_num_workers,
                pin_memory=torch.cuda.is_available(),
                **(
                    {
                        "persistent_workers": execution.persistent_workers,
                        "prefetch_factor": execution.prefetch_factor,
                    }
                    if resolved_num_workers > 0
                    else {}
                ),
            )

            logger.info(
                "Encoding %d tile images with '%s' (precision=%s, batch_size=%d)...",
                len(records),
                self._encoder.name,
                self._encoder.precision,
                self._encoder.batch_size,
            )
            embedding_label = "Embedding tiles"
            slide2vec_progress.emit_progress(
                "embedding.slide.started",
                sample_id=embedding_label,
                total_tiles=total_samples,
            )

            processed_samples = 0
            with torch.inference_mode():
                for batch_images, batch_ids in loader:
                    batch_images = batch_images.to(device, non_blocking=True)
                    features = encoder.encode_tiles(batch_images)  # (B, D)
                    features = features.float().cpu()
                    if feature_dim is None:
                        feature_dim = features.shape[1]
                    for feat, sample_id in zip(features, batch_ids):
                        torch.save(feat, out_dir / f"{sample_id}.pt")
                    processed_samples += len(batch_ids)

                    slide2vec_progress.emit_progress(
                        "embedding.tile.progress",
                        sample_id=embedding_label,
                        processed=processed_samples,
                        total=total_samples,
                        unit="tile",
                    )

            slide2vec_progress.emit_progress(
                "embedding.slide.finished",
                sample_id=embedding_label,
                num_tiles=total_samples,
            )
            slide2vec_progress.emit_progress(
                "embedding.finished",
                slide_count=1,
                slides_completed=1,
                tile_artifacts=processed_samples,
                slide_artifacts=0,
            )

        logger.info("Saved tile features to %s (dim=%s)", out_dir, feature_dim)

        if cache_resolution is not None and feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)
            record_sample_identity_signatures(
                cache_resolution,
                [record.sample_id for record in records],
            )

        return FeatureStore(out_dir)
