"""TileFeatureExtractor — encodes individual tile images into 1D feature vectors.

The Given-geometry entry point (ADR 0006): the dataset rows are *pre-cropped images*
that soma never asked for at any particular size — a public patch benchmark (BACH, CRC,
Gleason, BreakHis, MHIST, PCam), an exported ROI set — so the encoder's shipped transform
is the contract and no geometry is declared.

soma does not encode anything here. It resolves the cache (key, completeness, identity
signatures), hands slide2vec the images that need encoding, and points
``execution.output_dir`` at the resolved cache directory so slide2vec writes its payloads
straight into ``<cache_dir>/image_embeddings/`` — which *is* soma's ``features_dir``, not a
schema soma translates into (ADR 0007). Persistence, batching, multi-GPU sharding, resume
and progress all live upstream in :meth:`slide2vec.Model.embed_images`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from slide2vec import ImageSpec, Model

from soma.cache import (
    FeatureCacheResolution,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_image_cache,
    resolve_output_dtype,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
from soma.dataset import Dataset, SampleRecord
from soma.encoders.validation import resolve_encoder_precision
from soma.features import FeatureStore
from soma.slide2vec_adapter import build_execution_options

logger = logging.getLogger(__name__)


def _drop_stale_payloads(features_dir: Path, sample_ids: list[str]) -> None:
    """Remove on-disk artifacts for samples soma has decided must be re-encoded.

    slide2vec resumes on **sidecar existence**, which is the right rule for an interrupted
    run but not for a sample whose *identity* changed underneath a stable ``sample_id``
    (a re-pointed ``image_path``, say). soma's cache resolution is the authority on which
    samples are stale; clearing their payloads first is how that decision reaches a
    resume check that cannot see identity signatures. Without it slide2vec would skip the
    sample and soma would then stamp the new signature onto the old features.
    """
    for sample_id in sample_ids:
        for path in features_dir.glob(f"{sample_id}.*"):
            path.unlink(missing_ok=True)


class TileFeatureExtractor:
    """Encode individual tile images into 1D feature vectors using a tile encoder.

    This is the entry point for ``dataset_type="tile"`` pipelines: each sample's
    ``image_path`` points at one pre-cropped image, and the result is one 1-D ``.pt``
    feature vector per sample.

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
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)

    def run(self, feature_dir: str | Path) -> FeatureStore:
        """Encode all tile images and return a FeatureStore over the results.

        Args:
            feature_dir: Directory to write embeddings into (under an
                ``image_embeddings/`` subdirectory). Ignored when a complete cache hit
                is found.

        Returns:
            FeatureStore over the 1-D feature vectors.
        """
        feature_dir = Path(feature_dir).resolve()

        # On-disk feature dtype (#164): one resolved value folded into the cache key and
        # handed to slide2vec as output_dtype, so storage matches the key.
        dtype = resolve_output_dtype(
            self._cache.dtype,
            resolve_encoder_precision(self._encoder, encoder_name=self._encoder.name),
        )

        cache_resolution: FeatureCacheResolution | None = None
        if self._cache.enabled:
            cache_root = resolve_cache_root(self._cache, feature_dir=feature_dir)
            cache_resolution = resolve_image_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                execution=self._encoder,
                output_variant=self._encoder.output_variant,
                dtype=dtype,
                fingerprint_files=self._cache.fingerprint_files,
                validate_payloads=self._cache.validate_payloads,
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached tile features from %s",
                    cache_resolution.features_dir,
                )
                return FeatureStore(cache_resolution.features_dir)

        # slide2vec appends ``image_embeddings/`` to output_dir, and that subdirectory is
        # exactly the cache's features_dir — so the payload root is the cache dir itself.
        out_root = cache_resolution.cache_dir if cache_resolution is not None else feature_dir
        out_root.mkdir(parents=True, exist_ok=True)

        records = list(self._dataset.samples.values())
        pending: list[SampleRecord] = records
        if cache_resolution is not None:
            missing = cache_resolution.missing_sample_ids()
            _drop_stale_payloads(cache_resolution.features_dir, missing)
            wanted = set(missing)
            pending = [record for record in records if record.sample_id in wanted]

        if pending:
            logger.info(
                "Encoding %d tile images with '%s' (batch_size=%d)...",
                len(pending),
                self._encoder.name,
                self._encoder.batch_size,
            )
            execution = build_execution_options(
                self._encoder,
                execution=self._execution,
                encoder_name=self._encoder.name,
                output_dir=out_root,
                num_gpus=self._execution.num_gpus,
                save_tile_embeddings=True,
                output_dtype=dtype,
            )
            model = Model.from_preset(
                self._encoder.name,
                output_variant=self._encoder.output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            )
            artifacts = model.embed_images(
                [
                    ImageSpec(sample_id=record.sample_id, image_path=record.image_path)
                    for record in pending
                ],
                execution=execution,
            )
            feature_dim = int(artifacts[0].feature_dim) if artifacts else None
            logger.info("Saved tile features to %s (dim=%s)", out_root, feature_dim)

            if cache_resolution is not None and feature_dim is not None:
                record_feature_dim(cache_resolution, feature_dim)
                record_sample_identity_signatures(
                    cache_resolution,
                    [record.sample_id for record in records],
                )

        return FeatureStore(out_root)
