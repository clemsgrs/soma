"""Private tile-image extraction engine for pooled feature vectors.

The Given-geometry entry point (ADR 0006): the dataset rows are *pre-cropped images*
that soma never asked for at any particular size — a public patch benchmark (BACH, CRC,
Gleason, BreakHis, MHIST, PCam), an exported ROI set — so the encoder's shipped transform
is the contract and no geometry is declared.

soma does not encode anything here. It resolves the cache (key, completeness, identity
signatures), hands slide2vec every image that needs work in one
:meth:`slide2vec.Model.embed_images` call, and points ``execution.output_dir`` at the
resolved cache directory so slide2vec writes its payloads straight into
``<cache_dir>/image_embeddings/`` — which *is* soma's ``features_dir``, not a schema soma
translates into (ADR 0007). Persistence, batching, multi-GPU sharding, per-image resume
and progress all live upstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

from slide2vec import ExecutionOptions, ImageSpec, Model

from soma.cache import (
    FeatureIdentityCheck,
    commit_extracted_samples,
    invalidate_sample_identities,
    resolve_cache_dtype,
    resolve_cache_root,
    resolve_image_cache,
)
from soma.cache.compute_key import resolved_output_variant
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
from soma.data._legacy import LegacySamples, LegacyRecord
from soma.cache._types import PACKED_FILENAME
from soma.data import CachedSetSource
from soma.slide2vec_adapter import build_execution_options

logger = logging.getLogger(__name__)

# slide2vec checks each image's recorded source path on resume. soma's sample identity
# covers that path, so an artifact recorded for another source is stale by soma's own
# rule: slide2vec replaces it instead of refusing to resume.
_ON_IMAGE_MISMATCH = "reencode"


class _TileFeatureExtractor:
    """Encode individual tile images into 1D feature vectors using a tile encoder.

    This is the entry point for ``dataset_type="tile"`` pipelines: each sample's
    ``image_path`` points at one pre-cropped image, and the result is one 1-D ``.pt``
    feature vector per sample.

    Args:
        dataset: LegacySamples whose ``image_path`` fields point to tile images.
        encoder: Encoder configuration (name, precision, batch_size, etc.).
        cache: Optional cache configuration. When enabled, features are stored
            in a content-addressed cache directory and reused across runs.
    """

    def __init__(
        self,
        dataset: LegacySamples,
        encoder: EncoderConfig,
        *,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)

    def run(self, feature_dir: str | Path) -> CachedSetSource:
        """Encode all tile images and return a CachedSetSource over the results.

        Args:
            feature_dir: Directory to write embeddings into (under an
                ``image_embeddings/`` subdirectory). Ignored when a complete cache hit
                is found.

        Returns:
            CachedSetSource over the 1-D feature vectors.
        """
        feature_dir = Path(feature_dir).resolve()

        # On-disk feature dtype (#164): one resolved value folded into the cache key and
        # handed to slide2vec as output_dtype, so storage matches the key.
        dtype = resolve_cache_dtype(
            self._cache.dtype, self._encoder, encoder_name=self._encoder.name
        )
        # Key on the *resolved* output variant, as the pooled WSI path does, so a config
        # that leaves ``encoder.output_variant`` null shares the cache of one that names
        # the encoder's default variant explicitly.
        output_variant = resolved_output_variant(
            self._encoder.name, self._encoder.output_variant
        )
        logger.info(
            "Tile-image features: encoder=%s output_variant=%s dtype=%s",
            self._encoder.name,
            output_variant,
            dtype,
        )

        records = list(self._dataset.samples.values())
        if not self._cache.enabled:
            feature_dir.mkdir(parents=True, exist_ok=True)
            if records:
                # slide2vec may replace a reused output directory's vectors (a re-pointed
                # sample), so a pack of what the directory held before no longer holds.
                (feature_dir / "image_embeddings" / PACKED_FILENAME).unlink(missing_ok=True)
                self._embed(records, out_root=feature_dir, dtype=dtype)
            return CachedSetSource(feature_dir)

        cache_resolution = resolve_image_cache(
            cache_root=resolve_cache_root(self._cache, feature_dir=feature_dir),
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            execution=self._encoder,
            output_variant=output_variant,
            dtype=dtype,
            validate_payloads=self._cache.validate_payloads,
            feature_identity=self._feature_identity_check(feature_dir, dtype=dtype),
        )
        if cache_resolution.complete:
            logger.info(
                "Reusing cached tile features from %s",
                cache_resolution.features_dir,
            )
            return CachedSetSource(cache_resolution.features_dir)

        missing = set(cache_resolution.missing_sample_ids())
        pending_ids = [record.sample_id for record in records if record.sample_id in missing]
        if pending_ids:
            # Nothing may vouch for these samples while slide2vec may replace their
            # features: a failure from here on leaves them unsigned, for slide2vec to
            # validate or re-encode on the next run.
            cache_resolution = invalidate_sample_identities(cache_resolution, pending_ids)
            feature_dim = self._embed(
                [self._dataset.samples[sample_id] for sample_id in pending_ids],
                out_root=cache_resolution.cache_dir,
                dtype=dtype,
            )
            commit_extracted_samples(
                cache_resolution,
                pending_ids,
                feature_dim=feature_dim,
                validate_payloads=self._cache.validate_payloads,
            )
        return CachedSetSource(cache_resolution.cache_dir)

    def _embed(self, records: list[LegacyRecord], *, out_root: Path, dtype: str) -> int:
        """Hand ``records`` to slide2vec in one call and return the feature dimension.

        slide2vec writes under ``out_root/image_embeddings/``. It reuses an image whose
        payload and sidecar record its source path and the current feature identity, and
        encodes every other one.
        """
        logger.info(
            "Embedding %d tile images with '%s' (batch_size=%d)...",
            len(records),
            self._encoder.name,
            self._encoder.batch_size,
        )
        model = Model.from_preset(
            self._encoder.name,
            output_variant=self._encoder.output_variant,
            allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
        )
        artifacts = model.embed_images(
            [ImageSpec(sample_id=record.sample_id, image_path=record.image_path) for record in records],
            execution=self._execution_options(out_root, dtype=dtype),
        )
        requested = sorted(record.sample_id for record in records)
        returned = sorted(str(artifact.sample_id) for artifact in artifacts)
        if returned != requested:
            raise RuntimeError(
                f"slide2vec returned image embeddings for {len(returned)} samples where "
                f"{len(requested)} were requested, or for other samples; nothing is "
                "committed to the cache."
            )
        feature_dim = int(artifacts[0].feature_dim)
        logger.info("Saved tile features to %s (dim=%s)", out_root, feature_dim)
        return feature_dim

    def _execution_options(self, output_dir: Path, *, dtype: str) -> ExecutionOptions:
        return build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=self._encoder.name,
            output_dir=output_dir,
            num_gpus=self._execution.num_gpus,
            save_tile_embeddings=True,
            output_dtype=dtype,
            on_image_mismatch=_ON_IMAGE_MISMATCH,
        )

    def _feature_identity_check(self, feature_dir: Path, *, dtype: str) -> FeatureIdentityCheck:
        """Verify a cache against the identity slide2vec would give these images now."""

        def differing(recorded: dict) -> dict:
            model = Model.from_preset(
                self._encoder.name,
                output_variant=self._encoder.output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            )
            return model.pooled_identity_differences(
                recorded,
                # Given geometry: pre-cropped images declare no tiling.
                preprocessing=None,
                execution=self._execution_options(feature_dir, dtype=dtype),
            )

        return FeatureIdentityCheck(
            differing=differing,
            on_mismatch=self._cache.on_identity_mismatch,
        )
