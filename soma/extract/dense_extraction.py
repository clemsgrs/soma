"""Private dense-image extraction engine over caller-supplied images.

The Given-image counterpart of :class:`soma.extract.dense_slide_extraction.DenseSlideFeatureExtractor`:
where that one names ROI coordinates inside a slide, this one names whole pre-cropped
images (segmentation/detection tiles, patch benchmarks). Both delegate the extraction
itself to slide2vec — here :meth:`slide2vec.Model.embed_images_dense`, which reads each
image, runs the encoder's **normalization-only** dense transform, pads up to the encoder's
patch multiple, encodes whole-image or by sliding the encoder's native field, and writes
``dense_image_embeddings/<sample_id>.pt`` plus a geometry sidecar.

What stays soma's (ADR 0007, ADR 0008): the cache **key**, what counts as complete, the
identity signatures, and the recorded extraction geometry. slide2vec persists into the
directory soma resolves; it does not decide whether a cached grid may be reused.

Dense-input mode is a *derived* window-as-knob (design §5): ``window_size=None`` ⇒ one
padded forward; a smaller ``window_size`` (+ ``overlap``) slides the encoder over
patch-aligned windows and blends the token grids.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from slide2vec import DenseImageOptions, DenseOptions, ExecutionOptions, ImageSpec, Model
from slide2vec.encoders.registry import resolve_patch_size

from soma.extract.cache import (
    build_dense_cache_key,
    dense_extraction_geometry,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
    resolve_cache_dtype,
)
from soma.config import CacheOptions, EncoderLeaf
from soma.data.manifests import ImageManifest
from soma.extract.dense import DENSE_IMAGE_PAYLOAD_SUBDIR, CachedGridSource
from soma.data.geometry import compute_dense_geometry, normalize_hw
from soma.extract.dense.sliding import describe_dense_mode
from soma.extract.slide2vec_adapter import execution_options

logger = logging.getLogger(__name__)

_PAD_MODES = {"reflect", "constant", "zero", "replicate"}

# Env var naming a lock file that serializes dense-extraction bursts across processes.
DENSE_EXTRACT_LOCK_ENV = "SOMA_DENSE_EXTRACT_LOCK"


@contextmanager
def _dense_extract_lock():
    """Hold an exclusive cross-process lock over the extraction burst, if configured.

    Extraction is the memory-heaviest stage soma runs: encoder weights plus spawned
    loader workers plus a sustained dirty-page write burst into the feature cache. On a
    shared node whose container carries a memory cap, two concurrent extractions can push
    the cgroup over the cap and get OOM-killed — while training (read-mostly, clean page
    cache) coexists fine. Pointing :data:`DENSE_EXTRACT_LOCK_ENV` at a lock file shared
    by every run on the node makes each process hold an exclusive ``flock`` for the
    duration of its burst, so at most one extraction runs at a time; unset (the default)
    this is a no-op. The wait is blocking and logged, and the lock releases before
    training starts, so serialization costs idle time only while another burst is live.
    """
    lock_path = os.environ.get(DENSE_EXTRACT_LOCK_ENV)
    if not lock_path:
        yield
        return
    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.info(
                "Waiting for the dense-extraction lock at %s (another extraction is running)",
                path,
            )
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _DenseImageExtractor:
    """Encode tile images into dense ``(d, h, w)`` grids (the grid shape over pre-cropped tiles).

    Args:
        images: The image manifest whose ``image_path`` fields point to tile images. One
            extraction run must contain either raster images or spacing-readable
            pyramidal images, not a mixture; split mixed manifests into separate runs.
        encoder: The encoder leaf. For encoders that recommend ``dynamic_img_size=False``
            (H-optimus), set ``allow_non_recommended_settings`` in its params to opt into
            the variable input size dense extraction needs.
        dense: slide2vec's ``DenseOptions``: the read spacing, the supervision
            ``target_size``, padding, window and feature kind. They are handed to
            :meth:`slide2vec.Model.embed_images_dense` as its ``DenseImageOptions`` (the
            same fields) and keyed field by field.
        execution: slide2vec's ``ExecutionOptions`` (batch size, precision, GPUs).
    """

    def __init__(
        self,
        images: ImageManifest,
        encoder: EncoderLeaf,
        *,
        dense: DenseOptions,
        execution: ExecutionOptions | None = None,
        cache: CacheOptions | None = None,
    ) -> None:
        if dense.pad_mode not in _PAD_MODES:
            raise ValueError(
                f"unsupported pad_mode {dense.pad_mode!r}; expected one of {sorted(_PAD_MODES)}"
            )
        self._images = images
        self._encoder = encoder
        self._options = image_options(dense)
        self._target_size = normalize_hw(dense.target_size, name="target_size")
        self._execution = execution if execution is not None else ExecutionOptions()
        self._cache = cache or CacheOptions(enabled=False)

    def _dense_options(self) -> DenseImageOptions:
        """The read + encode recipe handed to slide2vec (strict post-read ``target_size``)."""
        return self._options

    def cache_dir(self, feature_dir: str | Path | None = None) -> Path | None:
        """The dense-image cache dir (``cache_root/dense_image/<key>``) this run uses.

        Resolved with **no side effects**: it loads no encoder (the key needs only the
        static ``patch_size`` slide2vec exposes as registry metadata — the #165
        check-before-load constant), scans no manifest (the dense key is
        manifest-independent), and creates no directories. Returns ``None`` when caching
        is disabled.

        Recomputes the *same* key ``run()`` resolves through ``resolve_dense_cache`` —
        ``build_dense_cache_key`` is the single source of truth for both, and the
        ``cache_root / "dense_image" / key`` layout mirrors ``_resolve_cache`` — so an offline
        re-scorer can address the *exact* grids this run trained on rather than guessing
        among sibling cache-key dirs. ``feature_dir`` is consulted only when
        ``cache.root_dir`` is unset (then the root is its parent ``feature_cache/``).
        """
        if not self._cache.enabled:
            return None
        patch_size = resolve_patch_size(self._encoder.name)
        dense_dtype = resolve_cache_dtype(
            self._cache.dtype, self._execution, encoder_name=self._encoder.name
        )
        cache_root = resolve_cache_root(
            self._cache, feature_dir=feature_dir if feature_dir is not None else Path.cwd()
        )
        key = build_dense_cache_key(
            tile_encoder_name=self._encoder.name,
            patch_size=patch_size,
            dense=self._options,
            execution=self._execution,
            dtype=dense_dtype,
        )
        return cache_root / "dense_image" / key

    def run(self, feature_dir: str | Path) -> CachedGridSource:
        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Check-before-load (#165): the dense cache key needs only patch_size, an
        # architectural constant slide2vec exposes as static registry metadata. Read it
        # without constructing the (multi-GB) encoder, so a full cache hit pays no ViT
        # load / CUDA context — the common case for campaign reruns. The model is
        # constructed on the miss path below, where extraction genuinely needs it.
        patch_size = resolve_patch_size(self._encoder.name)
        geometry = compute_dense_geometry(target_size=self._target_size, patch_size=patch_size)

        # Announce the resolved dense-input mode before the cache check, so it shows on a
        # cache hit too (extraction only runs on a miss).
        logger.info(
            "Dense extraction mode: %s",
            describe_dense_mode(self._options.window_size, self._options.overlap),
        )

        # Resolve the grid storage dtype from the shared cache.dtype umbrella (#164):
        # None ⇒ follow the compute precision; 'fp16'/'fp32' force it. Folded into the cache
        # key and handed to slide2vec as the write dtype, so storage matches the key.
        dense_dtype = resolve_cache_dtype(
            self._cache.dtype, self._execution, encoder_name=self._encoder.name
        )
        logger.info("Dense grid storage dtype resolved to %s", dense_dtype)

        cache_resolution = None
        out_root = feature_dir
        payload_dir = feature_dir / DENSE_IMAGE_PAYLOAD_SUBDIR
        if self._cache.enabled:
            cache_root = resolve_cache_root(self._cache, feature_dir=feature_dir)
            cache_resolution = resolve_dense_cache(
                cache_root=cache_root,
                images=self._images,
                tile_encoder_name=self._encoder.name,
                patch_size=patch_size,
                dense=self._options,
                execution=self._execution,
                dtype=dense_dtype,
                validate_payloads=self._cache.validate_payloads,
                cache_kind="dense_image",
                extraction_geometry=dense_extraction_geometry(
                    encoder_name=self._encoder.name,
                    target_size_px=self._target_size,
                    window_size=self._options.window_size,
                ),
            )
            payload_dir = cache_resolution.features_dir
            if cache_resolution.complete:
                logger.info("Reusing cached dense grids from %s", cache_resolution.features_dir)
                return CachedGridSource(payload_dir)
            # slide2vec appends its own payload subdir, so it is handed the cache dir.
            out_root = cache_resolution.cache_dir

        records = list(self._images)
        # Resume: encode only the images absent from the cache (the missing set comes from
        # the shared FeatureCacheResolution contract — no inline missing-logic). Present
        # grids are left untouched. Cache disabled ⇒ encode all.
        #
        # soma's cache is the authority on reuse; slide2vec runs its own recipe-aware resume
        # over whatever it is handed, but it only ever sees the images soma already decided
        # to encode, so the two cannot disagree about a skip. Batch size is deliberately
        # outside cache identity: it is runtime tuning, and resumed payloads need only meet
        # slide2vec's numerical-equivalence contract rather than be byte-identical.
        if cache_resolution is not None:
            wanted = set(cache_resolution.missing_sample_ids())
            if not wanted:
                return CachedGridSource(payload_dir)
            records = [record for record in records if record.sample_id in wanted]

        with _dense_extract_lock():
            model = Model.from_preset(self._encoder.name, **self._encoder.params)
            execution = execution_options(
                self._execution,
                encoder_name=self._encoder.name,
                output_dir=out_root,
                save_tile_embeddings=True,
                # soma resolves cache.dtype → 'fp16'/'fp32' once and passes the resolved
                # value, so the on-disk grid is cast to exactly the dtype folded into the
                # cache key above (key and storage can never drift).
                output_dtype=dense_dtype,
            )
            logger.info(
                "Encoding %d tiles into dense grids with '%s' at target_size=%s, patch=%s -> grid %s",
                len(records),
                self._encoder.name,
                self._target_size,
                patch_size,
                geometry.grid_shape,
            )
            artifacts = model.embed_images_dense(
                [
                    ImageSpec(
                        sample_id=str(record.sample_id),
                        image_path=record.image_path,
                        spacing_at_level_0=record.spacing_at_level_0,
                    )
                    for record in records
                ],
                dense=self._dense_options(),
                execution=execution,
            )

        if cache_resolution is not None and artifacts:
            cache_resolution = record_feature_dim(
                cache_resolution,
                int(artifacts[0].feature_dim),
                validate_payloads=self._cache.validate_payloads,
            )
            cache_resolution = record_sample_identity_signatures(
                cache_resolution,
                [record.sample_id for record in records],
                validate_payloads=self._cache.validate_payloads,
            )
        return CachedGridSource(payload_dir)


def image_options(dense: DenseOptions) -> DenseImageOptions:
    """slide2vec's ``DenseImageOptions`` with every field of ``dense`` (the two share
    their fields; the image route reads each image whole instead of at a region)."""
    return DenseImageOptions(
        **{field.name: getattr(dense, field.name) for field in dataclasses.fields(DenseOptions)}
    )
