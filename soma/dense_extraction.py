"""Private dense-image extraction engine over caller-supplied images.

The Given-image counterpart of :class:`soma.dense_slide_extraction.DenseSlideFeatureExtractor`:
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

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from slide2vec.api import DenseImageOptions, ImageSpec, Model
from slide2vec.encoders.registry import resolve_patch_size

from soma.cache import (
    build_dense_cache_key,
    dense_extraction_geometry,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
    resolve_output_dtype,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.dense import (
    DENSE_IMAGE_PAYLOAD_SUBDIR,
    DenseFeatureStore,
    compute_dense_geometry,
    normalize_hw,
)
from soma.dense.sliding import describe_dense_mode
from soma.slide2vec_adapter import build_execution_options

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
            print(f"… waiting for dense-extraction lock: {path}")
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _DenseImageExtractor:
    """Encode tile images into dense ``(d, h, w)`` grids (``dataset_type="segmentation"``).

    Args:
        dataset: Dataset whose ``image_path`` fields point to tile images. One
            extraction run must contain either raster images or spacing-readable
            pyramidal images, not a mixture; split mixed manifests into separate runs.
        encoder: Encoder configuration. For encoders that recommend
            ``dynamic_img_size=False`` (H-optimus), set
            ``allow_non_recommended_settings=True`` to opt into the variable input
            size dense extraction needs.
        target_size: The supervision tile/mask size (int or ``(h, w)``). Fixed per
            run (v1).
        pad_mode: How to pad up to the patch multiple — ``"reflect"`` (default, no
            out-of-distribution constant region), ``"constant"``/``"zero"``, or
            ``"replicate"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        *,
        target_size: int | tuple[int, int],
        spacing_um: float,
        backend: str = "auto",
        tolerance: float = 0.05,
        pad_mode: str = "reflect",
        window_size: int | None = None,
        overlap: float = 0.0,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")
        if window_size is not None and int(window_size) <= 0:
            raise ValueError(f"window_size must be a positive int or None, got {window_size!r}")
        if not (0.0 <= float(overlap) < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
        self._dataset = dataset
        self._encoder = encoder
        self._target_size = normalize_hw(target_size, name="target_size")
        self._spacing_um = float(spacing_um)
        self._backend = backend
        self._tolerance = float(tolerance)
        self._pad_mode = pad_mode
        self._window_size = None if window_size is None else int(window_size)
        self._overlap = float(overlap)
        self._dense_input_mode = "whole" if window_size is None else "sliding_window"
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)
        # The run's preprocessing identity (spacing/tolerance/backend/tile size) is
        # folded into the dense cache key — so grids read at different spacings can
        # never alias to one cache entry (mirrors the pooled/bag tile path).
        self._preprocessing = preprocessing
        # feature_kind + attention knobs (design — attention-pixel segmentation §7).
        # patch_features → the ViT patch grid; cls_attention → per-head prefix-token
        # self-attention. slide2vec picks the per-window encode from this.
        self._feature_kind = (
            (preprocessing.feature_kind or "patch_features")
            if preprocessing is not None
            else "patch_features"
        )
        if preprocessing is not None and self._feature_kind == "cls_attention":
            self._attention_blocks = tuple(preprocessing.attention.blocks)
            self._attention_include_registers = bool(preprocessing.attention.include_registers)
        else:
            self._attention_blocks = (-1,)
            self._attention_include_registers = False

    def _image_pad_value(self) -> float | None:
        # Only meaningful for constant/zero padding; None (N/A) for reflect/replicate.
        return 0.0 if self._pad_mode in ("constant", "zero") else None

    def _dense_options(self) -> DenseImageOptions:
        """The read + encode recipe handed to slide2vec.

        ``target_size`` is passed as an ``int`` when square so the option object is
        spelled the way a square run reads, and as ``(h, w)`` otherwise; slide2vec treats
        it as a strict post-read declaration either way — it never resizes to fit.
        """
        height, width = self._target_size
        return DenseImageOptions(
            target_size=int(height) if height == width else (int(height), int(width)),
            # soma always states a spacing (the pipeline requires one for a dense run), so
            # slide2vec's ``None`` behaviour — resolve the encoder's registry default for a
            # spacing-readable source — is unreachable from soma's config surface.
            spacing_um=self._spacing_um,
            tolerance=self._tolerance,
            backend=self._backend,
            pad_mode=self._pad_mode,
            image_pad_value=self._image_pad_value(),
            # window_size=None ⇒ one whole-image forward; a smaller window slides the
            # encoder over patch-aligned windows of each padded image and blends the grids.
            # Sliding is required for encoders that only accept their native input size.
            window_size=self._window_size,
            overlap=self._overlap,
            feature_kind=self._feature_kind,
            attention_blocks=self._attention_blocks,
            attention_include_registers=self._attention_include_registers,
        )

    def cache_dir(self, feature_dir: str | Path | None = None) -> Path | None:
        """The dense-image cache dir (``cache_root/dense_image/<key>``) this run uses.

        Resolved with **no side effects**: it loads no encoder (the key needs only the
        static ``patch_size`` slide2vec exposes as registry metadata — the #165
        check-before-load constant), scans no dataset (the dense key is
        dataset-independent), and creates no directories. Returns ``None`` when caching
        is disabled.

        Recomputes the *same* key ``run()`` resolves through ``resolve_dense_cache`` —
        ``build_dense_cache_key`` is the single source of truth for both, and the
        ``cache_root / "dense_image" / key`` layout mirrors ``_resolve_cache`` — so an offline
        re-scorer can address the *exact* grids this run trained on rather than guessing
        among sibling cache-key dirs (an empty orphan from a since-changed key looks just
        like the real one to a blind glob). ``feature_dir`` is consulted only when
        ``cache.root_dir`` is unset (then the root is its parent ``feature_cache/``);
        the pipeline always sets ``root_dir`` for dense runs, so it is usually moot.
        """
        if not self._cache.enabled:
            return None
        patch_size = resolve_patch_size(self._encoder.name)
        dense_dtype = resolve_output_dtype(
            self._cache.dtype, self._execution.precision or self._encoder.precision
        )
        cache_root = resolve_cache_root(
            self._cache, feature_dir=feature_dir if feature_dir is not None else Path.cwd()
        )
        key = build_dense_cache_key(
            tile_encoder_name=self._encoder.name,
            target_size=self._target_size,
            patch_size=patch_size,
            pad_mode=self._pad_mode,
            execution=self._encoder,
            preprocessing=self._preprocessing,
            dense_input_mode=self._dense_input_mode,
            window_size=self._window_size,
            overlap=self._overlap,
            feature_kind=self._feature_kind,
            attention_blocks=self._attention_blocks,
            attention_include_registers=self._attention_include_registers,
            dtype=dense_dtype,
        )
        return cache_root / "dense_image" / key

    def run(self, feature_dir: str | Path) -> DenseFeatureStore:
        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Check-before-load (#165): the dense cache key needs only patch_size, an
        # architectural constant slide2vec exposes as static registry metadata. Read it
        # without constructing the (multi-GB) encoder, so a full cache hit pays no ViT
        # load / CUDA context — the common case for campaign reruns. The model is
        # constructed on the miss path below, where extraction genuinely needs it.
        patch_size = resolve_patch_size(self._encoder.name)
        geometry = compute_dense_geometry(target_size=self._target_size, patch_size=patch_size)

        # Announce the resolved dense-input mode before the cache check, so it always shows
        # (cache hit too — extraction only runs on a miss) regardless of logging config.
        # print, not logger, so the user never has to opt into seeing it.
        print(f"Dense extraction mode: {describe_dense_mode(self._window_size, self._overlap)}")

        # Resolve the grid storage dtype from the shared cache.dtype umbrella (#164):
        # None ⇒ follow the compute precision; 'fp16'/'fp32' force it. Folded into the cache
        # key (guarded so fp32 keys stay byte-stable) and handed to slide2vec as the write
        # dtype, so storage matches the key.
        dense_dtype = resolve_output_dtype(
            self._cache.dtype, self._execution.precision or self._encoder.precision
        )

        cache_resolution = None
        out_root = feature_dir
        payload_dir = feature_dir / DENSE_IMAGE_PAYLOAD_SUBDIR
        if self._cache.enabled:
            cache_root = resolve_cache_root(self._cache, feature_dir=feature_dir)
            cache_resolution = resolve_dense_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                target_size=self._target_size,
                patch_size=patch_size,
                pad_mode=self._pad_mode,
                execution=self._encoder,
                preprocessing=self._preprocessing,
                dense_input_mode=self._dense_input_mode,
                window_size=self._window_size,
                overlap=self._overlap,
                feature_kind=self._feature_kind,
                attention_blocks=self._attention_blocks,
                attention_include_registers=self._attention_include_registers,
                dtype=dense_dtype,
                validate_payloads=self._cache.validate_payloads,
                cache_kind="dense_image",
                extraction_geometry=dense_extraction_geometry(
                    encoder_name=self._encoder.name,
                    target_size_px=self._target_size,
                    window_size=self._window_size,
                ),
            )
            payload_dir = cache_resolution.features_dir
            if cache_resolution.complete:
                logger.info("Reusing cached dense grids from %s", cache_resolution.features_dir)
                return DenseFeatureStore(payload_dir)
            # slide2vec appends its own payload subdir, so it is handed the cache dir.
            out_root = cache_resolution.cache_dir

        records = list(self._dataset.samples.values())
        # Resume: encode only the images absent from the cache (the missing set comes from
        # the shared FeatureCacheResolution contract — no inline missing-logic). Present
        # grids are left untouched. Cache disabled ⇒ encode all.
        #
        # soma's cache is the authority on reuse; slide2vec runs its own recipe-aware resume
        # over whatever it is handed, but it only ever sees the images soma already decided
        # to encode, so the two cannot disagree about a skip. The one thing upstream's recipe
        # deliberately does not key on is batch size — and soma's dense grids are only
        # byte-stable at a fixed batch size, which is why the standing rule is to resume a
        # dense cache at the batch size that built it (docs/caching.rst).
        if cache_resolution is not None:
            wanted = set(cache_resolution.missing_sample_ids())
            if not wanted:
                return DenseFeatureStore(payload_dir)
            records = [record for record in records if record.sample_id in wanted]

        with _dense_extract_lock():
            model = Model.from_preset(
                self._encoder.name,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            )
            execution = build_execution_options(
                self._encoder,
                execution=self._execution,
                encoder_name=self._encoder.name,
                output_dir=out_root,
                num_gpus=self._execution.num_gpus,
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
        return DenseFeatureStore(payload_dir)
