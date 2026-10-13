"""Slide-manifest dense extraction: slides + annotation masks → cached ROI grids.

The slide-region counterpart of soma's internal dense given-image engine
(which reads *pre-cropped* tile images). Here the manifest rows are **whole slides** plus an
annotation vocabulary in hs2p's ``MaskSamplingConfig``; soma:

1. runs **hs2p annotation sampling** (``tile_slide(masks=…)``, ``merged`` output mode) per
   slide to get ROI coordinates (the union of tiles passing any class threshold);
2. asks **slide2vec** to encode each ROI region into a dense ``(d, gh, gw)`` grid
   (:meth:`slide2vec.Model.embed_regions_dense` — region reads, the encoder's
   whole/sliding dense forward, and the write of each grid + geometry sidecar); soma never
   reads slide regions, runs the encoder, or defines the artifact schema, mirroring how the
   pooled path defers to slide2vec;
3. **caches** the grids via soma's own dense cache layer, with the sampling spec folded into
   the cache key so distinct ``min_coverage``/spacing/strategy never alias.

Persistence is not caching (ADR 0007): slide2vec writes the payloads, into soma's resolved
cache directory, while the key, the completeness decision, the missing set and the identity
signatures stay here.

Splits stay outside extraction. Each sampled ROI is a private
:class:`~soma.extract.rois.RoiEntry` (its parent slide, its level-0 origin, its stored
mask crop); callers see the ROI ids and their parent sample, and derive the ROI cohort
from it (:meth:`soma.data.Cohort.derived`). No pixels are written to disk: the grids and
the mask crops are the only cached payloads.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from slide2vec import DenseOptions, ExecutionOptions, Model, PreprocessingConfig, SlideRegions

from soma.extract.cache import (
    dense_extraction_geometry,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
    resolve_cache_dtype,
    roi_sampling_signature,
)
from soma.extract.cache.options import CacheOptions
from soma.models.encoders import EncoderLeaf
from soma.data.manifests import AnnotationManifest, ImageManifest
from soma.extract.dense import CachedGridSource
from soma.data.geometry import normalize_hw
from soma.models.encoders.validation import resolve_filter_config
from soma.extract.rois import RoiEntry
from soma.extract.slide2vec_adapter import execution_options
from soma.extract.spacing import resolve_effective_spacing_um

logger = logging.getLogger(__name__)

def effective_spacing_um(
    preprocessing: PreprocessingConfig, spacing_at_level_0: float | None, *, spacing_policy: str
) -> float:
    """The spacing one slide is read at: the requested one, or its native spacing when the
    policy allows it and the slide is coarser (see :func:`soma.extract.spacing.resolve_effective_spacing_um`)."""
    tiling = preprocessing.tiling
    return resolve_effective_spacing_um(
        requested_spacing_um=tiling.requested_spacing_um,
        spacing_at_level_0=spacing_at_level_0,
        tolerance=tiling.tolerance,
        policy=spacing_policy,
    )


def sample_slide_rois(
    images: ImageManifest,
    annotations: AnnotationManifest,
    *,
    preprocessing: PreprocessingConfig,
    spacing_policy: str = "strict",
    sample_ids: Collection[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Run hs2p annotation sampling per slide; return level-0 ROI coords by slide id.

    hs2p tiles each slide's annotation raster with ``preprocessing.tiling`` (read at the
    slide's effective spacing), ``preprocessing.masks`` and ``preprocessing.filter`` (its
    contour-area threshold and pixel QC; unset resolves, as the roi_sampling key does, to
    ``ref_tile_size`` at the tile size and ``a_t=0``, keeping every annotated component).
    ``sample_ids`` restricts sampling to that subset of ``images``' slides (manifest
    order preserved) — the roi_sampling cache's partial-miss path samples only the
    missing slides without touching the hits. ``None`` (the default) samples every slide.
    """
    from hs2p import SlideSpec, tile_slide

    masks = preprocessing.masks
    filtering = resolve_filter_config(
        preprocessing.filter,
        tile_size_px=preprocessing.tiling.requested_tile_size_px,
        annotation_sampling=True,
    )
    if masks.output_mode != "merged":
        raise ValueError(
            f"slide-manifest dense extraction supports output_mode='merged' only, got "
            f"{masks.output_mode!r} (per_annotation extraction is deferred — soma #86)."
        )
    wanted: set[str] | None = None
    if sample_ids is not None:
        wanted = {str(sample_id) for sample_id in sample_ids}
        unknown = sorted(wanted - set(images.sample_ids))
        if unknown:
            raise ValueError(f"sample_ids not in the slide manifest: {unknown}")

    coords_by_slide: dict[str, list[tuple[int, int]]] = {}
    for record in images:
        sid = record.sample_id
        if wanted is not None and sid not in wanted:
            continue
        label_mask_path = annotations[sid].label_mask_path if sid in annotations else None
        if label_mask_path is None:
            raise ValueError(
                f"slide '{sid}' has no label_mask_path; a slide-manifest row needs one."
            )
        tiling = replace(
            preprocessing.tiling,
            requested_spacing_um=effective_spacing_um(
                preprocessing, record.spacing_at_level_0, spacing_policy=spacing_policy
            ),
        )
        # The annotation raster drives ROI sampling (per-class coverage), so it is the
        # sampling mask hs2p sees here.
        result = tile_slide(
            SlideSpec(
                sample_id=sid,
                image_path=record.image_path,
                mask_path=label_mask_path,
                spacing_at_level_0=record.spacing_at_level_0,
            ),
            tiling=tiling,
            filtering=filtering,
            masks=masks,
        )
        # MERGED collapses to a {None: merged} dict (one result per slide).
        merged = result[None] if isinstance(result, dict) else result
        coords_by_slide[sid] = [
            (int(x), int(y)) for x, y in zip(merged.tiles.x.tolist(), merged.tiles.y.tolist())
        ]
    return coords_by_slide


def write_roi_mask_crops(
    records: Sequence[RoiEntry],
    *,
    spacing_um_by_sample_id: Mapping[str, float],
    preprocessing: PreprocessingConfig,
) -> None:
    """Write each ROI's mask crop to its ``label_mask_crop_path``.

    The one place a slide-manifest ROI's annotation window is read: each slide's raster is
    opened and aligned once for all its ROIs, and each ROI is read at its grid's recorded
    spacing (``spacing_um_by_sample_id``) and the run's tile size — the read the
    segmentation head used to repeat on every step. The crop keeps the raw values and only
    the ROI's in-slide part (see :func:`soma.extract.dense.reader.read_mask_crop`).
    """
    from soma.extract.dense.reader import read_mask_regions_within_slide, write_mask_crop

    tiling = preprocessing.tiling
    if tiling is None:
        raise ValueError("Writing ROI mask crops requires a resolved preprocessing.tiling.")
    height, width = normalize_hw(int(tiling.requested_tile_size_px), name="tile_size_px")
    by_slide: dict[tuple, list] = defaultdict(list)
    for record in records:
        if record.label_mask_crop_path is None:
            raise ValueError(
                f"ROI '{record.sample_id}' needs a label_mask_crop_path to write its mask crop."
            )
        by_slide[
            (str(record.label_mask_path), str(record.slide.image_path), record.slide.spacing_at_level_0)
        ].append(record)
    for (label_mask_path, image_path, spacing_at_level_0), slide_records in by_slide.items():
        try:
            crops = read_mask_regions_within_slide(
                label_mask_path,
                regions=[
                    (record.region, float(spacing_um_by_sample_id[record.sample_id]))
                    for record in slide_records
                ],
                size=(width, height),
                reference_path=image_path,
                reference_backend=tiling.backend,
                spacing_at_level_0=spacing_at_level_0,
                pixel_mapping=preprocessing.masks.pixel_mapping,
                backend=tiling.mask_backend,
            )
            # strict: exhausts the reader, which closes the raster.
            for record, crop in zip(slide_records, crops, strict=True):
                try:
                    write_mask_crop(record.label_mask_crop_path, crop)
                except ValueError as error:
                    raise ValueError(f"segmentation ROI '{record.sample_id}': {error}") from error
        except ValueError as error:
            raise ValueError(f"annotation raster '{label_mask_path}': {error}") from error


class _SlideRegionExtractor:
    """Encode annotation-sampled ROIs into cached dense grids (slide-manifest segmentation).

    Takes the ROIs (:class:`~soma.extract.rois.RoiEntry`), groups them by slide, and
    asks slide2vec to encode each slide's regions into ``(d, gh, gw)`` grids;
    persists each grid via soma's dense store and resolves/populates soma's dense cache.
    """

    def __init__(
        self,
        rois: Sequence[RoiEntry],
        encoder: EncoderLeaf,
        *,
        preprocessing: PreprocessingConfig,
        dense: DenseOptions,
        spacing_policy: str = "strict",
        execution: ExecutionOptions | None = None,
        cache: CacheOptions | None = None,
    ) -> None:
        self._rois = list(rois)
        self._images = ImageManifest([roi.image for roi in self._rois])
        self._encoder = encoder
        self._preprocessing = preprocessing
        self._dense = dense
        self._spacing_policy = spacing_policy
        self._execution = execution if execution is not None else ExecutionOptions()
        self._cache = cache or CacheOptions(enabled=False)
        self._target_size = normalize_hw(int(dense.target_size), name="target_size")

    def _payload_stems(self) -> dict[str, str]:
        """ROI id → its path inside ``dense_embeddings/``, as ``<slide_id>/<x>_<y>``.

        slide2vec namespaces ROI grids by their parent slide, so an ROI's on-disk address
        is (slide, x, y) — all three read off the ROI entry, never parsed back out of the
        ROI id.
        """
        return {roi.sample_id: f"{roi.slide_id}/{int(roi.region[0])}_{int(roi.region[1])}" for roi in self._rois}

    def run(self, feature_dir: str | Path) -> CachedGridSource:
        from slide2vec.encoders.registry import resolve_patch_size

        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)
        payload_stems = self._payload_stems()

        # Check-before-load (#165): the dense cache key needs only patch_size, which
        # slide2vec exposes as static registry metadata — read it without constructing
        # the (multi-GB) encoder so a full ROI-grid cache hit pays no ViT load. load_model
        # is deferred to the miss path below. The static value is parity-tested against the
        # runtime encoder.patch_size in slide2vec (and re-asserted by load_model), so the
        # cache key is byte-identical to the pre-change key.
        patch_size = resolve_patch_size(self._encoder.name)
        signature = roi_sampling_signature(self._preprocessing, spacing_policy=self._spacing_policy)

        # Resolve the grid storage dtype from the shared cache.dtype umbrella (#164):
        # None ⇒ follow the encoder's resolved compute precision (override, else registry
        # recommendation); 'fp16'/'fp32' force it. Same resolver as the pooled path, so
        # the key never aliases an fp16 cache with an fp32 one.
        dense_dtype = resolve_cache_dtype(
            self._cache.dtype, self._execution, encoder_name=self._encoder.name
        )
        logger.info("Dense grid storage dtype resolved to %s", dense_dtype)

        cache_resolution = None
        # slide2vec appends ``dense_embeddings/`` to output_dir, and that subdirectory is
        # exactly the dense cache's features_dir — so the payload root is the cache dir.
        out_root = feature_dir
        if self._cache.enabled:
            cache_root = resolve_cache_root(self._cache, feature_dir=feature_dir)
            cache_resolution = resolve_dense_cache(
                cache_root=cache_root,
                images=self._images,
                tile_encoder_name=self._encoder.name,
                patch_size=patch_size,
                dense=self._dense,
                execution=self._execution,
                dtype=dense_dtype,
                sampling_signature=signature,
                validate_payloads=self._cache.validate_payloads,
                payload_stem_by_id=payload_stems,
                extraction_geometry=dense_extraction_geometry(
                    encoder_name=self._encoder.name,
                    target_size_px=self._target_size,
                    window_size=self._dense.window_size,
                ),
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached ROI dense grids from %s",
                    cache_resolution.features_dir,
                )
                return CachedGridSource(cache_resolution.cache_dir, payload_stems=payload_stems)
            out_root = cache_resolution.cache_dir

        # Resume: encode only the ROIs absent from the cache (the missing set comes
        # from the shared FeatureCacheResolution contract — no inline missing-logic).
        # A slide whose every ROI is already cached is dropped here, so it is never
        # opened/read; its grids on disk stay untouched. Cache disabled ⇒ encode all.
        wanted: set[str] | None = None
        if cache_resolution is not None:
            wanted = set(cache_resolution.missing_sample_ids())
            if not wanted:
                return CachedGridSource(out_root, payload_stems=payload_stems)

        # Group ROI coordinates by parent slide: one SlideRegions per slide, so slide2vec
        # opens and reads each slide once.
        coords_by_slide: dict[str, list[tuple[int, int]]] = defaultdict(list)
        image_path_by_slide: dict[str, Path] = {}
        spacing_by_slide: dict[str, float | None] = {}
        for roi in self._rois:
            if wanted is not None and roi.sample_id not in wanted:
                continue
            slide_id = roi.slide_id
            coords_by_slide[slide_id].append(roi.region)
            image_path_by_slide[slide_id] = roi.slide.image_path
            spacing_by_slide[slide_id] = roi.slide.spacing_at_level_0

        if not coords_by_slide:
            return CachedGridSource(out_root, payload_stems=payload_stems)

        # Cache miss (or cache disabled): extraction needs the encoder, so load it now.
        model = Model.from_preset(self._encoder.name, **self._encoder.params)
        execution = execution_options(
            self._execution,
            encoder_name=self._encoder.name,
            output_dir=out_root,
            save_tile_embeddings=True,
            # soma resolves cache.dtype → 'fp16'/'fp32' once and passes the resolved value,
            # so the on-disk grid is cast to exactly the dtype folded into the cache key
            # above (key and storage can never drift).
            output_dtype=dense_dtype,
        )
        regions_by_effective_spacing: dict[float, list[SlideRegions]] = defaultdict(list)
        for slide_id, coords in coords_by_slide.items():
            effective_spacing = effective_spacing_um(
                self._preprocessing, spacing_by_slide[slide_id], spacing_policy=self._spacing_policy
            )
            regions_by_effective_spacing[effective_spacing].append(
                SlideRegions(
                    sample_id=slide_id,
                    image_path=image_path_by_slide[slide_id],
                    coordinates=coords,
                    spacing_at_level_0=spacing_by_slide[slide_id],
                )
            )

        artifacts = []
        for effective_spacing, regions in regions_by_effective_spacing.items():
            # A slide read at its native spacing (native_if_coarser) is encoded at it too.
            dense = replace(self._dense, spacing_um=float(effective_spacing))
            artifacts.extend(
                model.embed_regions_dense(
                    regions,
                    dense=dense,
                    execution=execution,
                )
            )

        if cache_resolution is not None and artifacts:
            cache_resolution = record_feature_dim(
                cache_resolution,
                int(artifacts[0].feature_dim),
                validate_payloads=self._cache.validate_payloads,
            )
            cache_resolution = record_sample_identity_signatures(
                cache_resolution,
                [roi.sample_id for roi in self._rois],
                validate_payloads=self._cache.validate_payloads,
            )
        return CachedGridSource(out_root, payload_stems=payload_stems)
