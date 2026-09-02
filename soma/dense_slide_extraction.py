"""Slide-manifest dense extraction: slides + annotation masks → cached ROI grids.

The slide-region counterpart of soma's internal dense given-image engine
(which reads *pre-cropped* tile images). Here the dataset rows are **whole slides** plus a
``masks:``/``sampling:`` config; soma:

1. runs **hs2p annotation sampling** (``tile_slide(sampling=…)``, ``merged`` output mode) per
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

Splits stay outside extraction. The ROI manifest records one row per sampled tile, with
its parent ``slide_id``, paths, and ``region_x``/``region_y`` origin; callers project their
existing splits through that explicit ancestry. The ROI manifest is a coordinate manifest,
not a tile dump: no pixels are written to disk, and the grids are the only cached payloads.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from slide2vec import DenseOptions, Model, SlideRegions

from soma.cache import (
    dense_extraction_geometry,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
    resolve_cache_dtype,
)
from soma.config import (
    CacheConfig,
    EncoderConfig,
    ExecutionConfig,
    MasksConfig,
    PreprocessingConfig,
    SamplingConfig,
)
from soma.dense import DenseFeatureStore, normalize_hw
from soma.slide2vec_adapter import build_execution_options

if TYPE_CHECKING:
    from soma.dataset import SegmentationManifest

logger = logging.getLogger(__name__)

# soma SamplingConfig → hs2p coordinate-selection strategy strings.
_STRATEGY_MAP = {"joint": "joint_sampling", "independent": "independent_sampling"}


def _build_tiling_config(preprocessing: PreprocessingConfig, sampling: SamplingConfig):
    """The composed hs2p ``TilingConfig`` for slide-manifest ROI sampling (ADR 0009).

    The geometry comes from the one composition soma has, so this seam cannot drift from the
    pooled one or from the cache key. Only the sampling strategy is restated, because the
    caller resolves it (a manifest run may sample under a strategy the config leaves unset).

    soma's ``preprocessing.min_coverage`` is the tissue threshold in hs2p's own masks-shaped
    form; the per-class sampling map still flows separately through
    ``_resolve_sampling_spec_from_masks(masks, …)``, and this tiling-level entry only feeds
    the result's binary ``min_tissue_fraction`` provenance.
    """
    return replace(
        preprocessing.tiling_config(),
        independent_sampling=(sampling.strategy == "independent"),
    )


def sampling_signature(
    masks: MasksConfig, sampling: SamplingConfig, preprocessing: PreprocessingConfig
) -> dict:
    """Canonical signature of the sampling spec, for the dense cache key.

    Different ``min_coverage``/strategy/spacing/tile-size ⇒ different ROIs ⇒ a distinct
    cache, even when two specs happen to yield colliding coordinates.
    """
    signature = {
        "pixel_mapping": dict(masks.pixel_mapping),
        "min_coverage": dict(masks.min_coverage),
        "colors": (
            None
            if masks.colors is None
            else {k: (None if v is None else list(v)) for k, v in masks.colors.items()}
        ),
        "strategy": str(sampling.strategy),
        "output_mode": str(sampling.output_mode),
        "spacing_um": float(preprocessing.requested_spacing_um),
        "tile_size_px": list(
            normalize_hw(int(preprocessing.requested_tile_size_px), name="tile_size_px")
        ),
        "overlap": float(preprocessing.overlap),
    }
    if preprocessing.spacing_policy != "strict":
        # Preserve legacy strict cache identities; only the opt-in behavior
        # needs an additional discriminator.
        signature["spacing_policy"] = str(preprocessing.spacing_policy)
    return signature


def sample_slide_rois(
    dataset: "SegmentationManifest",
    *,
    masks: MasksConfig,
    sampling: SamplingConfig,
    preprocessing: PreprocessingConfig,
    sample_ids: Collection[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Run hs2p annotation sampling per slide; return level-0 ROI coords by slide id.

    ``sample_ids`` restricts sampling to that subset of the manifest's slides (manifest
    order preserved) — the roi_sampling cache's partial-miss path samples only the
    missing slides without touching the hits. ``None`` (the default) samples every slide.
    """
    from hs2p import SlideSpec, tile_slide
    from hs2p.configs.resolvers import _resolve_sampling_spec_from_masks
    from hs2p.wsi.types import CoordinateOutputMode

    if sampling.output_mode != "merged":
        raise ValueError(
            f"slide-manifest dense extraction supports output_mode='merged' only, got "
            f"{sampling.output_mode!r} (per_annotation extraction is deferred — soma #86)."
        )
    wanted: set[str] | None = None
    if sample_ids is not None:
        wanted = {str(sample_id) for sample_id in sample_ids}
        unknown = sorted(wanted - set(dataset.samples))
        if unknown:
            raise ValueError(f"sample_ids not in the slide manifest: {unknown}")
    strategy = _STRATEGY_MAP[sampling.strategy]

    coords_by_slide: dict[str, list[tuple[int, int]]] = {}
    for sid, record in dataset.samples.items():
        if wanted is not None and sid not in wanted:
            continue
        if record.label_mask_path is None:
            raise ValueError(
                f"slide '{sid}' has no label_mask_path; a slide-manifest row needs one."
            )
        effective_preprocessing = replace(
            preprocessing,
            requested_spacing_um=preprocessing.effective_spacing_um(record.spacing_at_level_0),
        )
        tiling = _build_tiling_config(effective_preprocessing, sampling)
        spec = _resolve_sampling_spec_from_masks(masks, tiling=tiling)
        # The annotation raster drives ROI sampling (per-class coverage), so it is the
        # sampling mask hs2p sees here.
        result = tile_slide(
            SlideSpec(
                sample_id=sid,
                image_path=record.image_path,
                mask_path=record.label_mask_path,
                spacing_at_level_0=record.spacing_at_level_0,
            ),
            tiling=tiling,
            sampling=spec,
            selection_strategy=strategy,
            output_mode=CoordinateOutputMode.MERGED,
        )
        # MERGED collapses to a {None: merged} dict (one result per slide).
        merged = result[None] if isinstance(result, dict) else result
        coords_by_slide[sid] = [
            (int(x), int(y)) for x, y in zip(merged.tiles.x.tolist(), merged.tiles.y.tolist())
        ]
    return coords_by_slide


def build_roi_dataset(
    dataset: "SegmentationManifest",
    coords_by_slide: dict[str, list[tuple[int, int]]],
    *,
    out_dir: Path,
) -> Path:
    """Persist the deterministic effective ROI dataset without depending on splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "dataset.csv"
    rows: list[dict] = []
    for slide_id in dataset.sample_ids:
        record = dataset.samples[slide_id]
        for x, y in coords_by_slide.get(slide_id, []):
            rows.append(
                {
                    "sample_id": f"{slide_id}__x{x}_y{y}",
                    "slide_id": slide_id,
                    "image_path": str(record.image_path),
                    "mask_path": ("" if record.mask_path is None else str(record.mask_path)),
                    "label_mask_path": str(record.label_mask_path),
                    "patient_id": ("" if record.patient_id is None else record.patient_id),
                    "spacing_at_level_0": record.spacing_at_level_0,
                    "region_x": int(x),
                    "region_y": int(y),
                }
            )
    _write_csv(
        manifest_path,
        [
            "sample_id",
            "slide_id",
            "image_path",
            "mask_path",
            "label_mask_path",
            "patient_id",
            "spacing_at_level_0",
            "region_x",
            "region_y",
        ],
        rows,
    )
    return manifest_path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _SlideRegionExtractor:
    """Encode annotation-sampled ROIs into cached dense grids (slide-manifest segmentation).

    Reads the ROI coordinate manifest (built by :func:`build_roi_dataset`), groups ROIs by
    slide, and asks slide2vec to encode each slide's regions into ``(d, gh, gw)`` grids;
    persists each grid via soma's dense store and resolves/populates soma's dense cache.
    """

    def __init__(
        self,
        roi_dataset: "SegmentationManifest",
        encoder: EncoderConfig,
        *,
        masks: MasksConfig,
        sampling: SamplingConfig,
        preprocessing: PreprocessingConfig,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
    ) -> None:
        self._dataset = roi_dataset
        self._encoder = encoder
        self._masks = masks
        self._sampling = sampling
        self._preprocessing = preprocessing
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)
        self._target_size = normalize_hw(
            int(preprocessing.requested_tile_size_px), name="target_size"
        )
        self._spacing_um = float(preprocessing.requested_spacing_um)
        self._pad_mode = "reflect"
        # Encoder-window knobs (design §5): window_size=None ⇒ one whole-region forward;
        # a smaller window slides the encoder over patch-aligned windows of each padded ROI
        # and blends the grids. Both are delegated to slide2vec's unified dense primitive
        # (the same path the pre-cropped dense extractor uses). Sliding is required for
        # encoders that only accept their native input size (e.g. phikon at 224).
        self._window_size = preprocessing.dense_window_size
        self._overlap = float(preprocessing.dense_window_overlap)
        self._feature_kind = preprocessing.feature_kind or "patch_features"
        if self._feature_kind == "cls_attention":
            self._attention_blocks = tuple(preprocessing.attention.blocks)
            self._attention_include_registers = bool(preprocessing.attention.include_registers)
        else:
            self._attention_blocks = (-1,)
            self._attention_include_registers = False

    def _payload_stems(self) -> dict[str, str]:
        """ROI id → its path inside ``dense_embeddings/``, as ``<slide_id>/<x>_<y>``.

        slide2vec namespaces ROI grids by their parent slide, so an ROI's on-disk address
        is (slide, x, y) — all three read off the manifest row, never parsed back out of
        the ROI id.
        """
        stems: dict[str, str] = {}
        for record in self._dataset.samples.values():
            if record.region is None:
                raise ValueError(
                    f"ROI '{record.sample_id}' has no region; the ROI manifest must carry "
                    "region_x/region_y."
                )
            if record.slide_id is None:
                raise ValueError(
                    f"ROI '{record.sample_id}' has no slide_id; the ROI manifest must carry "
                    "the parent slide id (it is part of the grid's on-disk address)."
                )
            x, y = record.region
            stems[record.sample_id] = f"{record.slide_id}/{int(x)}_{int(y)}"
        return stems

    def run(self, feature_dir: str | Path) -> DenseFeatureStore:
        from slide2vec.encoders.registry import resolve_patch_size

        dense_input_mode = "whole" if self._window_size is None else "sliding_window"

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
        signature = sampling_signature(self._masks, self._sampling, self._preprocessing)

        # Resolve the grid storage dtype from the shared cache.dtype umbrella (#164):
        # None ⇒ follow the encoder's resolved compute precision (override, else registry
        # recommendation); 'fp16'/'fp32' force it. Same resolver as the pooled path, so
        # the key never aliases an fp16 cache with an fp32 one.
        dense_dtype = resolve_cache_dtype(
            self._cache.dtype, self._encoder, encoder_name=self._encoder.name
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
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                target_size=self._target_size,
                patch_size=patch_size,
                pad_mode=self._pad_mode,
                execution=self._encoder,
                preprocessing=self._preprocessing,
                dense_input_mode=dense_input_mode,
                window_size=self._window_size,
                overlap=self._overlap,
                feature_kind=self._feature_kind,
                attention_blocks=self._attention_blocks,
                attention_include_registers=self._attention_include_registers,
                dtype=dense_dtype,
                sampling_signature=signature,
                validate_payloads=self._cache.validate_payloads,
                payload_stem_by_id=payload_stems,
                extraction_geometry=dense_extraction_geometry(
                    encoder_name=self._encoder.name,
                    target_size_px=self._target_size,
                    window_size=self._window_size,
                ),
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached ROI dense grids from %s",
                    cache_resolution.features_dir,
                )
                return DenseFeatureStore(cache_resolution.cache_dir, payload_stems=payload_stems)
            out_root = cache_resolution.cache_dir

        # Resume: encode only the ROIs absent from the cache (the missing set comes
        # from the shared FeatureCacheResolution contract — no inline missing-logic).
        # A slide whose every ROI is already cached is dropped here, so it is never
        # opened/read; its grids on disk stay untouched. Cache disabled ⇒ encode all.
        wanted: set[str] | None = None
        if cache_resolution is not None:
            wanted = set(cache_resolution.missing_sample_ids())
            if not wanted:
                return DenseFeatureStore(out_root, payload_stems=payload_stems)

        # Group ROI coordinates by parent slide: one SlideRegions per slide, so slide2vec
        # opens and reads each slide once.
        coords_by_slide: dict[str, list[tuple[int, int]]] = defaultdict(list)
        image_path_by_slide: dict[str, Path] = {}
        spacing_by_slide: dict[str, float | None] = {}
        for record in self._dataset.samples.values():
            if wanted is not None and record.sample_id not in wanted:
                continue
            slide_id = str(record.slide_id)
            coords_by_slide[slide_id].append(record.region)
            image_path_by_slide[slide_id] = record.image_path
            if (
                slide_id in spacing_by_slide
                and spacing_by_slide[slide_id] != record.spacing_at_level_0
            ):
                raise ValueError(
                    f"ROI rows for slide '{slide_id}' disagree on spacing_at_level_0: "
                    f"{spacing_by_slide[slide_id]} vs {record.spacing_at_level_0}."
                )
            spacing_by_slide[slide_id] = record.spacing_at_level_0

        if not coords_by_slide:
            return DenseFeatureStore(out_root, payload_stems=payload_stems)

        # Cache miss (or cache disabled): extraction needs the encoder, so load it now.
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
            # soma resolves cache.dtype → 'fp16'/'fp32' once and passes the resolved value,
            # so the on-disk grid is cast to exactly the dtype folded into the cache key
            # above (key and storage can never drift).
            output_dtype=dense_dtype,
        )
        regions_by_effective_spacing: dict[float, list[SlideRegions]] = defaultdict(list)
        for slide_id, coords in coords_by_slide.items():
            effective_spacing = self._preprocessing.effective_spacing_um(spacing_by_slide[slide_id])
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
            dense = DenseOptions(
                spacing_um=effective_spacing,
                target_size=int(self._target_size[0]),
                tolerance=float(self._preprocessing.tolerance),
                backend=self._preprocessing.backend,
                pad_mode=self._pad_mode,
                # window_size=None ⇒ one whole-region forward; a smaller window slides the
                # encoder over patch-aligned windows of each padded ROI and blends the grids.
                # Sliding is required for encoders that only accept their native input size.
                window_size=self._window_size,
                overlap=self._overlap,
                feature_kind=self._feature_kind,
                attention_blocks=self._attention_blocks,
                attention_include_registers=self._attention_include_registers,
            )
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
                [record.sample_id for record in self._dataset.samples.values()],
                validate_payloads=self._cache.validate_payloads,
            )
        return DenseFeatureStore(out_root, payload_stems=payload_stems)
