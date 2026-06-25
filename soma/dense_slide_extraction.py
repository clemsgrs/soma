"""Slide-manifest dense extraction: slides + annotation masks → cached ROI grids.

The slide-manifest counterpart of :class:`soma.dense_extraction.DenseTileFeatureExtractor`
(which reads *pre-cropped* tile images). Here the dataset rows are **whole slides** plus a
``masks:``/``sampling:`` config; soma:

1. runs **hs2p annotation sampling** (``tile_slide(sampling=…)``, ``merged`` output mode) per
   slide to get ROI coordinates (the union of tiles passing any class threshold);
2. asks **slide2vec** to encode each ROI region into a dense ``(d, gh, gw)`` grid
   (:func:`slide2vec.runtime.dense_regions.encode_regions_dense` — region reads + the
   encoder's ``encode_tiles_dense``); soma never reads slide regions or runs the encoder
   itself, mirroring how the pooled path defers to slide2vec;
3. **caches** the grids via soma's own dense cache layer, with the sampling spec folded into
   the cache key so distinct ``min_coverage``/spacing/strategy never alias.

Splits stay slide-level and user-provided; the ROI manifest (one row per sampled tile, with
its parent slide's ``image_path``/``mask_path`` + a ``region_x``/``region_y`` origin) and the
ROI splits (each ROI inherits its parent slide's split/fold) are *derived* here — soma never
creates splits. The ROI manifest is a coordinate manifest, not a tile dump: no pixels are
written to disk, the grids are the only cached artifact.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from soma.cache import (
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, MasksConfig, PreprocessingConfig, SamplingConfig
from soma.dense import DenseFeatureStore, compute_dense_geometry, dense_grid_metadata, normalize_hw, write_dense_grid
from soma.slide2vec_adapter import build_execution_options

if TYPE_CHECKING:
    from soma.dataset import SampleRecord, SegmentationManifest

logger = logging.getLogger(__name__)

# soma SamplingConfig → hs2p coordinate-selection strategy strings.
_STRATEGY_MAP = {"joint": "joint_sampling", "independent": "independent_sampling"}


def _build_tiling_config(preprocessing: PreprocessingConfig, sampling: SamplingConfig):
    """hs2p ``TilingConfig`` from soma preprocessing (spacing/tile size/overlap/backend)."""
    from hs2p.configs import TilingConfig

    if preprocessing.requested_tile_size_px is None:
        raise ValueError(
            "slide-manifest segmentation requires preprocessing.requested_tile_size_px "
            "(the supervision tile size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "slide-manifest segmentation requires preprocessing.requested_spacing_um — set it "
            "or use an encoder advertising a single supported spacing."
        )
    # soma expresses the tissue threshold as ``preprocessing.min_coverage["tissue"]``, the same
    # masks-shaped map hs2p's ``TilingConfig.min_coverage`` expects. The per-class sampling map
    # still flows separately through ``_resolve_sampling_spec_from_masks(masks, …)``; this
    # tiling-level entry only feeds the result's binary ``min_tissue_fraction`` provenance.
    return TilingConfig(
        requested_spacing_um=float(preprocessing.requested_spacing_um),
        requested_tile_size_px=int(preprocessing.requested_tile_size_px),
        tolerance=float(preprocessing.tolerance),
        overlap=float(preprocessing.overlap),
        min_coverage={"tissue": float(preprocessing.min_coverage.get("tissue") or 0.0)},
        backend=preprocessing.backend,
        independent_sampling=(sampling.strategy == "independent"),
    )


def sampling_signature(
    masks: MasksConfig, sampling: SamplingConfig, preprocessing: PreprocessingConfig
) -> dict:
    """Canonical signature of the sampling spec, for the dense cache key.

    Different ``min_coverage``/strategy/spacing/tile-size ⇒ different ROIs ⇒ a distinct
    cache, even when two specs happen to yield colliding coordinates.
    """
    return {
        "pixel_mapping": dict(masks.pixel_mapping),
        "min_coverage": dict(masks.min_coverage),
        "colors": None if masks.colors is None else {k: (None if v is None else list(v)) for k, v in masks.colors.items()},
        "strategy": str(sampling.strategy),
        "output_mode": str(sampling.output_mode),
        "spacing_um": float(preprocessing.requested_spacing_um),
        "tile_size_px": list(normalize_hw(int(preprocessing.requested_tile_size_px), name="tile_size_px")),
        "overlap": float(preprocessing.overlap),
    }


def sample_slide_rois(
    dataset: "SegmentationManifest",
    *,
    masks: MasksConfig,
    sampling: SamplingConfig,
    preprocessing: PreprocessingConfig,
) -> dict[str, list[tuple[int, int]]]:
    """Run hs2p annotation sampling per slide; return level-0 ROI coords by slide id."""
    from hs2p import SlideSpec, tile_slide
    from hs2p.configs.resolvers import _resolve_sampling_spec_from_masks
    from hs2p.wsi.types import CoordinateOutputMode

    if sampling.output_mode != "merged":
        raise ValueError(
            f"slide-manifest dense extraction supports output_mode='merged' only, got "
            f"{sampling.output_mode!r} (per_annotation extraction is deferred — soma #86)."
        )
    tiling = _build_tiling_config(preprocessing, sampling)
    spec = _resolve_sampling_spec_from_masks(masks, tiling=tiling)
    strategy = _STRATEGY_MAP[sampling.strategy]

    coords_by_slide: dict[str, list[tuple[int, int]]] = {}
    for sid, record in dataset.samples.items():
        if record.mask_path is None:
            raise ValueError(f"slide '{sid}' has no mask_path; a slide-manifest row needs one.")
        result = tile_slide(
            SlideSpec(sample_id=sid, image_path=record.image_path, mask_path=record.mask_path),
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


def build_roi_manifest(
    dataset: "SegmentationManifest",
    slide_splits_csv: str | Path,
    coords_by_slide: dict[str, list[tuple[int, int]]],
    *,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write the derived ROI coordinate manifest + ROI splits CSVs; return their paths.

    One ROI row per sampled tile (``sample_id = <slide>__x<X>_y<Y>``) carrying its parent
    slide's ``image_path``/``mask_path`` and a ``region_x``/``region_y`` origin. The ROI
    splits CSV is expanded **directly from the slide splits CSV** — every ROI inherits its
    parent slide row's split (and ``fold``, if present) verbatim (propagation, not creation;
    soma never partitions). Reading the slide splits CSV (rather than a parsed ``Splits``)
    preserves the exact fold labels and any extra test-split names.
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "roi_manifest.csv"
    splits_path = out_dir / "roi_splits.csv"

    slide_splits = pd.read_csv(slide_splits_csv)
    has_fold = "fold" in slide_splits.columns
    # slide sample_id -> list of its split rows (split[, fold]) to clone onto each ROI.
    rows_by_slide: dict[str, list[dict]] = defaultdict(list)
    for _, row in slide_splits.iterrows():
        entry = {"split": str(row["split"])}
        if has_fold:
            entry["fold"] = int(row["fold"])
        rows_by_slide[str(row["sample_id"])].append(entry)

    manifest_rows: list[dict] = []
    splits_rows: list[dict] = []
    for slide_id, coords in coords_by_slide.items():
        record = dataset.samples[slide_id]
        for x, y in coords:
            roi_id = f"{slide_id}__x{x}_y{y}"
            manifest_rows.append(
                {
                    "sample_id": roi_id,
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                    "region_x": int(x),
                    "region_y": int(y),
                }
            )
            for entry in rows_by_slide.get(slide_id, []):
                splits_rows.append({"sample_id": roi_id, **entry})

    _write_csv(manifest_path, ["sample_id", "image_path", "mask_path", "region_x", "region_y"], manifest_rows)
    splits_fields = ["sample_id", "split", "fold"] if has_fold else ["sample_id", "split"]
    _write_csv(splits_path, splits_fields, splits_rows)
    return manifest_path, splits_path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SlideManifestDenseExtractor:
    """Encode annotation-sampled ROIs into cached dense grids (slide-manifest segmentation).

    Reads the ROI coordinate manifest (built by :func:`build_roi_manifest`), groups ROIs by
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
        self._target_size = normalize_hw(int(preprocessing.requested_tile_size_px), name="target_size")
        self._spacing_um = float(preprocessing.requested_spacing_um)
        self._pad_mode = "reflect"
        # Encoder-window knobs (design §5): window_size=None ⇒ one whole-region forward
        # (delegated to slide2vec); a smaller window slides the encoder over patch-aligned
        # windows of each padded ROI and blends the grids (soma's encode_dense_sliding —
        # the same path the pre-cropped dense extractor uses). Sliding is required for
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

    def run(self, feature_dir: str | Path) -> DenseFeatureStore:
        from slide2vec.inference import load_model
        from slide2vec.runtime.dense_regions import encode_regions_dense

        dense_input_mode = "whole" if self._window_size is None else "sliding_window"

        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)

        loaded = load_model(
            name=self._encoder.name,
            output_variant=self._encoder.output_variant,
            allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            dynamic_img_size=True,
        )
        model = loaded.model
        device = loaded.device
        patch_size = model.patch_size
        geometry = compute_dense_geometry(target_size=self._target_size, patch_size=patch_size)
        signature = sampling_signature(self._masks, self._sampling, self._preprocessing)

        cache_resolution = None
        out_dir = feature_dir
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
                sampling_signature=signature,
                fingerprint_files=self._cache.fingerprint_files,
                validate_payloads=self._cache.validate_payloads,
            )
            if cache_resolution.complete:
                logger.info("Reusing cached ROI dense grids from %s", cache_resolution.features_dir)
                return DenseFeatureStore(cache_resolution.cache_dir)
            out_dir = cache_resolution.features_dir

        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=self._encoder.name,
            output_dir=out_dir,
            num_gpus=1,
            save_tile_embeddings=True,
        )

        # Group ROI records by parent slide so each slide is opened/read once.
        records_by_slide: dict[Path, list["SampleRecord"]] = defaultdict(list)
        for record in self._dataset.samples.values():
            if record.region is None:
                raise ValueError(
                    f"ROI '{record.sample_id}' has no region; the ROI manifest must carry "
                    "region_x/region_y."
                )
            records_by_slide[record.image_path].append(record)

        from hs2p.wsi.wsi import WSI

        feature_dim: int | None = None
        for image_path, records in records_by_slide.items():
            wsi = WSI(Path(image_path), backend=self._preprocessing.backend)
            coords = [record.region for record in records]
            if self._window_size is None:
                # Whole-region forward — delegated to slide2vec (the extraction layer).
                grids = encode_regions_dense(
                    model=model,
                    device=device,
                    wsi=wsi,
                    coordinates=coords,
                    requested_spacing_um=self._spacing_um,
                    target_size=self._target_size,
                    tolerance=float(self._preprocessing.tolerance),
                    pad_mode=self._pad_mode,
                    feature_kind=self._feature_kind,
                    attention_blocks=self._attention_blocks,
                    attention_include_registers=self._attention_include_registers,
                    batch_size=self._encoder.batch_size,
                    precision=execution.precision,
                )
            else:
                # Sliding-window: read each padded ROI region and slide the encoder over
                # patch-aligned windows (reusing soma's tested encode_dense_sliding), so
                # native-only encoders (phikon@224) can serve a 512 supervision tile.
                grids = self._encode_regions_sliding(
                    model=model,
                    device=device,
                    wsi=wsi,
                    coordinates=coords,
                    geometry=geometry,
                    precision=execution.precision,
                )
            for record, grid in zip(records, grids):
                if feature_dim is None:
                    feature_dim = int(grid.shape[0])
                metadata = dense_grid_metadata(
                    geometry,
                    feature_dim=int(grid.shape[0]),
                    pad_mode=self._pad_mode,
                    dense_input_mode=dense_input_mode,
                    window_size=self._window_size,
                    overlap=self._overlap,
                    spacing_um=self._spacing_um,
                    feature_kind=self._feature_kind,
                    attention_blocks=self._attention_blocks,
                    attention_include_registers=self._attention_include_registers,
                )
                write_dense_grid(out_dir, record.sample_id, torch.from_numpy(grid), metadata)

        if cache_resolution is not None and feature_dim is not None:
            cache_resolution = record_feature_dim(
                cache_resolution,
                feature_dim,
                validate_payloads=self._cache.validate_payloads,
            )
            cache_resolution = record_sample_identity_signatures(
                cache_resolution,
                [record.sample_id for record in self._dataset.samples.values()],
                validate_payloads=self._cache.validate_payloads,
            )
        return DenseFeatureStore(out_dir)

    def _encode_regions_sliding(
        self,
        *,
        model,
        device,
        wsi,
        coordinates,
        geometry,
        precision: str,
    ):
        """Encode ROI regions with the encoder slid over patch-aligned windows.

        The sliding counterpart of slide2vec's whole-region ``encode_regions_dense``:
        each ROI is read at the run spacing/target_size, normalization-transformed, padded
        to ``encoded_size``, then fed through soma's :func:`encode_dense_sliding` (the same
        blended stitch the pre-cropped dense extractor uses). Returns ``(N, d, gh, gw)``.
        """
        import numpy as np
        from PIL import Image
        from slide2vec.runtime.dense_regions import pad_image_to_encoded
        from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx

        from soma.dense.sliding import encode_dense_sliding

        target_h, target_w = geometry.target_size
        dense_transform = model.get_dense_transform()
        if self._feature_kind == "cls_attention":
            blocks = self._attention_blocks
            include_reg = self._attention_include_registers

            def encode_fn(window):
                return model.encode_tiles_attention(window, blocks=blocks, include_registers=include_reg)
        else:
            encode_fn = model.encode_tiles_dense

        def _read_padded(location):
            region = wsi.read_region_at_spacing(
                location,
                float(self._spacing_um),
                (target_w, target_h),  # hs2p size is (width, height)
                tolerance=float(self._preprocessing.tolerance),
                interpolation="area",
            )
            region = np.ascontiguousarray(np.asarray(region)[..., :3])
            tensor = torch.as_tensor(dense_transform(Image.fromarray(region))).as_subclass(torch.Tensor)
            if tensor.ndim != 3 or tuple(int(s) for s in tensor.shape[-2:]) != (target_h, target_w):
                raise ValueError(
                    f"region at {location} is {tuple(int(s) for s in tensor.shape)} after the dense "
                    f"transform; expected (C, {target_h}, {target_w}) (normalization-only transform)."
                )
            return pad_image_to_encoded(
                tensor, geometry, pad_mode=self._pad_mode, image_pad_value=None
            )

        coords = [(int(x), int(y)) for x, y in coordinates]
        batch_size = max(1, int(self._encoder.batch_size))
        out: list[np.ndarray] = []
        with torch.inference_mode(), slide_encode_autocast_ctx(device, precision):
            for start in range(0, len(coords), batch_size):
                chunk = coords[start : start + batch_size]
                batch = torch.stack([_read_padded(loc) for loc in chunk]).to(device, non_blocking=True)
                grids = encode_dense_sliding(
                    model,
                    batch,
                    geometry=geometry,
                    window_size=self._window_size,
                    overlap=self._overlap,
                    encode_fn=encode_fn,
                )
                out.append(grids.detach().float().cpu().numpy())
        import numpy as _np

        return _np.concatenate(out, axis=0)
