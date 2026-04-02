"""FeatureExtractor — preprocesses slides and extracts features."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import hs2p.preprocessing as _shared_preprocessing

from soma.cache import (
    record_feature_dim,
    resolve_cache_root,
    resolve_slide_cache,
    resolve_tile_cache,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.distributed import SlideTask, extract_dataset
from soma.encoders.base import SlideEncoder
import soma.encoders.models  # noqa: F401
from soma.encoders.registry import encoder_registry
from soma.encoders.registry import (
    resolve_encoder_level,
    resolve_encoder_output,
    resolve_tile_dependency_output,
)
from soma.encoders.validation import resolve_preprocessing_config, validate_encoder_config
from soma.features import FeatureStore
from soma.preprocessing.io import (
    load_tiling_result,
    save_tiling_result,
    validate_tiling_result_provenance,
)
from soma.preprocessing.tiling import (
    TilingResult,
    generate_tiles,
    expand_regions_to_subtiles,
    resolve_base_spacing_um,
)
from soma.preprocessing.tissue import detect_contours, segment_tissue
from soma.wsi.reader import open_slide, select_level_for_downsample


def _load_precomputed_tissue_mask(
    *,
    mask_path: str | Path,
    slide,
    seg_level: int,
    tissue_value: int,
) -> tuple[object, int, float]:
    original_open_slide = _shared_preprocessing.open_slide
    _shared_preprocessing.open_slide = (
        lambda path, backend="auto", **kwargs: open_slide(path, backend, **kwargs)
    )
    try:
        return _shared_preprocessing.load_precomputed_tissue_mask(
            mask_path=mask_path,
            slide=slide,
            seg_level=seg_level,
            tissue_value=tissue_value,
        )
    finally:
        _shared_preprocessing.open_slide = original_open_slide


class FeatureExtractor:
    """Preprocesses slides and extracts features for all samples in a dataset.

    Args:
        dataset: Dataset with sample records (image_path, sample_id).
        encoder: Encoder configuration (model name, precision, batch_size, etc.).
        preprocessing: Preprocessing configuration (tiling, tissue segmentation).
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        preprocessing: PreprocessingConfig = PreprocessingConfig(),
        cache: CacheConfig = CacheConfig(),
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._preprocessing = preprocessing
        self._cache = cache

    def _resolved_preprocessing(self) -> PreprocessingConfig:
        encoder_info = encoder_registry.info(self._encoder.name)
        return resolve_preprocessing_config(
            self._encoder,
            self._preprocessing,
            model_metadata=encoder_info,
        )

    def _resolved_output(self) -> dict[str, object]:
        encoder_info = encoder_registry.info(self._encoder.name)
        return resolve_encoder_output(
            self._encoder.name,
            requested_output_variant=self._encoder.output_variant,
            metadata=encoder_info,
        )

    def preprocess(
        self,
        output_dir: str | Path,
        *,
        skip_existing: bool = True,
        backend: str = "auto",
    ) -> None:
        """Preprocess all slides: tissue segmentation -> contours -> tiles.

        Saves TilingResult artifacts per sample to output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._resolved_preprocessing()

        for record in self._dataset.samples.values():
            npz_path = output_dir / f"{record.sample_id}.coordinates.npz"
            meta_path = output_dir / f"{record.sample_id}.coordinates.meta.json"
            expected_mask_value = (
                int(cfg.tissue_mask_tissue_value)
                if record.tissue_mask_path is not None
                else None
            )
            if skip_existing and npz_path.exists() and meta_path.exists():
                existing = load_tiling_result(npz_path, meta_path)
                validate_tiling_result_provenance(
                    existing,
                    sample_id=record.sample_id,
                    image_path=record.image_path,
                    tissue_mask_path=record.tissue_mask_path,
                    tissue_mask_tissue_value=expected_mask_value,
                )
                continue

            slide = open_slide(record.image_path, backend)
            try:
                seg_level = select_level_for_downsample(
                    cfg.seg_downsample, slide.level_downsamples
                )
                if record.tissue_mask_path is not None:
                    mask, mask_level, mask_spacing_um = _load_precomputed_tissue_mask(
                        mask_path=record.tissue_mask_path,
                        slide=slide,
                        seg_level=seg_level,
                        tissue_value=int(cfg.tissue_mask_tissue_value),
                    )
                    tissue_method = "precomputed_mask"
                else:
                    seg_size = slide.level_dimensions[seg_level]
                    seg_image = slide.read_region((0, 0), seg_level, seg_size)
                    mask = segment_tissue(seg_image, method=cfg.tissue_method)
                    mask_level = None
                    mask_spacing_um = None
                    tissue_method = cfg.tissue_method
                seg_spacing_um = slide.spacing * slide.level_downsamples[seg_level]

                contours = detect_contours(
                    mask,
                    slide_dimensions=slide.dimensions,
                    ref_tile_size_px=cfg.ref_tile_size_px,
                    requested_spacing_um=cfg.requested_spacing_um,
                    a_t=cfg.a_t,
                    base_spacing_um=slide.spacing,
                    level_downsamples=slide.level_downsamples,
                    tolerance=cfg.tolerance,
                )

                tiling = generate_tiles(
                    slide_dimensions=slide.dimensions,
                    contours=contours,
                    requested_tile_size_px=cfg.requested_tile_size_px,
                    requested_spacing_um=cfg.requested_spacing_um,
                    base_spacing_um=slide.spacing,
                    level_downsamples=slide.level_downsamples,
                    overlap=cfg.overlap,
                    use_padding=cfg.use_padding,
                    min_tissue_fraction=cfg.min_tissue_fraction,
                    tolerance=cfg.tolerance,
                )
                step_px_lv0 = max(1, round(tiling.tile_size_lv0 * (1.0 - cfg.overlap)))
                tiling = TilingResult(
                    coordinates=tiling.coordinates,
                    tissue_fractions=tiling.tissue_fractions,
                    requested_tile_size_px=tiling.requested_tile_size_px,
                    requested_spacing_um=tiling.requested_spacing_um,
                    read_level=tiling.read_level,
                    effective_tile_size_px=tiling.effective_tile_size_px,
                    effective_spacing_um=tiling.effective_spacing_um,
                    tile_size_lv0=tiling.tile_size_lv0,
                    is_within_tolerance=tiling.is_within_tolerance,
                    use_padding=tiling.use_padding,
                    tile_index=tiling.tile_index,
                    tissue_mask=tiling.tissue_mask,
                    sample_id=record.sample_id,
                    image_path=str(record.image_path),
                    backend=slide.backend_name,
                    requested_backend=backend,
                    base_spacing_um=slide.spacing,
                    slide_dimensions=list(slide.dimensions),
                    level_downsamples=list(slide.level_downsamples),
                    overlap=cfg.overlap,
                    min_tissue_fraction=cfg.min_tissue_fraction,
                    step_px_lv0=step_px_lv0,
                    tissue_method=tissue_method,
                    seg_downsample=cfg.seg_downsample,
                    seg_level=seg_level,
                    seg_spacing_um=seg_spacing_um,
                    ref_tile_size_px=cfg.ref_tile_size_px,
                    a_t=cfg.a_t,
                    tissue_mask_path=(
                        None
                        if record.tissue_mask_path is None
                        else str(record.tissue_mask_path)
                    ),
                    tissue_mask_tissue_value=expected_mask_value,
                    mask_level=mask_level,
                    mask_spacing_um=mask_spacing_um,
                )
            finally:
                slide.close()
            save_tiling_result(tiling, output_dir, record.sample_id)

    def extract(
        self,
        output_dir: str | Path,
        *,
        tiling_dir: str | Path | None = None,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
        """Extract features for all preprocessed slides.

        Args:
            output_dir: Directory to save feature .pt files.
            tiling_dir: Directory with tiling artifacts. If None, preprocesses
                in-memory (saves tiling to output_dir/.tiling/).
            skip_existing: Skip samples with existing .pt files.
            num_gpus: Number of GPUs for distributed extraction.
            backend: WSI backend for slide reading.

        Returns:
            FeatureStore pointing to output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if tiling_dir is None:
            tiling_dir = output_dir / ".tiling"
            self.preprocess(tiling_dir, skip_existing=skip_existing, backend=backend)

        tiling_dir = Path(tiling_dir)
        encoder_info = encoder_registry.info(self._encoder.name)
        resolved_output = resolve_encoder_output(
            self._encoder.name,
            requested_output_variant=self._encoder.output_variant,
            metadata=encoder_info,
        )
        tile_dependency_output = resolve_tile_dependency_output(
            self._encoder.name,
            metadata=encoder_info,
        )
        resolved_preprocessing = resolve_preprocessing_config(
            self._encoder,
            self._preprocessing,
            model_metadata=encoder_info,
        )
        for warning in validate_encoder_config(
            self._encoder,
            encoder_info,
            preprocessing_config=resolved_preprocessing,
        ):
            warnings.warn(warning, stacklevel=2)

        slide_tasks = []
        tilings: dict[str, object] = {}
        for record in self._dataset.samples.values():
            npz_path = tiling_dir / f"{record.sample_id}.coordinates.npz"
            meta_path = tiling_dir / f"{record.sample_id}.coordinates.meta.json"
            tiling = load_tiling_result(npz_path, meta_path)
            validate_tiling_result_provenance(
                tiling,
                sample_id=record.sample_id,
                image_path=record.image_path,
                tissue_mask_path=record.tissue_mask_path,
                tissue_mask_tissue_value=(
                    int(resolved_preprocessing.tissue_mask_tissue_value)
                    if record.tissue_mask_path is not None
                    else None
                ),
            )
            # Hierarchical mode: expand region coordinates into subtile coordinates
            if self._preprocessing.hierarchical and self._preprocessing.npatch is not None:
                tiling = expand_regions_to_subtiles(tiling, self._preprocessing.npatch)
            tilings[record.sample_id] = tiling
            for warning in validate_encoder_config(
                self._encoder,
                encoder_info,
                preprocessing_config=resolved_preprocessing,
                tiling_result=tiling,
            ):
                warnings.warn(warning, stacklevel=2)
            slide_tasks.append(
                SlideTask(
                    slide_path=str(record.image_path),
                    tiling_result=tiling,
                    slide_id=record.sample_id,
                )
            )

        if not self._cache.enabled:
            extract_dataset(
                encoder_name=self._encoder.name,
                output_variant=str(resolved_output["output_variant"]),
                tile_output_variant=(
                    str(tile_dependency_output["output_variant"])
                    if resolve_encoder_level(self._encoder.name, encoder_info) == "slide"
                    else None
                ),
                slides=slide_tasks,
                output_dir=output_dir,
                batch_size=self._encoder.batch_size,
                adaptive_batching=self._encoder.adaptive_batching,
                num_workers=self._encoder.num_workers,
                precision=self._encoder.precision,
                skip_existing=skip_existing,
                num_gpus=num_gpus,
                backend=backend,
                save_tile_features=self._encoder.save_tile_features,
            )
            return FeatureStore(output_dir)

        cache_root = resolve_cache_root(self._cache, output_dir=output_dir)
        level = resolve_encoder_level(self._encoder.name, encoder_info)
        if level == "tile":
            cache_resolution = resolve_tile_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                preprocessing=resolved_preprocessing,
                execution=self._encoder,
                output_variant=str(resolved_output["output_variant"]),
            )
            self._populate_tile_cache(
                cache_resolution,
                slide_tasks,
                skip_existing=skip_existing,
                num_gpus=num_gpus,
                backend=backend,
            )
            return FeatureStore(cache_resolution.cache_dir)

        tile_encoder_name = str(encoder_info["tile_encoder"])
        tile_cache = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=tile_encoder_name,
            preprocessing=resolved_preprocessing,
            execution=self._encoder,
            output_variant=str(tile_dependency_output["output_variant"]),
        )
        self._populate_tile_cache(
            tile_cache,
            slide_tasks,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
            backend=backend,
        )

        slide_cache = resolve_slide_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            slide_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_cache_key=tile_cache.key,
            execution=self._encoder,
            output_variant=str(resolved_output["output_variant"]),
        )
        self._populate_slide_cache(
            slide_cache,
            tile_cache,
            slide_tasks,
            tilings=tilings,
            backend=backend,
        )
        return FeatureStore(slide_cache.cache_dir)

    def run(
        self,
        output_dir: str | Path,
        *,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
        """Preprocess + extract in one call.

        Tiling artifacts saved to output_dir/.tiling/.
        Feature .pt files saved to output_dir/.
        """
        output_dir = Path(output_dir)
        tiling_dir = output_dir / ".tiling"
        self.preprocess(tiling_dir, skip_existing=skip_existing, backend=backend)
        return self.extract(
            output_dir,
            tiling_dir=tiling_dir,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
            backend=backend,
        )

    def _populate_tile_cache(
        self,
        cache_resolution,
        slide_tasks: list[SlideTask],
        *,
        skip_existing: bool,
        num_gpus: int | None,
        backend: str,
    ) -> None:
        missing = set(cache_resolution.missing_sample_ids())
        if not missing:
            return
        tasks_to_run = [task for task in slide_tasks if task.slide_id in missing]
        extract_dataset(
            encoder_name=cache_resolution.metadata["encoder_name"],
            output_variant=cache_resolution.metadata["execution"].get("output_variant"),
            slides=tasks_to_run,
            output_dir=cache_resolution.features_dir,
            batch_size=self._encoder.batch_size,
            adaptive_batching=self._encoder.adaptive_batching,
            num_workers=self._encoder.num_workers,
            precision=self._encoder.precision,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
            backend=backend,
            save_tile_features=False,
        )
        sample_path = cache_resolution.features_dir / f"{tasks_to_run[0].slide_id}.pt"
        tensor = torch.load(sample_path, weights_only=True, map_location="cpu")
        record_feature_dim(
            cache_resolution,
            tensor.shape[1] if tensor.ndim == 2 else tensor.shape[0],
        )

    def _populate_slide_cache(
        self,
        slide_cache,
        tile_cache,
        slide_tasks: list[SlideTask],
        *,
        tilings: dict[str, object],
        backend: str,
    ) -> None:
        missing = set(slide_cache.missing_sample_ids())
        if not missing:
            return
        encoder_cls = encoder_registry.get(self._encoder.name)
        slide_encoder = encoder_cls(
            output_variant=slide_cache.metadata["execution"].get("output_variant")
        ).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        assert isinstance(slide_encoder, SlideEncoder)

        feature_dim: int | None = None
        for task in slide_tasks:
            if task.slide_id not in missing:
                continue
            tile_features = torch.load(
                tile_cache.features_dir / f"{task.slide_id}.pt",
                weights_only=True,
                map_location="cpu",
            )
            tiling = tilings[task.slide_id]
            coordinates = torch.as_tensor(
                tiling.coordinates,
                dtype=torch.long,
            )
            prepared = slide_encoder.prepare_coordinates(
                coordinates,
                base_spacing_um=resolve_base_spacing_um(tiling),  # type: ignore[arg-type]
                target_spacing_um=float(tiling.requested_spacing_um),  # type: ignore[attr-defined]
            )
            slide_features = slide_encoder.encode_slide(
                tile_features.to(slide_encoder.device),
                prepared.to(slide_encoder.device),
                tile_size_lv0=int(tiling.tile_size_lv0),  # type: ignore[attr-defined]
            ).detach().float().cpu()
            if slide_features.ndim > 1:
                slide_features = slide_features.squeeze(0)
            torch.save(slide_features, slide_cache.features_dir / f"{task.slide_id}.pt")
            feature_dim = slide_features.shape[0]

        if feature_dim is not None:
            record_feature_dim(slide_cache, feature_dim)
