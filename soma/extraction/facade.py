"""Canonical type-directed orchestration for persistent frozen features."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
from soma.dataset import (
    Dataset,
    DetectionManifest,
    SegmentationManifest,
    SpatialExpressionManifest,
    TileDataset,
)
from soma.extraction_contracts import (
    ExtractionArtifacts,
    FeatureDataset,
    FeatureExtractionResult,
    FeatureProvenance,
)
from soma.encoders.validation import resolve_preprocessing_config


class FeatureExtractor:
    """Fully configured persistent extraction operation.

    Dataset type and preprocessing configuration select the representation. Runtime
    arguments are intentionally absent from :meth:`extract`: construction fixes both
    behavior and artifact layout.
    """

    def __init__(
        self,
        dataset: FeatureDataset,
        encoder: EncoderConfig,
        preprocessing: PreprocessingConfig = PreprocessingConfig(),
        *,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig = CacheConfig(),
        output_root: str | Path,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._preprocessing = preprocessing
        self._execution = execution
        self._cache = cache
        self._output_root = Path(output_root).resolve()
        self._extract_impl = self._resolve_extractor()

    def _resolve_extractor(self) -> Callable[[], FeatureExtractionResult]:
        if type(self._dataset) is TileDataset or isinstance(
            self._dataset, SpatialExpressionManifest
        ):
            return self._extract_given_images
        if type(self._dataset) is Dataset:
            return self._extract_pooled
        if isinstance(self._dataset, SegmentationManifest):
            region_flags = [
                record.region is not None and record.slide_id is not None
                for record in self._dataset.samples.values()
            ]
            if any(region_flags):
                if not all(region_flags) or self._preprocessing.masks is None:
                    raise TypeError(
                        "Unsupported dataset/config combination: explicit slide regions "
                        "require region_x/region_y, slide_id, and preprocessing.masks on "
                        "every sample."
                    )
                return self._extract_dense_regions
            return (
                self._extract_annotation_rois
                if self._preprocessing.masks is not None
                else self._extract_dense_images
            )
        if isinstance(self._dataset, DetectionManifest):
            if self._preprocessing.masks is not None:
                raise TypeError(
                    "Unsupported dataset/config combination: annotation sampling is only "
                    "defined for SegmentationManifest inputs."
                )
            return self._extract_dense_images
        raise TypeError(
            "Unsupported dataset/config combination for persistent extraction: "
            f"dataset={type(self._dataset).__name__}."
        )

    def extract(self) -> FeatureExtractionResult:
        self._output_root.mkdir(parents=True, exist_ok=True)
        (self._output_root / "extraction_provenance.json").unlink(missing_ok=True)
        return self._extract_impl()

    def _dense_preprocessing(self) -> PreprocessingConfig:
        """Resolve only missing dense geometry, preserving explicit cache identities."""
        if (
            self._preprocessing.requested_tile_size_px is not None
            and self._preprocessing.requested_spacing_um is not None
        ):
            return self._preprocessing
        from slide2vec.encoders.registry import encoder_registry

        return resolve_preprocessing_config(
            self._encoder,
            self._preprocessing,
            model_metadata=encoder_registry.info(self._encoder.name),
        )

    def _extract_given_images(self) -> FeatureExtractionResult:
        from soma.tile_extraction import _TileFeatureExtractor
        from soma.extraction.orchestration import _release_parent_cuda_state

        extractor = _TileFeatureExtractor(
            self._dataset,
            self._encoder,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(self._output_root / "features")
            store.validate_coverage(list(self._dataset.sample_ids))
        finally:
            _release_parent_cuda_state()
        return self._completed_result(
            source=store,
            dataset=self._dataset,
            provenance=FeatureProvenance(
                kind="pooled_image",
                encoder_name=self._encoder.name,
            ),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _extract_pooled(self) -> FeatureExtractionResult:
        from slide2vec.encoders.registry import (
            encoder_registry,
            resolve_encoder_level,
        )

        from soma.extraction.extractor import _PooledFeatureExtractor
        from soma.extraction.orchestration import _release_parent_cuda_state

        extractor = _PooledFeatureExtractor(
            self._dataset,
            self._encoder,
            self._preprocessing,
            output_root=self._output_root,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(feature_dir="features")
            empty_sample_ids = tuple(store.empty_feature_samples)
            level = resolve_encoder_level(
                self._encoder.name,
                encoder_registry.info(self._encoder.name),
            )
            expected = (
                list(self._dataset.patient_groups)
                if level == "patient"
                else [
                    sample_id
                    for sample_id in self._dataset.sample_ids
                    if sample_id not in set(empty_sample_ids)
                ]
            )
            store.validate_coverage(expected)
        finally:
            _release_parent_cuda_state()
        kind = (
            "hierarchical"
            if store.is_hierarchical
            else "pooled_slide" if store.is_slide_level else "pooled_bag"
        )
        return self._completed_result(
            source=store,
            dataset=self._dataset,
            provenance=FeatureProvenance(
                kind=kind,
                encoder_name=self._encoder.name,
                zero_sample_ids=empty_sample_ids,
            ),
            artifacts=ExtractionArtifacts(
                feature_dir=store.feature_dir,
                tiling_dir=self._output_root / "tiling",
            ),
        )

    def _extract_dense_images(self) -> FeatureExtractionResult:
        from soma.dense import (
            CacheBackedDenseSource,
            DenseSourceProvenance,
        )
        from soma.dense_extraction import _DenseImageExtractor
        from soma.extraction.orchestration import _release_parent_cuda_state

        preprocessing = self._dense_preprocessing()
        if preprocessing.requested_tile_size_px is None:
            raise ValueError(
                "Dense extraction requires preprocessing.requested_tile_size_px."
            )
        if preprocessing.requested_spacing_um is None:
            raise ValueError(
                "Dense extraction requires preprocessing.requested_spacing_um."
            )
        extractor = _DenseImageExtractor(
            self._dataset,
            self._encoder,
            target_size=int(preprocessing.requested_tile_size_px),
            spacing_um=float(preprocessing.requested_spacing_um),
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
            window_size=preprocessing.dense_window_size,
            overlap=float(preprocessing.dense_window_overlap),
            execution=self._execution,
            cache=self._cache,
            preprocessing=preprocessing,
        )
        try:
            store = extractor.run(self._output_root / "features")
            store.validate_coverage(list(self._dataset.sample_ids))
        finally:
            _release_parent_cuda_state()
        source = CacheBackedDenseSource(
            store,
            provenance=DenseSourceProvenance(
                kind="dense_cache",
                feature_dir=store.feature_dir,
                dataset_csv=getattr(self._dataset, "_path", None),
            ),
        )
        return self._completed_result(
            source=source,
            dataset=self._dataset,
            provenance=FeatureProvenance(
                kind="dense_image",
                encoder_name=self._encoder.name,
            ),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _extract_annotation_rois(self) -> FeatureExtractionResult:
        from soma.cache import (
            resolve_cache_root,
            resolve_roi_sampling_cache,
            write_roi_sampling_coords,
        )
        from soma.config import SamplingConfig
        from soma.dataset import SegmentationManifest
        from soma.dense import CacheBackedDenseSource, DenseSourceProvenance
        from soma.dense_slide_extraction import (
            _SlideRegionExtractor,
            build_roi_dataset,
            sample_slide_rois,
        )
        from soma.extraction.orchestration import _release_parent_cuda_state

        preprocessing = self._dense_preprocessing()
        masks = preprocessing.masks
        assert masks is not None
        sampling = preprocessing.sampling or SamplingConfig()
        features_root = self._output_root / "features"

        if self._cache.enabled:
            sampling_cache = resolve_roi_sampling_cache(
                cache_root=resolve_cache_root(
                    self._cache,
                    feature_dir=features_root,
                    output_root=self._output_root,
                ),
                dataset=self._dataset,
                preprocessing=preprocessing,
            )
            fresh: dict[str, list[tuple[int, int]]] = {}
            if sampling_cache.miss_sample_ids:
                fresh = sample_slide_rois(
                    self._dataset,
                    masks=masks,
                    sampling=sampling,
                    preprocessing=preprocessing,
                    sample_ids=sampling_cache.miss_sample_ids,
                )
                write_roi_sampling_coords(
                    cache_resolution=sampling_cache,
                    coords_by_sample_id=fresh,
                )
            merged = {**sampling_cache.coords_by_id, **fresh}
        else:
            merged = sample_slide_rois(
                self._dataset,
                masks=masks,
                sampling=sampling,
                preprocessing=preprocessing,
            )
        coords_by_slide = {
            sample_id: merged[sample_id] for sample_id in self._dataset.sample_ids
        }
        zero_sample_ids = tuple(
            sample_id
            for sample_id in self._dataset.sample_ids
            if not coords_by_slide[sample_id]
        )
        effective_csv = build_roi_dataset(
            self._dataset,
            coords_by_slide,
            out_dir=self._output_root / "segmentation_rois",
        )
        effective_dataset = SegmentationManifest(effective_csv)
        extractor = _SlideRegionExtractor(
            effective_dataset,
            self._encoder,
            masks=masks,
            sampling=sampling,
            preprocessing=preprocessing,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(features_root)
            store.validate_coverage(list(effective_dataset.sample_ids))
        finally:
            _release_parent_cuda_state()
        source = CacheBackedDenseSource(
            store,
            provenance=DenseSourceProvenance(
                kind="slide_manifest_dense_cache",
                feature_dir=store.feature_dir,
                dataset_csv=effective_csv,
                parent_dataset_csv=getattr(self._dataset, "_path", None),
            ),
        )
        return self._completed_result(
            source=source,
            dataset=effective_dataset,
            provenance=FeatureProvenance(
                kind="slide_manifest_dense",
                encoder_name=self._encoder.name,
                zero_sample_ids=zero_sample_ids,
            ),
            artifacts=ExtractionArtifacts(
                feature_dir=store.feature_dir,
                dataset_csv=effective_csv,
            ),
        )

    def _extract_dense_regions(self) -> FeatureExtractionResult:
        from soma.config import SamplingConfig
        from soma.dense import CacheBackedDenseSource, DenseSourceProvenance
        from soma.dense_slide_extraction import _SlideRegionExtractor
        from soma.extraction.orchestration import _release_parent_cuda_state

        preprocessing = self._dense_preprocessing()
        masks = preprocessing.masks
        assert masks is not None
        extractor = _SlideRegionExtractor(
            self._dataset,
            self._encoder,
            masks=masks,
            sampling=preprocessing.sampling or SamplingConfig(),
            preprocessing=preprocessing,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(self._output_root / "features")
            store.validate_coverage(list(self._dataset.sample_ids))
        finally:
            _release_parent_cuda_state()
        source = CacheBackedDenseSource(
            store,
            provenance=DenseSourceProvenance(
                kind="slide_region_dense_cache",
                feature_dir=store.feature_dir,
                dataset_csv=getattr(self._dataset, "_path", None),
            ),
        )
        return self._completed_result(
            source=source,
            dataset=self._dataset,
            provenance=FeatureProvenance(
                kind="slide_region_dense",
                encoder_name=self._encoder.name,
            ),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _completed_result(
        self,
        *,
        source: object,
        dataset: FeatureDataset,
        provenance: FeatureProvenance,
        artifacts: ExtractionArtifacts,
    ) -> FeatureExtractionResult:
        """Publish provenance only after source coverage has been validated."""
        provenance_json = self._output_root / "extraction_provenance.json"
        completed_artifacts = ExtractionArtifacts(
            feature_dir=artifacts.feature_dir,
            tiling_dir=artifacts.tiling_dir,
            dataset_csv=artifacts.dataset_csv,
            provenance_json=provenance_json,
        )
        payload = {
            "status": "completed",
            "provenance": asdict(provenance),
            "artifacts": {
                key: None if value is None else str(value)
                for key, value in asdict(completed_artifacts).items()
            },
        }
        temporary = provenance_json.with_name(f".{provenance_json.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(provenance_json)
        return FeatureExtractionResult(
            source=source,
            dataset=dataset,
            provenance=provenance,
            artifacts=completed_artifacts,
        )
