"""``soma.extract``: the single way features enter soma (ADR 0021).

The verb reads an :class:`~soma.data.ImageManifest` (and, to sample ROIs from whole
slides, an :class:`~soma.data.AnnotationManifest`), never records, and returns the
cache-backed source the extraction produced: a :class:`~soma.data.SetSource` (one tensor
per sample) or, with ``dense``, a :class:`~soma.data.GridSource`. The extraction setting
picks the engine: ``unit="tile"`` reads each row as one pre-cropped image; ``unit="slide"``
tiles whole slides (set) or, under an annotation vocabulary in ``extraction.masks``,
samples ROIs from them (grid). ROI sampling stays inside extraction: the grid source it
returns is keyed by ROI id and its ``rois`` (:class:`~soma.extract.rois.RoiSet`) name
each ROI's parent sample and serve its label maps and pixels.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal

from slide2vec import DenseOptions, ExecutionOptions, PreprocessingConfig

from soma._internal.logging import ensure_default_logging
from soma.extract.cache.options import CacheOptions
from soma.extract.preprocessing import is_annotation_sampling
from soma.models.encoders import EncoderLeaf
from soma.data import AnnotationManifest, ImageManifest, require_coverage
from soma.models.encoders.validation import resolve_preprocessing_config
from soma.extract.rois import RoiSet
from soma.extract.contracts import ExtractionArtifacts, FeatureProvenance
from soma.extract.preprocessing.resolution import reconcile_dense_tiling

__all__ = ["extract"]

Unit = Literal["slide", "tile"]


def extract(
    images: ImageManifest,
    *,
    encoder: EncoderLeaf | str,
    cache: CacheOptions,
    annotations: AnnotationManifest | None = None,
    unit: Unit = "slide",
    rois: RoiSet | None = None,
    execution: ExecutionOptions | None = None,
    extraction: PreprocessingConfig | None = None,
    dense: DenseOptions | None = None,
    spacing_policy: str = "strict",
    output_dir: str | Path,
):
    """Extract (or reuse from the feature cache) ``encoder``'s features of ``images``.

    Args:
        images: The samples' pixels: one row per whole slide (``unit="slide"``) or per
            pre-cropped image (``unit="tile"``). The cache keys each sample on its row.
        encoder: A slide2vec encoder name or ``{name, params}`` leaf.
        cache: The feature and tiling cache policy (``CacheOptions(enabled=False)`` writes
            under ``output_dir`` only).
        annotations: The annotation rasters ROI sampling reads (``label_mask_path``);
            needed only to sample ROIs from whole slides.
        unit: What one row is: a whole slide or a pre-cropped image.
        rois: The ROIs of an earlier annotation-sampled extraction (its source's
            ``rois``): their grids are encoded again, without sampling. Their label crops
            were stored at the read spacing and window they were sampled at, so pass the
            same ``extraction``, ``dense`` and ``spacing_policy``; another geometry is
            refused.
        execution: slide2vec's ``ExecutionOptions`` (batch size, precision, GPUs).
        extraction: slide2vec's ``PreprocessingConfig`` (tiling, segmentation, masks);
            unset fields come from the encoder's recommendation.
        dense: slide2vec's ``DenseOptions``: given, the features are token grids read
            and encoded at it; else one tensor per sample.
        spacing_policy: How a grid run reads a sample coarser than ``dense.spacing_um``
            (``native_if_coarser`` reads it at its own spacing).
        output_dir: Where the run's tiling, extraction record and uncached features go.

    Returns:
        A :class:`~soma.data.SetSource` (``CachedSetSource``) or, with ``dense`` or
        ``rois``, a :class:`~soma.data.GridSource` (``CachedGridSource``). An
        annotation-sampled grid source is keyed by ROI id and carries the
        :class:`~soma.extract.rois.RoiSet` as ``rois``.
    """
    return _Extraction(
        images,
        encoder=encoder,
        cache=cache,
        annotations=annotations,
        unit=unit,
        rois=rois,
        execution=execution,
        extraction=extraction,
        dense=dense,
        spacing_policy=spacing_policy,
        output_dir=output_dir,
    ).run()


class _Extraction:
    """One fully configured extraction: construction validates and picks the engine."""

    def __init__(
        self,
        images: ImageManifest,
        *,
        encoder: EncoderLeaf | str,
        cache: CacheOptions,
        annotations: AnnotationManifest | None,
        unit: str,
        rois: RoiSet | None,
        execution: ExecutionOptions | None,
        extraction: PreprocessingConfig | None,
        dense: DenseOptions | None,
        spacing_policy: str,
        output_dir: str | Path,
    ) -> None:
        ensure_default_logging()
        if not isinstance(images, ImageManifest):
            raise TypeError(f"soma.extract reads an ImageManifest, got {type(images).__name__}.")
        if annotations is not None and not isinstance(annotations, AnnotationManifest):
            raise TypeError(
                f"annotations must be an AnnotationManifest, got {type(annotations).__name__}."
            )
        if unit not in ("slide", "tile"):
            raise ValueError(f"unit must be 'slide' or 'tile', got {unit!r}")
        grid = dense is not None or rois is not None
        if (grid or unit != "slide") and images.supplies_coordinates:
            # Only the pooled path embeds supplied tiles: pre-cropped images are read whole
            # and a dense grid over whole slides samples its own ROIs from the masks.
            raise ValueError(
                "Manifest column 'coordinates_path' (user-supplied tile coordinates) only "
                "applies to set-shaped extraction over whole slides (no dense options, "
                f"unit='slide'); got {'a grid' if grid else 'a set'} over unit={unit!r}. "
                "Drop the column."
            )
        self._images = images
        self._annotations = annotations
        self._unit = unit
        self._rois = rois
        self._grid = grid
        self._encoder = EncoderLeaf(name=encoder) if isinstance(encoder, str) else encoder
        self._preprocessing = extraction if extraction is not None else PreprocessingConfig()
        self._dense = dense
        self._spacing_policy = spacing_policy
        self._execution = execution if execution is not None else ExecutionOptions()
        self._cache = cache
        self._output_root = Path(output_dir).resolve()
        self._engine = self._resolve_engine()

    def _resolve_engine(self) -> Callable[[], Any]:
        annotation_sampling = is_annotation_sampling(self._preprocessing.masks)
        if not self._grid:
            if annotation_sampling and self._unit == "tile":
                raise TypeError(
                    "Unsupported extraction: annotation sampling needs whole slides (unit='slide')."
                )
            return self._extract_given_images if self._unit == "tile" else self._extract_pooled
        if self._rois is not None:
            if self._unit != "slide":
                raise TypeError("ROIs are windows of whole slides: extract them with unit='slide'.")
            mismatched = sorted(
                {
                    entry.slide_id
                    for entry in self._rois
                    if entry.slide_id not in self._images or self._images[entry.slide_id] != entry.slide
                }
            )
            if mismatched:
                raise ValueError(
                    f"The ROIs' slide(s) {mismatched[:20]} have no row in the image manifest or a "
                    "row that differs from the one the ROIs were sampled from."
                )
            return self._extract_rois
        if self._unit == "tile":
            if annotation_sampling:
                raise TypeError(
                    "Unsupported extraction: annotation sampling needs whole slides (unit='slide')."
                )
            return self._extract_dense_images
        if not annotation_sampling:
            raise TypeError(
                "Unsupported extraction: a grid over whole slides needs an annotation "
                "vocabulary in extraction.masks (annotation sampling); pass unit='tile' for "
                "pre-cropped images."
            )
        annotations = self._annotations
        if annotations is None or any(
            sample_id not in annotations or annotations[sample_id].label_mask_path is None
            for sample_id in self._images.sample_ids
        ):
            raise TypeError(
                "Unsupported extraction: annotation sampling reads each slide's annotation "
                "raster; pass annotations with a label_mask_path for every slide."
            )
        return self._extract_annotation_rois

    def run(self):
        self._output_root.mkdir(parents=True, exist_ok=True)
        (self._output_root / "extraction_provenance.json").unlink(missing_ok=True)
        return self._engine()

    def _dense_preprocessing(self) -> PreprocessingConfig:
        """The extraction with its tiling reconciled with ``dense`` (ROIs are sampled at the
        geometry the grid reads them at), and the tiling and filter the encoder decides
        filled in."""
        from slide2vec.encoders.registry import encoder_registry

        return resolve_preprocessing_config(
            reconcile_dense_tiling(self._preprocessing, self._dense),
            encoder_name=self._encoder.name,
            model_metadata=encoder_registry.info(self._encoder.name),
        )

    def _dense_options(self, preprocessing: PreprocessingConfig) -> DenseOptions:
        """``dense``, or slide2vec's dense defaults at the resolved tiling."""
        if self._dense is not None:
            return self._dense
        tiling = preprocessing.tiling
        return DenseOptions(
            spacing_um=float(tiling.requested_spacing_um),
            target_size=int(tiling.requested_tile_size_px),
            tolerance=float(tiling.tolerance),
            backend=tiling.backend,
        )

    def _extract_given_images(self):
        from soma.extract.orchestration import _release_parent_cuda_state
        from soma.extract.tile_extraction import _TileFeatureExtractor

        extractor = _TileFeatureExtractor(
            self._images,
            self._encoder,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(self._output_root / "features")
            require_coverage(store, list(self._images.sample_ids))
        finally:
            _release_parent_cuda_state()
        return self._completed(
            store,
            provenance=FeatureProvenance(kind="pooled_image", encoder_name=self._encoder.name),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _extract_pooled(self):
        from slide2vec.encoders.registry import encoder_registry, resolve_encoder_level

        from soma.extract.extractor import _PooledFeatureExtractor
        from soma.extract.orchestration import _release_parent_cuda_state

        extractor = _PooledFeatureExtractor(
            self._images,
            self._encoder,
            self._preprocessing,
            output_root=self._output_root,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(feature_dir="features")
            empty_sample_ids = tuple(store.empty_feature_samples)
            level = resolve_encoder_level(self._encoder.name, encoder_registry.info(self._encoder.name))
            empty = set(empty_sample_ids)
            expected = (
                list(dict.fromkeys(str(entry.patient_id) for entry in self._images))
                if level == "patient"
                else [sample_id for sample_id in self._images.sample_ids if sample_id not in empty]
            )
            require_coverage(store, expected)
        finally:
            _release_parent_cuda_state()
        kind = "hierarchical" if store.rank == 3 else "pooled_slide" if store.rank == 1 else "pooled_bag"
        return self._completed(
            store,
            provenance=FeatureProvenance(
                kind=kind, encoder_name=self._encoder.name, zero_sample_ids=empty_sample_ids
            ),
            artifacts=ExtractionArtifacts(
                feature_dir=store.feature_dir, tiling_dir=self._output_root / "tiling"
            ),
        )

    def _extract_dense_images(self):
        from soma.extract.dense import DenseSourceProvenance
        from soma.extract.dense_extraction import _DenseImageExtractor
        from soma.extract.orchestration import _release_parent_cuda_state

        extractor = _DenseImageExtractor(
            self._images,
            self._encoder,
            dense=self._dense,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(self._output_root / "features")
            require_coverage(store, list(self._images.sample_ids))
        finally:
            _release_parent_cuda_state()
        store.provenance = DenseSourceProvenance(kind="dense_cache", feature_dir=store.feature_dir)
        return self._completed(
            store,
            provenance=FeatureProvenance(kind="dense_image", encoder_name=self._encoder.name),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _extract_annotation_rois(self):
        from soma.extract.cache import (
            resolve_cache_root,
            resolve_roi_sampling_cache,
            roi_sampling_signature,
            write_roi_sampling_coords,
        )
        from soma.extract.dense import DenseSourceProvenance
        from soma.extract.dense_slide_extraction import sample_slide_rois, write_roi_mask_crops
        from soma.extract.rois import roi_entries, write_roi_table

        preprocessing = self._dense_preprocessing()
        features_root = self._output_root / "features"
        slides, annotations = self._images, self._annotations
        if self._cache.enabled:
            sampling_cache = resolve_roi_sampling_cache(
                cache_root=resolve_cache_root(self._cache, feature_dir=features_root, output_root=self._output_root),
                images=slides,
                annotations=annotations,
                preprocessing=preprocessing,
                spacing_policy=self._spacing_policy,
            )
            fresh: dict[str, list[tuple[int, int]]] = {}
            if sampling_cache.miss_sample_ids:
                fresh = sample_slide_rois(
                    slides,
                    annotations,
                    preprocessing=preprocessing,
                    spacing_policy=self._spacing_policy,
                    sample_ids=sampling_cache.miss_sample_ids,
                )
                write_roi_sampling_coords(cache_resolution=sampling_cache, coords_by_sample_id=fresh)
            coords_by_slide = {**sampling_cache.coords_by_id, **fresh}
            mask_crop_dirs = {sid: sampling_cache.mask_crop_dir(sid) for sid in slides.sample_ids}
            # A hit slide's crops are all present; only freshly sampled slides need theirs.
            crop_slide_ids = set(fresh)
        else:
            coords_by_slide = sample_slide_rois(
                slides, annotations, preprocessing=preprocessing, spacing_policy=self._spacing_policy
            )
            crops_root = self._output_root / "segmentation_rois" / "masks"
            mask_crop_dirs = {sid: crops_root / sid for sid in slides.sample_ids}
            crop_slide_ids = set(slides.sample_ids)
        zero_sample_ids = tuple(sid for sid in slides.sample_ids if not coords_by_slide[sid])
        entries = roi_entries(slides, annotations, coords_by_slide, mask_crop_dirs=mask_crop_dirs)
        roi_table = write_roi_table(entries, out_dir=self._output_root / "segmentation_rois")
        store = self._encode_rois(entries, preprocessing)
        # Crops are read at each grid's recorded spacing (a slide within tolerance is read
        # natively), so they are written once the grids exist; a crash before this point
        # leaves the slide's coords without crops, a miss that re-samples next launch.
        crop_entries = [entry for entry in entries if entry.slide_id in crop_slide_ids]
        write_roi_mask_crops(
            crop_entries,
            spacing_um_by_sample_id={entry.sample_id: store.spacing(entry.sample_id) for entry in crop_entries},
            preprocessing=preprocessing,
        )
        sampling = roi_sampling_signature(preprocessing, spacing_policy=self._spacing_policy)
        store.rois = RoiSet(entries, sampling=sampling)
        store.provenance = DenseSourceProvenance(
            kind="slide_manifest_dense_cache", feature_dir=store.feature_dir, dataset_csv=roi_table
        )
        return self._completed(
            store,
            provenance=FeatureProvenance(
                kind="slide_manifest_dense", encoder_name=self._encoder.name, zero_sample_ids=zero_sample_ids
            ),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir, dataset_csv=roi_table),
        )

    def _extract_rois(self):
        from soma.extract.cache import roi_sampling_signature
        from soma.extract.dense import DenseSourceProvenance

        preprocessing = self._dense_preprocessing()
        sampled = self._rois.sampling
        if sampled is not None:
            # Each ROI's label crop was stored at the geometry it was sampled at; a grid read
            # at another one would not align with it.
            replay = roi_sampling_signature(preprocessing, spacing_policy=self._spacing_policy)
            differing = [
                f"extraction.{field}"
                for field, value in sampled["preprocessing"].items()
                if replay["preprocessing"].get(field) != value
            ] + (["spacing_policy"] if sampled["spacing_policy"] != replay["spacing_policy"] else [])
            if differing:
                raise ValueError(
                    "The ROIs were sampled at another extraction than this one reads them at "
                    f"(differing: {', '.join(differing)}), so their label crops would not align "
                    "with the grids. Pass the extraction, dense and spacing_policy they were "
                    "sampled with, or sample again with annotations=."
                )
        entries = list(self._rois)
        store = self._encode_rois(entries, preprocessing)
        store.rois = RoiSet(entries, sampling=sampled)
        store.provenance = DenseSourceProvenance(kind="slide_region_dense_cache", feature_dir=store.feature_dir)
        return self._completed(
            store,
            provenance=FeatureProvenance(kind="slide_region_dense", encoder_name=self._encoder.name),
            artifacts=ExtractionArtifacts(feature_dir=store.feature_dir),
        )

    def _encode_rois(self, entries, preprocessing: PreprocessingConfig):
        from soma.extract.dense_slide_extraction import _SlideRegionExtractor
        from soma.extract.orchestration import _release_parent_cuda_state

        extractor = _SlideRegionExtractor(
            entries,
            self._encoder,
            preprocessing=preprocessing,
            dense=self._dense_options(preprocessing),
            spacing_policy=self._spacing_policy,
            execution=self._execution,
            cache=self._cache,
        )
        try:
            store = extractor.run(self._output_root / "features")
            require_coverage(store, [entry.sample_id for entry in entries])
        finally:
            _release_parent_cuda_state()
        return store

    def _completed(self, source, *, provenance: FeatureProvenance, artifacts: ExtractionArtifacts):
        """Publish ``extraction_provenance.json`` once the source's coverage is validated,
        then hand the source back."""
        provenance_json = self._output_root / "extraction_provenance.json"
        completed = ExtractionArtifacts(
            feature_dir=artifacts.feature_dir,
            tiling_dir=artifacts.tiling_dir,
            dataset_csv=artifacts.dataset_csv,
            provenance_json=provenance_json,
        )
        payload = {
            "status": "completed",
            "provenance": asdict(provenance),
            "artifacts": {
                key: None if value is None else str(value) for key, value in asdict(completed).items()
            },
        }
        temporary = provenance_json.with_name(f".{provenance_json.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(provenance_json)
        return source
