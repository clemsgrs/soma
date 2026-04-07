"""FeatureExtractor — delegates generic extraction to slide2vec."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from slide2vec import (
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_encoder_level,
    resolve_encoder_output,
    resolve_tile_dependency_output,
)
from slide2vec.encoders.validation import (
    validate_encoder_config as validate_slide2vec_encoder_config,
)

from soma.cache import (
    build_tile_artifacts_from_cache_payload,
    record_feature_dim,
    resolve_cache_root,
    resolve_hierarchical_cache,
    resolve_slide_cache,
    resolve_tile_cache,
    write_cache_payload,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_preprocessing_config
from soma.features import FeatureStore
from soma.slide2vec_adapter import (
    LoadedTiling,
    build_execution_options,
    build_preprocessing_config,
    build_slide_specs,
    ensure_supported_mask_value,
    load_tilings,
)


def _validate_runtime(
    *,
    encoder_name: str,
    output_variant: str | None,
    encoder: EncoderConfig,
    tiling_results: Sequence[object],
) -> None:
    if not tiling_results:
        return
    first = tiling_results[0]
    validate_slide2vec_encoder_config(
        encoder_name,
        target_tile_size_px=int(first.requested_tile_size_px),
        target_spacing_um=float(first.requested_spacing_um),
        precision=encoder.precision,
        output_variant=output_variant,
        allow_non_recommended=False,
    )


def _embed_tiles(
    *,
    model_name: str,
    output_variant: str,
    slides: Sequence[object],
    tiling_results: Sequence[object],
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
) -> list:
    model = Model.from_preset(model_name, output_variant=output_variant)
    return model.embed_tiles(
        list(slides),
        list(tiling_results),
        preprocessing=preprocessing,
        execution=execution,
    )


def _run_with_coordinates(
    *,
    model_name: str,
    output_variant: str,
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
    tiling_dir: Path,
    slides: Sequence[object],
):
    return Pipeline(
        Model.from_preset(model_name, output_variant=output_variant),
        preprocessing,
        execution=execution,
    ).run_with_coordinates(
        tiling_dir,
        slides=list(slides),
    )


def _aggregate_tiles(
    *,
    model_name: str,
    output_variant: str,
    tile_artifacts,
    preprocessing: Slide2VecPreprocessingConfig | None,
    execution: ExecutionOptions,
):
    model = Model.from_preset(model_name, output_variant=output_variant)
    return model.aggregate_tiles(
        tile_artifacts,
        preprocessing=preprocessing,
        execution=execution,
    )


class FeatureExtractor:
    """Preprocesses slides and extracts features for all samples in a dataset."""

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
        """Preprocess all slides via slide2vec/hs2p tiling orchestration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._resolved_preprocessing()
        ensure_supported_mask_value(self._dataset, cfg)
        process_list_path = output_dir / "process_list.csv"
        if skip_existing and process_list_path.is_file():
            return
        preprocessing = build_preprocessing_config(cfg, backend=backend)
        pipeline = Pipeline(
            Model.from_preset(self._encoder.name),
            preprocessing,
            execution=ExecutionOptions(
                output_dir=Path(output_dir),
                num_gpus=1,
                precision="fp32",
            ),
        )
        pipeline.run(slides=build_slide_specs(self._dataset), tiling_only=True)

    def extract(
        self,
        output_dir: str | Path,
        *,
        tiling_dir: str | Path | None = None,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
        """Extract features using slide2vec and adapt outputs for soma."""
        del skip_existing
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if tiling_dir is None:
            tiling_dir = output_dir / ".tiling"
            self.preprocess(tiling_dir, skip_existing=True, backend=backend)
        tiling_dir = Path(tiling_dir)

        encoder_info = encoder_registry.info(self._encoder.name)
        level = resolve_encoder_level(self._encoder.name, encoder_info)
        resolved_output = self._resolved_output()
        resolved_preprocessing = self._resolved_preprocessing()
        ensure_supported_mask_value(self._dataset, resolved_preprocessing)
        is_hierarchical = resolved_preprocessing.region_tile_multiple is not None

        loaded_tilings = load_tilings(
            dataset=self._dataset,
            tiling_dir=tiling_dir,
            tissue_mask_tissue_value=int(resolved_preprocessing.tissue_mask_tissue_value),
        )

        if is_hierarchical and level != "tile":
            raise ValueError(
                "Hierarchical preprocessing is only supported for tile-level extraction."
            )

        prepared_tilings: list[object] = [loaded.tiling_result for loaded in loaded_tilings]
        output_variant = str(resolved_output["output_variant"])
        s2v_preprocessing = build_preprocessing_config(resolved_preprocessing, backend=backend)

        _validate_runtime(
            encoder_name=self._encoder.name,
            output_variant=output_variant,
            encoder=self._encoder,
            tiling_results=prepared_tilings,
        )

        if not self._cache.enabled:
            self._extract_uncached(
                output_dir=output_dir,
                loaded_tilings=loaded_tilings,
                prepared_tilings=prepared_tilings,
                tiling_dir=tiling_dir,
                preprocessing=s2v_preprocessing,
                level=level,
                output_variant=output_variant,
                num_gpus=num_gpus,
                hierarchical=is_hierarchical,
            )
            return FeatureStore(output_dir)

        cache_root = resolve_cache_root(self._cache, output_dir=output_dir)
        if is_hierarchical:
            return self._extract_hierarchical_cached(
                cache_root=cache_root,
                loaded_tilings=loaded_tilings,
                prepared_tilings=prepared_tilings,
                tiling_dir=tiling_dir,
                preprocessing=s2v_preprocessing,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=output_variant,
                num_gpus=num_gpus,
            )

        if level == "tile":
            return self._extract_tile_cached(
                cache_root=cache_root,
                loaded_tilings=loaded_tilings,
                prepared_tilings=prepared_tilings,
                tiling_dir=tiling_dir,
                preprocessing=s2v_preprocessing,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=output_variant,
                num_gpus=num_gpus,
            )

        return self._extract_slide_cached(
            cache_root=cache_root,
            encoder_info=encoder_info,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=s2v_preprocessing,
            resolved_preprocessing=resolved_preprocessing,
            resolved_output=resolved_output,
            output_variant=output_variant,
            num_gpus=num_gpus,
        )

    def _extract_uncached(
        self,
        *,
        output_dir: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        level: str,
        output_variant: str,
        num_gpus: int | None,
        hierarchical: bool = False,
    ) -> None:
        execution = build_execution_options(
            self._encoder,
            output_dir=output_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=(level == "tile" or self._encoder.save_tile_features or hierarchical),
        )
        slides = [loaded.slide for loaded in loaded_tilings]
        model_name = self._encoder.name

        if hierarchical:
            if num_gpus is not None and num_gpus > 1:
                _run_with_coordinates(
                    model_name=model_name,
                    output_variant=output_variant,
                    preprocessing=preprocessing,
                    execution=execution,
                    tiling_dir=tiling_dir,
                    slides=slides,
                )
                return
            _embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                slides=slides,
                tiling_results=prepared_tilings,
                preprocessing=preprocessing,
                execution=execution,
            )
            return

        if num_gpus is not None and num_gpus > 1:
            _run_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=slides,
            )
            return

        if level == "tile":
            _embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                slides=slides,
                tiling_results=prepared_tilings,
                preprocessing=preprocessing,
                execution=execution,
            )
            return

        if self._encoder.save_tile_features:
            tile_artifacts = _embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                slides=slides,
                tiling_results=prepared_tilings,
                preprocessing=preprocessing,
                execution=execution,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="soma-tiles-") as tmp_dir:
                temp_execution = build_execution_options(
                    self._encoder,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                )
                tile_artifacts = _embed_tiles(
                    model_name=model_name,
                    output_variant=output_variant,
                    slides=slides,
                    tiling_results=prepared_tilings,
                    preprocessing=preprocessing,
                    execution=temp_execution,
                )

        _aggregate_tiles(
            model_name=model_name,
            output_variant=output_variant,
            tile_artifacts=tile_artifacts,
            preprocessing=preprocessing,
            execution=execution,
        )

    def _extract_tile_cached(
        self,
        *,
        cache_root: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        output_variant: str,
        num_gpus: int | None,
    ) -> FeatureStore:
        cache_resolution = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._encoder,
            output_variant=output_variant,
        )
        if cache_resolution.complete:
            return FeatureStore(cache_resolution.cache_dir)

        self._populate_tile_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=preprocessing,
            encoder_name=self._encoder.name,
            output_variant=output_variant,
            num_gpus=num_gpus,
        )
        return FeatureStore(cache_resolution.cache_dir)

    def _extract_hierarchical_cached(
        self,
        *,
        cache_root: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        output_variant: str,
        num_gpus: int | None,
    ) -> FeatureStore:
        cache_resolution = resolve_hierarchical_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._encoder,
            output_variant=output_variant,
        )
        if cache_resolution.complete:
            return FeatureStore(cache_resolution.cache_dir)

        self._populate_hierarchical_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=preprocessing,
            encoder_name=self._encoder.name,
            output_variant=output_variant,
            num_gpus=num_gpus,
        )
        return FeatureStore(cache_resolution.cache_dir)

    def _extract_slide_cached(
        self,
        *,
        cache_root: Path,
        encoder_info: dict,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        resolved_output: dict[str, object],
        output_variant: str,
        num_gpus: int | None,
    ) -> FeatureStore:
        tile_encoder_name = str(encoder_info["tile_encoder"])
        tile_dependency_output = resolve_tile_dependency_output(
            self._encoder.name,
            metadata=encoder_info,
        )
        tile_cache = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=tile_encoder_name,
            preprocessing=resolved_preprocessing,
            execution=self._encoder,
            output_variant=str(tile_dependency_output["output_variant"]),
        )
        slide_cache = resolve_slide_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            slide_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_cache_key=tile_cache.key,
            execution=self._encoder,
            output_variant=output_variant,
        )
        if tile_cache.complete and slide_cache.complete:
            return FeatureStore(slide_cache.cache_dir)

        if num_gpus is not None and num_gpus > 1:
            self._populate_slide_and_tile_caches_distributed(
                tile_cache=tile_cache,
                slide_cache=slide_cache,
                loaded_tilings=loaded_tilings,
                tiling_dir=tiling_dir,
                preprocessing=preprocessing,
                model_name=self._encoder.name,
                output_variant=output_variant,
                num_gpus=num_gpus,
            )
            return FeatureStore(slide_cache.cache_dir)

        self._populate_tile_cache(
            cache_resolution=tile_cache,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=preprocessing,
            encoder_name=tile_encoder_name,
            output_variant=str(tile_dependency_output["output_variant"]),
            num_gpus=num_gpus,
        )
        self._populate_slide_cache(
            slide_cache=slide_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded_tilings,
            model_name=self._encoder.name,
            output_variant=str(slide_cache.metadata["execution"]["output_variant"]),
            num_gpus=num_gpus,
        )
        return FeatureStore(slide_cache.cache_dir)

    def _populate_tile_cache(
        self,
        *,
        cache_resolution,
        loaded_tilings: Sequence[LoadedTiling],
        prepared_tilings: Sequence[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        encoder_name: str,
        output_variant: str,
        num_gpus: int | None,
    ) -> None:
        missing = cache_resolution.missing_sample_ids()
        if not missing:
            return
        wanted = set(missing)
        selected_loaded = [loaded for loaded in loaded_tilings if loaded.slide.sample_id in wanted]
        selected_tilings = [
            tiling
            for loaded, tiling in zip(loaded_tilings, prepared_tilings)
            if loaded.slide.sample_id in wanted
        ]
        with tempfile.TemporaryDirectory(prefix="soma-cache-tile-") as tmp_dir:
            execution = build_execution_options(
                self._encoder,
                output_dir=Path(tmp_dir),
                num_gpus=num_gpus,
                save_tile_embeddings=True,
            )
            if num_gpus is not None and num_gpus > 1:
                artifacts = _run_with_coordinates(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    preprocessing=preprocessing,
                    execution=execution,
                    tiling_dir=tiling_dir,
                    slides=[loaded.slide for loaded in selected_loaded],
                ).tile_artifacts
            else:
                artifacts = _embed_tiles(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    slides=[loaded.slide for loaded in selected_loaded],
                    tiling_results=selected_tilings,
                    preprocessing=preprocessing,
                    execution=execution,
                )
            feature_dim = write_cache_payload(
                artifacts,
                output_dir=cache_resolution.features_dir,
            )
        if feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)

    def _populate_hierarchical_cache(
        self,
        *,
        cache_resolution,
        loaded_tilings: Sequence[LoadedTiling],
        prepared_tilings: Sequence[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        encoder_name: str,
        output_variant: str,
        num_gpus: int | None,
    ) -> None:
        missing = cache_resolution.missing_sample_ids()
        if not missing:
            return
        wanted = set(missing)
        selected_loaded = [loaded for loaded in loaded_tilings if loaded.slide.sample_id in wanted]
        selected_tilings = [
            tiling
            for loaded, tiling in zip(loaded_tilings, prepared_tilings)
            if loaded.slide.sample_id in wanted
        ]
        with tempfile.TemporaryDirectory(prefix="soma-cache-hierarchical-") as tmp_dir:
            execution = build_execution_options(
                self._encoder,
                output_dir=Path(tmp_dir),
                num_gpus=num_gpus,
                save_tile_embeddings=True,
            )
            if num_gpus is not None and num_gpus > 1:
                result = _run_with_coordinates(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    preprocessing=preprocessing,
                    execution=execution,
                    tiling_dir=tiling_dir,
                    slides=[loaded.slide for loaded in selected_loaded],
                )
            else:
                result = _embed_tiles(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    slides=[loaded.slide for loaded in selected_loaded],
                    tiling_results=selected_tilings,
                    preprocessing=preprocessing,
                    execution=execution,
                )
            artifacts = getattr(result, "hierarchical_artifacts", None)
            if artifacts is None:
                raise ValueError(
                    "slide2vec did not return hierarchical_artifacts for hierarchical cache population"
                )
            feature_dim = write_cache_payload(
                artifacts,
                output_dir=cache_resolution.features_dir,
            )
        if feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)

    def _populate_slide_and_tile_caches_distributed(
        self,
        *,
        tile_cache,
        slide_cache,
        loaded_tilings: Sequence[LoadedTiling],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        model_name: str,
        output_variant: str,
        num_gpus: int,
    ) -> None:
        tile_missing = set(tile_cache.missing_sample_ids())
        slide_missing = set(slide_cache.missing_sample_ids())
        run_ids = tile_missing | slide_missing
        if not run_ids:
            return
        selected_loaded = [loaded for loaded in loaded_tilings if loaded.slide.sample_id in run_ids]
        with tempfile.TemporaryDirectory(prefix="soma-cache-slide-dist-") as tmp_dir:
            run_result = _run_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                preprocessing=preprocessing,
                execution=build_execution_options(
                    self._encoder,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                ),
                tiling_dir=tiling_dir,
                slides=[loaded.slide for loaded in selected_loaded],
            )
            tile_feature_dim = write_cache_payload(
                [a for a in run_result.tile_artifacts if a.sample_id in tile_missing],
                output_dir=tile_cache.features_dir,
            )
            slide_feature_dim = write_cache_payload(
                [a for a in run_result.slide_artifacts if a.sample_id in slide_missing],
                output_dir=slide_cache.features_dir,
            )
        if tile_feature_dim is not None:
            record_feature_dim(tile_cache, tile_feature_dim)
        if slide_feature_dim is not None:
            record_feature_dim(slide_cache, slide_feature_dim)

    def _populate_slide_cache(
        self,
        *,
        slide_cache,
        tile_cache,
        loaded_tilings: Sequence[LoadedTiling],
        model_name: str,
        output_variant: str,
        num_gpus: int | None,
    ) -> None:
        missing = set(slide_cache.missing_sample_ids())
        if not missing:
            return
        selected_loaded = [loaded for loaded in loaded_tilings if loaded.slide.sample_id in missing]
        with tempfile.TemporaryDirectory(prefix="soma-cache-slide-") as tmp_dir:
            artifact_dir = Path(tmp_dir)
            tile_artifacts = build_tile_artifacts_from_cache_payload(
                features_dir=tile_cache.features_dir,
                loaded_tilings=selected_loaded,
                work_dir=artifact_dir / "tile_metadata",
            )
            slide_artifacts = _aggregate_tiles(
                model_name=model_name,
                output_variant=output_variant,
                tile_artifacts=tile_artifacts,
                preprocessing=None,
                execution=build_execution_options(
                    self._encoder,
                    output_dir=artifact_dir,
                    num_gpus=num_gpus,
                    save_tile_embeddings=False,
                ),
            )
            feature_dim = write_cache_payload(
                slide_artifacts,
                output_dir=slide_cache.features_dir,
            )
        if feature_dim is not None:
            record_feature_dim(slide_cache, feature_dim)

    def run(
        self,
        output_dir: str | Path,
        *,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
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
