"""FeatureExtractor — delegates generic extraction to slide2vec."""

from __future__ import annotations

from pathlib import Path

from soma.cache import (
    resolve_cache_root,
    resolve_slide_cache,
    resolve_tile_cache,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.registry import (
    encoder_registry,
    resolve_encoder_level,
    resolve_encoder_output,
    resolve_tile_dependency_output,
)
from soma.encoders.validation import resolve_preprocessing_config
from soma.features import FeatureStore
from soma.preprocessing.tiling import expand_regions_to_subtiles
from soma.slide2vec_adapter import (
    SLIDE2VEC_RUNTIME,
    build_preprocessing_config,
    build_slide_specs,
    ensure_supported_mask_value,
    load_tilings,
    Slide2VecRuntime,
    to_slide2vec_tiling_result,
    to_soma_tiling_result,
    validate_runtime,
)


def _expand_tiling_result_for_hierarchy(tiling_result: object, npatch: int) -> object:
    return to_slide2vec_tiling_result(
        expand_regions_to_subtiles(
            to_soma_tiling_result(tiling_result),
            npatch=npatch,
        )
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
        SLIDE2VEC_RUNTIME.preprocess(
            model_name=self._encoder.name,
            slides=build_slide_specs(self._dataset),
            preprocessing=build_preprocessing_config(cfg, backend=backend),
            output_dir=output_dir,
        )

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

        loaded_tilings = load_tilings(
            dataset=self._dataset,
            tiling_dir=tiling_dir,
            tissue_mask_tissue_value=int(resolved_preprocessing.tissue_mask_tissue_value),
        )

        prepared_tilings: list[object] = []
        for loaded in loaded_tilings:
            tiling_result = loaded.tiling_result
            if resolved_preprocessing.hierarchical:
                if level != "tile":
                    raise ValueError(
                        "Hierarchical preprocessing is only supported for tile-level extraction."
                    )
                if resolved_preprocessing.npatch is None:
                    raise ValueError("Hierarchical preprocessing requires npatch.")
                expanded = _expand_tiling_result_for_hierarchy(
                    tiling_result,
                    resolved_preprocessing.npatch,
                )
                prepared_tilings.append(expanded)
            else:
                prepared_tilings.append(tiling_result)

        validate_runtime(
            encoder_name=self._encoder.name,
            output_variant=str(resolved_output["output_variant"]),
            encoder=self._encoder,
            tiling_results=prepared_tilings,
        )
        if resolved_preprocessing.hierarchical and num_gpus is not None and num_gpus > 1:
            raise ValueError(
                "Hierarchical preprocessing is not yet supported with num_gpus > 1."
            )

        if not self._cache.enabled:
            SLIDE2VEC_RUNTIME.extract_uncached(
                output_dir=output_dir,
                loaded_tilings=loaded_tilings,
                prepared_tilings=prepared_tilings,
                tiling_dir=tiling_dir,
                encoder=self._encoder,
                preprocessing=build_preprocessing_config(
                    resolved_preprocessing,
                    backend=backend,
                ),
                level=level,
                model_name=self._encoder.name,
                output_variant=str(resolved_output["output_variant"]),
                num_gpus=num_gpus,
            )
            return FeatureStore(output_dir)

        cache_root = resolve_cache_root(self._cache, output_dir=output_dir)
        if level == "tile":
            cache_resolution = resolve_tile_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                preprocessing=resolved_preprocessing,
                execution=self._encoder,
                output_variant=str(resolved_output["output_variant"]),
            )
            SLIDE2VEC_RUNTIME.populate_tile_cache(
                cache_resolution=cache_resolution,
                loaded_tilings=loaded_tilings,
                prepared_tilings=prepared_tilings,
                tiling_dir=tiling_dir,
                encoder=self._encoder,
                preprocessing=build_preprocessing_config(
                    resolved_preprocessing,
                    backend=backend,
                ),
                encoder_name=self._encoder.name,
                output_variant=str(resolved_output["output_variant"]),
                num_gpus=num_gpus,
            )
            return FeatureStore(cache_resolution.cache_dir)

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
            output_variant=str(resolved_output["output_variant"]),
        )
        if num_gpus is not None and num_gpus > 1:
            SLIDE2VEC_RUNTIME.populate_slide_and_tile_caches_distributed(
                tile_cache=tile_cache,
                slide_cache=slide_cache,
                loaded_tilings=loaded_tilings,
                tiling_dir=tiling_dir,
                encoder=self._encoder,
                preprocessing=build_preprocessing_config(
                    resolved_preprocessing,
                    backend=backend,
                ),
                model_name=self._encoder.name,
                output_variant=str(resolved_output["output_variant"]),
                num_gpus=num_gpus,
            )
            return FeatureStore(slide_cache.cache_dir)

        SLIDE2VEC_RUNTIME.populate_tile_cache(
            cache_resolution=tile_cache,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            encoder=self._encoder,
            preprocessing=build_preprocessing_config(
                resolved_preprocessing,
                backend=backend,
            ),
            encoder_name=tile_encoder_name,
            output_variant=str(tile_dependency_output["output_variant"]),
            num_gpus=num_gpus,
        )
        SLIDE2VEC_RUNTIME.populate_slide_cache(
            slide_cache=slide_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded_tilings,
            encoder=self._encoder,
            model_name=self._encoder.name,
            output_variant=str(slide_cache.metadata["execution"]["output_variant"]),
            num_gpus=num_gpus,
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
