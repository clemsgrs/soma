"""Helpers for driving slide2vec from soma."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from hs2p import SlideSpec
from hs2p.preprocessing import validate_tiling_result_provenance
from slide2vec import (
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
from slide2vec.artifacts import (
    SlideEmbeddingArtifact,
    TileEmbeddingArtifact,
    load_array,
)
from slide2vec.encoders.validation import (
    validate_encoder_config as validate_slide2vec_encoder_config,
)
from slide2vec.utils.tiling_io import load_process_df, load_tiling_result_from_row

from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset, SampleRecord
from soma.preprocessing.tiling import TilingResult as SomaTilingResult


@dataclass(frozen=True)
class LoadedTiling:
    slide: SlideSpec
    tiling_result: object


@dataclass(frozen=True)
class Slide2VecArtifactAdapter:
    def resolve_feature_payload_dir(self, path: Path | str) -> Path:
        root = Path(path)
        if (root / "cache_metadata.json").is_file() and (root / "features").is_dir():
            return root / "features"
        slide_embeddings_dir = root / "slide_embeddings"
        tile_embeddings_dir = root / "tile_embeddings"
        if slide_embeddings_dir.is_dir():
            return slide_embeddings_dir
        if tile_embeddings_dir.is_dir():
            return tile_embeddings_dir
        return root

    def write_cache_payload(
        self,
        artifacts: Sequence[TileEmbeddingArtifact | SlideEmbeddingArtifact],
        *,
        output_dir: Path,
    ) -> int | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_dim: int | None = None
        for artifact in artifacts:
            array = load_array(artifact.path)
            tensor = array if torch.is_tensor(array) else torch.as_tensor(array)
            torch.save(tensor, output_dir / f"{artifact.sample_id}.pt")
            feature_dim = int(tensor.shape[0] if tensor.ndim == 1 else tensor.shape[1])
        return feature_dim

    def build_tile_artifacts_from_cache_payload(
        self,
        *,
        features_dir: Path,
        loaded_tilings: Sequence[LoadedTiling],
        work_dir: Path,
    ) -> list[TileEmbeddingArtifact]:
        work_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[TileEmbeddingArtifact] = []
        for loaded in loaded_tilings:
            feature_path = features_dir / f"{loaded.slide.sample_id}.pt"
            tensor = torch.load(feature_path, weights_only=True, map_location="cpu")
            metadata_path = work_dir / f"{loaded.slide.sample_id}.meta.json"
            metadata = {
                "sample_id": loaded.slide.sample_id,
                "artifact_type": "tile_embeddings",
                "format": "pt",
                "feature_dim": int(tensor.shape[1]),
                "num_tiles": int(tensor.shape[0]),
                "image_path": str(loaded.slide.image_path),
                "mask_path": str(loaded.slide.mask_path) if loaded.slide.mask_path is not None else "",
                "coordinates_npz_path": str(getattr(loaded.tiling_result, "coordinates_npz_path", "")),
                "coordinates_meta_path": str(getattr(loaded.tiling_result, "coordinates_meta_path", "")),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifacts.append(
                TileEmbeddingArtifact(
                    sample_id=loaded.slide.sample_id,
                    path=feature_path,
                    metadata_path=metadata_path,
                    format="pt",
                    feature_dim=int(tensor.shape[1]),
                    num_tiles=int(tensor.shape[0]),
                )
            )
        return artifacts


SLIDE2VEC_ARTIFACT_ADAPTER = Slide2VecArtifactAdapter()


class Slide2VecRuntime:
    def __init__(self, artifact_adapter: Slide2VecArtifactAdapter) -> None:
        self._artifact_adapter = artifact_adapter

    def preprocess(
        self,
        *,
        model_name: str,
        slides: Sequence[SlideSpec],
        preprocessing: Slide2VecPreprocessingConfig,
        output_dir: Path,
    ) -> None:
        pipeline = Pipeline(
            Model.from_preset(model_name),
            preprocessing,
            execution=ExecutionOptions(
                output_dir=Path(output_dir),
                num_gpus=1,
                precision="fp32",
            ),
        )
        pipeline.run(slides=list(slides), tiling_only=True)

    def extract_uncached(
        self,
        *,
        output_dir: Path,
        loaded_tilings: Sequence[LoadedTiling],
        prepared_tilings: Sequence[object],
        tiling_dir: Path,
        encoder: EncoderConfig,
        preprocessing: Slide2VecPreprocessingConfig,
        level: str,
        model_name: str,
        output_variant: str,
        num_gpus: int | None,
    ) -> None:
        execution = build_execution_options(
            encoder,
            output_dir=output_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=(level == "tile" or encoder.save_tile_features),
        )
        slides = [loaded.slide for loaded in loaded_tilings]
        if num_gpus is not None and num_gpus > 1:
            self._run_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=slides,
            )
            return
        if level == "tile":
            self._embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                slides=slides,
                tiling_results=prepared_tilings,
                preprocessing=preprocessing,
                execution=execution,
            )
            return
        if encoder.save_tile_features:
            tile_artifacts = self._embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                slides=slides,
                tiling_results=prepared_tilings,
                preprocessing=preprocessing,
                execution=execution,
            )
        else:
            with make_tempdir("soma-slide2vec-tiles-") as tmp_dir:
                temp_execution = build_execution_options(
                    encoder,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                )
                tile_artifacts = self._embed_tiles(
                    model_name=model_name,
                    output_variant=output_variant,
                    slides=slides,
                    tiling_results=prepared_tilings,
                    preprocessing=preprocessing,
                    execution=temp_execution,
                )
        self._aggregate_tiles(
            model_name=model_name,
            output_variant=output_variant,
            tile_artifacts=tile_artifacts,
            preprocessing=preprocessing,
            execution=execution,
        )

    def populate_tile_cache(
        self,
        *,
        cache_resolution,
        loaded_tilings: Sequence[LoadedTiling],
        prepared_tilings: Sequence[object],
        tiling_dir: Path,
        encoder: EncoderConfig,
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
        with make_tempdir("soma-slide2vec-cache-tile-") as tmp_dir:
            execution = build_execution_options(
                encoder,
                output_dir=Path(tmp_dir),
                num_gpus=num_gpus,
                save_tile_embeddings=True,
            )
            if num_gpus is not None and num_gpus > 1:
                artifacts = self._run_with_coordinates(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    preprocessing=preprocessing,
                    execution=execution,
                    tiling_dir=tiling_dir,
                    slides=[loaded.slide for loaded in selected_loaded],
                ).tile_artifacts
            else:
                artifacts = self._embed_tiles(
                    model_name=encoder_name,
                    output_variant=output_variant,
                    slides=[loaded.slide for loaded in selected_loaded],
                    tiling_results=selected_tilings,
                    preprocessing=preprocessing,
                    execution=execution,
                )
            feature_dim = self._artifact_adapter.write_cache_payload(
                artifacts,
                output_dir=cache_resolution.features_dir,
            )
        if feature_dim is not None:
            from soma.cache import record_feature_dim

            record_feature_dim(cache_resolution, feature_dim)

    def populate_slide_and_tile_caches_distributed(
        self,
        *,
        tile_cache,
        slide_cache,
        loaded_tilings: Sequence[LoadedTiling],
        tiling_dir: Path,
        encoder: EncoderConfig,
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
        with make_tempdir("soma-slide2vec-cache-slide-dist-") as tmp_dir:
            run_result = self._run_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                preprocessing=preprocessing,
                execution=build_execution_options(
                    encoder,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                ),
                tiling_dir=tiling_dir,
                slides=[loaded.slide for loaded in selected_loaded],
            )
            tile_feature_dim = self._artifact_adapter.write_cache_payload(
                [
                    artifact
                    for artifact in run_result.tile_artifacts
                    if artifact.sample_id in tile_missing
                ],
                output_dir=tile_cache.features_dir,
            )
            slide_feature_dim = self._artifact_adapter.write_cache_payload(
                [
                    artifact
                    for artifact in run_result.slide_artifacts
                    if artifact.sample_id in slide_missing
                ],
                output_dir=slide_cache.features_dir,
            )
        if tile_feature_dim is not None:
            from soma.cache import record_feature_dim

            record_feature_dim(tile_cache, tile_feature_dim)
        if slide_feature_dim is not None:
            from soma.cache import record_feature_dim

            record_feature_dim(slide_cache, slide_feature_dim)

    def populate_slide_cache(
        self,
        *,
        slide_cache,
        tile_cache,
        loaded_tilings: Sequence[LoadedTiling],
        encoder: EncoderConfig,
        model_name: str,
        output_variant: str,
        num_gpus: int | None,
    ) -> None:
        missing = set(slide_cache.missing_sample_ids())
        if not missing:
            return
        selected_loaded = [loaded for loaded in loaded_tilings if loaded.slide.sample_id in missing]
        with make_tempdir("soma-slide2vec-cache-slide-") as tmp_dir:
            artifact_dir = Path(tmp_dir)
            tile_artifacts = self._artifact_adapter.build_tile_artifacts_from_cache_payload(
                features_dir=tile_cache.features_dir,
                loaded_tilings=selected_loaded,
                work_dir=artifact_dir / "tile_metadata",
            )
            slide_artifacts = self._aggregate_tiles(
                model_name=model_name,
                output_variant=output_variant,
                tile_artifacts=tile_artifacts,
                preprocessing=None,
                execution=build_execution_options(
                    encoder,
                    output_dir=artifact_dir,
                    num_gpus=num_gpus,
                    save_tile_embeddings=False,
                ),
            )
            feature_dim = self._artifact_adapter.write_cache_payload(
                slide_artifacts,
                output_dir=slide_cache.features_dir,
            )
        if feature_dim is not None:
            from soma.cache import record_feature_dim

            record_feature_dim(slide_cache, feature_dim)

    def _run_with_coordinates(
        self,
        *,
        model_name: str,
        output_variant: str,
        preprocessing: Slide2VecPreprocessingConfig,
        execution: ExecutionOptions,
        tiling_dir: Path,
        slides: Sequence[SlideSpec],
    ):
        return Pipeline(
            Model.from_preset(model_name, output_variant=output_variant),
            preprocessing,
            execution=execution,
        ).run_with_coordinates(
            tiling_dir,
            slides=list(slides),
        )

    def _embed_tiles(
        self,
        *,
        model_name: str,
        output_variant: str,
        slides: Sequence[SlideSpec],
        tiling_results: Sequence[object],
        preprocessing: Slide2VecPreprocessingConfig,
        execution: ExecutionOptions,
    ):
        model = Model.from_preset(model_name, output_variant=output_variant)
        return model.embed_tiles(
            list(slides),
            list(tiling_results),
            preprocessing=preprocessing,
            execution=execution,
        )

    def _aggregate_tiles(
        self,
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

SLIDE2VEC_RUNTIME = Slide2VecRuntime(SLIDE2VEC_ARTIFACT_ADAPTER)


def to_soma_tiling_result(tiling_result: object) -> SomaTilingResult:
    x_values = np.asarray(getattr(tiling_result, "x"), dtype=np.int64)
    y_values = np.asarray(getattr(tiling_result, "y"), dtype=np.int64)
    mask_path = getattr(
        tiling_result,
        "tissue_mask_path",
        getattr(tiling_result, "mask_path", None),
    )
    image_path = getattr(tiling_result, "image_path", None)
    return SomaTilingResult(
        coordinates=np.column_stack((x_values, y_values)),
        tissue_fractions=np.asarray(getattr(tiling_result, "tissue_fractions"), dtype=np.float32),
        requested_tile_size_px=int(getattr(tiling_result, "requested_tile_size_px")),
        requested_spacing_um=float(getattr(tiling_result, "requested_spacing_um")),
        read_level=int(getattr(tiling_result, "read_level")),
        effective_tile_size_px=int(getattr(tiling_result, "effective_tile_size_px")),
        effective_spacing_um=float(getattr(tiling_result, "effective_spacing_um")),
        tile_size_lv0=int(getattr(tiling_result, "tile_size_lv0")),
        is_within_tolerance=bool(getattr(tiling_result, "is_within_tolerance")),
        use_padding=bool(getattr(tiling_result, "use_padding", True)),
        tile_index=getattr(tiling_result, "tile_index", None),
        sample_id=getattr(tiling_result, "sample_id", None),
        image_path=None if image_path is None else str(image_path),
        backend=getattr(tiling_result, "backend", None),
        requested_backend=getattr(tiling_result, "requested_backend", None),
        base_spacing_um=getattr(tiling_result, "base_spacing_um", None),
        slide_dimensions=getattr(tiling_result, "slide_dimensions", None),
        level_downsamples=getattr(tiling_result, "level_downsamples", None),
        overlap=getattr(tiling_result, "overlap", None),
        min_tissue_fraction=getattr(tiling_result, "min_tissue_fraction", None),
        step_px_lv0=getattr(tiling_result, "step_px_lv0", None),
        tissue_method=getattr(tiling_result, "tissue_method", None),
        seg_downsample=getattr(tiling_result, "seg_downsample", None),
        seg_level=getattr(tiling_result, "seg_level", None),
        seg_spacing_um=getattr(tiling_result, "seg_spacing_um", None),
        ref_tile_size_px=getattr(tiling_result, "ref_tile_size_px", None),
        a_t=getattr(tiling_result, "a_t", None),
        tissue_mask_path=None if mask_path is None else str(mask_path),
        tissue_mask_tissue_value=getattr(tiling_result, "tissue_mask_tissue_value", None),
        mask_level=getattr(tiling_result, "mask_level", None),
        mask_spacing_um=getattr(tiling_result, "mask_spacing_um", None),
        config_hash=getattr(tiling_result, "config_hash", None),
        hierarchical=bool(getattr(tiling_result, "hierarchical", False)),
        npatch=getattr(tiling_result, "npatch", None),
        region_index=getattr(tiling_result, "region_index", None),
        region_coordinates=getattr(tiling_result, "region_coordinates", None),
        requested_region_size_px=getattr(tiling_result, "requested_region_size_px", None),
    )


def to_slide2vec_tiling_result(tiling_result: SomaTilingResult) -> object:
    payload = dict(vars(tiling_result))
    coordinates = np.asarray(payload.pop("coordinates"), dtype=np.int64)
    payload["x"] = coordinates[:, 0]
    payload["y"] = coordinates[:, 1]
    payload["coordinates"] = coordinates
    for attr in (
        "coordinates_npz_path",
        "coordinates_meta_path",
        "tiles_tar_path",
        "mask_preview_path",
        "tiling_preview_path",
    ):
        payload.pop(attr, None)
    return SimpleNamespace(**payload)


def to_slide_spec(record: SampleRecord) -> SlideSpec:
    return SlideSpec(
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        spacing_at_level_0=record.metadata.get("spacing_at_level_0"),
    )


def build_slide_specs(dataset: Dataset) -> list[SlideSpec]:
    return [to_slide_spec(record) for record in dataset.samples.values()]


def ensure_supported_mask_value(
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
) -> None:
    if int(preprocessing.tissue_mask_tissue_value) == 1:
        return
    if not any(record.mask_path is not None for record in dataset.samples.values()):
        return
    raise ValueError(
        "slide2vec-backed tiling currently supports only tissue_mask_tissue_value=1 "
        "when mask_path is provided."
    )


def build_preprocessing_config(
    preprocessing: PreprocessingConfig,
    *,
    backend: str,
) -> Slide2VecPreprocessingConfig:
    if preprocessing.requested_tile_size_px is None:
        raise ValueError("requested_tile_size_px must be resolved before extraction")
    if preprocessing.requested_spacing_um is None:
        raise ValueError("requested_spacing_um must be resolved before extraction")
    return Slide2VecPreprocessingConfig(
        backend=backend,
        target_spacing_um=float(preprocessing.requested_spacing_um),
        target_tile_size_px=int(preprocessing.requested_tile_size_px),
        tolerance=float(preprocessing.tolerance),
        overlap=float(preprocessing.overlap),
        tissue_threshold=float(preprocessing.min_tissue_fraction),
        on_the_fly=True,
        adaptive_batching=False,
        use_supertiles=True,
        segmentation={
            "downsample": int(preprocessing.seg_downsample),
            "use_hsv": preprocessing.tissue_method == "hsv",
        },
        filtering={
            "ref_tile_size": int(
                preprocessing.ref_tile_size_px
                if preprocessing.ref_tile_size_px is not None
                else preprocessing.requested_tile_size_px
            ),
            "a_t": int(preprocessing.a_t),
        },
    )


def build_execution_options(
    encoder: EncoderConfig,
    *,
    output_dir: Path,
    num_gpus: int | None,
    save_tile_embeddings: bool,
) -> ExecutionOptions:
    return ExecutionOptions(
        output_dir=Path(output_dir),
        output_format="pt",
        batch_size=int(encoder.batch_size),
        num_workers=int(encoder.num_workers),
        num_preprocessing_workers=8,
        num_gpus=1 if num_gpus is None else int(num_gpus),
        precision=encoder.precision,
        save_tile_embeddings=save_tile_embeddings,
        save_latents=False,
    )


def run_tiling(
    *,
    model_name: str,
    slides: Sequence[SlideSpec],
    preprocessing: Slide2VecPreprocessingConfig,
    output_dir: Path,
) -> None:
    pipeline = Pipeline(
        Model.from_preset(model_name),
        preprocessing,
        execution=ExecutionOptions(output_dir=Path(output_dir), num_gpus=1, precision="fp32"),
    )
    pipeline.run(slides=list(slides), tiling_only=True)


def load_tilings(
    *,
    dataset: Dataset,
    tiling_dir: Path,
    tissue_mask_tissue_value: int,
) -> list[LoadedTiling]:
    process_list_path = Path(tiling_dir) / "process_list.csv"
    if not process_list_path.is_file():
        raise ValueError(
            f"Tiling directory '{tiling_dir}' is missing process_list.csv; "
            "soma now expects slide2vec/hs2p tiling outputs."
        )
    process_df = load_process_df(process_list_path)
    rows_by_sample_id = {
        str(row["sample_id"]): row
        for row in process_df.to_dict("records")
    }
    loaded: list[LoadedTiling] = []
    for record in dataset.samples.values():
        row = rows_by_sample_id.get(record.sample_id)
        if row is None:
            raise ValueError(f"No tiling result found for sample_id={record.sample_id}")
        if row["tiling_status"] != "success":
            raise RuntimeError(
                f"Tiling failed for {record.sample_id}: {row.get('error', '')}"
            )
        tiling_result = load_tiling_result_from_row(row)
        validate_tiling_result_provenance(
            tiling_result,
            sample_id=record.sample_id,
            image_path=record.image_path,
            tissue_mask_path=record.mask_path,
            tissue_mask_tissue_value=(
                int(tissue_mask_tissue_value) if record.mask_path is not None else None
            ),
        )
        loaded.append(LoadedTiling(slide=to_slide_spec(record), tiling_result=tiling_result))
    return loaded


def validate_runtime(
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


def make_tempdir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=prefix)
