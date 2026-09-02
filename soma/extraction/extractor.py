"""FeatureExtractor: preprocessing, per-level extraction, and cache management."""

from __future__ import annotations

import contextlib
import csv
import itertools
import json
import os
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch
from slide2vec import (
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
import slide2vec.progress as slide2vec_progress

from soma.cache import (
    FeatureCacheResolution,
    pooled_extraction_geometry,
    build_tile_artifacts_from_cache_payload,
    preprocessing_backend_provenance,
    record_empty_sample_ids,
    record_feature_dim,
    record_sample_identity_signatures,
    probe_resolved_backends,
    resolve_cache_root,
    resolve_hierarchical_cache,
    resolve_cache_dtype,
    resolve_patient_cache,
    resolve_slide_cache,
    resolve_tiling_cache,
    resolve_tiling_cache_root,
    resolve_tile_cache,
    write_tiling_cache_payload,
    write_tiling_cache_stub,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
from soma.dataset import Dataset, ensure_filename_safe_id
from soma.encoders.validation import resolve_encoder_precision, resolve_preprocessing_config
from soma.extraction.commit import (
    DEFAULT_SLIDES_PER_GPU_COMMIT_EVERY,
    commit_chunks,
    resolve_commit_every,
)
from soma.extraction.orchestration import (
    _aggregate_patients,
    _aggregate_tiles,
    _embed_hierarchical_artifacts_with_coordinates,
    _embed_tile_artifacts_with_coordinates,
    _embed_tiles,
    _load_model,
    _release_parent_cuda_state,
    _run_with_coordinates,
)
from soma.extraction.reporters import (
    _forward_tiling_progress_ctx,
    _make_extraction_reporter_ctx,
    _suppress_logger_noise_ctx,
)
from soma.extraction.slide_aggregation_spawn import spawn_slide_aggregation_workers
from soma.features import PACKED_FILENAME, FeatureStore
from soma.slide2vec_adapter import (
    LoadedTiling,
    build_execution_options,
    build_preprocessing_config,
    build_slide_specs,
    ensure_supported_mask_value,
    load_tilings,
    tiling_num_tiles,
)


def _validate_runtime(
    *,
    encoder_name: str,
    output_variant: str | None,
    encoder: EncoderConfig,
    preprocessing: PreprocessingConfig,
    tiling_results: Sequence[object],
) -> None:
    if not tiling_results:
        return
    validate_slide2vec_encoder_config(
        encoder_name,
        requested_tile_size_px=int(preprocessing.requested_tile_size_px),
        requested_spacing_um=float(preprocessing.requested_spacing_um),
        precision=resolve_encoder_precision(encoder, encoder_name=encoder_name),
        output_variant=output_variant,
        allow_non_recommended=bool(encoder.allow_non_recommended_settings),
    )


def _validate_preprocessing_runtime(
    *,
    encoder_name: str,
    encoder: EncoderConfig,
    preprocessing: PreprocessingConfig,
) -> None:
    validate_slide2vec_encoder_config(
        encoder_name,
        requested_tile_size_px=int(preprocessing.requested_tile_size_px),
        requested_spacing_um=float(preprocessing.requested_spacing_um),
        precision=resolve_encoder_precision(encoder, encoder_name=encoder_name),
        allow_non_recommended=bool(encoder.allow_non_recommended_settings),
    )


def _feature_kind_from_rank(feature_rank: int) -> str:
    if int(feature_rank) == 1:
        return "slide"
    if int(feature_rank) == 2:
        return "bag"
    if int(feature_rank) == 3:
        return "hierarchical"
    raise ValueError(f"Unsupported feature rank {feature_rank}")


def _resolve_output_subpath(output_root: Path, path: str | Path, *, field: str) -> Path:
    """Resolve a safe relative output path beneath ``output_root``."""
    relative = Path(path)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to output_root, got {path!r}")
    resolved = (output_root / relative).resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"{field} must stay within output_root; parent traversal is not allowed: {path!r}"
        ) from exc
    return resolved


def _feature_rank_from_type(feature_type: str) -> int:
    if feature_type == "tile":
        return 1
    if feature_type == "bag":
        return 2
    if feature_type in {"slide", "patient"}:
        return 1
    if feature_type == "hierarchical":
        return 3
    raise ValueError(f"Unsupported feature type {feature_type}")


def _feature_rank_from_artifact_type(artifact_type: str) -> int:
    normalized = artifact_type.removesuffix("_embeddings")
    if normalized in {"slide", "patient"}:
        return 1
    if normalized == "tile":
        return 2
    if normalized == "hierarchical":
        return 3
    raise ValueError(f"Unsupported artifact_type {artifact_type}")


def _feature_summary_from_sidecar(feature_dir: Path) -> tuple[int, int] | None:
    sample_ids = sorted(
        path.stem for path in feature_dir.glob("*.pt") if path.name != PACKED_FILENAME
    )
    if not sample_ids:
        return None
    meta_path = feature_dir / f"{sample_ids[0]}.meta.json"
    if not meta_path.is_file():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_dim = metadata.get("feature_dim")
    artifact_type = str(metadata.get("artifact_type", ""))
    if feature_dim is None or not artifact_type:
        return None
    return _feature_rank_from_artifact_type(artifact_type), int(feature_dim)


def _backend_provenance_from_mapping(
    *,
    requested_backend: str,
    backend_by_sample_id: dict[str, str],
) -> dict[str, object]:
    unique_backends = sorted(set(backend_by_sample_id.values()))
    return {
        "requested_backend": str(requested_backend),
        "backend": unique_backends[0] if len(unique_backends) == 1 else None,
        "backend_by_sample_id": dict(backend_by_sample_id),
    }


def _resolve_num_gpus(num_gpus: int | None) -> int:
    if num_gpus is not None:
        return int(num_gpus)
    if torch.cuda.is_available():
        return max(1, int(torch.cuda.device_count()))
    return 1


def _empty_sample_ids_from_loaded_tilings(loaded_tilings: Sequence[LoadedTiling]) -> list[str]:
    return [
        str(loaded.slide.sample_id)
        for loaded in loaded_tilings
        if tiling_num_tiles(loaded.tiling_result) == 0
    ]


def _non_empty_loaded_tilings(loaded_tilings: Sequence[LoadedTiling]) -> list[LoadedTiling]:
    return [loaded for loaded in loaded_tilings if tiling_num_tiles(loaded.tiling_result) > 0]


class _PooledFeatureExtractor:
    """Preprocesses slides and extracts features for all samples in a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        preprocessing: PreprocessingConfig = PreprocessingConfig(),
        *,
        output_root: str | Path,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig = CacheConfig(),
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._preprocessing = preprocessing
        self._execution = execution
        self._cache = cache
        self._output_root = Path(output_root).resolve()

    def _resolved_preprocessing(self) -> PreprocessingConfig:
        encoder_info = encoder_registry.info(self._encoder.name)
        return resolve_preprocessing_config(
            self._encoder,
            self._preprocessing,
            model_metadata=encoder_info,
        )

    def _feature_subpath(self, feature_dir: str | Path | None) -> Path:
        if feature_dir is not None:
            return Path(feature_dir)
        encoder_name = ensure_filename_safe_id(
            self._encoder.name,
            field="encoder name",
        )
        return Path("features") / encoder_name

    def _effective_preprocessing(self) -> PreprocessingConfig:
        """Resolve preprocessing, then prefer precomputed masks when every slide has one."""
        preprocessing = self._resolved_preprocessing()
        has_precomputed_masks = all(
            record.mask_path is not None for record in self._dataset.samples.values()
        )
        if preprocessing.tissue_method == "precomputed_mask" or has_precomputed_masks:
            return replace(preprocessing, tissue_method="precomputed_mask")
        if preprocessing.tissue_method is None:
            raise ValueError(
                "tissue_method is required when no precomputed tissue mask is provided "
                "for every sample in the dataset."
            )
        return preprocessing

    def _resolved_output(self) -> dict[str, object]:
        encoder_info = encoder_registry.info(self._encoder.name)
        return resolve_encoder_output(
            self._encoder.name,
            requested_output_variant=self._encoder.output_variant,
            metadata=encoder_info,
        )

    def _resolved_execution_for_cache(
        self,
        *,
        encoder_name: str,
        resolved_preprocessing: PreprocessingConfig | None,
        output_variant: str | None,
    ) -> EncoderConfig:
        return replace(
            self._encoder,
            output_variant=output_variant if output_variant is not None else self._encoder.output_variant,
        )

    def _slides_per_commit(self, *, num_gpus: int | None) -> int:
        """Slides per slide2vec call on the WSI paths (see ``cache.commit_every``)."""
        return resolve_commit_every(
            self._cache.commit_every,
            default=DEFAULT_SLIDES_PER_GPU_COMMIT_EVERY * max(1, int(num_gpus or 1)),
        )

    def _resolved_dtype(self, *, encoder_name: str | None = None) -> str:
        """Resolve the on-disk feature dtype ('fp16'/'fp32') from cache.dtype + precision.

        The single value soma folds into the cache key AND passes to slide2vec's writer
        (via ExecutionOptions.output_dtype), so the stored dtype always matches the key.
        ``encoder_name`` selects whose compute precision a ``cache.dtype=None`` follows —
        the tile encoder for a tile-dependency cache, the slide/patient encoder for its own.
        """
        name = encoder_name or self._encoder.name
        return resolve_cache_dtype(self._cache.dtype, self._encoder, encoder_name=name)

    def preprocess(
        self,
        tiling_dir: str | Path,
        *,
        skip_existing: bool = True,
    ) -> None:
        """Preprocess all slides via slide2vec/hs2p tiling orchestration."""
        tiling_dir = Path(tiling_dir).resolve()
        tiling_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._effective_preprocessing()
        ensure_supported_mask_value(self._dataset, cfg)
        _validate_preprocessing_runtime(
            encoder_name=self._encoder.name,
            encoder=self._encoder,
            preprocessing=cfg,
        )
        process_list_path = tiling_dir / "process_list.csv"
        if not self._cache.enabled:
            if skip_existing and process_list_path.is_file():
                return
            preprocessing = build_preprocessing_config(cfg)
            execution = build_execution_options(
                self._encoder,
                execution=self._execution,
                encoder_name=self._encoder.name,
                output_dir=tiling_dir,
                num_gpus=self._execution.num_gpus,
                save_tile_embeddings=False,
            )
            pipeline = Pipeline(
                _load_model(
                    self._encoder.name,
                    output_variant=None,
                    allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                ),
                preprocessing,
                execution=execution,
            )
            with _suppress_logger_noise_ctx("cucim"):
                with _forward_tiling_progress_ctx():
                    pipeline.run(slides=build_slide_specs(self._dataset), tiling_only=True)
            return

        backend_by_sample_id = probe_resolved_backends(
            dataset=self._dataset,
            requested_backend=cfg.backend,
        )
        backend_provenance = _backend_provenance_from_mapping(
            requested_backend=cfg.backend,
            backend_by_sample_id=backend_by_sample_id,
        )
        cache_root = resolve_tiling_cache_root(
            self._cache,
            tiling_dir=tiling_dir,
            output_root=self._output_root,
        )
        cache_resolution = resolve_tiling_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            preprocessing=cfg,
            backend_provenance=backend_provenance,
            encoder_name=self._encoder.name,
            requested_preprocessing=asdict(self._preprocessing),
        )
        if cache_resolution.complete:
            write_tiling_cache_stub(tiling_dir, cache_resolution=cache_resolution)
            return

        preprocessing = build_preprocessing_config(cfg)
        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=self._encoder.name,
            output_dir=tiling_dir,
            num_gpus=self._execution.num_gpus,
            save_tile_embeddings=False,
        )
        pipeline = Pipeline(
            _load_model(
                self._encoder.name,
                output_variant=None,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            ),
            preprocessing,
            execution=execution,
        )
        with _suppress_logger_noise_ctx("cucim"):
            with _forward_tiling_progress_ctx():
                pipeline.run(slides=build_slide_specs(self._dataset), tiling_only=True)
        can_publish = str(cache_resolution.metadata.get("requested_backend")) == str(
            backend_provenance.get("requested_backend")
        )
        if can_publish and process_list_path.is_file():
            write_tiling_cache_payload(
                live_dir=tiling_dir,
                cache_resolution=cache_resolution,
            )
            refreshed = resolve_tiling_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                preprocessing=cfg,
                backend_provenance=backend_provenance,
                encoder_name=self._encoder.name,
                requested_preprocessing=asdict(self._preprocessing),
                complete_state="populated",
            )
            if refreshed.complete:
                write_tiling_cache_stub(tiling_dir, cache_resolution=refreshed)

    def extract(
        self,
        feature_dir: str | Path | None = None,
        *,
        tiling_dir: str | Path | None = None,
        num_gpus: int | None = None,
    ) -> FeatureStore:
        """Extract features using slide2vec and adapt outputs for soma.

        When ``feature_dir`` is omitted, it defaults to
        ``<output_root>/features/<encoder-name>``. A supplied ``feature_dir``
        is resolved relative to ``output_root``; absolute paths and parent
        traversal are rejected.
        """
        feature_dir = _resolve_output_subpath(
            self._output_root,
            self._feature_subpath(feature_dir),
            field="feature_dir",
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        if tiling_dir is None:
            tiling_dir = feature_dir / "tiling"
            self.preprocess(tiling_dir=tiling_dir, skip_existing=True)
        tiling_dir = Path(tiling_dir).resolve()

        encoder_info = encoder_registry.info(self._encoder.name)
        level = resolve_encoder_level(self._encoder.name, encoder_info)
        resolved_output = self._resolved_output()
        resolved_preprocessing = self._effective_preprocessing()
        ensure_supported_mask_value(self._dataset, resolved_preprocessing)
        is_hierarchical = resolved_preprocessing.region_tile_multiple is not None

        loaded_tilings = load_tilings(
            dataset=self._dataset,
            tiling_dir=tiling_dir,
            requested_seg_downsample=int(resolved_preprocessing.seg_downsample),
            tissue_mask_tissue_value=int(resolved_preprocessing.tissue_mask_tissue_value),
            masks_active=resolved_preprocessing.masks is not None,
        )

        if is_hierarchical and level != "tile":
            raise ValueError(
                "Hierarchical preprocessing is only supported for tile-level extraction."
            )

        prepared_tilings: list[object] = [loaded.tiling_result for loaded in loaded_tilings]
        backend_provenance = preprocessing_backend_provenance(
            requested_backend=resolved_preprocessing.backend,
            loaded_tilings=loaded_tilings,
        )
        resolved_output_variant = str(resolved_output["output_variant"])
        runtime_output_variant = resolved_output_variant
        s2v_preprocessing = build_preprocessing_config(resolved_preprocessing)
        allow_non_recommended_settings = bool(self._encoder.allow_non_recommended_settings)

        # slide2vec validates the *requested* override (``None`` = "use the default");
        # handing it the resolved default trips its fixed-variant guard for single-variant
        # slide/patient encoders. The resolved variant itself was already checked by
        # ``_resolved_output``.
        _validate_runtime(
            encoder_name=self._encoder.name,
            output_variant=self._encoder.output_variant,
            encoder=self._encoder,
            preprocessing=resolved_preprocessing,
            tiling_results=prepared_tilings,
        )

        effective_num_gpus = _resolve_num_gpus(
            num_gpus if num_gpus is not None else self._execution.num_gpus
        )
        with _suppress_logger_noise_ctx("cucim"):
            with _make_extraction_reporter_ctx(feature_dir):
                if not self._cache.enabled:
                    self._extract_uncached(
                        feature_dir=feature_dir,
                        loaded_tilings=loaded_tilings,
                        prepared_tilings=prepared_tilings,
                        tiling_dir=tiling_dir,
                        preprocessing=s2v_preprocessing,
                        level=level,
                        output_variant=runtime_output_variant,
                        allow_non_recommended_settings=allow_non_recommended_settings,
                        num_gpus=effective_num_gpus,
                        hierarchical=is_hierarchical,
                    )
                    store = FeatureStore(feature_dir)
                else:
                    cache_root = resolve_cache_root(
                        self._cache,
                        feature_dir=feature_dir,
                        output_root=self._output_root,
                    )
                    if is_hierarchical:
                        store = self._extract_hierarchical_cached(
                            feature_dir=feature_dir,
                            cache_root=cache_root,
                            loaded_tilings=loaded_tilings,
                            prepared_tilings=prepared_tilings,
                            tiling_dir=tiling_dir,
                            preprocessing=s2v_preprocessing,
                            resolved_preprocessing=resolved_preprocessing,
                            backend_provenance=backend_provenance,
                            resolved_output_variant=resolved_output_variant,
                            num_gpus=effective_num_gpus,
                        )
                    elif level == "tile":
                        store = self._extract_tile_cached(
                            feature_dir=feature_dir,
                            cache_root=cache_root,
                            loaded_tilings=loaded_tilings,
                            prepared_tilings=prepared_tilings,
                            tiling_dir=tiling_dir,
                            preprocessing=s2v_preprocessing,
                            resolved_preprocessing=resolved_preprocessing,
                            backend_provenance=backend_provenance,
                            resolved_output_variant=resolved_output_variant,
                            num_gpus=effective_num_gpus,
                        )
                    elif level == "patient":
                        store = self._extract_patient_cached(
                            feature_dir=feature_dir,
                            cache_root=cache_root,
                            encoder_info=encoder_info,
                            loaded_tilings=loaded_tilings,
                            prepared_tilings=prepared_tilings,
                            tiling_dir=tiling_dir,
                            preprocessing=s2v_preprocessing,
                            resolved_preprocessing=resolved_preprocessing,
                            resolved_output=resolved_output,
                            backend_provenance=backend_provenance,
                            resolved_output_variant=resolved_output_variant,
                            runtime_output_variant=runtime_output_variant,
                            num_gpus=effective_num_gpus,
                        )
                    else:
                        store = self._extract_slide_cached(
                            feature_dir=feature_dir,
                            cache_root=cache_root,
                            encoder_info=encoder_info,
                            loaded_tilings=loaded_tilings,
                            prepared_tilings=prepared_tilings,
                            tiling_dir=tiling_dir,
                            preprocessing=s2v_preprocessing,
                            resolved_preprocessing=resolved_preprocessing,
                            resolved_output=resolved_output,
                            backend_provenance=backend_provenance,
                            resolved_output_variant=resolved_output_variant,
                            runtime_output_variant=runtime_output_variant,
                            num_gpus=effective_num_gpus,
                        )

                if level != "patient":
                    self._write_feature_manifest(
                        feature_dir=feature_dir,
                        store=store,
                        loaded_tilings=loaded_tilings,
                        encoder_name=self._encoder.name,
                        output_variant=resolved_output_variant,
                    )
                store = FeatureStore(store.feature_dir)

        return store

    def _write_feature_manifest(
        self,
        *,
        feature_dir: Path,
        store: FeatureStore,
        loaded_tilings: Sequence[LoadedTiling],
        encoder_name: str,
        output_variant: str,
    ) -> None:
        if store.feature_manifest_path is not None and store.feature_manifest_path.is_file():
            return
        manifest_roots = {feature_dir.resolve()}
        feature_root = store.feature_dir.resolve()
        if feature_root != feature_dir.resolve():
            manifest_roots.add(feature_root.parent.resolve())
        feature_rank: int | None = None
        feature_dim: int | None = None
        if store.feature_manifest_path is not None and store.feature_manifest_path.is_file():
            with store.feature_manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rank_value = row.get("feature_rank")
                    dim_value = row.get("feature_dim")
                    if not rank_value or not dim_value:
                        continue
                    feature_rank = int(rank_value)
                    feature_dim = int(dim_value)
                    break
        if feature_rank is None or feature_dim is None:
            summary = _feature_summary_from_sidecar(feature_root)
            if summary is None:
                raise ValueError(
                    "Could not determine feature rank and dimensionality from the current "
                    "manifest or artifact sidecars"
                )
            feature_rank, feature_dim = summary
        feature_kind = _feature_kind_from_rank(feature_rank)

        rows = []
        for loaded in loaded_tilings:
            num_tiles = tiling_num_tiles(loaded.tiling_result)
            feature_status = "empty" if num_tiles == 0 else "success"
            feature_path = ""
            if feature_status == "success":
                feature_path = str((feature_root / f"{loaded.slide.sample_id}.pt").resolve())
            annotation = str(getattr(loaded.tiling_result, "annotation", "tissue") or "tissue")
            rows.append(
                {
                    "sample_id": loaded.slide.sample_id,
                    "annotation": annotation,
                    "feature_status": feature_status,
                    "feature_path": feature_path,
                    "num_tiles": num_tiles,
                    "feature_rank": feature_rank,
                    "feature_dim": feature_dim,
                    "encoder_name": encoder_name,
                    "output_variant": output_variant,
                    "feature_kind": feature_kind,
                }
            )

        for root in manifest_roots:
            manifest_path = root / "process_list.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "annotation",
                        "feature_status",
                        "feature_path",
                        "num_tiles",
                        "feature_rank",
                        "feature_dim",
                        "encoder_name",
                        "output_variant",
                        "feature_kind",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

    def _write_cache_marker(
        self,
        feature_dir: Path,
        *,
        cache_resolution: FeatureCacheResolution,
    ) -> None:
        """Leave a short marker in the requested feature directory when cache is used."""
        marker_path = feature_dir / "README.txt"
        if marker_path.exists():
            return
        cache_dir = cache_resolution.cache_dir.resolve()
        marker_path.write_text(
            (
                "This directory is a cache-backed feature location placeholder.\n"
                f"Actual embedding payloads are stored under: {cache_dir}\n"
                "Configure CacheConfig.root_dir to control the cache location.\n"
            ),
            encoding="utf-8",
        )

    def _write_cached_process_list(
        self,
        feature_dir: Path,
        *,
        cache_resolution: FeatureCacheResolution,
    ) -> None:
        """Write canonical cache and run-local manifests for cached feature payloads."""
        manifest_roots = {
            feature_dir.resolve(),
            cache_resolution.cache_dir.resolve(),
        }
        metadata = cache_resolution.metadata
        sample_ids = self._cache_ids_for_resolution(cache_resolution)
        empty_sample_ids = cache_resolution.empty_sample_ids
        artifact_kind = {
            "tile": "tile_embeddings",
            "slide": "slide_embeddings",
            "hierarchical": "hierarchical_embeddings",
            "patient": "patient_embeddings",
        }[cache_resolution.cache_kind]
        cache_dir = cache_resolution.cache_dir.resolve()
        feature_type = str(metadata["feature_type"])
        feature_rank = _feature_rank_from_type(feature_type)
        feature_dim = metadata.get("feature_dim")
        output_variant = metadata.get("execution", {}).get("output_variant")
        feature_kind = feature_type
        fieldnames = [
            "sample_id",
            "annotation",
            "feature_status",
            "feature_path",
            "artifact_kind",
            "cache_kind",
            "cache_key",
            "cache_dir",
            "encoder_name",
            "output_variant",
            "feature_kind",
            "feature_rank",
            "feature_dim",
        ]
        rows = []
        for sample_id in sample_ids:
            feature_status = "empty" if sample_id in empty_sample_ids else "success"
            feature_path = ""
            if feature_status == "success":
                feature_path = str(self._feature_path_for_cache_id(cache_resolution, sample_id).resolve())
            rows.append(
                {
                    "sample_id": sample_id,
                    "annotation": "tissue",
                    "feature_status": feature_status,
                    "feature_path": feature_path,
                    "artifact_kind": artifact_kind,
                    "cache_kind": cache_resolution.cache_kind,
                    "cache_key": metadata["cache_key"],
                    "cache_dir": str(cache_dir),
                    "encoder_name": metadata["encoder_name"],
                    "output_variant": output_variant,
                    "feature_kind": feature_kind,
                    "feature_rank": feature_rank,
                    "feature_dim": feature_dim,
                }
            )
        for root in manifest_roots:
            root.mkdir(parents=True, exist_ok=True)
            with (root / "process_list.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _materialize_feature_dir_from_cache(
        self,
        feature_dir: Path,
        *,
        cache_resolution: FeatureCacheResolution,
    ) -> None:
        """Keep the run-local feature dir as a lightweight pointer to the shared cache."""
        feature_dir.mkdir(parents=True, exist_ok=True)
        for existing in itertools.chain(
            feature_dir.glob("*.pt"),
            feature_dir.glob("*.meta.json"),
            feature_dir.glob("*.npz"),
        ):
            existing.unlink()

    @staticmethod
    def _filter_tilings_for_ids(
        loaded_tilings: list[LoadedTiling],
        wanted_ids: set[str],
        empty_ids: set[str],
    ) -> list[LoadedTiling]:
        return [
            loaded
            for loaded in loaded_tilings
            if loaded.slide.sample_id in wanted_ids and loaded.slide.sample_id not in empty_ids
        ]

    @staticmethod
    def _cache_ids_for_resolution(cache_resolution: FeatureCacheResolution) -> list[str]:
        cache_ids = getattr(cache_resolution, "cache_ids", None)
        if cache_ids is not None:
            return [str(sample_id) for sample_id in cache_ids]
        return [str(sample_id) for sample_id in cache_resolution.metadata.get("sample_ids", [])]

    @staticmethod
    def _feature_path_for_cache_id(
        cache_resolution: FeatureCacheResolution,
        cache_id: str,
    ) -> Path:
        if hasattr(cache_resolution, "feature_path_for_id"):
            return cache_resolution.feature_path_for_id(cache_id)
        return cache_resolution.features_dir / f"{cache_id}.pt"

    def _extract_uncached(
        self,
        *,
        feature_dir: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        level: str,
        output_variant: str,
        allow_non_recommended_settings: bool,
        num_gpus: int | None,
        hierarchical: bool = False,
    ) -> None:
        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=self._encoder.name,
            output_dir=feature_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=(level == "tile" or self._encoder.save_tile_features or hierarchical),
            output_dtype=self._resolved_dtype(),
        )
        embeddable_tilings = _non_empty_loaded_tilings(loaded_tilings)
        slides = [loaded.slide for loaded in embeddable_tilings]
        embedding_tiling_results = [loaded.tiling_result for loaded in embeddable_tilings]
        model_name = self._encoder.name

        if hierarchical:
            _embed_hierarchical_artifacts_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=allow_non_recommended_settings,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=slides,
            )
            return

        if level == "tile":
            _embed_tile_artifacts_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=allow_non_recommended_settings,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=slides,
            )
            return

        if num_gpus is not None and num_gpus > 1:
            _run_with_coordinates(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=allow_non_recommended_settings,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=slides,
            )
            return

        tile_dir_context = (
            contextlib.nullcontext(None)
            if self._encoder.save_tile_features
            else tempfile.TemporaryDirectory(prefix="soma-tiles-")
        )
        with tile_dir_context as tmp_dir:
            tile_execution = execution
            if tmp_dir is not None:
                tile_execution = build_execution_options(
                    self._encoder,
                    execution=self._execution,
                    encoder_name=self._encoder.name,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                    output_dtype=self._resolved_dtype(),
                )
            tile_artifacts = _embed_tiles(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=allow_non_recommended_settings,
                slides=slides,
                tiling_results=embedding_tiling_results,
                preprocessing=preprocessing,
                execution=tile_execution,
            )
            _aggregate_tiles(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=allow_non_recommended_settings,
                tile_artifacts=tile_artifacts,
                preprocessing=preprocessing,
                execution=execution,
            )


    def _extraction_geometry(
        self,
        *,
        resolved_preprocessing: PreprocessingConfig,
        loaded_tilings: list[LoadedTiling] | None = None,
    ) -> dict[str, object] | None:
        """The geometry record for this pooled run (ADR 0008), or None if undeclarable.

        The per-slide read sizes come from the tilings that were just loaded — the same
        objects extraction is about to encode — so the record describes what this run
        actually read, not what a config implies.
        """
        read_by_id: dict[str, int] | None = None
        if loaded_tilings:
            read_by_id = {}
            for loaded in loaded_tilings:
                tiles = getattr(getattr(loaded, "tiling_result", None), "tiles", None)
                read_size = getattr(tiles, "read_tile_size_px", None)
                if read_size is not None:
                    read_by_id[str(loaded.slide.sample_id)] = int(read_size)
        return pooled_extraction_geometry(
            encoder_name=self._encoder.name,
            requested_tile_size_px=resolved_preprocessing.requested_tile_size_px,
            allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            read_tile_size_px_by_id=read_by_id or None,
        )

    def _extract_tile_cached(
        self,
        *,
        feature_dir: Path,
        cache_root: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        backend_provenance: dict[str, object],
        resolved_output_variant: str,
        num_gpus: int | None,
    ) -> FeatureStore:
        cache_resolution = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
            extraction_geometry=self._extraction_geometry(
                resolved_preprocessing=resolved_preprocessing,
                loaded_tilings=loaded_tilings,
            ),
        )
        self._write_cache_marker(feature_dir, cache_resolution=cache_resolution)
        if cache_resolution.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
            self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=cache_resolution)
            return FeatureStore(cache_resolution.cache_dir)

        self._populate_tile_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=preprocessing,
            encoder_name=self._encoder.name,
            output_variant=resolved_output_variant,
            num_gpus=num_gpus,
        )
        refreshed = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            complete_state="populated",
            validate_payloads=self._cache.validate_payloads,
            extraction_geometry=self._extraction_geometry(
                resolved_preprocessing=resolved_preprocessing,
                loaded_tilings=loaded_tilings,
            ),
        )
        self._write_cached_process_list(feature_dir, cache_resolution=refreshed)
        self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=refreshed)
        return FeatureStore(refreshed.cache_dir)

    def _extract_hierarchical_cached(
        self,
        *,
        feature_dir: Path,
        cache_root: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        backend_provenance: dict[str, object],
        resolved_output_variant: str,
        num_gpus: int | None,
    ) -> FeatureStore:
        cache_resolution = resolve_hierarchical_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
            extraction_geometry=self._extraction_geometry(
                resolved_preprocessing=resolved_preprocessing,
                loaded_tilings=loaded_tilings,
            ),
        )
        self._write_cache_marker(feature_dir, cache_resolution=cache_resolution)
        if cache_resolution.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
            self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=cache_resolution)
            return FeatureStore(cache_resolution.cache_dir)

        self._populate_hierarchical_cache(
            cache_resolution=cache_resolution,
            loaded_tilings=loaded_tilings,
            prepared_tilings=prepared_tilings,
            tiling_dir=tiling_dir,
            preprocessing=preprocessing,
            encoder_name=self._encoder.name,
            output_variant=resolved_output_variant,
            num_gpus=num_gpus,
        )
        refreshed = resolve_hierarchical_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=self._encoder.name,
            preprocessing=resolved_preprocessing,
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            complete_state="populated",
            validate_payloads=self._cache.validate_payloads,
            extraction_geometry=self._extraction_geometry(
                resolved_preprocessing=resolved_preprocessing,
                loaded_tilings=loaded_tilings,
            ),
        )
        self._write_cached_process_list(feature_dir, cache_resolution=refreshed)
        self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=refreshed)
        return FeatureStore(refreshed.cache_dir)

    def _extract_slide_cached(
        self,
        *,
        feature_dir: Path,
        cache_root: Path,
        encoder_info: dict,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        resolved_output: dict[str, object],
        backend_provenance: dict[str, object],
        resolved_output_variant: str,
        runtime_output_variant: str | None,
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
            execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            output_variant=str(tile_dependency_output["output_variant"]),
            dtype=self._resolved_dtype(encoder_name=tile_encoder_name),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
        )
        slide_cache = resolve_slide_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            slide_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_preprocessing=resolved_preprocessing,
            tile_execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            tile_output_variant=str(tile_dependency_output["output_variant"]),
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
        )
        self._write_cache_marker(feature_dir, cache_resolution=slide_cache)
        if tile_cache.complete and slide_cache.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=slide_cache)
            self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=slide_cache)
            return FeatureStore(slide_cache.cache_dir)

        # Always-aggregate: ensure the tile-dependency cache is complete first,
        # then aggregate from it. This reuses the incremental, crash-safe tile
        # pipeline and keeps slide extraction memory-bounded — mirrors the
        # patient path (_extract_patient_cached). The same tile_cache resolution
        # object is populated then re-resolved and read, so there is no second,
        # independent cache key that could mismatch.
        if not tile_cache.complete:
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
            tile_cache = resolve_tile_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=tile_encoder_name,
                preprocessing=resolved_preprocessing,
                execution=self._resolved_execution_for_cache(
                    encoder_name=tile_encoder_name,
                    resolved_preprocessing=resolved_preprocessing,
                    output_variant=str(tile_dependency_output["output_variant"]),
                ),
                output_variant=str(tile_dependency_output["output_variant"]),
                dtype=self._resolved_dtype(encoder_name=tile_encoder_name),
                backend_provenance=backend_provenance,
                complete_state="populated",
                validate_payloads=self._cache.validate_payloads,
            )

        self._populate_slide_cache(
            tile_cache=tile_cache,
            slide_cache=slide_cache,
            loaded_tilings=loaded_tilings,
            preprocessing=preprocessing,
            model_name=self._encoder.name,
            output_variant=runtime_output_variant,
            num_gpus=num_gpus,
        )
        refreshed = resolve_slide_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            slide_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_preprocessing=resolved_preprocessing,
            tile_execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            tile_output_variant=str(tile_dependency_output["output_variant"]),
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            complete_state="populated",
            validate_payloads=self._cache.validate_payloads,
        )
        self._write_cached_process_list(feature_dir, cache_resolution=refreshed)
        self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=refreshed)
        return FeatureStore(refreshed.cache_dir)

    def _extract_patient_cached(
        self,
        *,
        feature_dir: Path,
        cache_root: Path,
        encoder_info: dict,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        resolved_output: dict[str, object],
        backend_provenance: dict[str, object],
        resolved_output_variant: str,
        runtime_output_variant: str | None,
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
            execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            output_variant=str(tile_dependency_output["output_variant"]),
            dtype=self._resolved_dtype(encoder_name=tile_encoder_name),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
        )
        patient_cache = resolve_patient_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            patient_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_preprocessing=resolved_preprocessing,
            tile_execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            tile_output_variant=str(tile_dependency_output["output_variant"]),
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            validate_payloads=self._cache.validate_payloads,
        )
        self._write_cache_marker(feature_dir, cache_resolution=patient_cache)
        if tile_cache.complete and patient_cache.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=patient_cache)
            self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=patient_cache)
            return FeatureStore(patient_cache.cache_dir)

        patient_id_map = self._patient_id_map_for_patient_encoder()

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
        tile_cache = resolve_tile_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            tile_encoder_name=tile_encoder_name,
            preprocessing=resolved_preprocessing,
            execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            output_variant=str(tile_dependency_output["output_variant"]),
            dtype=self._resolved_dtype(encoder_name=tile_encoder_name),
            backend_provenance=backend_provenance,
            complete_state="populated",
            validate_payloads=self._cache.validate_payloads,
        )
        self._populate_patient_cache(
            patient_cache=patient_cache,
            tile_cache=tile_cache,
            loaded_tilings=loaded_tilings,
            patient_id_map=patient_id_map,
            model_name=self._encoder.name,
            output_variant=runtime_output_variant,
            num_gpus=num_gpus,
        )
        refreshed = resolve_patient_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            patient_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_preprocessing=resolved_preprocessing,
            tile_execution=self._resolved_execution_for_cache(
                encoder_name=tile_encoder_name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=str(tile_dependency_output["output_variant"]),
            ),
            tile_output_variant=str(tile_dependency_output["output_variant"]),
            execution=self._resolved_execution_for_cache(
                encoder_name=self._encoder.name,
                resolved_preprocessing=resolved_preprocessing,
                output_variant=resolved_output_variant,
            ),
            output_variant=resolved_output_variant,
            dtype=self._resolved_dtype(),
            backend_provenance=backend_provenance,
            complete_state="populated",
            validate_payloads=self._cache.validate_payloads,
        )
        self._write_cached_process_list(feature_dir, cache_resolution=refreshed)
        self._materialize_feature_dir_from_cache(feature_dir, cache_resolution=refreshed)
        return FeatureStore(refreshed.cache_dir)

    def _populate_patient_cache(
        self,
        *,
        patient_cache: FeatureCacheResolution,
        tile_cache: FeatureCacheResolution,
        loaded_tilings: Sequence[LoadedTiling],
        patient_id_map: dict[str, str],
        model_name: str,
        output_variant: str | None,
        num_gpus: int | None,
    ) -> None:
        """Compute patient embeddings from cached tile features and write them to the patient cache.

        Two-phase: tile features → slide embeddings (via encode_slide) → patient embeddings
        (via encode_patient, grouped by patient_id).
        """
        missing_sample_ids = set(tile_cache.missing_sample_ids())
        empty_sample_ids = set(tile_cache.empty_sample_ids)
        unavailable_sample_ids = missing_sample_ids | empty_sample_ids
        selected_loaded = [
            loaded for loaded in loaded_tilings
            if loaded.slide.sample_id not in unavailable_sample_ids
        ]
        patient_to_loaded_ids: dict[str, set[str]] = {}
        for loaded in loaded_tilings:
            sample_id = loaded.slide.sample_id
            if sample_id in patient_id_map:
                patient_to_loaded_ids.setdefault(patient_id_map[sample_id], set()).add(sample_id)
        empty_patient_ids = sorted(
            patient_id
            for patient_id, sample_ids in patient_to_loaded_ids.items()
            if sample_ids and sample_ids.issubset(empty_sample_ids)
        )
        if empty_patient_ids:
            record_empty_sample_ids(patient_cache, empty_patient_ids)
        if not selected_loaded:
            return
        slide_exec = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=model_name,
            output_dir=patient_cache.cache_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=False,
            output_dtype=self._resolved_dtype(encoder_name=model_name),
        )
        patient_exec = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=model_name,
            output_dir=patient_cache.cache_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=False,
            output_dtype=self._resolved_dtype(encoder_name=model_name),
        )
        # The scratch ``.meta.json`` sidecars are consumed by _aggregate_patients,
        # so the temp dir must outlive that call (not just the build step).
        with tempfile.TemporaryDirectory(prefix="soma-patient-agg-") as tile_meta_tmp:
            tile_artifacts = build_tile_artifacts_from_cache_payload(
                features_dir=tile_cache.features_dir,
                loaded_tilings=selected_loaded,
                work_dir=Path(tile_meta_tmp),
                feature_path_by_sample_id={
                    loaded.slide.sample_id: tile_cache.feature_path_for_id(loaded.slide.sample_id)
                    for loaded in selected_loaded
                },
            )
            patient_artifacts = _aggregate_patients(
                model_name=model_name,
                output_variant=output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                tile_artifacts=tile_artifacts,
                patient_id_map=patient_id_map,
                preprocessing=None,
                slide_execution=slide_exec,
                patient_execution=patient_exec,
            )
        feature_dim = self._write_artifacts_to_cache_resolution(
            artifacts=patient_artifacts,
            cache_resolution=patient_cache,
            id_attr="patient_id",
        )
        if feature_dim is not None:
            record_feature_dim(patient_cache, feature_dim)

    def _patient_id_map_for_patient_encoder(self) -> dict[str, str]:
        try:
            patient_groups = self._dataset.patient_groups
        except ValueError as exc:
            raise ValueError(
                f"Encoder '{self._encoder.name}' is a patient-level encoder, so every "
                "dataset row must have a patient_id."
            ) from exc
        return {
            record.sample_id: patient_id
            for patient_id, records in patient_groups.items()
            for record in records
        }

    def _write_artifacts_to_cache_resolution(
        self,
        *,
        artifacts: Sequence[object],
        cache_resolution: FeatureCacheResolution,
        id_attr: str = "sample_id",
    ) -> int | None:
        feature_dim: int | None = None
        written_ids: set[str] = set()
        cache_stem_by_id = getattr(cache_resolution, "cache_stem_by_id", None)
        for artifact in artifacts:
            cache_id = str(getattr(artifact, id_attr))
            if cache_stem_by_id is not None and cache_id not in cache_stem_by_id:
                continue
            source = Path(artifact.path)
            destination = self._feature_path_for_cache_id(cache_resolution, cache_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                if destination.exists():
                    destination.unlink()
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            if feature_dim is None:
                dim = getattr(artifact, "feature_dim", None)
                if dim is not None:
                    feature_dim = int(dim)
            written_ids.add(cache_id)
        if written_ids:
            record_sample_identity_signatures(cache_resolution, sorted(written_ids))
        return feature_dim

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
        empty_sample_ids = _empty_sample_ids_from_loaded_tilings(loaded_tilings)
        wanted = set(missing)
        selected_loaded = self._filter_tilings_for_ids(
            list(loaded_tilings),
            wanted_ids=wanted,
            empty_ids=set(empty_sample_ids),
        )
        if not selected_loaded:
            if empty_sample_ids:
                record_empty_sample_ids(cache_resolution, empty_sample_ids)
            return
        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=encoder_name,
            output_dir=cache_resolution.cache_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=True,
            output_dtype=self._resolved_dtype(encoder_name=encoder_name),
        )
        # Commit per chunk of slides so an interrupted run keeps every finished chunk
        # (signatures are what make a payload trusted on resume).
        for chunk in commit_chunks(
            selected_loaded, self._slides_per_commit(num_gpus=execution.num_gpus)
        ):
            artifacts = _embed_tile_artifacts_with_coordinates(
                model_name=encoder_name,
                output_variant=output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=[loaded.slide for loaded in chunk],
            )
            feature_dim = self._write_artifacts_to_cache_resolution(
                artifacts=artifacts,
                cache_resolution=cache_resolution,
            )
            if feature_dim is not None:
                record_feature_dim(cache_resolution, feature_dim)
        if empty_sample_ids:
            record_empty_sample_ids(cache_resolution, empty_sample_ids)

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
        empty_sample_ids = _empty_sample_ids_from_loaded_tilings(loaded_tilings)
        wanted = set(missing)
        selected_loaded = self._filter_tilings_for_ids(
            list(loaded_tilings),
            wanted_ids=wanted,
            empty_ids=set(empty_sample_ids),
        )
        if not selected_loaded:
            if empty_sample_ids:
                record_empty_sample_ids(cache_resolution, empty_sample_ids)
            return
        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=encoder_name,
            output_dir=cache_resolution.cache_dir,
            num_gpus=num_gpus,
            save_tile_embeddings=True,
            output_dtype=self._resolved_dtype(encoder_name=encoder_name),
        )
        # Commit per chunk of slides so an interrupted run keeps every finished chunk
        # (signatures are what make a payload trusted on resume).
        for chunk in commit_chunks(
            selected_loaded, self._slides_per_commit(num_gpus=execution.num_gpus)
        ):
            artifacts = _embed_hierarchical_artifacts_with_coordinates(
                model_name=encoder_name,
                output_variant=output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                preprocessing=preprocessing,
                execution=execution,
                tiling_dir=tiling_dir,
                slides=[loaded.slide for loaded in chunk],
            )
            feature_dim = self._write_artifacts_to_cache_resolution(
                artifacts=artifacts,
                cache_resolution=cache_resolution,
            )
            if feature_dim is not None:
                record_feature_dim(cache_resolution, feature_dim)
        if empty_sample_ids:
            record_empty_sample_ids(cache_resolution, empty_sample_ids)

    def _populate_slide_cache(
        self,
        *,
        tile_cache,
        slide_cache,
        loaded_tilings: Sequence[LoadedTiling],
        preprocessing: Slide2VecPreprocessingConfig,
        model_name: str,
        output_variant: str | None,
        num_gpus: int | None,
    ) -> None:
        tile_missing = set(tile_cache.missing_sample_ids())
        slide_missing = set(slide_cache.missing_sample_ids())
        # Invariant: the tile-dependency cache is fully populated before this
        # method runs (see _extract_slide_cached). Slide extraction is therefore
        # always pure aggregation from cached tile features — never a direct
        # embed. This keeps host memory bounded and the run crash-safe. This is a
        # runtime, state-dependent invariant (a slide can tile but fail to
        # embed), so raise rather than assert (asserts are stripped under -O).
        if tile_missing:
            raise RuntimeError(
                "_populate_slide_cache requires a complete tile dependency cache, "
                f"but {len(tile_missing)} sample(s) are still missing tile features: "
                f"{', '.join(sorted(tile_missing)[:10])}"
                f"{' …' if len(tile_missing) > 10 else ''}. "
                "The tile cache must be populated before slide aggregation."
            )
        empty_sample_ids = set(_empty_sample_ids_from_loaded_tilings(loaded_tilings))
        selected_loaded = self._filter_tilings_for_ids(loaded_tilings, slide_missing, empty_sample_ids)
        if selected_loaded:
            slide_execution = build_execution_options(
                self._encoder,
                execution=self._execution,
                encoder_name=model_name,
                output_dir=slide_cache.cache_dir,
                num_gpus=num_gpus,
                save_tile_embeddings=False,
                output_dtype=self._resolved_dtype(encoder_name=model_name),
            )
            slide_feature_dim: int | None = None
            if slide_execution.num_gpus > 1 and len(selected_loaded) > 1:
                num_workers = min(int(slide_execution.num_gpus), len(selected_loaded))
                shard_payloads: list[list[dict[str, str]]] = [[] for _ in range(num_workers)]
                for idx, loaded in enumerate(selected_loaded):
                    shard_payloads[idx % num_workers].append(
                        {
                            "sample_id": str(loaded.slide.sample_id),
                            "feature_path": str(tile_cache.feature_path_for_id(loaded.slide.sample_id)),
                            "image_path": str(loaded.slide.image_path),
                            "mask_path": (
                                str(loaded.slide.mask_path) if loaded.slide.mask_path is not None else ""
                            ),
                            "coordinates_npz_path": str(
                                getattr(loaded.tiling_result, "coordinates_npz_path")
                            ),
                            "coordinates_meta_path": str(
                                getattr(loaded.tiling_result, "coordinates_meta_path")
                            ),
                        }
                    )
                total_slides = len(selected_loaded)
                slide2vec_progress.emit_progress(
                    "embedding.started",
                    slide_count=total_slides,
                )

                def _on_aggregation_progress(processed: int, total: int) -> None:
                    slide2vec_progress.emit_progress(
                        "embedding.progress",
                        slide_count=total,
                        slides_completed=processed,
                    )

                # Commit each worker's slides as it finishes (per-shard), so a
                # crash partway through preserves shards that already completed
                # instead of forcing the whole aggregation step to rerun.
                def _commit_shard(shard_ids: list[str], shard_dim: int | None) -> None:
                    if shard_dim is not None:
                        record_feature_dim(slide_cache, shard_dim)
                    record_sample_identity_signatures(slide_cache, sorted(shard_ids))

                written_ids, slide_feature_dim = spawn_slide_aggregation_workers(
                    num_workers=num_workers,
                    model_name=model_name,
                    output_variant=output_variant,
                    allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                    execution_precision=slide_execution.precision,
                    execution_batch_size=slide_execution.batch_size,
                    execution_num_workers_per_gpu=slide_execution.resolved_num_workers_per_gpu(),
                    execution_output_dtype=slide_execution.output_dtype,
                    execution_prefetch_factor=slide_execution.prefetch_factor,
                    output_dir=slide_cache.cache_dir,
                    shard_payloads_by_rank=shard_payloads,
                    on_progress=_on_aggregation_progress,
                    on_shard_complete=_commit_shard,
                )
                slide2vec_progress.emit_progress(
                    "embedding.finished",
                    slide_count=total_slides,
                    slides_completed=len(written_ids),
                    tile_artifacts=0,
                    slide_artifacts=len(written_ids),
                )
            else:
                with tempfile.TemporaryDirectory(prefix="soma-slide-agg-") as tile_meta_tmp:
                    tile_artifacts = build_tile_artifacts_from_cache_payload(
                        features_dir=tile_cache.features_dir,
                        loaded_tilings=selected_loaded,
                        work_dir=Path(tile_meta_tmp),
                        feature_path_by_sample_id={
                            loaded.slide.sample_id: tile_cache.feature_path_for_id(loaded.slide.sample_id)
                            for loaded in selected_loaded
                        },
                    )
                    slide_artifacts = _aggregate_tiles(
                        model_name=model_name,
                        output_variant=output_variant,
                        allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
                        tile_artifacts=tile_artifacts,
                        preprocessing=None,
                        execution=slide_execution,
                    )
                slide_feature_dim = self._write_artifacts_to_cache_resolution(
                    artifacts=slide_artifacts,
                    cache_resolution=slide_cache,
                )
            if slide_feature_dim is not None:
                record_feature_dim(slide_cache, slide_feature_dim)
        if empty_sample_ids:
            record_empty_sample_ids(tile_cache, sorted(empty_sample_ids))
            record_empty_sample_ids(slide_cache, sorted(empty_sample_ids))

    def run(
        self,
        feature_dir: str | Path | None = None,
        *,
        skip_existing: bool = True,
        num_gpus: int | None = None,
    ) -> FeatureStore:
        feature_subpath = self._feature_subpath(feature_dir)
        resolved_feature_dir = _resolve_output_subpath(
            self._output_root,
            feature_subpath,
            field="feature_dir",
        )
        tiling_dir = resolved_feature_dir.parent / "tiling"
        self.preprocess(tiling_dir=tiling_dir, skip_existing=skip_existing)
        return self.extract(
            feature_dir=feature_subpath,
            tiling_dir=tiling_dir,
            num_gpus=num_gpus,
        )
