"""FeatureExtractor — delegates generic extraction to slide2vec."""

from __future__ import annotations

import contextlib
import csv
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
import hs2p.progress as hs2p_progress
from slide2vec import (
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
import slide2vec.progress as slide2vec_progress
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
    FeatureCacheResolution,
    build_tile_artifacts_from_cache_payload,
    probe_resolved_backends,
    record_feature_dim,
    preprocessing_backend_provenance,
    resolve_cache_root,
    resolve_hierarchical_cache,
    resolve_slide_cache,
    resolve_tiling_cache,
    resolve_tiling_cache_root,
    resolve_tile_cache,
    write_tiling_cache_payload,
    write_tiling_cache_stub,
    write_cache_payload,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_encoder_precision, resolve_preprocessing_config
from soma.features import FeatureStore
from soma.slide2vec_adapter import (
    LoadedTiling,
    build_execution_options,
    build_preprocessing_config,
    build_slide_specs,
    ensure_supported_mask_value,
    load_tilings,
    tiling_num_tiles,
)


class _DeduplicateLogFilter(logging.Filter):
    """Suppress repeated log messages from the same logger."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg in self._seen:
            return False
        self._seen.add(msg)
        return True


@contextlib.contextmanager
def _suppress_logger_noise_ctx(*logger_names: str):
    """Temporarily raise selected logger trees to WARNING."""
    original_levels: dict[str, int] = {}
    try:
        for logger_name in logger_names:
            for name, logger in logging.root.manager.loggerDict.items():
                if not isinstance(name, str):
                    continue
                if name != logger_name and not name.startswith(f"{logger_name}."):
                    continue
                original_levels[name] = logging.getLogger(name).level
                logging.getLogger(name).setLevel(logging.WARNING)
            original_levels[logger_name] = logging.getLogger(logger_name).level
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


class _SomaExtractionReporter:
    """
    Progress reporter wrapper for soma's extraction step.

    Wraps a slide2vec reporter to:
    - Deduplicate identical write_log messages
    - Delegate embedding progress tracking to slide2vec's rich UI
      (one bar per GPU with per-slide updates)
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._seen_logs: set[str] = set()

    def emit(self, event) -> None:
        self._inner.emit(event)

    def write_log(self, message: str, *, stream=None) -> None:
        if message in self._seen_logs:
            return
        self._seen_logs.add(message)
        self._inner.write_log(message, stream=stream)

    def close(self) -> None:
        self._inner.close()


class _TilingProgressBridgeReporter:
    """Forward live hs2p tiling updates into slide2vec's active reporter."""

    def emit(self, event) -> None:
        if getattr(event, "kind", None) != "tiling.progress":
            return
        slide2vec_progress.get_progress_reporter().emit(
            slide2vec_progress.ProgressEvent(kind=event.kind, payload=dict(event.payload))
        )

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        target = stream or None
        slide2vec_progress.emit_progress_log(message, stream=target)


@contextlib.contextmanager
def _make_extraction_reporter_ctx(feature_dir: Path):
    """
    Context manager that activates a *SomaExtractionReporter* when no
    reporter is already active, and installs a log-dedup filter on
    slide2vec's inference logger to suppress repeated warnings.
    """
    dedup_filter = _DeduplicateLogFilter()
    inference_logger = logging.getLogger("slide2vec.inference")
    inference_logger.addFilter(dedup_filter)
    try:
        with _suppress_logger_noise_ctx("cucim"):
            active = slide2vec_progress.get_progress_reporter()
            if not isinstance(active, slide2vec_progress.NullProgressReporter):
                yield
                return
            inner = slide2vec_progress.create_api_progress_reporter(output_dir=feature_dir)
            if isinstance(inner, slide2vec_progress.NullProgressReporter):
                yield
                return
            with slide2vec_progress.activate_progress_reporter(_SomaExtractionReporter(inner)):
                yield
    finally:
        inference_logger.removeFilter(dedup_filter)


@contextlib.contextmanager
def _forward_tiling_progress_ctx():
    """Forward hs2p tiling progress into the active slide2vec reporter."""
    with hs2p_progress.activate_progress_reporter(_TilingProgressBridgeReporter()):
        yield


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
        precision=resolve_encoder_precision(encoder, encoder_name=encoder_name),
        output_variant=output_variant,
        allow_non_recommended=False,
    )


def _runtime_output_variant(*, level: str, resolved_output: dict[str, object]) -> str | None:
    if level == "slide":
        return None
    return str(resolved_output["output_variant"])


def _feature_kind_from_rank(feature_rank: int) -> str:
    if int(feature_rank) == 1:
        return "slide"
    if int(feature_rank) == 2:
        return "tile"
    if int(feature_rank) == 3:
        return "hierarchical"
    raise ValueError(f"Unsupported feature rank {feature_rank}")


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


_PENDING_FEATURE_PATH_SENTINEL = "__soma_pending_feature_path__"


def _rewrite_process_list_rows(
    process_list_path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    with process_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_process_list_for_embedding(process_list_path: Path) -> None:
    if not process_list_path.is_file():
        return
    with process_list_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        return

    changed = False
    for column in ("feature_status", "aggregation_status"):
        if column not in fieldnames:
            continue
        if any((row.get(column) or "").strip() for row in rows):
            continue
        for row in rows:
            row[column] = "tbp"
        changed = True
    if "feature_path" in fieldnames:
        if not any((row.get("feature_path") or "").strip() for row in rows):
            for row in rows:
                row["feature_path"] = _PENDING_FEATURE_PATH_SENTINEL
            changed = True

    if changed:
        _rewrite_process_list_rows(
            process_list_path,
            fieldnames=fieldnames,
            rows=rows,
        )


def _restore_process_list_after_embedding(process_list_path: Path) -> None:
    if not process_list_path.is_file():
        return
    with process_list_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "feature_path" not in fieldnames or not rows:
        return
    changed = False
    for row in rows:
        if row.get("feature_path") == _PENDING_FEATURE_PATH_SENTINEL:
            row["feature_path"] = ""
            changed = True
    if changed:
        _rewrite_process_list_rows(
            process_list_path,
            fieldnames=fieldnames,
            rows=rows,
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
    staged_process_list = Path(execution.output_dir) / "process_list.csv"
    source_process_list = tiling_dir / "process_list.csv"
    if source_process_list.is_file():
        _normalize_process_list_for_embedding(source_process_list)
        if not staged_process_list.exists():
            staged_process_list.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_process_list, staged_process_list)
    try:
        return Pipeline(
            Model.from_preset(model_name, output_variant=output_variant),
            preprocessing,
            execution=execution,
        ).run_with_coordinates(
            tiling_dir,
            slides=list(slides),
        )
    finally:
        if source_process_list.is_file():
            _restore_process_list_after_embedding(source_process_list)
            source_resolved = source_process_list.resolve()
            staged_resolved = staged_process_list.resolve()
            if source_resolved != staged_resolved:
                shutil.copyfile(source_process_list, staged_process_list)


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
        output_root: str | Path | None = None,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._preprocessing = preprocessing
        self._cache = cache
        self._output_root = Path(output_root).resolve() if output_root is not None else None

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
        tiling_dir: str | Path,
        *,
        skip_existing: bool = True,
    ) -> None:
        """Preprocess all slides via slide2vec/hs2p tiling orchestration."""
        tiling_dir = Path(tiling_dir).resolve()
        tiling_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._resolved_preprocessing()
        ensure_supported_mask_value(self._dataset, cfg)
        process_list_path = tiling_dir / "process_list.csv"
        if not self._cache.enabled:
            if skip_existing and process_list_path.is_file():
                return
            preprocessing = build_preprocessing_config(cfg)
            pipeline = Pipeline(
                Model.from_preset(self._encoder.name),
                preprocessing,
                execution=ExecutionOptions(
                    output_dir=tiling_dir,
                    num_gpus=1,
                    precision="fp32",
                ),
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
            raw_preprocessing=asdict(self._preprocessing),
        )
        if cache_resolution.complete:
            write_tiling_cache_stub(tiling_dir, cache_resolution=cache_resolution)
            return

        preprocessing = build_preprocessing_config(cfg)
        pipeline = Pipeline(
            Model.from_preset(self._encoder.name),
            preprocessing,
            execution=ExecutionOptions(
                output_dir=tiling_dir,
                num_gpus=1,
                precision="fp32",
            ),
        )
        with _suppress_logger_noise_ctx("cucim"):
            with _forward_tiling_progress_ctx():
                pipeline.run(slides=build_slide_specs(self._dataset), tiling_only=True)
        can_publish = (
            dict(cache_resolution.metadata.get("backend_by_sample_id", {}))
            == dict(backend_provenance["backend_by_sample_id"])
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
                raw_preprocessing=asdict(self._preprocessing),
            )
            if refreshed.complete:
                write_tiling_cache_stub(tiling_dir, cache_resolution=refreshed)

    def extract(
        self,
        feature_dir: str | Path,
        *,
        tiling_dir: str | Path | None = None,
        skip_existing: bool = True,
        num_gpus: int | None = None,
    ) -> FeatureStore:
        """Extract features using slide2vec and adapt outputs for soma."""
        del skip_existing
        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)
        if tiling_dir is None:
            tiling_dir = feature_dir / ".tiling"
            self.preprocess(tiling_dir=tiling_dir, skip_existing=True)
        tiling_dir = Path(tiling_dir).resolve()

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
        backend_provenance = preprocessing_backend_provenance(
            requested_backend=resolved_preprocessing.backend,
            loaded_tilings=loaded_tilings,
        )
        output_variant = str(resolved_output["output_variant"])
        runtime_output_variant = _runtime_output_variant(
            level=level,
            resolved_output=resolved_output,
        )
        s2v_preprocessing = build_preprocessing_config(resolved_preprocessing)

        _validate_runtime(
            encoder_name=self._encoder.name,
            output_variant=runtime_output_variant,
            encoder=self._encoder,
            tiling_results=prepared_tilings,
        )

        n_slides = len(loaded_tilings)
        effective_num_gpus = _resolve_num_gpus(num_gpus)
        should_delegate_embedding_progress = effective_num_gpus > 1
        with _suppress_logger_noise_ctx("cucim"):
            with _make_extraction_reporter_ctx(feature_dir):
                if not should_delegate_embedding_progress:
                    slide2vec_progress.emit_progress("embedding.started", slide_count=n_slides)

                if not self._cache.enabled:
                    self._extract_uncached(
                        feature_dir=feature_dir,
                        loaded_tilings=loaded_tilings,
                        prepared_tilings=prepared_tilings,
                        tiling_dir=tiling_dir,
                        preprocessing=s2v_preprocessing,
                        level=level,
                        output_variant=runtime_output_variant,
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
                            output_variant=output_variant,
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
                            output_variant=output_variant,
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
                            output_variant=output_variant,
                            runtime_output_variant=runtime_output_variant,
                            num_gpus=effective_num_gpus,
                        )

                self._write_feature_manifest(
                    feature_dir=feature_dir,
                    store=store,
                    loaded_tilings=loaded_tilings,
                    encoder_name=self._encoder.name,
                    output_variant=output_variant,
                )
                store = FeatureStore(store.feature_dir)

                if not should_delegate_embedding_progress:
                    slide2vec_progress.emit_progress(
                        "embedding.finished",
                        slide_count=n_slides,
                        slides_completed=n_slides,
                        tile_artifacts=n_slides,
                        slide_artifacts=0,
                    )

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
        manifest_roots = {feature_dir.resolve()}
        feature_root = store.feature_dir.resolve()
        if feature_root != feature_dir.resolve():
            manifest_roots.add(feature_root.parent.resolve())
        feature_kind = _feature_kind_from_rank(store.feature_rank)

        rows = []
        for loaded in loaded_tilings:
            num_tiles = tiling_num_tiles(loaded.tiling_result)
            feature_status = "empty" if num_tiles == 0 else "success"
            feature_path = ""
            if feature_status == "success":
                feature_path = str((feature_root / f"{loaded.slide.sample_id}.pt").resolve())
            rows.append(
                {
                    "sample_id": loaded.slide.sample_id,
                    "feature_status": feature_status,
                    "feature_path": feature_path,
                    "num_tiles": num_tiles,
                    "feature_rank": store.feature_rank,
                    "feature_dim": store.feature_dim,
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
        """Write a run-local manifest that points back to the shared cache payloads."""
        process_list_path = feature_dir / "process_list.csv"
        metadata = cache_resolution.metadata
        sample_ids = [str(sample_id) for sample_id in metadata["sample_ids"]]
        artifact_kind = {
            "tile": "tile_embeddings",
            "slide": "slide_embeddings",
            "hierarchical": "hierarchical_embeddings",
        }[cache_resolution.kind]
        cache_dir = cache_resolution.cache_dir.resolve()
        feature_rank = int(metadata["feature_rank"])
        feature_dim = metadata.get("feature_dim")
        output_variant = metadata.get("execution", {}).get("output_variant")
        feature_kind = _feature_kind_from_rank(feature_rank)
        with process_list_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
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
                ],
            )
            writer.writeheader()
            for sample_id in sample_ids:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "feature_status": "success",
                        "feature_path": str((cache_resolution.features_dir / f"{sample_id}.pt").resolve()),
                        "artifact_kind": artifact_kind,
                        "cache_kind": cache_resolution.kind,
                        "cache_key": metadata["cache_key"],
                        "cache_dir": str(cache_dir),
                        "encoder_name": metadata["encoder_name"],
                        "output_variant": output_variant,
                        "feature_kind": feature_kind,
                        "feature_rank": feature_rank,
                        "feature_dim": feature_dim,
                    }
                )

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
        num_gpus: int | None,
        hierarchical: bool = False,
    ) -> None:
        execution = build_execution_options(
            self._encoder,
            encoder_name=self._encoder.name,
            output_dir=feature_dir,
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
                    encoder_name=self._encoder.name,
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
        feature_dir: Path,
        cache_root: Path,
        loaded_tilings: list[LoadedTiling],
        prepared_tilings: list[object],
        tiling_dir: Path,
        preprocessing: Slide2VecPreprocessingConfig,
        resolved_preprocessing: PreprocessingConfig,
        backend_provenance: dict[str, object],
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
            backend_provenance=backend_provenance,
        )
        self._write_cache_marker(feature_dir, cache_resolution=cache_resolution)
        if cache_resolution.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
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
        self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
        return FeatureStore(cache_resolution.cache_dir)

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
            backend_provenance=backend_provenance,
        )
        self._write_cache_marker(feature_dir, cache_resolution=cache_resolution)
        if cache_resolution.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
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
        self._write_cached_process_list(feature_dir, cache_resolution=cache_resolution)
        return FeatureStore(cache_resolution.cache_dir)

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
        output_variant: str,
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
            execution=self._encoder,
            output_variant=str(tile_dependency_output["output_variant"]),
            backend_provenance=backend_provenance,
        )
        slide_cache = resolve_slide_cache(
            cache_root=cache_root,
            dataset=self._dataset,
            slide_encoder_name=self._encoder.name,
            tile_encoder_name=tile_encoder_name,
            tile_cache_key=tile_cache.key,
            execution=self._encoder,
            output_variant=output_variant,
            backend_provenance=backend_provenance,
        )
        self._write_cache_marker(feature_dir, cache_resolution=slide_cache)
        if tile_cache.complete and slide_cache.complete:
            self._write_cached_process_list(feature_dir, cache_resolution=slide_cache)
            return FeatureStore(slide_cache.cache_dir)

        if num_gpus is not None and num_gpus > 1:
            self._populate_slide_and_tile_caches_distributed(
                tile_cache=tile_cache,
                slide_cache=slide_cache,
                loaded_tilings=loaded_tilings,
                tiling_dir=tiling_dir,
                preprocessing=preprocessing,
                model_name=self._encoder.name,
                output_variant=runtime_output_variant,
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
            output_variant=runtime_output_variant,
            num_gpus=num_gpus,
        )
        self._write_cached_process_list(feature_dir, cache_resolution=slide_cache)
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
                encoder_name=encoder_name,
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
                feature_dir=cache_resolution.features_dir,
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
                encoder_name=encoder_name,
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
                feature_dir=cache_resolution.features_dir,
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
        output_variant: str | None,
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
                    encoder_name=model_name,
                    output_dir=Path(tmp_dir),
                    num_gpus=num_gpus,
                    save_tile_embeddings=True,
                ),
                tiling_dir=tiling_dir,
                slides=[loaded.slide for loaded in selected_loaded],
            )
            tile_feature_dim = write_cache_payload(
                [a for a in run_result.tile_artifacts if a.sample_id in tile_missing],
                feature_dir=tile_cache.features_dir,
            )
            slide_feature_dim = write_cache_payload(
                [a for a in run_result.slide_artifacts if a.sample_id in slide_missing],
                feature_dir=slide_cache.features_dir,
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
        output_variant: str | None,
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
                    encoder_name=model_name,
                    output_dir=artifact_dir,
                    num_gpus=num_gpus,
                    save_tile_embeddings=False,
                ),
            )
            feature_dim = write_cache_payload(
                slide_artifacts,
                feature_dir=slide_cache.features_dir,
            )
        if feature_dim is not None:
            record_feature_dim(slide_cache, feature_dim)

    def run(
        self,
        feature_dir: str | Path,
        *,
        skip_existing: bool = True,
        num_gpus: int | None = None,
    ) -> FeatureStore:
        feature_dir = Path(feature_dir).resolve()
        tiling_dir = feature_dir / ".tiling"
        self.preprocess(tiling_dir=tiling_dir, skip_existing=skip_existing)
        return self.extract(
            feature_dir=feature_dir,
            tiling_dir=tiling_dir,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
        )
