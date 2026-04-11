"""Shared cache utilities for feature and tiling artifacts."""

from __future__ import annotations

from abc import ABC
import csv
import contextlib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from hs2p.preprocessing import validate_tiling_result_provenance
from hs2p.wsi.reader import resolve_backend
from slide2vec.artifacts import TileEmbeddingArtifact
import slide2vec.progress as slide2vec_progress
from slide2vec.utils.tiling_io import load_tiling_process_df, load_tiling_result_from_row

from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_encoder_precision

CACHE_METADATA_NAME = "cache_metadata.json"
MANIFEST_NAME = "manifest.csv"
PROCESS_LIST_NAME = "process_list.csv"
SCHEMA_VERSION = "v1"


def _resolve_encoder_precision(
    encoder_config: EncoderConfig,
    *,
    encoder_name: str | None = None,
) -> str:
    return resolve_encoder_precision(encoder_config, encoder_name=encoder_name)


@dataclass(frozen=True)
class BaseCacheResolution(ABC):
    key: str
    cache_dir: Path
    metadata_path: Path
    manifest_path: Path
    reused: bool
    complete: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CacheValidationResult:
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class FeatureCacheResolution(BaseCacheResolution):
    cache_kind: str
    features_dir: Path

    @property
    def empty_sample_ids(self) -> set[str]:
        return {str(s) for s in self.metadata.get("empty_sample_ids", [])}

    def missing_sample_ids(self) -> list[str]:
        expected = self.metadata["sample_ids"]
        empty = self.empty_sample_ids
        missing: list[str] = []
        for sample_id in expected:
            if str(sample_id) in empty:
                continue
            if not (self.features_dir / f"{sample_id}.pt").is_file():
                missing.append(str(sample_id))
        return missing


@dataclass(frozen=True)
class TilingCacheResolution(BaseCacheResolution):
    process_list_path: Path
    artifacts_dir: Path


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def dataset_manifest_rows(dataset: Dataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in sorted(dataset.sample_ids):
        sample = dataset.samples[sample_id]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path)
                if sample.mask_path is not None
                else None,
            }
        )
    return rows


def manifest_digest(manifest_rows: Iterable[dict[str, object]]) -> str:
    normalized = sorted(
        [
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "mask_path": row.get("mask_path"),
            }
            for row in manifest_rows
        ],
        key=lambda row: str(row["sample_id"]),
    )
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()[:16]


def preprocessing_signature(config: PreprocessingConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "requested_tile_size_px": config.requested_tile_size_px,
        "requested_spacing_um": config.requested_spacing_um,
        "requested_region_size_px": config.requested_region_size_px,
        "region_tile_multiple": config.region_tile_multiple,
        "read_tile_size_px": config.read_tile_size_px,
        "read_region_size_px": config.read_region_size_px,
        "tissue_method": config.tissue_method,
        "tissue_mask_tissue_value": config.tissue_mask_tissue_value,
        "tissue_threshold": config.tissue_threshold,
        "overlap": config.overlap,
        "seg_downsample": config.seg_downsample,
        "tolerance": config.tolerance,
        "ref_tile_size_px": config.ref_tile_size_px,
        "a_t": config.a_t,
        "hierarchical": config.hierarchical,
        "npatch": config.npatch,
        "hierarchical_patch_size_px": config.hierarchical_patch_size_px,
    }


def preprocessing_backend_provenance(
    *,
    requested_backend: str,
    loaded_tilings: Sequence[object],
) -> dict[str, Any]:
    backend_by_sample_id: dict[str, str] = {}
    for loaded in loaded_tilings:
        slide = getattr(loaded, "slide", None)
        sample_id = getattr(slide, "sample_id", getattr(loaded, "sample_id", None))
        if sample_id is None:
            continue
        backend = getattr(loaded, "backend", None)
        if backend is None:
            backend = getattr(loaded, "requested_backend", requested_backend)
        backend_by_sample_id[str(sample_id)] = str(backend)

    unique_backends = sorted(set(backend_by_sample_id.values()))
    actual_backend: str | None = unique_backends[0] if len(unique_backends) == 1 else None
    return {
        "requested_backend": str(requested_backend),
        "backend": actual_backend,
        "backend_by_sample_id": backend_by_sample_id,
    }


def probe_resolved_backends(
    *,
    dataset: Dataset,
    requested_backend: str,
) -> dict[str, str]:
    requested_backend = str(requested_backend)
    if requested_backend != "auto":
        return {
            sample_id: requested_backend
            for sample_id in sorted(dataset.sample_ids)
        }
    mapping: dict[str, str] = {}
    for sample_id in sorted(dataset.sample_ids):
        sample = dataset.samples[sample_id]
        selection = resolve_backend(
            requested_backend,
            wsi_path=sample.image_path,
            mask_path=sample.mask_path,
        )
        mapping[sample_id] = str(selection.backend)
    return mapping


def execution_signature(
    encoder_config: EncoderConfig,
    *,
    encoder_name: str | None = None,
    output_variant: str | None = None,
) -> dict[str, Any]:
    return {
        "precision": _resolve_encoder_precision(encoder_config, encoder_name=encoder_name),
        "input_size": encoder_config.input_size,
        "spacing_um": encoder_config.spacing_um,
        "output_variant": output_variant if output_variant is not None else encoder_config.output_variant,
    }


def build_tile_cache_key(
    *,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tile",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "tile_encoder_name": tile_encoder_name,
        "preprocessing": preprocessing_signature(preprocessing),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_slide_cache_key(
    *,
    dataset: Dataset,
    slide_encoder_name: str,
    tile_cache_key: str,
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "slide",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "slide_encoder_name": slide_encoder_name,
        "tile_cache_key": tile_cache_key,
        "execution": execution_signature(
            execution,
            encoder_name=slide_encoder_name,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_hierarchical_cache_key(
    *,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hierarchical",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "tile_encoder_name": tile_encoder_name,
        "preprocessing": preprocessing_signature(preprocessing),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_tiling_cache_key(
    *,
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tiling",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "preprocessing": preprocessing_signature(preprocessing),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def resolve_cache_root(
    cache_config: CacheConfig,
    *,
    feature_dir: Path | str,
    output_root: Path | str | None = None,
) -> Path:
    if cache_config.root_dir is not None:
        return Path(cache_config.root_dir)
    if output_root is not None:
        return Path(output_root) / "feature_cache"
    return Path(feature_dir).parent / "feature_cache"


def resolve_tiling_cache_root(
    cache_config: CacheConfig,
    *,
    tiling_dir: Path | str,
    output_root: Path | str | None = None,
) -> Path:
    feature_root = resolve_cache_root(
        cache_config,
        feature_dir=Path(tiling_dir).parent / "features",
        output_root=output_root,
    )
    return feature_root.parent / "tiling_cache"


def resolve_feature_payload_dir(path: Path | str) -> Path:
    """Resolve the directory containing feature .pt files.

    Handles soma cache dirs (cache_metadata.json + features/),
    slide2vec artifact dirs (slide_embeddings/, hierarchical_embeddings/, tile_embeddings/),
    and plain directories.
    """
    root = Path(path)
    if (root / "cache_metadata.json").is_file() and (root / "features").is_dir():
        return root / "features"
    for subdir in ("slide_embeddings", "hierarchical_embeddings", "tile_embeddings"):
        candidate = root / subdir
        if candidate.is_dir():
            return candidate
    return root


def _feature_dim_from_tensor(tensor: torch.Tensor) -> int:
    return int(tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1])


def _materialize_pt_artifact(*, artifact_path: Path, output_path: Path) -> torch.Tensor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    try:
        os.link(artifact_path, output_path)
    except OSError:
        shutil.copyfile(artifact_path, output_path)
        with contextlib.suppress(OSError):
            artifact_path.unlink()
    return torch.load(output_path, weights_only=True, map_location="cpu")


def write_cache_payload(
    artifacts: Sequence[object],
    *,
    feature_dir: Path,
) -> int | None:
    """Write slide2vec artifacts to a soma cache directory as .pt files."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_dim: int | None = None
    for artifact in artifacts:
        artifact_path = Path(artifact.path)
        output_path = feature_dir / f"{artifact.sample_id}.pt"
        if artifact_path.suffix != ".pt" or not artifact_path.is_file():
            raise ValueError(f"Expected a .pt artifact for cache materialization, got: {artifact_path}")
        tensor = _materialize_pt_artifact(
            artifact_path=artifact_path,
            output_path=output_path,
        )
        feature_dim = _feature_dim_from_tensor(tensor)
    return feature_dim


def build_tile_artifacts_from_cache_payload(
    *,
    features_dir: Path,
    loaded_tilings: Sequence[object],
    work_dir: Path,
) -> list[TileEmbeddingArtifact]:
    """Reconstruct TileEmbeddingArtifact objects from cached .pt files."""
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
            "feature_dim": _feature_dim_from_tensor(tensor),
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
                feature_dim=_feature_dim_from_tensor(tensor),
                num_tiles=int(tensor.shape[0]),
            )
        )
    return artifacts


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    value_str = str(value)
    if not value_str or value_str.lower() == "nan":
        return None
    return Path(value_str)


def _build_tiling_cache_metadata(
    *,
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    raw_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": "tiling",
        "cache_key": build_tiling_cache_key(
            dataset=dataset,
            preprocessing=preprocessing,
        ),
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "sample_ids": sorted(dataset.sample_ids),
        "preprocessing": preprocessing_signature(preprocessing),
        "requested_backend": str(backend_provenance["requested_backend"]),
        "backend": backend_provenance.get("backend"),
        "backend_by_sample_id": dict(backend_provenance["backend_by_sample_id"]),
    }
    if encoder_name is not None:
        metadata["resolved_by_encoder_name"] = str(encoder_name)
    if raw_preprocessing is not None:
        metadata["raw_preprocessing"] = raw_preprocessing
    return metadata


def _compare_tiling_metadata(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    ignore_keys = {"backend", "backend_by_sample_id", "resolved_by_encoder_name", "raw_preprocessing"}
    comparable_existing = {key: value for key, value in existing.items() if key not in ignore_keys}
    comparable_expected = {key: value for key, value in expected.items() if key not in ignore_keys}
    if comparable_existing != comparable_expected:
        raise ValueError("Tiling cache metadata mismatch")


def _tiling_cache_dir(cache_root: Path, key: str) -> Path:
    return cache_root / key


def _canonical_artifact_destination(
    *,
    sample_id: str,
    column_name: str,
    source_path: Path,
    artifacts_dir: Path,
) -> Path:
    suffix = "".join(source_path.suffixes) if source_path.suffixes else source_path.suffix
    stem = f"{sample_id}.{column_name}"
    return artifacts_dir / f"{stem}{suffix}"


def _copy_file_to_cache(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _clear_directory_for_stub(tiling_dir: Path) -> None:
    tiling_dir.mkdir(parents=True, exist_ok=True)
    for path in list(tiling_dir.iterdir()):
        if path.name in {PROCESS_LIST_NAME, "README.txt"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _write_tiling_stub_marker(*, tiling_dir: Path, cache_dir: Path) -> None:
    (tiling_dir / "README.txt").write_text(
        (
            "This directory is a cache-backed tiling location placeholder.\n"
            f"Actual tiling payloads are stored under: {cache_dir.resolve()}\n"
            "Configure CacheConfig.root_dir to control the shared cache location.\n"
        ),
        encoding="utf-8",
    )


def _validate_tiling_cache_contents(
    *,
    dataset: Dataset,
    process_list_path: Path,
    artifacts_dir: Path,
    metadata: dict[str, Any],
    preprocessing: PreprocessingConfig,
    expected_backend_provenance: dict[str, Any] | None,
) -> CacheValidationResult:
    if not process_list_path.is_file():
        return CacheValidationResult(complete=False, reason="missing process_list.csv")
    try:
        process_df = load_tiling_process_df(process_list_path)
    except Exception:
        return CacheValidationResult(complete=False, reason="process_list.csv could not be loaded")
    rows_by_sample_id = {
        str(row["sample_id"]): row
        for row in process_df.to_dict("records")
    }
    for sample_id in metadata["sample_ids"]:
        sample = dataset.samples[str(sample_id)]
        row = rows_by_sample_id.get(str(sample_id))
        if row is None or row.get("tiling_status") != "success":
            return CacheValidationResult(complete=False, reason=f"invalid tiling row for {sample_id}")
        for column_name in (
            "coordinates_npz_path",
            "coordinates_meta_path",
            "tiles_tar_path",
            "mask_preview_path",
            "tiling_preview_path",
        ):
            candidate = _optional_path(row.get(column_name))
            if candidate is None:
                continue
            if not candidate.is_file():
                return CacheValidationResult(complete=False, reason=f"missing artifact for {sample_id}")
            try:
                candidate.resolve().relative_to(artifacts_dir.resolve())
            except ValueError:
                return CacheValidationResult(
                    complete=False,
                    reason=f"artifact path escapes cache entry for {sample_id}",
                )
        try:
            tiling_result = load_tiling_result_from_row(row)
            validate_tiling_result_provenance(
                tiling_result,
                sample_id=sample.sample_id,
                image_path=sample.image_path,
                mask_path=sample.mask_path,
                tissue_mask_tissue_value=(
                    int(preprocessing.tissue_mask_tissue_value)
                    if sample.mask_path is not None
                    else None
                ),
            )
        except Exception:
            return CacheValidationResult(complete=False, reason=f"invalid tiling provenance for {sample_id}")
        if int(getattr(tiling_result, "requested_tile_size_px", -1)) != int(preprocessing.requested_tile_size_px):
            return CacheValidationResult(complete=False, reason=f"tile size mismatch for {sample_id}")
        if float(getattr(tiling_result, "requested_spacing_um", -1.0)) != float(preprocessing.requested_spacing_um):
            return CacheValidationResult(complete=False, reason=f"spacing mismatch for {sample_id}")
        expected_backend = metadata.get("backend_by_sample_id", {}).get(str(sample_id))
        actual_backend = str(getattr(tiling_result, "backend", row.get("backend")))
        if expected_backend is not None and str(expected_backend) != actual_backend:
            return CacheValidationResult(complete=False, reason=f"backend mismatch for {sample_id}")
    if expected_backend_provenance is None:
        return CacheValidationResult(complete=True)
    if dict(metadata.get("backend_by_sample_id", {})) != dict(expected_backend_provenance["backend_by_sample_id"]):
        return CacheValidationResult(complete=False, reason="backend mapping mismatch")
    return CacheValidationResult(complete=True)


def resolve_tiling_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    raw_preprocessing: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> TilingCacheResolution:
    metadata = _build_tiling_cache_metadata(
        dataset=dataset,
        preprocessing=preprocessing,
        backend_provenance=backend_provenance,
        encoder_name=encoder_name,
        raw_preprocessing=raw_preprocessing,
    )
    cache_dir = _tiling_cache_dir(cache_root, str(metadata["cache_key"]))
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    process_list_path = cache_dir / PROCESS_LIST_NAME
    artifacts_dir = cache_dir / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        _compare_tiling_metadata(existing, metadata)
        validation = _validate_tiling_cache_contents(
            dataset=dataset,
            process_list_path=process_list_path,
            artifacts_dir=artifacts_dir,
            metadata=existing,
            preprocessing=preprocessing,
            expected_backend_provenance=backend_provenance,
        )
        _emit_cache_state_log(
            cache_label="tiling",
            cache_dir=cache_dir,
            complete=validation.complete,
            complete_state=complete_state,
            reason=validation.reason,
        )
        return TilingCacheResolution(
            key=str(existing["cache_key"]),
            cache_dir=cache_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            reused=validation.complete,
            complete=validation.complete,
            metadata=existing,
            process_list_path=process_list_path,
            artifacts_dir=artifacts_dir,
        )

    _write_manifest(manifest_path, dataset_manifest_rows(dataset))
    _write_metadata(metadata_path, metadata)
    _emit_cache_state_log(
        cache_label="tiling",
        cache_dir=cache_dir,
        complete=False,
        complete_state=complete_state,
        reason="initializing",
    )
    return TilingCacheResolution(
        key=str(metadata["cache_key"]),
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        reused=False,
        complete=False,
        metadata=metadata,
        process_list_path=process_list_path,
        artifacts_dir=artifacts_dir,
    )


def write_tiling_cache_payload(
    *,
    live_dir: Path,
    cache_resolution: TilingCacheResolution,
) -> None:
    process_df = load_tiling_process_df(live_dir / PROCESS_LIST_NAME)
    rows: list[dict[str, Any]] = []
    for row in process_df.to_dict("records"):
        rewritten = dict(row)
        sample_id = str(row["sample_id"])
        for column_name in (
            "coordinates_npz_path",
            "coordinates_meta_path",
            "tiles_tar_path",
            "mask_preview_path",
            "tiling_preview_path",
        ):
            source_path = _optional_path(row.get(column_name))
            if source_path is None:
                rewritten[column_name] = None
                continue
            destination = _canonical_artifact_destination(
                sample_id=sample_id,
                column_name=column_name,
                source_path=source_path,
                artifacts_dir=cache_resolution.artifacts_dir,
            )
            _copy_file_to_cache(source=source_path, destination=destination)
            rewritten[column_name] = str(destination.resolve())
        rows.append(rewritten)
    with cache_resolution.process_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(process_df.columns))
        writer.writeheader()
        writer.writerows(rows)


def write_tiling_cache_stub(
    tiling_dir: Path | str,
    *,
    cache_resolution: TilingCacheResolution,
) -> None:
    tiling_dir = Path(tiling_dir).resolve()
    _clear_directory_for_stub(tiling_dir)
    shutil.copyfile(cache_resolution.process_list_path, tiling_dir / PROCESS_LIST_NAME)
    _write_tiling_stub_marker(tiling_dir=tiling_dir, cache_dir=cache_resolution.cache_dir)


def _cache_dir(cache_root: Path, cache_kind: str, key: str) -> Path:
    return cache_root / cache_kind / key


def _build_tile_cache_metadata(
    *,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tile",
        "cache_key": key,
        "encoder_name": tile_encoder_name,
        "encoder_level": "tile",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "sample_ids": sorted(dataset.sample_ids),
        "preprocessing": preprocessing_signature(preprocessing),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
        "feature_rank": 2,
        "feature_dim": None,
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_slide_cache_metadata(
    *,
    dataset: Dataset,
    slide_encoder_name: str,
    tile_encoder_name: str,
    tile_cache_key: str,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_slide_cache_key(
        dataset=dataset,
        slide_encoder_name=slide_encoder_name,
        tile_cache_key=tile_cache_key,
        execution=execution,
        output_variant=output_variant,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "slide",
        "cache_key": key,
        "encoder_name": slide_encoder_name,
        "encoder_level": "slide",
        "tile_encoder": tile_encoder_name,
        "tile_cache_key": tile_cache_key,
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "sample_ids": sorted(dataset.sample_ids),
        "execution": execution_signature(
            execution,
            encoder_name=slide_encoder_name,
            output_variant=output_variant,
        ),
        "feature_rank": 1,
        "feature_dim": None,
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_hierarchical_cache_metadata(
    *,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_hierarchical_cache_key(
        dataset=dataset,
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hierarchical",
        "cache_key": key,
        "encoder_name": tile_encoder_name,
        "encoder_level": "tile",
        "manifest_digest": manifest_digest(dataset_manifest_rows(dataset)),
        "sample_ids": sorted(dataset.sample_ids),
        "preprocessing": preprocessing_signature(preprocessing),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
        "feature_rank": 3,
        "feature_dim": None,
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(metadata)
    comparable.pop("feature_dim", None)
    comparable.pop("empty_sample_ids", None)
    return comparable


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "mask_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_feature_cache_contents(
    *,
    features_dir: Path,
    metadata: dict[str, Any],
) -> CacheValidationResult:
    expected_rank = int(metadata["feature_rank"])
    sample_ids = [str(s) for s in metadata["sample_ids"]]
    sample_ids_set = set(sample_ids)
    feature_dim = metadata.get("feature_dim")
    empty_sample_ids = {str(s) for s in metadata.get("empty_sample_ids", [])}
    if not empty_sample_ids.issubset(sample_ids_set):
        return CacheValidationResult(complete=False, reason="empty sample metadata mismatch")
    for sample_id in sample_ids:
        if sample_id in empty_sample_ids:
            if (features_dir / f"{sample_id}.pt").is_file():
                return CacheValidationResult(complete=False, reason=f"unexpected feature for empty sample {sample_id}")
            continue
        path = features_dir / f"{sample_id}.pt"
        if not path.is_file():
            return CacheValidationResult(complete=False, reason=f"missing feature for {sample_id}")
        try:
            tensor = torch.load(path, weights_only=True, map_location="cpu")
        except Exception:
            return CacheValidationResult(complete=False, reason=f"corrupt feature for {sample_id}")
        if tensor.ndim != expected_rank:
            return CacheValidationResult(complete=False, reason=f"rank mismatch for {sample_id}")
        if feature_dim is not None:
            inferred = tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1]
            if int(feature_dim) != int(inferred):
                return CacheValidationResult(complete=False, reason=f"dim mismatch for {sample_id}")
    return CacheValidationResult(complete=True)


def _emit_cache_state_log(
    *,
    cache_label: str,
    cache_dir: Path,
    complete: bool,
    complete_state: str = "hit",
    reason: str | None = None,
) -> None:
    cache_path = str(cache_dir.resolve())
    reporter = slide2vec_progress.get_progress_reporter()
    rich_viz = hasattr(reporter, "console") and hasattr(reporter, "progress")
    if complete:
        status = complete_state
        if rich_viz:
            status = f"\x1b[1;32m{complete_state}\x1b[0m"
        message = f"✓ {cache_label} cache {status}: {cache_path}"
    else:
        status = "miss"
        if rich_viz:
            status = "\x1b[1;31mmiss\x1b[0m"
        message = f"✗ {cache_label} cache {status}: {cache_path}"
        if reason is not None:
            message = f"{message} ({reason})"
    slide2vec_progress.emit_progress_log(message)


def _resolve_cache(
    *,
    cache_root: Path,
    cache_kind: str,
    key: str,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, object]],
    initial_reason: str | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    cache_dir = _cache_dir(cache_root, cache_kind, key)
    features_dir = cache_dir / "features"
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        if _comparable_metadata(existing) != _comparable_metadata(metadata):
            raise ValueError(f"Cache metadata mismatch for {cache_dir}")
        validation = _validate_feature_cache_contents(features_dir=features_dir, metadata=existing)
        _emit_cache_state_log(
            cache_label="feature",
            cache_dir=cache_dir,
            complete=validation.complete,
            complete_state=complete_state,
            reason=validation.reason,
        )
        return FeatureCacheResolution(
            key=key,
            cache_dir=cache_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            reused=validation.complete,
            complete=validation.complete,
            metadata=existing,
            cache_kind=cache_kind,
            features_dir=features_dir,
        )

    _write_manifest(manifest_path, manifest_rows)
    _write_metadata(metadata_path, metadata)
    _emit_cache_state_log(
        cache_label="feature",
        cache_dir=cache_dir,
        complete=False,
        reason=initial_reason,
    )
    return FeatureCacheResolution(
        key=key,
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        reused=False,
        complete=False,
        metadata=metadata,
        cache_kind=cache_kind,
        features_dir=features_dir,
    )


def resolve_tile_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    metadata = _build_tile_cache_metadata(
        dataset=dataset,
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="tile",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
    )


def resolve_slide_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    slide_encoder_name: str,
    tile_encoder_name: str,
    tile_cache_key: str,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    metadata = _build_slide_cache_metadata(
        dataset=dataset,
        slide_encoder_name=slide_encoder_name,
        tile_encoder_name=tile_encoder_name,
        tile_cache_key=tile_cache_key,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="slide",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
    )


def resolve_hierarchical_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    metadata = _build_hierarchical_cache_metadata(
        dataset=dataset,
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="hierarchical",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
    )


def record_feature_dim(resolution: FeatureCacheResolution, feature_dim: int) -> None:
    metadata = dict(resolution.metadata)
    metadata["feature_dim"] = int(feature_dim)
    _write_metadata(resolution.metadata_path, metadata)


def record_empty_sample_ids(resolution: FeatureCacheResolution, empty_sample_ids: Sequence[str]) -> None:
    metadata = dict(resolution.metadata)
    metadata["empty_sample_ids"] = sorted({str(sample_id) for sample_id in empty_sample_ids})
    _write_metadata(resolution.metadata_path, metadata)
