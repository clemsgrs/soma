"""Shared feature-cache utilities for tile and slide artifacts."""

from __future__ import annotations

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
from slide2vec.artifacts import TileEmbeddingArtifact

from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_encoder_precision

CACHE_METADATA_NAME = "cache_metadata.json"
MANIFEST_NAME = "manifest.csv"
SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class CacheResolution:
    kind: str
    key: str
    cache_dir: Path
    metadata_path: Path
    manifest_path: Path
    features_dir: Path
    reused: bool
    complete: bool
    metadata: dict[str, Any]

    def missing_sample_ids(self) -> list[str]:
        expected = self.metadata["sample_ids"]
        missing: list[str] = []
        for sample_id in expected:
            if not (self.features_dir / f"{sample_id}.pt").is_file():
                missing.append(str(sample_id))
        return missing


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
        "target_tile_size_px": config.target_tile_size_px,
        "target_spacing_um": config.target_spacing_um,
        "target_region_size_px": config.target_region_size_px,
        "region_tile_multiple": config.region_tile_multiple,
        "effective_tile_size_px": config.effective_tile_size_px,
        "effective_region_size_px": config.effective_region_size_px,
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


def execution_signature(
    encoder_config: EncoderConfig,
    *,
    encoder_name: str | None = None,
    output_variant: str | None = None,
) -> dict[str, Any]:
    return {
        "precision": resolve_encoder_precision(encoder_config, encoder_name=encoder_name),
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


def resolve_cache_root(
    cache_config: CacheConfig,
    *,
    output_dir: Path | str,
) -> Path:
    if cache_config.root_dir is not None:
        return Path(cache_config.root_dir)
    return Path(output_dir).parent / "feature_cache"


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
    output_dir: Path,
) -> int | None:
    """Write slide2vec artifacts to a soma cache directory as .pt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dim: int | None = None
    for artifact in artifacts:
        artifact_path = Path(artifact.path)
        output_path = output_dir / f"{artifact.sample_id}.pt"
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


def _cache_dir(cache_root: Path, kind: str, key: str) -> Path:
    return cache_root / kind / key


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


def _is_complete(features_dir: Path, metadata: dict[str, Any]) -> bool:
    expected_rank = int(metadata["feature_rank"])
    sample_ids = metadata["sample_ids"]
    feature_dim = metadata.get("feature_dim")
    for sample_id in sample_ids:
        path = features_dir / f"{sample_id}.pt"
        if not path.is_file():
            return False
        try:
            tensor = torch.load(path, weights_only=True, map_location="cpu")
        except Exception:
            return False
        if tensor.ndim != expected_rank:
            return False
        if feature_dim is not None:
            inferred = tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1]
            if int(feature_dim) != int(inferred):
                return False
    return True


def _resolve_cache(
    *,
    cache_root: Path,
    kind: str,
    key: str,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, object]],
) -> CacheResolution:
    cache_dir = _cache_dir(cache_root, kind, key)
    features_dir = cache_dir / "features"
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        if _comparable_metadata(existing) != _comparable_metadata(metadata):
            raise ValueError(f"Cache metadata mismatch for {cache_dir}")
        complete = _is_complete(features_dir, existing)
        return CacheResolution(
            kind=kind,
            key=key,
            cache_dir=cache_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            features_dir=features_dir,
            reused=complete,
            complete=complete,
            metadata=existing,
        )

    _write_manifest(manifest_path, manifest_rows)
    _write_metadata(metadata_path, metadata)
    return CacheResolution(
        kind=kind,
        key=key,
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        features_dir=features_dir,
        reused=False,
        complete=False,
        metadata=metadata,
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
) -> CacheResolution:
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
        kind="tile",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
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
) -> CacheResolution:
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
        kind="slide",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
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
) -> CacheResolution:
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
        kind="hierarchical",
        key=metadata["cache_key"],
        metadata=metadata,
        manifest_rows=dataset_manifest_rows(dataset),
    )


def record_feature_dim(resolution: CacheResolution, feature_dim: int) -> None:
    metadata = dict(resolution.metadata)
    metadata["feature_dim"] = int(feature_dim)
    _write_metadata(resolution.metadata_path, metadata)
