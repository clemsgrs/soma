"""Shared cache utilities for feature and tiling artifacts."""

from __future__ import annotations

from abc import ABC
import csv
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from hs2p.wsi.reader import resolve_backend
from slide2vec.artifacts import TileEmbeddingArtifact
import slide2vec.progress as slide2vec_progress
from slide2vec.utils.tiling_io import load_tiling_process_df

from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_encoder_precision

CACHE_METADATA_NAME = "cache_metadata.json"
MANIFEST_NAME = "manifest.csv"
PROCESS_LIST_NAME = "process_list.csv"
SCHEMA_VERSION = "v1"
_FEATURE_TYPE_TO_RANK = {
    "tile": 1,
    "bag": 2,
    "slide": 1,
    "patient": 1,
    "hierarchical": 3,
}

_CACHE_KIND_TO_FEATURES_SUBDIR = {
    "tile": "tile_embeddings",
    "slide": "slide_embeddings",
    "patient": "patient_embeddings",
    "hierarchical": "hierarchical_embeddings",
}


def _features_subdir_for_kind(cache_kind: str) -> str:
    try:
        return _CACHE_KIND_TO_FEATURES_SUBDIR[cache_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache_kind '{cache_kind}' for features_dir resolution") from exc


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
    cache_ids: tuple[str, ...]
    cache_stem_by_id: dict[str, str]

    @property
    def empty_sample_ids(self) -> set[str]:
        return {str(s) for s in self.metadata.get("empty_sample_ids", [])}

    def feature_path_for_id(self, cache_id: str) -> Path:
        return self.features_dir / f"{cache_id}.pt"

    def missing_sample_ids(self) -> list[str]:
        expected = self.cache_ids
        empty = self.empty_sample_ids
        cached_signature_by_id = {
            str(cache_id): str(signature)
            for cache_id, signature in self.metadata.get("sample_identity_signature_by_id", {}).items()
        }
        missing: list[str] = []
        for cache_id in expected:
            cache_id = str(cache_id)
            if cache_id in empty:
                continue
            expected_signature = str(self.cache_stem_by_id[cache_id])
            cached_signature = cached_signature_by_id.get(cache_id)
            if cached_signature is None or cached_signature != expected_signature:
                missing.append(cache_id)
                continue
            if not self.feature_path_for_id(cache_id).is_file():
                missing.append(cache_id)
        return missing


@dataclass(frozen=True)
class TilingCacheResolution(BaseCacheResolution):
    process_list_path: Path
    artifacts_dir: Path
    cache_ids: tuple[str, ...]
    cache_stem_by_id: dict[str, str]


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


def sample_identity_signature(
    *,
    sample_id: str,
    image_path: Path | str,
    mask_path: Path | str | None,
) -> str:
    payload = {
        "sample_id": str(sample_id),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _sample_cache_stem(*, sample_signature: str, identity_payload: dict[str, Any]) -> str:
    payload = dict(identity_payload)
    payload["sample_signature"] = str(sample_signature)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


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
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig | None = None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    feature_type: str = "bag",
) -> str:
    if feature_type not in _FEATURE_TYPE_TO_RANK:
        raise ValueError(f"Unsupported feature_type '{feature_type}' for tile cache key")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tile",
        "tile_encoder_name": tile_encoder_name,
        "feature_type": str(feature_type),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
    }
    if preprocessing is not None:
        payload["preprocessing"] = preprocessing_signature(preprocessing)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_slide_cache_key(
    *,
    slide_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "slide",
        "slide_encoder_name": slide_encoder_name,
        "tile_dependency_signature": tile_dependency_signature,
        "execution": execution_signature(
            execution,
            encoder_name=slide_encoder_name,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_patient_cache_key(
    *,
    patient_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "patient",
        "patient_encoder_name": patient_encoder_name,
        "tile_dependency_signature": tile_dependency_signature,
        "execution": execution_signature(
            execution,
            encoder_name=patient_encoder_name,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_hierarchical_cache_key(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hierarchical",
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
    preprocessing: PreprocessingConfig,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tiling",
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
    for subdir in ("patient_embeddings", "slide_embeddings", "hierarchical_embeddings", "tile_embeddings"):
        candidate = root / subdir
        if candidate.is_dir():
            return candidate
    return root


def _feature_dim_from_tensor(tensor: torch.Tensor) -> int:
    return int(tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1])


def _materialize_pt_artifact(*, artifact_path: Path, output_path: Path) -> torch.Tensor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        try:
            os.link(artifact_path, output_path)
        except OSError:
            shutil.copyfile(artifact_path, output_path)
            with contextlib.suppress(OSError):
                artifact_path.unlink()
    return torch.load(output_path, weights_only=True, map_location="cpu")


def write_feature_payload(
    *,
    feature_dir: Path,
    sample_id: str,
    tensor: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a sample-level feature tensor directly into a cache feature directory."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{sample_id}.pt"
    with tempfile.NamedTemporaryFile(prefix=f".{sample_id}.", suffix=".pt", dir=feature_dir, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(tensor.detach().cpu(), tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
    if metadata is not None:
        metadata_path = feature_dir / f"{sample_id}.meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_cache_payload(
    artifacts: Sequence[object],
    *,
    feature_dir: Path,
    id_attr: str = "sample_id",
) -> int | None:
    """Write slide2vec artifacts to a soma cache directory as .pt files.

    Args:
        id_attr: Attribute on each artifact used as the filename stem. Defaults to
            ``"sample_id"``; pass ``"patient_id"`` for PatientEmbeddingArtifact.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_dim: int | None = None
    for artifact in artifacts:
        artifact_path = Path(artifact.path)
        artifact_id = getattr(artifact, id_attr)
        output_path = feature_dir / f"{artifact_id}.pt"
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
    feature_path_by_sample_id: dict[str, Path] | None = None,
) -> list[TileEmbeddingArtifact]:
    """Reconstruct TileEmbeddingArtifact objects from cached .pt files."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[TileEmbeddingArtifact] = []
    for loaded in loaded_tilings:
        sample_id = str(loaded.slide.sample_id)
        if feature_path_by_sample_id is not None:
            feature_path = feature_path_by_sample_id[sample_id]
        else:
            feature_path = features_dir / f"{sample_id}.pt"
        tensor = torch.load(feature_path, weights_only=True, map_location="cpu")
        metadata_path = work_dir / f"{sample_id}.meta.json"
        metadata = {
            "sample_id": sample_id,
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
                sample_id=sample_id,
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
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    requested_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": "tiling",
        "cache_key": build_tiling_cache_key(
            preprocessing=preprocessing,
        ),
        "preprocessing": preprocessing_signature(preprocessing),
        "requested_backend": str(backend_provenance["requested_backend"]),
    }
    if encoder_name is not None:
        metadata["resolved_by_encoder_name"] = str(encoder_name)
    if requested_preprocessing is not None:
        metadata["requested_preprocessing"] = requested_preprocessing
    return metadata


def _format_cache_metadata_mismatch(
    *,
    cache_label: str,
    cache_dir: Path,
    existing: dict[str, Any],
    expected: dict[str, Any],
    ignore_keys: set[str] | frozenset[str] = frozenset(),
) -> str:
    base_ignore_keys = {
        "backend",
        "backend_by_sample_id",
        "resolved_by_encoder_name",
        "requested_preprocessing",
    }
    effective_ignore_keys = base_ignore_keys | set(ignore_keys)
    comparable_existing = {
        key: value for key, value in existing.items() if key not in effective_ignore_keys
    }
    comparable_expected = {
        key: value for key, value in expected.items() if key not in effective_ignore_keys
    }
    if comparable_existing == comparable_expected:
        return ""

    all_keys = sorted(set(comparable_existing) | set(comparable_expected))
    missing_keys: list[str] = []
    extra_keys: list[str] = []
    changed_keys: list[str] = []
    for key in all_keys:
        if key not in comparable_existing:
            missing_keys.append(f"{key}={comparable_expected[key]!r}")
        elif key not in comparable_expected:
            extra_keys.append(f"{key}={comparable_existing[key]!r}")
        elif comparable_existing[key] != comparable_expected[key]:
            changed_keys.append(
                f"{key}: existing={comparable_existing[key]!r}, expected={comparable_expected[key]!r}"
            )

    sections: list[str] = []
    if missing_keys:
        sections.append(f"missing=[{', '.join(missing_keys)}]")
    if extra_keys:
        sections.append(f"extra=[{', '.join(extra_keys)}]")
    if changed_keys:
        sections.append(f"changed=[{'; '.join(changed_keys)}]")
    details = "; ".join(sections)
    return f"{cache_label} metadata mismatch for {cache_dir}: {details}"


def _tiling_cache_dir(cache_root: Path, key: str) -> Path:
    return cache_root / key


def _canonical_artifact_destination(
    *,
    artifact_stem: str,
    column_name: str,
    source_path: Path,
    artifacts_dir: Path,
) -> Path:
    suffix = "".join(source_path.suffixes) if source_path.suffixes else source_path.suffix
    stem = f"{artifact_stem}.{column_name}"
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
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
    preprocessing: PreprocessingConfig,
    expected_backend_provenance: dict[str, Any] | None,
) -> CacheValidationResult:
    if not process_list_path.is_file():
        return CacheValidationResult(complete=False, reason="missing process_list.csv")
    try:
        process_df = load_tiling_process_df(process_list_path)
    except Exception:
        return CacheValidationResult(complete=False, reason="process_list.csv could not be loaded")
    rows = process_df.to_dict("records")
    rows_by_stem: dict[str, dict[str, Any]] = {}
    for row in rows:
        stem = row.get("sample_cache_stem")
        if stem is None or str(stem).strip() == "" or str(stem).lower() == "nan":
            stem = row.get("sample_id")
        if stem is None:
            continue
        rows_by_stem[str(stem)] = row

    for sample_id in cache_ids:
        sample_id = str(sample_id)
        sample = dataset.samples[sample_id]
        row = rows_by_stem.get(str(cache_stem_by_id[sample_id]))
        if row is None:
            row = rows_by_stem.get(sample_id)
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
        row_image_path = row.get("image_path")
        if row_image_path is not None and str(row_image_path) != str(sample.image_path):
            return CacheValidationResult(complete=False, reason=f"image path mismatch for {sample_id}")
        row_mask_path = row.get("mask_path")
        expected_mask_path = "" if sample.mask_path is None else str(sample.mask_path)
        if row_mask_path is not None:
            row_mask_str = str(row_mask_path)
            if row_mask_str.lower() == "nan":
                row_mask_str = ""
            if row_mask_str != expected_mask_path:
                return CacheValidationResult(complete=False, reason=f"mask path mismatch for {sample_id}")
        expected_requested_tile_size_px = (
            preprocessing.requested_region_size_px
            if preprocessing.requested_region_size_px is not None
            else preprocessing.requested_tile_size_px
        )
        row_tile_size = row.get("requested_tile_size_px")
        if row_tile_size is not None and str(row_tile_size).strip() not in {"", "nan", "NaN"}:
            if int(float(row_tile_size)) != int(expected_requested_tile_size_px):
                return CacheValidationResult(complete=False, reason=f"tile size mismatch for {sample_id}")
        row_spacing = row.get("requested_spacing_um")
        if row_spacing is not None and str(row_spacing).strip() not in {"", "nan", "NaN"}:
            if float(row_spacing) != float(preprocessing.requested_spacing_um):
                return CacheValidationResult(complete=False, reason=f"spacing mismatch for {sample_id}")
        expected_backend = None
        if expected_backend_provenance is not None:
            expected_backend = expected_backend_provenance.get("backend_by_sample_id", {}).get(str(sample_id))
        actual_backend = row.get("backend")
        if expected_backend is not None and str(expected_backend) != str(actual_backend):
            return CacheValidationResult(complete=False, reason=f"backend mismatch for {sample_id}")
    return CacheValidationResult(complete=True)


def resolve_tiling_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    requested_preprocessing: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> TilingCacheResolution:
    metadata = _build_tiling_cache_metadata(
        preprocessing=preprocessing,
        backend_provenance=backend_provenance,
        encoder_name=encoder_name,
        requested_preprocessing=requested_preprocessing,
    )
    cache_ids = tuple(sorted(dataset.sample_ids))
    cache_stem_by_id = _sample_stems_for_tiling(
        dataset=dataset,
        cache_key=str(metadata["cache_key"]),
    )
    cache_dir = _tiling_cache_dir(cache_root, str(metadata["cache_key"]))
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    process_list_path = cache_dir / PROCESS_LIST_NAME
    artifacts_dir = cache_dir / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _emit_cache_resolve_log(
        cache_label="tiling",
        cache_dir=cache_dir,
        key=str(metadata["cache_key"]),
        scope_name="samples",
        scope_count=len(cache_ids),
    )

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        mismatch_message = _format_cache_metadata_mismatch(
            cache_label="Tiling cache",
            cache_dir=cache_dir,
            existing=existing,
            expected=metadata,
        )
        if mismatch_message:
            raise ValueError(mismatch_message)
        validation = _validate_tiling_cache_contents(
            dataset=dataset,
            process_list_path=process_list_path,
            artifacts_dir=artifacts_dir,
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
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
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
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
        cache_ids=cache_ids,
        cache_stem_by_id=cache_stem_by_id,
    )


def write_tiling_cache_payload(
    *,
    live_dir: Path,
    cache_resolution: TilingCacheResolution,
) -> None:
    process_df = load_tiling_process_df(live_dir / PROCESS_LIST_NAME)
    fieldnames = list(process_df.columns)
    if "sample_cache_stem" not in fieldnames:
        try:
            sample_idx = fieldnames.index("sample_id")
            fieldnames.insert(sample_idx + 1, "sample_cache_stem")
        except ValueError:
            fieldnames.insert(0, "sample_cache_stem")

    rows_by_stem: dict[str, dict[str, Any]] = {}
    if cache_resolution.process_list_path.is_file():
        with cache_resolution.process_list_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stem = row.get("sample_cache_stem")
                if stem is None or str(stem).strip() == "" or str(stem).lower() == "nan":
                    stem = row.get("sample_id")
                if stem is None:
                    continue
                rows_by_stem[str(stem)] = dict(row)

    for row in process_df.to_dict("records"):
        rewritten = dict(row)
        sample_id = str(row["sample_id"])
        sample_cache_stem = cache_resolution.cache_stem_by_id.get(sample_id)
        if sample_cache_stem is None:
            continue
        rewritten["sample_cache_stem"] = sample_cache_stem
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
                artifact_stem=sample_cache_stem,
                column_name=column_name,
                source_path=source_path,
                artifacts_dir=cache_resolution.artifacts_dir,
            )
            _copy_file_to_cache(source=source_path, destination=destination)
            rewritten[column_name] = str(destination.resolve())
        rows_by_stem[sample_cache_stem] = rewritten

    extra_columns: list[str] = []
    for row in rows_by_stem.values():
        for key in row:
            if key not in fieldnames and key not in extra_columns:
                extra_columns.append(key)
    resolved_fieldnames = [*fieldnames, *extra_columns]
    rows = [rows_by_stem[key] for key in sorted(rows_by_stem)]
    with cache_resolution.process_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
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
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig | None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    feature_type: str = "bag",
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if feature_type not in _FEATURE_TYPE_TO_RANK:
        raise ValueError(f"Unsupported feature_type '{feature_type}' for tile cache metadata")
    key = build_tile_cache_key(
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        feature_type=feature_type,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "tile",
        "cache_key": key,
        "encoder_name": tile_encoder_name,
        "encoder_level": "tile",
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
        "feature_type": str(feature_type),
        "feature_dim": None,
        "sample_identity_signature_by_id": {},
    }
    if preprocessing is not None:
        metadata["preprocessing"] = preprocessing_signature(preprocessing)
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_slide_cache_metadata(
    *,
    slide_encoder_name: str,
    tile_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_slide_cache_key(
        slide_encoder_name=slide_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
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
        "tile_dependency_signature": dict(tile_dependency_signature),
        "execution": execution_signature(
            execution,
            encoder_name=slide_encoder_name,
            output_variant=output_variant,
        ),
        "feature_type": "slide",
        "feature_dim": None,
        "sample_identity_signature_by_id": {},
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_patient_cache_metadata(
    *,
    patient_encoder_name: str,
    tile_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_patient_cache_key(
        patient_encoder_name=patient_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "patient",
        "cache_key": key,
        "encoder_name": patient_encoder_name,
        "encoder_level": "patient",
        "tile_encoder": tile_encoder_name,
        "tile_dependency_signature": dict(tile_dependency_signature),
        "execution": execution_signature(
            execution,
            encoder_name=patient_encoder_name,
            output_variant=output_variant,
        ),
        "feature_type": "patient",
        "feature_dim": None,
        "sample_identity_signature_by_id": {},
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_hierarchical_cache_metadata(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_hierarchical_cache_key(
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
        "preprocessing": preprocessing_signature(preprocessing),
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            output_variant=output_variant,
        ),
        "feature_type": "hierarchical",
        "feature_dim": None,
        "sample_identity_signature_by_id": {},
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(metadata)
    comparable.pop("feature_dim", None)
    comparable.pop("empty_sample_ids", None)
    comparable.pop("sample_identity_signature_by_id", None)
    return comparable


def _sample_identity_payload(dataset: Dataset) -> dict[str, str]:
    return {
        sample_id: sample_identity_signature(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
        )
        for sample_id, sample in dataset.samples.items()
    }


def _sample_stems_for_kind(
    *,
    dataset: Dataset,
    cache_kind: str,
    static_identity_payload: dict[str, Any],
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(dataset)
    return {
        sample_id: _sample_cache_stem(
            sample_signature=signature_by_sample_id[sample_id],
            identity_payload={
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": cache_kind,
                **static_identity_payload,
            },
        )
        for sample_id in sorted(dataset.sample_ids)
    }


def _patient_stems_for_kind(
    *,
    dataset: Dataset,
    cache_kind: str,
    static_identity_payload: dict[str, Any],
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(dataset)
    sample_ids_by_patient: dict[str, list[str]] = {}
    for sample_id in sorted(dataset.sample_ids):
        record = dataset.samples[sample_id]
        if record.patient_id is None:
            continue
        sample_ids_by_patient.setdefault(str(record.patient_id), []).append(sample_id)
    stems: dict[str, str] = {}
    for patient_id, patient_sample_ids in sample_ids_by_patient.items():
        patient_sample_signatures = sorted(signature_by_sample_id[sample_id] for sample_id in patient_sample_ids)
        stems[patient_id] = _sample_cache_stem(
            sample_signature=hashlib.sha256(
                _canonical_json({"patient_sample_signatures": patient_sample_signatures}).encode("utf-8")
            ).hexdigest()[:16],
            identity_payload={
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": cache_kind,
                "patient_id": str(patient_id),
                **static_identity_payload,
            },
        )
    return stems


def _sample_stems_for_tiling(
    *,
    dataset: Dataset,
    cache_key: str,
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(dataset)
    return {
        sample_id: _sample_cache_stem(
            sample_signature=signature_by_sample_id[sample_id],
            identity_payload={
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "tiling_sample",
                "tiling_cache_key": str(cache_key),
            },
        )
        for sample_id in sorted(dataset.sample_ids)
    }


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
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
) -> tuple[CacheValidationResult, int, int]:
    feature_type = str(metadata.get("feature_type", ""))
    if feature_type not in _FEATURE_TYPE_TO_RANK:
        return CacheValidationResult(complete=False, reason="unsupported feature_type metadata"), 0, 0
    empty_sample_ids = {str(s) for s in metadata.get("empty_sample_ids", [])}
    expected_ids = {str(cache_id) for cache_id in cache_ids}
    if not empty_sample_ids.issubset(expected_ids):
        return CacheValidationResult(complete=False, reason="empty sample metadata mismatch"), 0, 0
    cached_signature_by_id = {
        str(cache_id): str(signature)
        for cache_id, signature in metadata.get("sample_identity_signature_by_id", {}).items()
    }
    expected = 0
    present = 0
    reason: str | None = None
    for cache_id in cache_ids:
        cache_id = str(cache_id)
        path = features_dir / f"{cache_id}.pt"
        if cache_id in empty_sample_ids:
            if path.is_file():
                return (
                    CacheValidationResult(complete=False, reason=f"unexpected feature for empty sample {cache_id}"),
                    present,
                    expected,
                )
            continue
        expected += 1
        expected_signature = str(cache_stem_by_id[cache_id])
        cached_signature = cached_signature_by_id.get(cache_id)
        if cached_signature is None:
            if reason is None:
                reason = f"missing cache identity for {cache_id}"
            continue
        if cached_signature != expected_signature:
            if reason is None:
                reason = f"cache identity mismatch for {cache_id}"
            continue
        if not path.is_file():
            if reason is None:
                reason = f"missing feature for {cache_id}"
            continue
        present += 1
    complete = reason is None
    return CacheValidationResult(complete=complete, reason=reason), present, expected


def _emit_cache_state_log(
    *,
    cache_label: str,
    cache_dir: Path,
    complete: bool,
    partial: bool = False,
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
        if partial:
            status = "partial"
            if rich_viz:
                status = "\x1b[1;33mpartial\x1b[0m"
            message = f"~ {cache_label} cache {status}: {cache_path}"
        else:
            status = "miss"
            if rich_viz:
                status = "\x1b[1;31mmiss\x1b[0m"
            message = f"✗ {cache_label} cache {status}: {cache_path}"
        if reason is not None:
            message = f"{message} ({reason})"
    slide2vec_progress.emit_progress_log(message)


def _emit_cache_resolve_log(
    *,
    cache_label: str,
    cache_dir: Path,
    key: str,
    scope_name: str,
    scope_count: int,
) -> None:
    slide2vec_progress.emit_progress_log(
        f"… resolving {cache_label} cache ({scope_name}={int(scope_count)}, key={str(key)[:16]}): {cache_dir.resolve()}"
    )


def _resolve_cache(
    *,
    cache_root: Path,
    cache_kind: str,
    key: str,
    metadata: dict[str, Any],
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
    manifest_rows: list[dict[str, object]] | None = None,
    initial_reason: str | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    cache_dir = _cache_dir(cache_root, cache_kind, key)
    features_dir = cache_dir / _features_subdir_for_kind(cache_kind)
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    _emit_cache_resolve_log(
        cache_label="feature",
        cache_dir=cache_dir,
        key=key,
        scope_name="artifacts",
        scope_count=len(cache_ids),
    )

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        mismatch_message = _format_cache_metadata_mismatch(
            cache_label="Feature cache",
            cache_dir=cache_dir,
            existing=_comparable_metadata(existing),
            expected=_comparable_metadata(metadata),
        )
        if mismatch_message:
            raise ValueError(mismatch_message)
        validation, present, expected = _validate_feature_cache_contents(
            features_dir=features_dir,
            metadata=existing,
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
        )
        partial = not validation.complete and present > 0 and expected > 0
        reason = validation.reason
        if partial:
            reason = f"{present}/{expected} present; {expected - present} missing"
        _emit_cache_state_log(
            cache_label="feature",
            cache_dir=cache_dir,
            complete=validation.complete,
            partial=partial,
            complete_state=complete_state,
            reason=reason,
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
            cache_ids=tuple(str(cache_id) for cache_id in cache_ids),
            cache_stem_by_id={str(cache_id): str(stem) for cache_id, stem in cache_stem_by_id.items()},
        )

    if manifest_rows is not None:
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
        cache_ids=tuple(str(cache_id) for cache_id in cache_ids),
        cache_stem_by_id={str(cache_id): str(stem) for cache_id, stem in cache_stem_by_id.items()},
    )


def resolve_tile_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig | None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    feature_type: str = "bag",
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    metadata = _build_tile_cache_metadata(
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        feature_type=feature_type,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="tile",
        static_identity_payload={"cache_key": metadata["cache_key"]},
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="tile",
        key=metadata["cache_key"],
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
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
    tile_preprocessing: PreprocessingConfig,
    tile_execution: EncoderConfig,
    tile_output_variant: str | None = None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    tile_dependency_signature = {
        "tile_encoder_name": str(tile_encoder_name),
        "tile_preprocessing": preprocessing_signature(tile_preprocessing),
        "tile_execution": execution_signature(
            tile_execution,
            encoder_name=tile_encoder_name,
            output_variant=tile_output_variant,
        ),
    }
    metadata = _build_slide_cache_metadata(
        slide_encoder_name=slide_encoder_name,
        tile_encoder_name=tile_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="slide",
        static_identity_payload={"cache_key": metadata["cache_key"]},
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="slide",
        key=metadata["cache_key"],
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
    )


def resolve_patient_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    patient_encoder_name: str,
    tile_encoder_name: str,
    tile_preprocessing: PreprocessingConfig,
    tile_execution: EncoderConfig,
    tile_output_variant: str | None = None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> FeatureCacheResolution:
    tile_dependency_signature = {
        "tile_encoder_name": str(tile_encoder_name),
        "tile_preprocessing": preprocessing_signature(tile_preprocessing),
        "tile_execution": execution_signature(
            tile_execution,
            encoder_name=tile_encoder_name,
            output_variant=tile_output_variant,
        ),
    }
    metadata = _build_patient_cache_metadata(
        patient_encoder_name=patient_encoder_name,
        tile_encoder_name=tile_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _patient_stems_for_kind(
        dataset=dataset,
        cache_kind="patient",
        static_identity_payload={"cache_key": metadata["cache_key"]},
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="patient",
        key=metadata["cache_key"],
        metadata=metadata,
        cache_ids=tuple(sorted(cache_stem_by_id.keys())),
        cache_stem_by_id=cache_stem_by_id,
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
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="hierarchical",
        static_identity_payload={"cache_key": metadata["cache_key"]},
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="hierarchical",
        key=metadata["cache_key"],
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
    )


def record_feature_dim(resolution: FeatureCacheResolution, feature_dim: int) -> None:
    metadata = (
        _load_metadata(resolution.metadata_path)
        if resolution.metadata_path.is_file()
        else dict(resolution.metadata)
    )
    metadata["feature_dim"] = int(feature_dim)
    _write_metadata(resolution.metadata_path, metadata)


def record_empty_sample_ids(resolution: FeatureCacheResolution, empty_sample_ids: Sequence[str]) -> None:
    metadata = (
        _load_metadata(resolution.metadata_path)
        if resolution.metadata_path.is_file()
        else dict(resolution.metadata)
    )
    empty_ids: set[str] = {str(s) for s in metadata.get("empty_sample_ids", [])}
    for sample_id in empty_sample_ids:
        sample_id = str(sample_id)
        if sample_id in resolution.cache_stem_by_id:
            empty_ids.add(sample_id)
    metadata["empty_sample_ids"] = sorted(empty_ids)
    _write_metadata(resolution.metadata_path, metadata)


def record_sample_identity_signatures(
    resolution: FeatureCacheResolution,
    cache_ids: Sequence[str],
) -> None:
    metadata_path = getattr(resolution, "metadata_path", None)
    if metadata_path is None:
        return
    metadata = (
        _load_metadata(metadata_path)
        if metadata_path.is_file()
        else dict(resolution.metadata)
    )
    signature_map = {
        str(cache_id): str(signature)
        for cache_id, signature in metadata.get("sample_identity_signature_by_id", {}).items()
    }
    for cache_id in cache_ids:
        cache_id = str(cache_id)
        if cache_id in resolution.cache_stem_by_id:
            signature_map[cache_id] = str(resolution.cache_stem_by_id[cache_id])
    metadata["sample_identity_signature_by_id"] = signature_map
    _write_metadata(metadata_path, metadata)
