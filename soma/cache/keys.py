"""Cache key and sample-stem derivation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from hs2p.wsi.reader import resolve_backend
from slide2vec.encoders.registry import encoder_registry

from soma.cache._types import SCHEMA_VERSION, _FEATURE_TYPE_TO_RANK
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.validation import resolve_encoder_precision


def _resolve_encoder_precision(
    encoder_config: EncoderConfig,
    *,
    encoder_name: str | None = None,
) -> str:
    return resolve_encoder_precision(encoder_config, encoder_name=encoder_name)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    fingerprint_files: bool = False,
) -> str:
    payload = {
        "sample_id": str(sample_id),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
    }
    if fingerprint_files:
        payload["image_sha256"] = _file_digest(image_path)
        payload["mask_sha256"] = _file_digest(mask_path) if mask_path is not None else None
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
    preprocessing: PreprocessingConfig | None = None,
    output_variant: str | None = None,
) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "precision": _resolve_encoder_precision(encoder_config, encoder_name=encoder_name),
        "output_variant": output_variant if output_variant is not None else encoder_config.output_variant,
    }
    effective_encoder_name = encoder_name or encoder_config.name
    try:
        metadata = encoder_registry.info(effective_encoder_name)
    except Exception:
        metadata = None

    if metadata is not None:
        recommended_input_size = metadata.get("input_size")
        if recommended_input_size is not None:
            signature["input_size"] = int(recommended_input_size)

    spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    if spacing_um is None and metadata is not None:
        recommended_spacing = metadata.get("supported_spacing_um")
        if isinstance(recommended_spacing, (int, float)):
            spacing_um = float(recommended_spacing)
    if spacing_um is not None:
        signature["spacing_um"] = float(spacing_um)

    return signature


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
            preprocessing=preprocessing,
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
            preprocessing=None,
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
            preprocessing=None,
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
            preprocessing=preprocessing,
            output_variant=output_variant,
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_dense_cache_key(
    *,
    tile_encoder_name: str,
    target_size: tuple[int, int],
    patch_size: tuple[int, int],
    pad_mode: str,
    execution: EncoderConfig,
    preprocessing: PreprocessingConfig | None = None,
    dense_input_mode: str = "whole",
) -> str:
    """Cache key for a dense ``(d, h, w)`` grid extraction.

    Distinct from the pooled tile key by ``feature_type="dense_grid"`` and by the
    dense geometry (``target_size``/``patch_size``/``pad_mode``) and
    ``dense_input_mode`` — so a 512px run and a 224px run, or whole-tile vs
    sliding-window, never collide. ``output_variant`` is intentionally omitted:
    the dense grid is pre-pooling (``forward_features`` → reshape), so the variant
    (which only changes pooling) does not affect it; keying on it would split
    identical caches.
    """
    execution_payload = execution_signature(
        execution,
        encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        output_variant=None,
    )
    execution_payload.pop("output_variant", None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "dense",
        "tile_encoder_name": tile_encoder_name,
        "feature_type": "dense_grid",
        "dense_input_mode": str(dense_input_mode),
        "target_size": [int(target_size[0]), int(target_size[1])],
        "patch_size": [int(patch_size[0]), int(patch_size[1])],
        "pad_mode": str(pad_mode),
        "execution": execution_payload,
    }
    if preprocessing is not None:
        payload["preprocessing"] = preprocessing_signature(preprocessing)
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


def _sample_identity_payload(
    dataset: Dataset,
    *,
    fingerprint_files: bool = False,
) -> dict[str, str]:
    return {
        sample_id: sample_identity_signature(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            fingerprint_files=fingerprint_files,
        )
        for sample_id, sample in dataset.samples.items()
    }


def _sample_stems_for_kind(
    *,
    dataset: Dataset,
    cache_kind: str,
    static_identity_payload: dict[str, Any],
    fingerprint_files: bool = False,
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(
        dataset,
        fingerprint_files=fingerprint_files,
    )
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
    fingerprint_files: bool = False,
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(
        dataset,
        fingerprint_files=fingerprint_files,
    )
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
    fingerprint_files: bool = False,
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(
        dataset,
        fingerprint_files=fingerprint_files,
    )
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
