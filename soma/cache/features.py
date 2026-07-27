"""Feature cache resolution: metadata builders, validation, resolvers, and recording."""

from __future__ import annotations

import csv
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from slide2vec.artifacts import load_array

logger = logging.getLogger(__name__)

from soma.cache._types import (
    CACHE_METADATA_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    _FEATURE_TYPE_TO_RANK,
    _features_subdir_for_kind,
    _list_feature_filenames,
    CacheValidationResult,
    FeatureCacheResolution,
)
from soma.cache.io import (
    _CacheValidationProgress,
    _emit_cache_resolve_log,
    _emit_cache_state_log,
    _format_cache_metadata_mismatch,
    _load_metadata,
    _normalized_manifest_rows,
    _write_manifest,
    _write_metadata,
)
from soma.cache.keys import (
    _patient_stems_for_kind,
    _sample_stems_for_kind,
    build_dense_cache_key,
    build_hierarchical_cache_key,
    build_patient_cache_key,
    build_slide_cache_key,
    build_tile_cache_key,
    dataset_manifest_rows,
    execution_signature,
    preprocessing_signature,
)
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset


def _cache_dir(cache_root: Path, cache_kind: str, key: str) -> Path:
    return cache_root / cache_kind / key


def _build_tile_cache_metadata(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig | None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    feature_type: str = "bag",
    dtype: str = "fp32",
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
        dtype=dtype,
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
            preprocessing=preprocessing,
            output_variant=output_variant,
        ),
        "feature_type": str(feature_type),
        "dtype": str(dtype),
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_slide_cache_key(
        slide_encoder_name=slide_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
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
            preprocessing=None,
            output_variant=output_variant,
        ),
        "feature_type": "slide",
        "dtype": str(dtype),
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_patient_cache_key(
        patient_encoder_name=patient_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
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
            preprocessing=None,
            output_variant=output_variant,
        ),
        "feature_type": "patient",
        "dtype": str(dtype),
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = build_hierarchical_cache_key(
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
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
            preprocessing=preprocessing,
            output_variant=output_variant,
        ),
        "feature_type": "hierarchical",
        "dtype": str(dtype),
        "feature_dim": None,
        "sample_identity_signature_by_id": {},
    }
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _build_dense_cache_metadata(
    *,
    tile_encoder_name: str,
    target_size: tuple[int, int],
    patch_size: tuple[int, int],
    pad_mode: str,
    execution: EncoderConfig,
    preprocessing: PreprocessingConfig | None = None,
    dense_input_mode: str = "whole",
    window_size: int | None,
    overlap: float,
    feature_kind: str = "patch_features",
    attention_blocks: tuple[int, ...] | None = None,
    attention_include_registers: bool = False,
    channel_dim: int = 0,
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    sampling_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Imported lazily so the cache layer has no load-time dependency on soma.dense
    # (soma.dense.store imports soma.cache; a top-level import here would cycle).
    from soma.dense.geometry import compute_dense_geometry

    key = build_dense_cache_key(
        tile_encoder_name=tile_encoder_name,
        target_size=target_size,
        patch_size=patch_size,
        pad_mode=pad_mode,
        execution=execution,
        preprocessing=preprocessing,
        dense_input_mode=dense_input_mode,
        window_size=window_size,
        overlap=overlap,
        feature_kind=feature_kind,
        attention_blocks=attention_blocks,
        attention_include_registers=attention_include_registers,
        dtype=dtype,
        sampling_signature=sampling_signature,
    )
    geometry = compute_dense_geometry(target_size=target_size, patch_size=patch_size)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "dense",
        "cache_key": key,
        "encoder_name": tile_encoder_name,
        "encoder_level": "tile",
        "execution": execution_signature(
            execution,
            encoder_name=tile_encoder_name,
            preprocessing=preprocessing,
            output_variant=None,
        ),
        "feature_type": "dense_grid",
        "dtype": str(dtype),
        "feature_kind": str(feature_kind),
        "attention_blocks": [int(b) for b in (attention_blocks or ())] if feature_kind != "patch_features" else None,
        "attention_include_registers": bool(attention_include_registers) if feature_kind != "patch_features" else None,
        "feature_dim": None,
        "channel_dim": int(channel_dim),
        "dense_input_mode": str(dense_input_mode),
        "window_size": None if window_size is None else int(window_size),
        "overlap": float(overlap),
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "pad_mode": str(pad_mode),
        "sample_identity_signature_by_id": {},
    }
    # output_variant is intentionally not part of the dense key (pre-pooling grid),
    # so drop it from the execution signature to keep metadata and key consistent.
    metadata["execution"].pop("output_variant", None)
    if preprocessing is not None:
        metadata["preprocessing"] = preprocessing_signature(preprocessing)
    if sampling_signature is not None:
        metadata["sampling"] = sampling_signature
    if backend_provenance is not None:
        metadata.update(backend_provenance)
    return metadata


def _comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(metadata)
    comparable.pop("feature_dim", None)
    comparable.pop("empty_sample_ids", None)
    comparable.pop("sample_identity_signature_by_id", None)
    # dtype is recorded for symmetry/provenance but excluded from the equality check: the
    # cache key already folds the dtype (guarded so fp32 keys stay byte-stable), so a dir's
    # dtype is fixed by its key. Excluding it lets legacy caches (whose metadata predates
    # the dtype field) still validate without a spurious "missing=[dtype=...]" mismatch.
    comparable.pop("dtype", None)
    return comparable


def _manifest_matches_dataset(manifest_path: Path, dataset: Dataset) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except Exception:
        logger.debug("Could not read cache manifest at %s", manifest_path, exc_info=True)
        return False
    return _normalized_manifest_rows(rows) == _normalized_manifest_rows(dataset_manifest_rows(dataset))


def _backfill_feature_cache_identity_metadata(
    *,
    metadata_path: Path,
    metadata: dict[str, Any],
    dataset: Dataset,
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
) -> dict[str, Any]:
    if not _manifest_matches_dataset(metadata_path.parent / MANIFEST_NAME, dataset):
        return metadata

    signature_map = {
        str(cache_id): str(signature)
        for cache_id, signature in metadata.get("sample_identity_signature_by_id", {}).items()
    }
    changed = False
    for cache_id in cache_ids:
        cache_id = str(cache_id)
        expected_signature = str(cache_stem_by_id[cache_id])
        if signature_map.get(cache_id) is None:
            signature_map[cache_id] = expected_signature
            changed = True

    if changed:
        updated = dict(metadata)
        updated["sample_identity_signature_by_id"] = signature_map
        _write_metadata(metadata_path, updated)
        return updated
    return metadata


def _normalized_dense_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_normalized_dense_field(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _validate_dense_sidecar_metadata(
    *,
    sidecar_path: Path,
    metadata: dict[str, Any],
    cache_id: str,
) -> str | None:
    try:
        sidecar = _load_metadata(sidecar_path)
    except Exception:
        logger.debug("Could not read dense sidecar at %s", sidecar_path, exc_info=True)
        return f"dense sidecar for {cache_id} could not be read"

    artifact_type = sidecar.get("artifact_type", sidecar.get("feature_type"))
    if artifact_type != "dense_grid":
        return (
            f"dense sidecar artifact_type mismatch for {cache_id}: "
            f"expected 'dense_grid', found {artifact_type!r}"
        )

    expected_fields = {
        "channel_dim": metadata.get("channel_dim"),
        "grid_shape": metadata.get("grid_shape"),
        "target_size": metadata.get("target_size"),
        "encoded_size": metadata.get("encoded_size"),
        "patch_size": metadata.get("patch_size"),
        "pad_mode": metadata.get("pad_mode"),
        "dense_input_mode": metadata.get("dense_input_mode"),
        "window_size": metadata.get("window_size"),
        "overlap": metadata.get("overlap"),
        "feature_kind": metadata.get("feature_kind"),
        "attention_blocks": metadata.get("attention_blocks"),
        "attention_include_registers": metadata.get("attention_include_registers"),
    }
    if metadata.get("feature_dim") is not None:
        expected_fields["feature_dim"] = metadata.get("feature_dim")
    for field, expected in expected_fields.items():
        expected = _normalized_dense_field(expected)
        observed = _normalized_dense_field(sidecar.get(field))
        if expected != observed:
            return (
                f"dense sidecar {field} mismatch for {cache_id}: "
                f"expected {expected!r}, found {observed!r}"
            )
    return None


def _validate_feature_cache_contents(
    *,
    features_dir: Path,
    metadata: dict[str, Any],
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
    validate_payloads: bool = False,
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
    expected_rank = _FEATURE_TYPE_TO_RANK[feature_type]
    expected_feature_dim = metadata.get("feature_dim")
    # Dense grids carry their shape in a per-sample sidecar, which DenseFeatureStore
    # requires to read them. The .pt-existence check alone would call an entry
    # "complete" that the store cannot actually load, so require the sidecar too.
    dense_sidecar_suffix: str | None = None
    if feature_type == "dense_grid":
        from soma.dense.store import DENSE_SIDECAR_SUFFIX

        dense_sidecar_suffix = DENSE_SIDECAR_SUFFIX
    # One directory listing answers the <id>.pt and <id>.meta.json existence
    # questions for every id; the hot loop below performs no per-id stat.
    existing_filenames = _list_feature_filenames(features_dir)
    expected = 0
    present = 0
    reason: str | None = None
    total = len(cache_ids)
    checked = 0
    progress = _CacheValidationProgress(cache_label="feature", total=total)
    progress.start()
    try:
        for cache_id in cache_ids:
            cache_id = str(cache_id)
            checked += 1
            progress.update(checked)
            path = features_dir / f"{cache_id}.pt"
            feature_present = f"{cache_id}.pt" in existing_filenames
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
            if cache_id in empty_sample_ids:
                if feature_present:
                    return (
                        CacheValidationResult(complete=False, reason=f"unexpected feature for empty sample {cache_id}"),
                        present,
                        expected,
                    )
                continue
            expected += 1
            if not feature_present:
                if reason is None:
                    reason = f"missing feature for {cache_id}"
                continue
            sidecar_path: Path | None = None
            if dense_sidecar_suffix is not None:
                # Existence is decided cheaply from the listing and always enforced:
                # a half-written sample (a ``.pt`` whose sidecar never landed) must
                # be treated as missing, never silently skipped into a load failure.
                if f"{cache_id}{dense_sidecar_suffix}" not in existing_filenames:
                    if reason is None:
                        reason = f"missing dense sidecar for {cache_id}"
                    continue
                sidecar_path = features_dir / f"{cache_id}{dense_sidecar_suffix}"
            if validate_payloads:
                # Sidecar *content* validation (the JSON read + shape cross-check)
                # is the one cost a listing cannot remove, so it joins the gated
                # deep-validation block. It runs before the (also-gated) tensor
                # load and short-circuits on mismatch, so a bad sidecar never pays
                # for a deserialize.
                if sidecar_path is not None:
                    sidecar_reason = _validate_dense_sidecar_metadata(
                        sidecar_path=sidecar_path,
                        metadata=metadata,
                        cache_id=cache_id,
                    )
                    if sidecar_reason is not None:
                        if reason is None:
                            reason = sidecar_reason
                        continue
                try:
                    payload = load_array(path)
                    tensor = payload if torch.is_tensor(payload) else torch.as_tensor(payload)
                except Exception as exc:
                    # A corrupt cached payload must not pass silently — surface
                    # it as a WARNING with the path so the user can fix or
                    # delete it. Returning ``complete=False`` lets the caller
                    # re-extract that sample.
                    logger.warning(
                        "Cached feature payload at %s is unreadable (%s: %s); "
                        "treating as missing.",
                        path,
                        type(exc).__name__,
                        exc,
                    )
                    if reason is None:
                        reason = f"invalid feature payload for {cache_id}"
                    continue
                if tensor.ndim != expected_rank:
                    if reason is None:
                        reason = (
                            f"feature rank mismatch for {cache_id}: "
                            f"expected {expected_rank}, found {tensor.ndim}"
                        )
                    continue
                if feature_type == "dense_grid":
                    # (channels, grid_h, grid_w): the feature dim is the channel
                    # axis (0 by convention), NOT the last axis (which is grid_w).
                    channel_dim = int(metadata.get("channel_dim", 0))
                    feature_dim = int(tensor.shape[channel_dim])
                else:
                    feature_dim = int(tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1])
                if expected_feature_dim is not None and int(expected_feature_dim) != feature_dim:
                    if reason is None:
                        reason = (
                            f"feature dim mismatch for {cache_id}: "
                            f"expected {int(expected_feature_dim)}, found {feature_dim}"
                        )
                    continue
                if feature_type == "dense_grid":
                    # Rank 3 alone can't catch a wrong grid (it equals hierarchical),
                    # so verify the spatial axes against the recorded grid_shape. Free:
                    # the tensor is already loaded above (only under validate_payloads).
                    expected_grid = metadata.get("grid_shape")
                    if expected_grid is not None:
                        grid_axes = [
                            int(size)
                            for axis, size in enumerate(tensor.shape)
                            if axis != int(metadata.get("channel_dim", 0))
                        ]
                        if grid_axes != [int(v) for v in expected_grid]:
                            if reason is None:
                                reason = (
                                    f"grid shape mismatch for {cache_id}: expected "
                                    f"{[int(v) for v in expected_grid]}, found {grid_axes}"
                                )
                            continue
            present += 1
        complete = reason is None
        return CacheValidationResult(complete=complete, reason=reason), present, expected
    finally:
        progress.finish()


def _refresh_feature_cache_resolution(
    resolution: FeatureCacheResolution,
    metadata: dict[str, Any],
    *,
    validate_payloads: bool = False,
) -> FeatureCacheResolution:
    validation, _present, _expected = _validate_feature_cache_contents(
        features_dir=resolution.features_dir,
        metadata=metadata,
        cache_ids=resolution.cache_ids,
        cache_stem_by_id=resolution.cache_stem_by_id,
        validate_payloads=validate_payloads,
    )
    return replace(
        resolution,
        reused=validation.complete,
        complete=validation.complete,
        metadata=metadata,
        validation=validation,
    )


def _resolve_cache(
    *,
    cache_root: Path,
    cache_kind: str,
    key: str,
    dataset: Dataset,
    metadata: dict[str, Any],
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
    manifest_rows: list[dict[str, object]] | None = None,
    initial_reason: str | None = None,
    complete_state: str = "hit",
    validate_payloads: bool = False,
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
        existing = _backfill_feature_cache_identity_metadata(
            metadata_path=metadata_path,
            metadata=existing,
            dataset=dataset,
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
        )
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
            validate_payloads=validate_payloads,
        )
        partial = not validation.complete and present > 0 and expected > 0
        reason = validation.reason
        if partial:
            missing = expected - present
            feature_word = "feature file" if present == 1 else "feature files"
            missing_word = "sample" if missing == 1 else "samples"
            reason = (
                f"{present}/{expected} {feature_word} already materialized on disk; "
                f"embedding the {missing} missing {missing_word}"
            )
            if validation.reason is not None:
                reason = f"{reason}; first issue: {validation.reason}"
        elif not validation.complete and expected > 0:
            missing = expected - present
            if missing > 0:
                feature_word = "feature file" if missing == 1 else "feature files"
                reason = f"{missing}/{expected} {feature_word} missing"
                if validation.reason is not None:
                    reason = f"{reason}; first issue: {validation.reason}"
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
            validation=validation,
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
        validation=CacheValidationResult(complete=False, reason=initial_reason),
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
    cache_kind: str = "tile",
    _precomputed_stems: dict[str, str] | None = None,
) -> FeatureCacheResolution:
    metadata = _build_tile_cache_metadata(
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        feature_type=feature_type,
        dtype=dtype,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _precomputed_stems if _precomputed_stems is not None else _sample_stems_for_kind(
        dataset=dataset,
        cache_kind=cache_kind,
        static_identity_payload={"cache_key": metadata["cache_key"]},
        fingerprint_files=fingerprint_files,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind=cache_kind,
        key=metadata["cache_key"],
        dataset=dataset,
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
        validate_payloads=validate_payloads,
    )


def resolve_image_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    execution: EncoderConfig,
    output_variant: str | None = None,
    dtype: str = "fp32",
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
) -> FeatureCacheResolution:
    """Resolve the cache for given-geometry images (pre-cropped patch datasets).

    One 1-D embedding per image, so the payload rank matches ``feature_type="tile"``; the
    ``image`` cache *kind* is what routes ``features_dir`` at ``image_embeddings/``, the
    directory :meth:`slide2vec.Model.embed_images` writes into. There is no
    ``preprocessing`` argument by construction: the Given regime has no tiling and no
    requested geometry to key on — the encoder's shipped transform is the contract, and it
    is already pinned by the encoder name and output variant that are in the key.
    """
    return resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name=tile_encoder_name,
        preprocessing=None,
        execution=execution,
        output_variant=output_variant,
        feature_type="tile",
        dtype=dtype,
        complete_state=complete_state,
        fingerprint_files=fingerprint_files,
        validate_payloads=validate_payloads,
        cache_kind="image",
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
    _precomputed_stems: dict[str, str] | None = None,
) -> FeatureCacheResolution:
    tile_dependency_signature = {
        "tile_encoder_name": str(tile_encoder_name),
        "tile_preprocessing": preprocessing_signature(tile_preprocessing),
        "tile_execution": execution_signature(
            tile_execution,
            encoder_name=tile_encoder_name,
            preprocessing=tile_preprocessing,
            output_variant=tile_output_variant,
        ),
    }
    metadata = _build_slide_cache_metadata(
        slide_encoder_name=slide_encoder_name,
        tile_encoder_name=tile_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _precomputed_stems if _precomputed_stems is not None else _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="slide",
        static_identity_payload={"cache_key": metadata["cache_key"]},
        fingerprint_files=fingerprint_files,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="slide",
        key=metadata["cache_key"],
        dataset=dataset,
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
        validate_payloads=validate_payloads,
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
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
    _precomputed_stems: dict[str, str] | None = None,
) -> FeatureCacheResolution:
    tile_dependency_signature = {
        "tile_encoder_name": str(tile_encoder_name),
        "tile_preprocessing": preprocessing_signature(tile_preprocessing),
        "tile_execution": execution_signature(
            tile_execution,
            encoder_name=tile_encoder_name,
            preprocessing=tile_preprocessing,
            output_variant=tile_output_variant,
        ),
    }
    metadata = _build_patient_cache_metadata(
        patient_encoder_name=patient_encoder_name,
        tile_encoder_name=tile_encoder_name,
        tile_dependency_signature=tile_dependency_signature,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _precomputed_stems if _precomputed_stems is not None else _patient_stems_for_kind(
        dataset=dataset,
        cache_kind="patient",
        static_identity_payload={"cache_key": metadata["cache_key"]},
        fingerprint_files=fingerprint_files,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="patient",
        key=metadata["cache_key"],
        dataset=dataset,
        metadata=metadata,
        cache_ids=tuple(sorted(cache_stem_by_id.keys())),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
        validate_payloads=validate_payloads,
    )


def resolve_hierarchical_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
    _precomputed_stems: dict[str, str] | None = None,
) -> FeatureCacheResolution:
    metadata = _build_hierarchical_cache_metadata(
        tile_encoder_name=tile_encoder_name,
        preprocessing=preprocessing,
        execution=execution,
        output_variant=output_variant,
        dtype=dtype,
        backend_provenance=backend_provenance,
    )
    cache_stem_by_id = _precomputed_stems if _precomputed_stems is not None else _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="hierarchical",
        static_identity_payload={"cache_key": metadata["cache_key"]},
        fingerprint_files=fingerprint_files,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="hierarchical",
        key=metadata["cache_key"],
        dataset=dataset,
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
        validate_payloads=validate_payloads,
    )


def resolve_dense_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    tile_encoder_name: str,
    target_size: tuple[int, int],
    patch_size: tuple[int, int],
    pad_mode: str,
    execution: EncoderConfig,
    preprocessing: PreprocessingConfig | None = None,
    dense_input_mode: str = "whole",
    window_size: int | None,
    overlap: float,
    feature_kind: str = "patch_features",
    attention_blocks: tuple[int, ...] | None = None,
    attention_include_registers: bool = False,
    channel_dim: int = 0,
    dtype: str = "fp32",
    backend_provenance: dict[str, Any] | None = None,
    sampling_signature: dict[str, Any] | None = None,
    complete_state: str = "hit",
    fingerprint_files: bool = False,
    validate_payloads: bool = False,
    _precomputed_stems: dict[str, str] | None = None,
) -> FeatureCacheResolution:
    metadata = _build_dense_cache_metadata(
        tile_encoder_name=tile_encoder_name,
        target_size=target_size,
        patch_size=patch_size,
        pad_mode=pad_mode,
        execution=execution,
        preprocessing=preprocessing,
        dense_input_mode=dense_input_mode,
        window_size=window_size,
        overlap=overlap,
        feature_kind=feature_kind,
        attention_blocks=attention_blocks,
        attention_include_registers=attention_include_registers,
        channel_dim=channel_dim,
        dtype=dtype,
        backend_provenance=backend_provenance,
        sampling_signature=sampling_signature,
    )
    cache_stem_by_id = _precomputed_stems if _precomputed_stems is not None else _sample_stems_for_kind(
        dataset=dataset,
        cache_kind="dense",
        static_identity_payload={"cache_key": metadata["cache_key"]},
        fingerprint_files=fingerprint_files,
    )
    return _resolve_cache(
        cache_root=cache_root,
        cache_kind="dense",
        key=metadata["cache_key"],
        dataset=dataset,
        metadata=metadata,
        cache_ids=tuple(sorted(dataset.sample_ids)),
        cache_stem_by_id=cache_stem_by_id,
        manifest_rows=dataset_manifest_rows(dataset),
        initial_reason="initializing",
        complete_state=complete_state,
        validate_payloads=validate_payloads,
    )


def record_feature_dim(
    resolution: FeatureCacheResolution,
    feature_dim: int,
    *,
    validate_payloads: bool = False,
) -> FeatureCacheResolution:
    metadata = (
        _load_metadata(resolution.metadata_path)
        if resolution.metadata_path.is_file()
        else dict(resolution.metadata)
    )
    metadata["feature_dim"] = int(feature_dim)
    _write_metadata(resolution.metadata_path, metadata)
    return _refresh_feature_cache_resolution(
        resolution,
        metadata,
        validate_payloads=validate_payloads,
    )


def record_empty_sample_ids(
    resolution: FeatureCacheResolution,
    empty_sample_ids: Sequence[str],
    *,
    validate_payloads: bool = False,
) -> FeatureCacheResolution:
    metadata = (
        _load_metadata(resolution.metadata_path)
        if resolution.metadata_path.is_file()
        else dict(resolution.metadata)
    )
    empty_ids: set[str] = {str(s) for s in metadata.get("empty_sample_ids", [])}
    signature_map = {
        str(cache_id): str(signature)
        for cache_id, signature in metadata.get("sample_identity_signature_by_id", {}).items()
    }
    for sample_id in empty_sample_ids:
        sample_id = str(sample_id)
        if sample_id in resolution.cache_stem_by_id:
            empty_ids.add(sample_id)
            signature_map[sample_id] = str(resolution.cache_stem_by_id[sample_id])
    metadata["empty_sample_ids"] = sorted(empty_ids)
    metadata["sample_identity_signature_by_id"] = signature_map
    _write_metadata(resolution.metadata_path, metadata)
    return _refresh_feature_cache_resolution(
        resolution,
        metadata,
        validate_payloads=validate_payloads,
    )


def record_sample_identity_signatures(
    resolution: FeatureCacheResolution,
    cache_ids: Sequence[str],
    *,
    validate_payloads: bool = False,
) -> FeatureCacheResolution:
    metadata_path = getattr(resolution, "metadata_path", None)
    if metadata_path is None:
        return resolution
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
    return _refresh_feature_cache_resolution(
        resolution,
        metadata,
        validate_payloads=validate_payloads,
    )
