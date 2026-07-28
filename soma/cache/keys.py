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


def annotation_sampling_signature(config: PreprocessingConfig) -> dict[str, Any] | None:
    """Selection-relevant projection of the annotation ``masks``/``sampling`` block.

    Returns ``None`` when no ``masks`` block is active (the default tissue path), so legacy
    tissue-only cache keys stay byte-stable. When a customized masks block governs tile
    selection (the annotation-restricted bag, #110), the projection folds the fields that
    change *which tiles are sampled* — the ``pixel_mapping`` vocabulary, per-class
    ``min_coverage`` (the active class set), and the ``sampling`` ``strategy`` /
    ``output_mode`` — into cache identity, so a tumor-restricted bag never aliases a
    full-tissue bag of the same slide/encoder. ``colors`` is cosmetic (preview overlay only)
    and is deliberately excluded.
    """
    masks = config.masks
    if masks is None:
        return None
    sampling = config.sampling
    return {
        "pixel_mapping": {k: masks.pixel_mapping[k] for k in sorted(masks.pixel_mapping)},
        "min_coverage": {k: masks.min_coverage[k] for k in sorted(masks.min_coverage)},
        "strategy": sampling.strategy if sampling is not None else "joint",
        "output_mode": sampling.output_mode if sampling is not None else "merged",
    }


def preprocessing_signature(config: PreprocessingConfig) -> dict[str, Any]:
    """Canonical preprocessing signature: hs2p's tiling vocabulary plus soma's own knobs.

    The tiling half is not restated here — it comes from ``config.tiling_values()``, the
    same mapping the composed hs2p ``TilingConfig`` is built from (ADR 0009), so the key
    cannot fall behind the geometry it is meant to key. What remains below is genuinely
    soma's: tissue segmentation, region (hierarchical) geometry and the read-size overrides.
    """
    signature: dict[str, Any] = {
        **config.tiling_values(),
        "requested_region_size_px": config.requested_region_size_px,
        "region_tile_multiple": config.region_tile_multiple,
        "read_tile_size_px": config.read_tile_size_px,
        "read_region_size_px": config.read_region_size_px,
        "tissue_method": config.tissue_method,
        "tissue_mask_tissue_value": config.tissue_mask_tissue_value,
        "seg_downsample": config.seg_downsample,
        "ref_tile_size_px": config.ref_tile_size_px,
        "a_t": config.a_t,
    }
    annotation = annotation_sampling_signature(config)
    if annotation is not None:
        # Injected only when a masks block is active — keeps tissue-only keys byte-stable.
        signature["annotation_sampling"] = annotation
    return signature


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


def _maybe_fold_dtype(payload: dict[str, Any], dtype: str) -> None:
    """Fold the on-disk dtype into a pooled cache-key payload, in place.

    Injected only for non-fp32 storage — legacy fp32 caches (the historical default,
    since soma force-upcast pooled features to fp32 before slide2vec grew a pooled
    ``output_dtype``) keep their byte-stable keys, while an fp16 cache resolves to a
    distinct key so the two can never be mixed. Mirrors the dense key's dtype guard.
    """
    if dtype != "fp32":
        payload["dtype"] = str(dtype)


def build_tile_cache_key(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig | None = None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    feature_type: str = "bag",
    dtype: str = "fp32",
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
    _maybe_fold_dtype(payload, dtype)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_slide_cache_key(
    *,
    slide_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
    dtype: str = "fp32",
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
    _maybe_fold_dtype(payload, dtype)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_patient_cache_key(
    *,
    patient_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: EncoderConfig,
    output_variant: str | None = None,
    dtype: str = "fp32",
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
    _maybe_fold_dtype(payload, dtype)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_hierarchical_cache_key(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig,
    output_variant: str | None = None,
    dtype: str = "fp32",
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
    _maybe_fold_dtype(payload, dtype)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


_FP16_ALIASES = {"fp16", "float16", "half", "16"}
_FP32_ALIASES = {"fp32", "float32", "32"}


def resolve_output_dtype(dtype: str | None, precision: str | None) -> str:
    """Resolve the on-disk feature dtype (``'fp16'``/``'fp32'``) for the cache key.

    The single resolver shared by the pooled and dense paths, governed by ``cache.dtype``.
    Mirrors slide2vec's ``output_dtype`` default: an explicit choice wins; ``None`` follows
    the compute ``precision`` — fp16 → fp16, else fp32 (fp32/bf16/unknown, since numpy has
    no bfloat16). Folding the result into the key keeps fp16 and fp32 caches from colliding.
    """
    if dtype is not None:
        d = str(dtype).lower()
        if d in _FP16_ALIASES:
            return "fp16"
        if d in _FP32_ALIASES:
            return "fp32"
        raise ValueError(f"cache.dtype must be 'fp16'/'fp32' or null, got {dtype!r}")
    return "fp16" if str(precision or "").lower() in _FP16_ALIASES else "fp32"


def build_dense_cache_key(
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
    dtype: str = "fp32",
    sampling_signature: dict[str, Any] | None = None,
) -> str:
    """Cache key for a dense ``(d, h, w)`` grid extraction.

    Distinct from the pooled tile key by ``feature_type="dense_grid"`` and by the
    dense geometry (``target_size``/``patch_size``/``pad_mode``) and
    ``dense_input_mode`` — so a 512px run and a 224px run, or whole-tile vs
    sliding-window, never collide. Sliding runs additionally key on
    ``window_size``/``overlap`` (different windows ⇒ different grids); these are
    injected **only** when ``window_size`` is set, so existing ``whole`` keys are
    unchanged. ``output_variant`` is intentionally omitted: the dense grid is
    pre-pooling (``forward_features`` → reshape), so the variant (which only changes
    pooling) does not affect it; keying on it would split identical caches.

    ``feature_kind`` discriminates a patch-feature grid from a CLS-attention grid (an
    attention grid must never alias a feature grid); the attention sub-knobs
    (``attention_blocks``/``attention_include_registers``) join the key too, since
    different blocks / register inclusion yield different channels. These are injected
    **only** for ``feature_kind != "patch_features"``, so legacy patch-feature keys are
    byte-stable.
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
    if window_size is not None:
        # Only sliding runs carry these — keeps legacy 'whole' keys byte-stable.
        payload["window_size"] = int(window_size)
        payload["overlap"] = float(overlap)
    if feature_kind != "patch_features":
        # Only attention (or future non-default) kinds carry these — keeps legacy
        # patch-feature keys byte-stable.
        payload["feature_kind"] = str(feature_kind)
        payload["attention_blocks"] = [int(b) for b in (attention_blocks or ())]
        payload["attention_include_registers"] = bool(attention_include_registers)
    if dtype != "fp32":
        # Injected only for non-fp32 storage — legacy fp32 caches (the historical default
        # when slide2vec force-upcast every grid) keep their byte-stable keys, while an
        # fp16 cache resolves to a distinct key so the two can never be mixed.
        payload["dtype"] = str(dtype)
    if preprocessing is not None:
        payload["preprocessing"] = preprocessing_signature(preprocessing)
    if sampling_signature is not None:
        # Slide-manifest segmentation: the ROIs (and thus the grids) are a function of the
        # annotation-sampling spec — pixel mapping, per-class coverage, strategy, output
        # mode, spacing/tile size. Folding it in means distinct sampling ⇒ distinct cache,
        # even when two specs happen to yield colliding coordinates. Absent (None) for
        # pre-cropped-tile inputs, which have no sampling step.
        payload["sampling"] = sampling_signature
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
