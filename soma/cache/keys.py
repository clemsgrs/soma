"""Cache key and sample-stem derivation.

Every key derives from the slide2vec and hs2p dataclasses soma holds (ADR 0019): the
pooled keys from slide2vec's ``PreprocessingConfig`` (hs2p's tiling, segmentation,
filter and mask-sampling configs) and the precision of slide2vec's ``ExecutionOptions``,
the dense keys from slide2vec's ``DenseOptions``. A field the upstream dataclasses gain
enters the key without a soma edit; only the fields that cannot change a feature are
left out, by name.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from hs2p.wsi.reader import resolve_backend
from slide2vec import DenseImageOptions, DenseOptions, ExecutionOptions, PreprocessingConfig
from slide2vec.encoders.registry import encoder_registry

from soma.cache._types import SCHEMA_VERSION, _FEATURE_TYPE_TO_RANK
from soma.data._legacy import LegacySamples
from soma.encoders.validation import resolve_encoder_precision
from soma.preprocessing.supplied_coordinates import coordinates_digest

#: slide2vec ``PreprocessingConfig`` fields that change how tiles are written, batched or
#: previewed, never which pixels a feature is computed from.
_RUNTIME_PREPROCESSING_FIELDS = frozenset(
    {"preview", "save_tiles", "adaptive_batching", "use_supertiles", "num_cucim_workers", "resume"}
)
#: Nested hs2p fields left out of the key: the requested-backend echoes restate
#: ``backend``/``mask_backend``; the SAM2 device and worker cap are runtime placement; the
#: mask colors only paint previews.
_RUNTIME_NESTED_FIELDS = {
    "tiling": frozenset({"requested_backend", "requested_mask_backend"}),
    "segmentation": frozenset({"sam2_device", "sam2_num_workers"}),
    "masks": frozenset({"colors"}),
}


def _resolve_encoder_precision(
    execution: ExecutionOptions | None,
    *,
    encoder_name: str,
) -> str:
    """The precision the features are computed at: ``execution.precision`` when set,
    else the encoder's recommendation."""
    return resolve_encoder_precision(
        None if execution is None else execution.precision,
        encoder_name=encoder_name,
    )


def _jsonable(value: Any) -> Any:
    """Canonical JSON primitives of a dataclass field value."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def dataset_manifest_rows(dataset: LegacySamples) -> list[dict[str, object]]:
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
                "spacing_at_level_0": sample.spacing_at_level_0,
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
                "spacing_at_level_0": row.get("spacing_at_level_0"),
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
    spacing_at_level_0: float | None = None,
    coordinates_digest: str | None = None,
) -> str:
    payload = {
        "sample_id": str(sample_id),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "spacing_at_level_0": spacing_at_level_0,
    }
    if coordinates_digest is not None:
        # User-supplied tiles are part of what a slide's features are: another tile set
        # is another sample. Injected only when set, so soma-tiled identities stay stable.
        payload["coordinates_digest"] = str(coordinates_digest)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _sample_cache_stem(*, sample_signature: str, identity_payload: dict[str, Any]) -> str:
    payload = dict(identity_payload)
    payload["sample_signature"] = str(sample_signature)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def preprocessing_signature(preprocessing: PreprocessingConfig) -> dict[str, Any]:
    """Canonical signature of a resolved slide2vec ``PreprocessingConfig``.

    Every field of slide2vec's config and of the hs2p configs it holds (tiling,
    segmentation, filter, mask sampling), except the ones named in
    :data:`_RUNTIME_PREPROCESSING_FIELDS` and :data:`_RUNTIME_NESTED_FIELDS`. The raw
    values a mask label merges are sorted: their order does not change what is sampled.
    """
    if preprocessing.tiling is None:
        raise ValueError(
            "preprocessing.tiling must be resolved before a cache key is derived from it."
        )
    signature: dict[str, Any] = {}
    for field in dataclasses.fields(preprocessing):
        if field.name in _RUNTIME_PREPROCESSING_FIELDS:
            continue
        value = _jsonable(getattr(preprocessing, field.name))
        if isinstance(value, dict) and field.name in _RUNTIME_NESTED_FIELDS:
            value = {
                key: item for key, item in value.items() if key not in _RUNTIME_NESTED_FIELDS[field.name]
            }
        signature[field.name] = value
    signature["masks"]["pixel_mapping"] = {
        label: sorted(values) for label, values in signature["masks"]["pixel_mapping"].items()
    }
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
    dataset: LegacySamples,
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
            spacing_override=sample.spacing_at_level_0,
        )
        mapping[sample_id] = str(selection.backend)
    return mapping


def execution_signature(
    execution: ExecutionOptions | None,
    *,
    encoder_name: str,
    preprocessing: PreprocessingConfig | None = None,
    output_variant: str | None = None,
) -> dict[str, Any]:
    """What the encoder decides about a pooled feature: the precision it runs at
    (``ExecutionOptions.precision``, else its recommendation), its output variant, its
    registry input size and the spacing it reads at."""
    signature: dict[str, Any] = {
        "precision": _resolve_encoder_precision(execution, encoder_name=encoder_name),
        "output_variant": output_variant,
    }
    try:
        metadata = encoder_registry.info(encoder_name)
    except Exception:
        metadata = None
    if metadata is not None and metadata.get("input_size") is not None:
        signature["input_size"] = int(metadata["input_size"])
    spacing_um = None
    if preprocessing is not None and preprocessing.tiling is not None:
        spacing_um = preprocessing.tiling.requested_spacing_um
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
    execution: ExecutionOptions | None,
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
        "dtype": str(dtype),
    }
    if preprocessing is not None:
        payload["preprocessing"] = preprocessing_signature(preprocessing)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_slide_cache_key(
    *,
    slide_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: ExecutionOptions | None,
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
        "dtype": str(dtype),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_patient_cache_key(
    *,
    patient_encoder_name: str,
    tile_dependency_signature: dict[str, Any],
    execution: ExecutionOptions | None,
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
        "dtype": str(dtype),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def build_hierarchical_cache_key(
    *,
    tile_encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions | None,
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
        "dtype": str(dtype),
    }
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


def resolve_cache_dtype(
    cache_dtype: str | None,
    execution: ExecutionOptions | None,
    *,
    encoder_name: str,
) -> str:
    """Resolve the on-disk dtype for a cache of ``encoder_name``'s features.

    The one entry point for the pooled, tile-image and dense paths: an explicit
    ``cache.dtype`` wins; otherwise the dtype follows the precision the encoder runs at
    (``execution.precision``, else the registry recommendation, else fp32).
    """
    precision = _resolve_encoder_precision(execution, encoder_name=encoder_name)
    return resolve_output_dtype(cache_dtype, precision)


def _hw(size: int | Sequence[int]) -> list[int]:
    if isinstance(size, (int,)) and not isinstance(size, bool):
        return [int(size), int(size)]
    height, width = size
    return [int(height), int(width)]


def dense_signature(dense: DenseOptions | DenseImageOptions) -> dict[str, Any]:
    """Every field of the slide2vec dense options a grid is read and encoded with, the
    ``target_size`` spelled ``[h, w]``."""
    signature = _jsonable(dense)
    signature["target_size"] = _hw(dense.target_size)
    return signature


def build_dense_cache_key(
    *,
    tile_encoder_name: str,
    patch_size: Sequence[int],
    dense: DenseOptions | DenseImageOptions,
    execution: ExecutionOptions | None,
    dtype: str = "fp32",
    sampling_signature: dict[str, Any] | None = None,
) -> str:
    """Cache key for a dense ``(d, h, w)`` grid extraction.

    Keys on the encoder, its static ``patch_size``, the precision it runs at, the
    on-disk dtype and every field of the slide2vec dense options (spacing, supervision
    size, padding, window, feature kind and attention recipe). An annotation-sampled ROI
    cache adds the sampling signature: the ROIs are a function of it.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "dense",
        "tile_encoder_name": tile_encoder_name,
        "feature_type": "dense_grid",
        "patch_size": _hw(patch_size),
        "precision": _resolve_encoder_precision(execution, encoder_name=tile_encoder_name),
        "dense": dense_signature(dense),
        "dtype": str(dtype),
    }
    if sampling_signature is not None:
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


def roi_sampling_signature(preprocessing: PreprocessingConfig, *, spacing_policy: str = "strict") -> dict[str, Any]:
    """What decides an annotation-sampled ROI set: the extraction's signature and the
    spacing policy that sets each slide's read spacing."""
    return {"preprocessing": preprocessing_signature(preprocessing), "spacing_policy": str(spacing_policy)}


def build_roi_sampling_cache_key(
    *,
    preprocessing: PreprocessingConfig,
    spacing_policy: str = "strict",
) -> str:
    """Directory key for the ``roi_sampling`` cache kind.

    Hashes ``{artifact_kind}`` plus :func:`roi_sampling_signature`. No ``SCHEMA_VERSION``
    and no engine version (consistent with the tiling key: an hs2p sampling-behavior
    change requires manual invalidation).
    """
    payload = {
        "artifact_kind": "roi_sampling",
        **roi_sampling_signature(preprocessing, spacing_policy=spacing_policy),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def roi_sampling_identity_signature(
    *,
    sample_id: str,
    image_path: Path | str,
    label_mask_path: Path | str | None,
    label_mask_size_bytes: int | None,
    label_mask_mtime_ns: int | None,
    spacing_at_level_0: float | None = None,
) -> str:
    """Per-slide identity stem for the ``roi_sampling`` cache kind.

    Styled after :func:`sample_identity_signature` but with this kind's own field set:
    the mask in the identity is the **annotation raster** (``label_mask_path``) — the
    file sampling actually reads — not the tissue mask. Path+spacing based, never
    content-hashed, consistent with every other soma cache — with one deliberate
    exception: the annotation raster's size and mtime (``None`` when it cannot be
    stat'ed) join the identity, because this entry stores the raster's content (the ROI
    mask crops, and the coords through per-class coverage). An annotation edited in place
    thus misses its coords and crops together; a full hash of a multi-GB raster per run
    would be too slow.
    """
    payload = {
        "sample_id": str(sample_id),
        "image_path": str(image_path),
        "label_mask_path": str(label_mask_path) if label_mask_path is not None else None,
        "label_mask_size_bytes": label_mask_size_bytes,
        "label_mask_mtime_ns": label_mask_mtime_ns,
        "spacing_at_level_0": spacing_at_level_0,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _file_size_and_mtime(path: Path | str | None) -> tuple[int | None, int | None]:
    if path is None:
        return None, None
    try:
        stat = Path(path).stat()
    except OSError:
        return None, None
    return int(stat.st_size), int(stat.st_mtime_ns)


def _sample_stems_for_roi_sampling(dataset: Any) -> dict[str, str]:
    """Duck-typed over ``samples``/``sample_ids`` so :class:`LegacySamples` fits."""
    stems: dict[str, str] = {}
    for sample_id, sample in sorted(dataset.samples.items()):
        size_bytes, mtime_ns = _file_size_and_mtime(sample.label_mask_path)
        stems[sample_id] = roi_sampling_identity_signature(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            label_mask_path=sample.label_mask_path,
            label_mask_size_bytes=size_bytes,
            label_mask_mtime_ns=mtime_ns,
            spacing_at_level_0=sample.spacing_at_level_0,
        )
    return stems


def _sample_identity_payload(dataset: LegacySamples) -> dict[str, str]:
    return {
        sample_id: sample_identity_signature(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            spacing_at_level_0=sample.spacing_at_level_0,
            coordinates_digest=(
                coordinates_digest(sample.coordinates_path, sample_id=sample.sample_id)
                if sample.coordinates_path is not None
                else None
            ),
        )
        for sample_id, sample in dataset.samples.items()
    }


def _sample_stems_for_kind(
    *,
    dataset: LegacySamples,
    cache_kind: str,
    static_identity_payload: dict[str, Any],
) -> dict[str, str]:
    signature_by_sample_id = _sample_identity_payload(dataset)
    return {
        sample_id: _sample_stem_for_kind(
            sample_signature=signature_by_sample_id[sample_id],
            cache_kind=cache_kind,
            static_identity_payload=static_identity_payload,
        )
        for sample_id in sorted(dataset.sample_ids)
    }


def _sample_stem_for_kind(
    *,
    sample_signature: str,
    cache_kind: str,
    static_identity_payload: dict[str, Any],
) -> str:
    return _sample_cache_stem(
        sample_signature=sample_signature,
        identity_payload={
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": cache_kind,
            **static_identity_payload,
        },
    )


def _patient_stems_for_kind(
    *,
    dataset: LegacySamples,
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
    dataset: LegacySamples,
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
