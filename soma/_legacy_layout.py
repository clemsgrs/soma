"""The YAML extraction blocks <-> :class:`~soma.config.FeaturesConfig` (slice 6b only).

soma holds slide2vec's and hs2p's config dataclasses as given (ADR 0019), but the YAML
layout still spells the extraction surface in soma's former vocabulary: the
``preprocessing:``, ``encoder:``, ``composite:``, ``execution:`` and ``cache:`` blocks
and ``data.feature_mode``. This module reads those blocks into the held upstream
dataclasses and writes them back for ``config.yaml``. Slice 6c rewrites the YAML layout
and deletes it.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import slide2vec
from hs2p import (
    DEFAULT_MASK_SAMPLING,
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
)

from soma.config import CacheConfig, EncoderLeaf, FeaturesConfig, is_annotation_sampling

__all__ = ["features_from_layout", "features_to_layout"]

_GRID_DATASET_TYPES = frozenset({"segmentation", "detection"})
_DEFAULT_FILTER = FilterConfig()

_PREPROCESSING_KEYS = frozenset(
    {
        "backend",
        "mask_backend",
        "requested_tile_size_px",
        "requested_spacing_um",
        "spacing_policy",
        "requested_region_size_px",
        "region_tile_multiple",
        "tissue_method",
        "min_coverage",
        "overlap",
        "seg_downsample",
        "sam2_device",
        "sam2_num_workers",
        "tolerance",
        "ref_tile_size_px",
        "a_t",
        "dense_window_size",
        "dense_window_overlap",
        "feature_kind",
        "attention",
        "preview",
        "masks",
        "sampling",
    }
)
_REMOVED_PREPROCESSING_KEYS = {
    "read_tile_size_px": "slide2vec reads at requested_tile_size_px",
    "read_region_size_px": "slide2vec reads at requested_region_size_px",
}
_ENCODER_KEYS = frozenset(
    {
        "name",
        "precision",
        "batch_size",
        "adaptive_batching",
        "output_variant",
        "allow_non_recommended_settings",
        "save_tile_features",
    }
)
_EXECUTION_KEYS = frozenset(
    {"num_gpus", "num_workers_per_gpu", "num_preprocessing_workers", "prefetch_factor", "precision"}
)
_COMPOSITE_KEYS = frozenset({"encoders", "concat_resolution", "concat_grid_size"})
_MEMBER_KEYS = frozenset(
    {
        "name",
        "feature_kind",
        "member_norm",
        "attention",
        "output_variant",
        "precision",
        "batch_size",
        "allow_non_recommended_settings",
        "dense_window_size",
        "dense_window_overlap",
    }
)


def _reject_unknown(block: dict[str, Any], allowed: frozenset[str], section: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(
            f"Config section '{section}' uses unsupported keys: {', '.join(map(str, unknown))}. "
            f"Supported: {', '.join(sorted(allowed))}."
        )


# ---------------------------------------------------------------------------
# YAML blocks -> FeaturesConfig
# ---------------------------------------------------------------------------


def features_from_layout(data: dict[str, Any]) -> FeaturesConfig:
    """Read a canonical (defaults-merged) YAML layout's extraction blocks."""
    data_section = data.get("data") or {}
    dataset_type = data_section.get("dataset_type")
    preprocessing = dict(data.get("preprocessing") or {})
    encoder_block = dict(data.get("encoder") or {})
    composite_block = dict(data.get("composite") or {})
    if encoder_block and composite_block:
        raise ValueError(
            "Set 'encoder' (single) XOR 'composite' (multi-encoder composite), not both."
        )
    for key, reason in _REMOVED_PREPROCESSING_KEYS.items():
        if key in preprocessing:
            raise ValueError(f"preprocessing.{key} was removed: {reason}.")
    # hs2p reads a tissue mask's tissue label as 1, the one value this key ever allowed.
    if preprocessing.pop("tissue_mask_tissue_value", 1) != 1:
        raise ValueError(
            "preprocessing.tissue_mask_tissue_value must be 1: hs2p reads a precomputed tissue "
            "mask's tissue label as 1."
        )
    _reject_unknown(preprocessing, _PREPROCESSING_KEYS, "preprocessing")
    _reject_unknown(encoder_block, _ENCODER_KEYS, "encoder")
    encoder = _encoder_leaf(encoder_block, composite_block)
    tiling = _tiling(preprocessing, encoder=encoder, dataset_type=dataset_type)
    extraction = slide2vec.PreprocessingConfig(
        tiling=tiling,
        segmentation=_segmentation(preprocessing),
        filter=_filter(preprocessing, tiling),
        preview=PreviewConfig(**dict(preprocessing.get("preview") or {})),
        masks=_masks(preprocessing),
        requested_region_size_px=_optional_int(preprocessing.get("requested_region_size_px")),
        region_tile_multiple=_optional_int(preprocessing.get("region_tile_multiple")),
        adaptive_batching=bool(encoder_block.get("adaptive_batching", False)),
    )
    return FeaturesConfig(
        kind=data_section.get("feature_mode", "cached"),
        encoder=encoder,
        extraction=extraction,
        execution=_execution(dict(data.get("execution") or {}), encoder_block),
        dense=_dense(preprocessing, tiling) if dataset_type in _GRID_DATASET_TYPES else None,
        cache=CacheConfig(**dict(data.get("cache") or {})),
        spacing_policy=preprocessing.get("spacing_policy", "strict"),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _encoder_leaf(encoder_block: dict[str, Any], composite_block: dict[str, Any]) -> EncoderLeaf | None:
    if composite_block:
        return EncoderLeaf(name="composite", params=_composite_params(composite_block))
    if not encoder_block:
        return None
    if not encoder_block.get("name"):
        raise ValueError("Config section 'encoder' needs a name (e.g. encoder: {name: uni2}).")
    params: dict[str, Any] = {}
    if encoder_block.get("output_variant") is not None:
        params["output_variant"] = str(encoder_block["output_variant"])
    if encoder_block.get("allow_non_recommended_settings"):
        params["allow_non_recommended_settings"] = True
    return EncoderLeaf(name=str(encoder_block["name"]), params=params)


def _composite_params(block: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(block, _COMPOSITE_KEYS, "composite")
    members = block.get("encoders")
    if not members or not isinstance(members, list):
        raise TypeError("composite.encoders must be a non-empty list of encoder mappings")
    params: dict[str, Any] = {
        "members": [_composite_member(dict(member)) for member in members],
        "concat_resolution": block.get("concat_resolution") or "grid",
    }
    if block.get("concat_grid_size") is not None:
        params["concat_grid_size"] = [int(v) for v in block["concat_grid_size"]]
    return params


def _composite_member(member: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(member, _MEMBER_KEYS, "composite.encoders[]")
    params: dict[str, Any] = {}
    if member.get("output_variant") is not None:
        params["output_variant"] = member["output_variant"]
    if member.get("allow_non_recommended_settings"):
        params["allow_non_recommended_settings"] = True
    dense: dict[str, Any] = {}
    if member.get("feature_kind") is not None:
        dense["feature_kind"] = member["feature_kind"]
    attention = dict(member.get("attention") or {})
    if "blocks" in attention:
        dense["attention_blocks"] = [int(b) for b in attention["blocks"]]
    if "include_registers" in attention:
        dense["attention_include_registers"] = bool(attention["include_registers"])
    if member.get("dense_window_size") is not None:
        dense["window_size"] = int(member["dense_window_size"])
    if member.get("dense_window_overlap") is not None:
        dense["overlap"] = float(member["dense_window_overlap"])
    execution = {key: member[key] for key in ("precision", "batch_size") if member.get(key) is not None}
    entry: dict[str, Any] = {"name": member["name"]}
    for key, value in (("params", params), ("dense", dense), ("execution", execution)):
        if value:
            entry[key] = value
    if member.get("member_norm") is not None:
        entry["member_norm"] = member["member_norm"]
    return entry


def _tiling(preprocessing: dict[str, Any], *, encoder: EncoderLeaf | None, dataset_type: str | None) -> TilingConfig | None:
    """hs2p's ``TilingConfig``, completing a missing spacing or tile size from the encoder.

    A ``TilingConfig`` is all or nothing, so the encoder fills what the YAML leaves out
    (as it did when the run resolved its preprocessing); tile datasets declare no tiling.
    """
    spacing = preprocessing.get("requested_spacing_um")
    tile_size = preprocessing.get("requested_tile_size_px")
    if (spacing is None or tile_size is None) and dataset_type != "tile" and encoder is not None:
        spacing, tile_size = _encoder_geometry(encoder, spacing, tile_size)
    if spacing is None or tile_size is None:
        return None
    return TilingConfig(
        requested_spacing_um=float(spacing),
        requested_tile_size_px=int(tile_size),
        tolerance=float(preprocessing.get("tolerance", 0.05)),
        overlap=float(preprocessing.get("overlap", 0.0)),
        backend=str(preprocessing.get("backend", "auto")),
        mask_backend=str(preprocessing.get("mask_backend", "auto")),
    )


def _encoder_geometry(encoder: EncoderLeaf, spacing: Any, tile_size: Any) -> tuple[Any, Any]:
    from slide2vec.encoders.registry import resolve_preprocessing_fields

    if encoder.name == "composite":
        if spacing is None:
            from soma.preprocessing.resolution import resolve_composite_spacing

            spacing = resolve_composite_spacing([member["name"] for member in encoder.params["members"]])
        return spacing, tile_size
    try:
        fields = resolve_preprocessing_fields(
            encoder.name,
            requested_spacing_um=None if spacing is None else float(spacing),
            requested_tile_size_px=None if tile_size is None else int(tile_size),
        )
    except (KeyError, ValueError):
        # The run resolves (and reports) what the encoder cannot decide.
        return spacing, tile_size
    return fields["spacing_um"], fields["tile_size_px"]


def _segmentation(preprocessing: dict[str, Any]) -> SegmentationConfig:
    method = preprocessing.get("tissue_method", "hsv")
    if not method:
        raise ValueError(
            "preprocessing.tissue_method must name a tissue segmentation method (hsv, otsu, "
            "threshold, sam2) or precomputed_mask."
        )
    values: dict[str, Any] = {
        "method": str(method),
        "downsample": int(preprocessing.get("seg_downsample", 64)),
        "sam2_device": str(preprocessing.get("sam2_device", "cpu")),
    }
    if preprocessing.get("sam2_num_workers") is not None:
        values["sam2_num_workers"] = int(preprocessing["sam2_num_workers"])
    return SegmentationConfig(**values)


def _filter(preprocessing: dict[str, Any], tiling: TilingConfig | None) -> FilterConfig | None:
    """hs2p's ``FilterConfig`` when the YAML states one; ``None`` (slide2vec's default:
    hs2p's filter at ``ref_tile_size`` = the tile size) otherwise."""
    reference = preprocessing.get("ref_tile_size_px")
    a_t = int(preprocessing.get("a_t", _DEFAULT_FILTER.a_t))
    if reference is None and a_t == _DEFAULT_FILTER.a_t:
        return None
    if reference is None and tiling is not None:
        reference = tiling.requested_tile_size_px
    if reference is None:
        raise ValueError(
            "preprocessing.a_t needs a reference tile size: set ref_tile_size_px or "
            "requested_tile_size_px."
        )
    return FilterConfig(ref_tile_size=int(reference), a_t=a_t)


def _masks(preprocessing: dict[str, Any]) -> MaskSamplingConfig:
    """hs2p's mask vocabulary: the annotation ``masks``/``sampling`` blocks when given
    (read by hs2p's own YAML resolver), else binary tissue tiling at ``min_coverage``."""
    masks = preprocessing.get("masks")
    sampling = dict(preprocessing.get("sampling") or {})
    if not masks:
        if sampling:
            raise ValueError(
                "sampling: requires a masks: block — it configures how ROIs are sampled from "
                "the annotation masks. Add masks: or drop sampling:."
            )
        min_coverage = dict(preprocessing.get("min_coverage") or {"tissue": 0.1})
        return replace(
            DEFAULT_MASK_SAMPLING,
            min_coverage={str(label): float(value) for label, value in min_coverage.items()},
        )
    _reject_unknown(sampling, frozenset({"strategy", "output_mode"}), "preprocessing.sampling")
    from hs2p.configs.resolvers import resolve_mask_sampling_config
    from omegaconf import OmegaConf

    block = {
        **dict(masks),
        "strategy": sampling.get("strategy", "joint"),
        "output_mode": sampling.get("output_mode", "merged"),
    }
    block.setdefault("min_coverage", {})
    return resolve_mask_sampling_config(OmegaConf.create({"tiling": {"masks": block}}))


def _dense(preprocessing: dict[str, Any], tiling: TilingConfig | None) -> slide2vec.DenseOptions | None:
    attention = dict(preprocessing.get("attention") or {})
    _reject_unknown(attention, frozenset({"blocks", "include_registers"}), "preprocessing.attention")
    if tiling is None:
        return None
    window = preprocessing.get("dense_window_size")
    return slide2vec.DenseOptions(
        spacing_um=float(tiling.requested_spacing_um),
        target_size=int(tiling.requested_tile_size_px),
        tolerance=float(tiling.tolerance),
        backend=tiling.backend,
        window_size=None if window is None else int(window),
        overlap=float(preprocessing.get("dense_window_overlap", 0.0)),
        feature_kind=preprocessing.get("feature_kind") or "patch_features",
        attention_blocks=tuple(int(b) for b in attention.get("blocks", (-1,))),
        attention_include_registers=bool(attention.get("include_registers", False)),
    )


def _execution(execution: dict[str, Any], encoder_block: dict[str, Any]) -> slide2vec.ExecutionOptions:
    _reject_unknown(execution, _EXECUTION_KEYS, "execution")
    values: dict[str, Any] = {
        "batch_size": int(encoder_block.get("batch_size", 32)),
        "save_tile_embeddings": bool(encoder_block.get("save_tile_features", False)),
    }
    precision = execution.get("precision") or encoder_block.get("precision")
    if precision is not None:
        values["precision"] = str(precision)
    for key in ("num_gpus", "num_workers_per_gpu", "num_preprocessing_workers", "prefetch_factor"):
        if execution.get(key) is not None:
            values[key] = int(execution[key])
    return slide2vec.ExecutionOptions(**values)


# ---------------------------------------------------------------------------
# FeaturesConfig -> YAML blocks
# ---------------------------------------------------------------------------


def features_to_layout(features: FeaturesConfig, *, dataset_type: str | None) -> dict[str, Any]:
    """The YAML blocks :func:`features_from_layout` reads back into ``features``.

    Upstream fields the former vocabulary cannot spell (hs2p filter knobs other than
    ``a_t``, a dense ``pad_mode``, ...) are not written; the experiment identity in
    ``run.yaml`` still carries them. ``dense`` without a tiling is written as the tiling
    it reads at (the layout derives the dense geometry from the tiling keys). A leaf
    encoder's unset tiling is written as the preset geometry the reader resolves it to,
    and a grid run's unset dense options as the defaults the reader gives it.
    """
    from soma.preprocessing.resolution import reconcile_dense_tiling

    extraction, execution, dense = features.extraction, features.execution, features.dense
    tiling = extraction.tiling
    if tiling is None:
        tiling = reconcile_dense_tiling(extraction, dense).tiling
    spacing_um = tile_size_px = None
    if tiling is not None:
        spacing_um, tile_size_px = tiling.requested_spacing_um, tiling.requested_tile_size_px
    elif dataset_type != "tile" and features.encoder is not None and not features.is_composite:
        spacing_um, tile_size_px = _encoder_geometry(features.encoder, None, None)
    if dense is None and dataset_type in _GRID_DATASET_TYPES and None not in (spacing_um, tile_size_px):
        # Read back, a grid run with a tiling gets slide2vec's default dense options.
        dense = slide2vec.DenseOptions(spacing_um=float(spacing_um), target_size=int(tile_size_px))
    segmentation = extraction.segmentation
    preprocessing: dict[str, Any] = {
        "backend": "auto" if tiling is None else tiling.backend,
        "mask_backend": "auto" if tiling is None else tiling.mask_backend,
        "requested_tile_size_px": tile_size_px,
        "requested_spacing_um": spacing_um,
        "spacing_policy": features.spacing_policy,
        "requested_region_size_px": extraction.requested_region_size_px,
        "region_tile_multiple": extraction.region_tile_multiple,
        "tissue_method": segmentation.method,
        "overlap": 0.0 if tiling is None else tiling.overlap,
        "seg_downsample": segmentation.downsample,
        "sam2_device": segmentation.sam2_device,
        "sam2_num_workers": segmentation.sam2_num_workers,
        "tolerance": 0.05 if tiling is None else tiling.tolerance,
        "ref_tile_size_px": None if extraction.filter is None else extraction.filter.ref_tile_size,
        "a_t": _DEFAULT_FILTER.a_t if extraction.filter is None else extraction.filter.a_t,
        "preview": _plain(asdict(extraction.preview)),
    }
    masks = extraction.masks
    if is_annotation_sampling(masks):
        preprocessing["masks"] = {
            "pixel_mapping": {label: list(values) for label, values in masks.pixel_mapping.items()},
            "min_coverage": dict(masks.min_coverage),
            "colors": None
            if masks.colors is None
            else {label: None if color is None else list(color) for label, color in masks.colors.items()},
        }
        preprocessing["sampling"] = {"strategy": masks.strategy, "output_mode": masks.output_mode}
    else:
        preprocessing["min_coverage"] = dict(masks.min_coverage)
    if dense is not None:
        preprocessing.update(
            {
                "dense_window_size": dense.window_size,
                "dense_window_overlap": dense.overlap,
                "feature_kind": dense.feature_kind,
                "attention": {
                    "blocks": list(dense.attention_blocks),
                    "include_registers": dense.attention_include_registers,
                },
            }
        )
    layout: dict[str, Any] = {
        "feature_mode": features.kind,
        "preprocessing": preprocessing,
        "execution": {
            "num_gpus": execution.num_gpus,
            "num_workers_per_gpu": execution.num_workers_per_gpu,
            "num_preprocessing_workers": execution.num_preprocessing_workers,
            "prefetch_factor": execution.prefetch_factor,
            "precision": execution.precision,
        },
        "cache": _plain(asdict(features.cache)),
        "encoder": None,
        "composite": None,
    }
    encoder = features.encoder
    if encoder is not None and features.is_composite:
        layout["composite"] = _composite_block(encoder.params)
    elif encoder is not None:
        layout["encoder"] = {
            "name": encoder.name,
            "precision": execution.precision,
            "batch_size": execution.batch_size,
            "adaptive_batching": extraction.adaptive_batching,
            "output_variant": encoder.output_variant,
            "allow_non_recommended_settings": encoder.allow_non_recommended_settings,
            "save_tile_features": execution.save_tile_embeddings,
        }
    return layout


def _composite_block(params: dict[str, Any]) -> dict[str, Any]:
    from soma.encoders import CompositeParams

    composite = CompositeParams(**params)
    members = []
    for member in composite.members:
        entry: dict[str, Any] = {"name": member.name, **dict(member.params)}
        dense = dict(member.dense)
        if "feature_kind" in dense:
            entry["feature_kind"] = dense["feature_kind"]
        attention = {}
        if "attention_blocks" in dense:
            attention["blocks"] = [int(b) for b in dense["attention_blocks"]]
        if "attention_include_registers" in dense:
            attention["include_registers"] = bool(dense["attention_include_registers"])
        if attention:
            entry["attention"] = attention
        if "window_size" in dense:
            entry["dense_window_size"] = dense["window_size"]
        if "overlap" in dense:
            entry["dense_window_overlap"] = dense["overlap"]
        entry.update(dict(member.execution))
        if member.member_norm is not None:
            entry["member_norm"] = member.member_norm
        members.append(entry)
    return {
        "encoders": members,
        "concat_resolution": composite.concat_resolution,
        "concat_grid_size": None if composite.concat_grid_size is None else list(composite.concat_grid_size),
    }


def _plain(value: Any) -> Any:
    """YAML-safe primitives (tuples to lists, paths to strings)."""
    from pathlib import Path

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value
