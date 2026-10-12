"""Public helper: recompute a feature-cache key from explicit operator inputs.

Mirrors the resolution chain that ``FeatureExtractor`` runs before calling the
underlying ``build_*_cache_key`` primitives, so downstream tools can ask
"what cache_key would soma produce for these inputs?" without running
extraction or loading model weights.

The arguments mean the same thing for every kind: ``preprocessing`` is the
**tile-level** slide2vec ``PreprocessingConfig`` soma will tile the slides with, and
``execution`` is the slide2vec ``ExecutionOptions`` the cache's own encoder runs under
(only its precision enters a key; ``None`` takes the encoder's recommendation). For
slide/patient kinds the upstream tile encoder is auto-derived from ``encoder_name`` via
the registry (the slide/patient encoder's ``tile_encoder`` metadata field) — pass
``tile_encoder_name`` to override the auto-derivation.

Contract (deliberately strict so the function stays a pure transform):
- The auto-promotion to ``segmentation.method="precomputed_mask"`` that
  ``FeatureExtractor`` performs by inspecting the dataset is not replicated here. If the
  caller knows the dataset has precomputed masks for every sample, pass
  ``has_precomputed_masks=True`` to mirror soma's promotion before hashing.
- ``output_variant`` may be ``None``; it falls back to the encoder's registry default via
  :func:`slide2vec.encoders.resolve_encoder_output`.
"""

from __future__ import annotations

from dataclasses import replace

from slide2vec import ExecutionOptions, PreprocessingConfig
from slide2vec.encoders.registry import encoder_registry, resolve_encoder_output

from soma.cache.keys import (
    build_hierarchical_cache_key,
    build_patient_cache_key,
    build_slide_cache_key,
    build_tile_cache_key,
    execution_signature,
    preprocessing_signature,
)
from soma.encoders.validation import resolve_preprocessing_config

_SUPPORTED_KINDS = ("tile", "hierarchical", "slide", "patient")


def resolved_output_variant(encoder_name: str, requested: str | None) -> str:
    """Resolve ``requested`` (``None`` = the encoder's default) to a concrete variant name."""
    info = encoder_registry.info(encoder_name)
    resolved = resolve_encoder_output(
        encoder_name,
        requested_output_variant=requested,
        metadata=info,
    )
    return str(resolved["output_variant"])


def _resolve(
    encoder_name: str,
    preprocessing: PreprocessingConfig,
    *,
    has_precomputed_masks: bool = False,
) -> PreprocessingConfig:
    """Return preprocessing with encoder-driven defaults filled in."""
    resolved = resolve_preprocessing_config(
        preprocessing,
        encoder_name=encoder_name,
        model_metadata=encoder_registry.info(encoder_name),
    )
    if has_precomputed_masks:
        resolved = replace(
            resolved, segmentation=replace(resolved.segmentation, method="precomputed_mask")
        )
    return resolved


def compute_cache_key(
    *,
    kind: str,
    encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions | None = None,
    output_variant: str | None = None,
    has_precomputed_masks: bool = False,
    tile_encoder_name: str | None = None,
    dtype: str = "fp32",
) -> str:
    """Recompute the cache_key soma would assign to the given inputs.

    See module docstring for the explicit-inputs contract.

    For slide/patient kinds, ``preprocessing`` / ``execution`` /
    ``output_variant`` / ``has_precomputed_masks`` all describe the upstream
    tile recipe; the slide- or patient-stage precision is the encoder's
    recommendation. ``tile_encoder_name`` defaults to the slide/patient encoder's
    registry-declared ``tile_encoder``.

    ``dtype`` is the on-disk storage precision (``cache.dtype`` resolved to
    ``'fp16'``/``'fp32'``).
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported kind '{kind}'; expected one of {_SUPPORTED_KINDS}")

    if kind in ("tile", "hierarchical"):
        resolved_prep = _resolve(encoder_name, preprocessing, has_precomputed_masks=has_precomputed_masks)
        builder = build_tile_cache_key if kind == "tile" else build_hierarchical_cache_key
        return builder(
            tile_encoder_name=encoder_name,
            preprocessing=resolved_prep,
            execution=execution,
            output_variant=resolved_output_variant(encoder_name, output_variant),
            dtype=dtype,
        )

    # slide / patient — wrap an upstream tile recipe.
    resolved_tile_encoder = tile_encoder_name or encoder_registry.info(encoder_name)["tile_encoder"]
    resolved_tile_prep = _resolve(
        resolved_tile_encoder, preprocessing, has_precomputed_masks=has_precomputed_masks,
    )
    tile_dep = {
        "tile_encoder_name": resolved_tile_encoder,
        "tile_preprocessing": preprocessing_signature(resolved_tile_prep),
        "tile_execution": execution_signature(
            execution,
            encoder_name=resolved_tile_encoder,
            preprocessing=resolved_tile_prep,
            output_variant=resolved_output_variant(resolved_tile_encoder, output_variant),
        ),
    }
    # The slide/patient stage runs the cache's own encoder over the upstream features —
    # no preprocessing of its own, at the precision the registry recommends.
    builder = build_slide_cache_key if kind == "slide" else build_patient_cache_key
    encoder_kw = "slide_encoder_name" if kind == "slide" else "patient_encoder_name"
    return builder(
        **{encoder_kw: encoder_name},
        tile_dependency_signature=tile_dep,
        execution=None,
        output_variant=resolved_output_variant(encoder_name, None),
        dtype=dtype,
    )
