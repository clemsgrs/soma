"""Public helper: recompute a feature-cache key from explicit operator inputs.

Mirrors the resolution chain that ``FeatureExtractor`` runs before calling the
underlying ``build_*_cache_key`` primitives, so downstream tools can ask
"what cache_key would soma produce for these inputs?" without running
extraction or loading model weights.

The arguments mean the same thing for every kind: ``preprocessing`` is the
**tile-level** preprocessing soma will tile the slides with, and
``execution`` is the execution config the cache's own encoder runs under.
For slide/patient kinds the upstream tile encoder is auto-derived from
``encoder_name`` via the registry (the slide/patient encoder's
``tile_encoder`` metadata field), and its execution defaults to
``EncoderConfig(name=tile_encoder_name)`` — pass ``tile_encoder_name`` to
override the auto-derivation.

Contract (deliberately strict so the function stays a pure transform):
- ``preprocessing.tissue_method`` must be set. The auto-promotion to
  ``"precomputed_mask"`` that ``FeatureExtractor`` performs by inspecting the
  dataset is intentionally not replicated here — caller is explicit. If the
  caller knows the dataset has precomputed masks for every sample, pass
  ``has_precomputed_masks=True`` to mirror soma's promotion; the resolved
  preprocessing will be forced to ``tissue_method="precomputed_mask"`` before
  hashing.
- ``execution.output_variant`` may be ``None``; it falls back to the encoder's
  registry default via :func:`slide2vec.encoders.resolve_encoder_output`.
"""

from __future__ import annotations

from dataclasses import replace

from soma.config import EncoderConfig, PreprocessingConfig
from soma.encoders import encoder_registry
from soma.encoders.validation import resolve_preprocessing_config
from soma.cache.keys import (
    build_hierarchical_cache_key,
    build_patient_cache_key,
    build_slide_cache_key,
    build_tile_cache_key,
    execution_signature,
    preprocessing_signature,
)
from slide2vec.encoders.registry import resolve_encoder_output


_SUPPORTED_KINDS = ("tile", "hierarchical", "slide", "patient")


def _resolved_output_variant(encoder_name: str, requested: str | None) -> str:
    info = encoder_registry.info(encoder_name)
    resolved = resolve_encoder_output(
        encoder_name,
        requested_output_variant=requested,
        metadata=info,
    )
    return str(resolved["output_variant"])


def _require_tissue_method(preprocessing: PreprocessingConfig) -> None:
    if preprocessing.tissue_method is None:
        raise ValueError(
            "preprocessing.tissue_method must be set explicitly when computing a "
            "cache_key — the auto-promotion soma performs against a Dataset is "
            "not replicated here."
        )


def _resolve(
    encoder_name: str,
    preprocessing: PreprocessingConfig,
    *,
    has_precomputed_masks: bool = False,
) -> PreprocessingConfig:
    """Return preprocessing with encoder-driven defaults filled in."""
    _require_tissue_method(preprocessing)
    resolved_prep = resolve_preprocessing_config(
        EncoderConfig(name=encoder_name),
        preprocessing,
        model_metadata=encoder_registry.info(encoder_name),
    )
    if has_precomputed_masks:
        resolved_prep = replace(resolved_prep, tissue_method="precomputed_mask")
    return resolved_prep


def _exec_with_output_variant(execution: EncoderConfig, output_variant: str) -> EncoderConfig:
    if execution.output_variant is None:
        return replace(execution, output_variant=output_variant)
    return execution


def compute_cache_key(
    *,
    kind: str,
    encoder_name: str,
    preprocessing: PreprocessingConfig,
    execution: EncoderConfig | None = None,
    output_variant: str | None = None,
    has_precomputed_masks: bool = False,
    tile_encoder_name: str | None = None,
) -> str:
    """Recompute the cache_key soma would assign to the given inputs.

    See module docstring for the explicit-inputs contract.

    For slide/patient kinds, ``preprocessing`` / ``execution`` /
    ``output_variant`` / ``has_precomputed_masks`` all describe the upstream
    tile recipe; the slide- or patient-stage execution is auto-derived from
    ``encoder_name`` using registry defaults. ``tile_encoder_name`` defaults
    to the slide/patient encoder's registry-declared ``tile_encoder``.
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported kind '{kind}'; expected one of {_SUPPORTED_KINDS}")

    if kind in ("tile", "hierarchical"):
        resolved_prep = _resolve(encoder_name, preprocessing, has_precomputed_masks=has_precomputed_masks)
        ov = _resolved_output_variant(encoder_name, output_variant)
        exec_for_cache = _exec_with_output_variant(
            execution if execution is not None else EncoderConfig(name=encoder_name),
            ov,
        )
        builder = build_tile_cache_key if kind == "tile" else build_hierarchical_cache_key
        return builder(
            tile_encoder_name=encoder_name,
            preprocessing=resolved_prep,
            execution=exec_for_cache,
            output_variant=ov,
        )

    # slide / patient — wrap an upstream tile recipe.
    resolved_tile_encoder = tile_encoder_name or encoder_registry.info(encoder_name)["tile_encoder"]
    resolved_tile_prep = _resolve(
        resolved_tile_encoder, preprocessing, has_precomputed_masks=has_precomputed_masks,
    )
    tile_ov = _resolved_output_variant(resolved_tile_encoder, output_variant)
    tile_exec = _exec_with_output_variant(
        execution if execution is not None else EncoderConfig(name=resolved_tile_encoder),
        tile_ov,
    )
    tile_dep = {
        "tile_encoder_name": resolved_tile_encoder,
        "tile_preprocessing": preprocessing_signature(resolved_tile_prep),
        "tile_execution": execution_signature(
            tile_exec,
            encoder_name=resolved_tile_encoder,
            preprocessing=resolved_tile_prep,
            output_variant=tile_ov,
        ),
    }
    # The slide/patient stage runs the cache's own encoder over the upstream
    # features — no preprocessing of its own, execution defaults from registry.
    stage_ov = _resolved_output_variant(encoder_name, None)
    stage_exec = _exec_with_output_variant(EncoderConfig(name=encoder_name), stage_ov)
    builder = build_slide_cache_key if kind == "slide" else build_patient_cache_key
    encoder_kw = "slide_encoder_name" if kind == "slide" else "patient_encoder_name"
    return builder(
        **{encoder_kw: encoder_name},
        tile_dependency_signature=tile_dep,
        execution=stage_exec,
        output_variant=stage_ov,
    )
