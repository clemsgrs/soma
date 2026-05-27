"""Public helper: recompute a feature-cache key from explicit operator inputs.

Mirrors the resolution chain that ``FeatureExtractor`` runs before calling the
underlying ``build_*_cache_key`` primitives, so downstream tools can ask
"what cache_key would soma produce for these inputs?" without running
extraction or loading model weights.

Contract (deliberately strict so the function stays a pure transform):
- ``preprocessing.tissue_method`` must be set. The auto-promotion to
  ``"precomputed_mask"`` that ``FeatureExtractor`` performs by inspecting the
  dataset is intentionally not replicated here — caller is explicit.
- ``execution.output_variant`` may be ``None``; it falls back to the encoder's
  registry default via :func:`slide2vec.encoders.resolve_encoder_output`.
- Slide and patient kinds require an upstream tile recipe
  (``tile_preprocessing`` + ``tile_execution``); ``tile_encoder_name`` defaults
  to the slide/patient encoder's registry-declared ``tile_encoder``.
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


def _resolve(encoder_name: str, preprocessing: PreprocessingConfig) -> tuple[PreprocessingConfig, EncoderConfig]:
    """Return (resolved_preprocessing, encoder_config-with-output_variant)."""
    _require_tissue_method(preprocessing)
    execution_with_name = EncoderConfig(name=encoder_name)
    resolved_prep = resolve_preprocessing_config(
        execution_with_name,
        preprocessing,
        model_metadata=encoder_registry.info(encoder_name),
    )
    return resolved_prep, execution_with_name


def _tile_dependency_signature(
    *,
    tile_encoder_name: str,
    tile_preprocessing: PreprocessingConfig,
    tile_execution: EncoderConfig,
    tile_output_variant: str | None,
) -> dict:
    resolved_prep, _ = _resolve(tile_encoder_name, tile_preprocessing)
    ov = _resolved_output_variant(tile_encoder_name, tile_output_variant)
    return {
        "tile_encoder_name": tile_encoder_name,
        "tile_preprocessing": preprocessing_signature(resolved_prep),
        "tile_execution": execution_signature(
            replace(tile_execution, output_variant=ov),
            encoder_name=tile_encoder_name,
            preprocessing=resolved_prep,
            output_variant=ov,
        ),
    }


def compute_cache_key(
    *,
    kind: str,
    encoder_name: str,
    preprocessing: PreprocessingConfig | None,
    execution: EncoderConfig,
    output_variant: str | None = None,
    tile_encoder_name: str | None = None,
    tile_preprocessing: PreprocessingConfig | None = None,
    tile_execution: EncoderConfig | None = None,
    tile_output_variant: str | None = None,
) -> str:
    """Recompute the cache_key soma would assign to the given inputs.

    See module docstring for the explicit-inputs contract.
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(f"unsupported kind '{kind}'; expected one of {_SUPPORTED_KINDS}")

    if kind in ("tile", "hierarchical"):
        if preprocessing is None:
            raise ValueError(f"preprocessing is required for kind='{kind}'")
        resolved_prep, _ = _resolve(encoder_name, preprocessing)
        ov = _resolved_output_variant(encoder_name, output_variant)
        exec_with_ov = replace(execution, output_variant=ov) if execution.output_variant is None else execution
        if kind == "tile":
            return build_tile_cache_key(
                tile_encoder_name=encoder_name,
                preprocessing=resolved_prep,
                execution=exec_with_ov,
                output_variant=ov,
            )
        return build_hierarchical_cache_key(
            tile_encoder_name=encoder_name,
            preprocessing=resolved_prep,
            execution=exec_with_ov,
            output_variant=ov,
        )

    # slide / patient — need upstream tile recipe.
    if tile_preprocessing is None or tile_execution is None:
        raise ValueError(
            f"kind='{kind}' requires tile_preprocessing and tile_execution "
            "for the upstream tile cache recipe."
        )
    resolved_tile_encoder = tile_encoder_name or encoder_registry.info(encoder_name)["tile_encoder"]
    tile_dep = _tile_dependency_signature(
        tile_encoder_name=resolved_tile_encoder,
        tile_preprocessing=tile_preprocessing,
        tile_execution=tile_execution,
        tile_output_variant=tile_output_variant,
    )
    ov = _resolved_output_variant(encoder_name, output_variant)
    exec_with_ov = replace(execution, output_variant=ov) if execution.output_variant is None else execution
    if kind == "slide":
        return build_slide_cache_key(
            slide_encoder_name=encoder_name,
            tile_dependency_signature=tile_dep,
            execution=exec_with_ov,
            output_variant=ov,
        )
    return build_patient_cache_key(
        patient_encoder_name=encoder_name,
        tile_dependency_signature=tile_dep,
        execution=exec_with_ov,
        output_variant=ov,
    )
