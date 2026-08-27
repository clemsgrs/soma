"""Resolve a pipeline's preprocessing geometry from its declared components."""

from __future__ import annotations

from dataclasses import replace

from soma.config import CompositeConfig, PipelineConfig, PreprocessingConfig
from soma.encoders.validation import resolve_preprocessing_config
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def resolve_composite_spacing(composite: CompositeConfig) -> float:
    """Resolve the one spacing shared by every v1 composite member."""
    from slide2vec.encoders.registry import resolve_preprocessing_requirements

    per_member: dict[str, float] = {}
    for member in composite.encoders:
        spacing = resolve_preprocessing_requirements(member.name)["spacing_um"]
        if isinstance(spacing, (list, tuple)):
            if len(spacing) != 1:
                raise ValueError(
                    f"composite member '{member.name}' supports multiple spacings "
                    f"{list(spacing)}; set preprocessing.requested_spacing_um explicitly."
                )
            spacing = spacing[0]
        per_member[member.name] = float(spacing)
    unique = sorted(set(per_member.values()))
    if len(unique) != 1:
        raise ValueError(
            "composite members do not share a single supported spacing "
            f"({per_member}); set preprocessing.requested_spacing_um explicitly "
            "(v1 reads every member at the same µm/px)."
        )
    return unique[0]


def resolve_pipeline_preprocessing(config: PipelineConfig) -> PreprocessingConfig:
    """Return the authoritative preprocessing config used by every pipeline path."""
    if config.dataset_type == "tile":
        return PreprocessingConfig()
    preprocessing = derive_preprocessing_for_aggregator(
        config.preprocessing, config.aggregator
    )
    if config.encoder is not None:
        return resolve_preprocessing_config(config.encoder, preprocessing)
    if config.composite is not None and preprocessing.requested_spacing_um is None:
        return replace(
            preprocessing,
            requested_spacing_um=resolve_composite_spacing(config.composite),
        )
    return preprocessing


__all__ = ["resolve_composite_spacing", "resolve_pipeline_preprocessing"]
