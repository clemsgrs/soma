"""Resolve a pipeline's extraction settings from its declared components."""

from __future__ import annotations

from slide2vec import DenseOptions, PreprocessingConfig

from soma.config import PipelineConfig
from soma.encoders.validation import resolve_preprocessing_config
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def resolve_composite_spacing(member_names: list[str]) -> float:
    """Resolve the one spacing shared by every v1 composite member."""
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    per_member = {name: resolve_preprocessing_defaults(name)["spacing_um"] for name in member_names}
    unique = sorted(set(per_member.values()))
    if len(unique) != 1:
        raise ValueError(
            "composite members do not share a single default spacing "
            f"({per_member}); set preprocessing.requested_spacing_um explicitly "
            "(v1 reads every member at the same µm/px)."
        )
    return unique[0]


def resolve_pipeline_preprocessing(config: PipelineConfig) -> PreprocessingConfig:
    """The slide2vec ``PreprocessingConfig`` every pipeline path extracts with.

    The held ``features.extraction`` with what the run decides filled in: HIPT's region
    geometry from its aggregator params, and the tiling and filter the encoder decides
    when they were left unset. Tile datasets declare no tiling, so theirs is returned as
    held; so is a composite's without a tiling (its members read at the dense options).
    """
    features = config.features
    extraction = features.extraction
    if config.dataset_type == "tile":
        return extraction
    extraction = derive_preprocessing_for_aggregator(extraction, config.aggregator)
    encoder = features.encoder
    if encoder is None:
        return extraction
    if features.is_composite:
        if extraction.tiling is None:
            return extraction
        from soma.encoders import CompositeParams

        first_member = CompositeParams(**encoder.params).members[0].name
        return resolve_preprocessing_config(extraction, encoder_name=first_member)
    return resolve_preprocessing_config(extraction, encoder_name=encoder.name)


def resolve_pipeline_dense(config: PipelineConfig) -> DenseOptions:
    """The slide2vec ``DenseOptions`` a grid run reads and encodes with.

    ``features.dense`` when set; otherwise slide2vec's dense defaults at the resolved
    tiling's spacing, tile size, tolerance and backend.
    """
    if config.features.dense is not None:
        return config.features.dense
    tiling = resolve_pipeline_preprocessing(config).tiling
    if tiling is None:
        raise ValueError(
            "A grid run needs a read spacing and a supervision tile size: set "
            "features.dense, or preprocessing.requested_spacing_um and "
            "requested_tile_size_px (or use an encoder that advertises them)."
        )
    return DenseOptions(
        spacing_um=float(tiling.requested_spacing_um),
        target_size=int(tiling.requested_tile_size_px),
        tolerance=float(tiling.tolerance),
        backend=tiling.backend,
    )


__all__ = ["resolve_composite_spacing", "resolve_pipeline_dense", "resolve_pipeline_preprocessing"]
