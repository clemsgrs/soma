"""Resolve a pipeline's extraction settings from its declared components."""

from __future__ import annotations

from dataclasses import replace

from hs2p import TilingConfig
from slide2vec import DenseOptions, PreprocessingConfig

from soma.encoders.validation import resolve_preprocessing_config
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def reconcile_dense_tiling(preprocessing: PreprocessingConfig, dense: DenseOptions | None) -> PreprocessingConfig:
    """``preprocessing`` with a tiling that names ``dense``'s read geometry.

    A grid run samples ROIs and crops their masks at the tiling, and reads and encodes
    them at ``dense``, so both must agree on the spacing, tile size, tolerance and
    backend. An unset tiling takes them from ``dense``; a set one that disagrees is
    refused. Without ``dense``, ``preprocessing`` is returned as held.
    """
    if dense is None:
        return preprocessing
    geometry = (float(dense.spacing_um), int(dense.target_size), float(dense.tolerance), dense.backend)
    tiling = preprocessing.tiling
    if tiling is None:
        spacing_um, tile_size_px, tolerance, backend = geometry
        return replace(
            preprocessing,
            tiling=TilingConfig(
                requested_spacing_um=spacing_um,
                requested_tile_size_px=tile_size_px,
                tolerance=tolerance,
                overlap=0.0,
                backend=backend,
            ),
        )
    held = (
        float(tiling.requested_spacing_um),
        int(tiling.requested_tile_size_px),
        float(tiling.tolerance),
        tiling.backend,
    )
    if held != geometry:
        raise ValueError(
            "The tiling and the dense options disagree on the read geometry (spacing µm/px, "
            f"tile size px, tolerance, backend): tiling {held}, dense {geometry}. A grid run "
            "samples ROIs at the tiling and reads them at the dense options; leave the "
            "tiling unset or make them agree."
        )
    return preprocessing


def resolve_pipeline_preprocessing(config) -> PreprocessingConfig:
    """The slide2vec ``PreprocessingConfig`` every pipeline path extracts with.

    ``config`` is an :class:`~soma.config.ExperimentConfig` or its
    :class:`~soma.plan.Plan`. The held ``features.extraction`` with what the run decides
    filled in: HIPT's region geometry from its aggregator params, and the tiling and
    filter the encoder decides when they were left unset. A grid run's tiling is
    reconciled with ``features.dense`` first (:func:`reconcile_dense_tiling`). A set run
    over tiles declares no tiling, so its extraction is returned as held; so is a
    composite's without a tiling (its members read at the dense options).
    """
    from soma.plan import learner_shape, sample_unit

    features = config.features
    extraction = features.extraction
    shape = learner_shape(config)
    if shape == "set" and sample_unit(config) == "tile":
        return extraction
    if shape == "grid":
        extraction = reconcile_dense_tiling(extraction, features.dense)
    aggregator = None if config.learner is None else config.learner.params.get("aggregator")
    extraction = derive_preprocessing_for_aggregator(extraction, aggregator)
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


def resolve_pipeline_dense(config) -> DenseOptions:
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
            "features.dense, or features.extraction.tiling (or use an encoder that "
            "advertises them)."
        )
    return DenseOptions(
        spacing_um=float(tiling.requested_spacing_um),
        target_size=int(tiling.requested_tile_size_px),
        tolerance=float(tiling.tolerance),
        backend=tiling.backend,
    )


__all__ = [
    "reconcile_dense_tiling",
    "resolve_pipeline_dense",
    "resolve_pipeline_preprocessing",
]
