"""Resolve a run's extraction settings from its plan's declared components."""

from __future__ import annotations

from slide2vec import DenseOptions, PreprocessingConfig

from soma.extract.preprocessing.hierarchy import derive_preprocessing_for_aggregator
from soma.extract.preprocessing.resolution import reconcile_dense_tiling
from soma.models.encoders.validation import resolve_preprocessing_config
from soma.plan import learner_shape, sample_unit

__all__ = ["resolve_pipeline_dense", "resolve_pipeline_preprocessing"]


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
        from soma.models.encoders import CompositeParams

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
