"""Presets: the beginner path into soma (design §8.4).

Each preset is a plain function that returns a complete
:class:`~soma.config.ExperimentConfig` from a few typed arguments. There is no override
mechanism: change the returned config by assigning its fields
(``cfg.trainer.max_epochs = 20``), or print it as YAML with ``soma init <preset>``, edit
the file and run it with ``soma run``. ``soma run --preset`` builds the config from a
preset and the cohort, splits, output directory and seed given on the command line.

A preset is named after what its head trains on: one tile image's embedding (``tile_*``),
an aggregator's pooling of a slide's tiles (``mil_*``) or a pretrained slide encoder's
embedding (``slide_*``); ``segmentation`` and ``detection`` decode each tile's token grid.
Each preset checks its encoder's level when it is built and names the preset to use for
an encoder of another level. No preset covers patient-level encoders yet.

The cohort paths of a preset config are placeholders (``dataset.csv``, ``splits.csv``):
point ``cfg.cohort`` at your files. The registered benchmarks build their configs from
these presets and then write their protocol values, so every benchmark run exercises them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal

from hs2p import MaskSamplingConfig, TilingConfig
from slide2vec import PreprocessingConfig
from slide2vec.encoders.registry import resolve_preprocessing_fields

from soma.config import (
    CohortConfig,
    Component,
    ExperimentConfig,
    FeaturesConfig,
    LearnerConfig,
    LoadersConfig,
)
from soma.plan import encoder_level
from soma.registry import Registry, signature_params

__all__ = [
    "detection",
    "mil_classification",
    "mil_regression",
    "mil_survival",
    "preset_registry",
    "segmentation",
    "slide_classification",
    "slide_regression",
    "slide_survival",
    "tile_classification",
    "tile_regression",
]

#: The ``presets`` domain of :func:`soma.components`. An entry is the preset function; its
#: params are the function's arguments (``None`` for a required one, listed in ``required``).
#: The presets list in the order they are registered here, grouped by what the head trains
#: on: tile images, a slide's pooled tiles, a slide encoder's embedding, a dense token grid.
preset_registry = Registry("presets", listed_by_name=False)


def _register(preset: Callable[..., ExperimentConfig]) -> Callable[..., ExperimentConfig]:
    signature = inspect.signature(preset)
    preset_registry.register(
        preset.__name__,
        preset,
        params=signature_params(preset, name=f"{preset.__name__.title().replace('_', '')}Params"),
        metadata={
            "description": inspect.getdoc(preset).splitlines()[0],
            "required": [name for name, p in signature.parameters.items() if p.default is p.empty],
        },
    )
    return preset


#: Why a dense preset needs a tile-level encoder.
_DENSE = "the decoder decodes each tile's token grid"


def _cohort() -> CohortConfig:
    return CohortConfig(records="dataset.csv", splits="splits.csv")


def _features(encoder: str, level: str, instead: str) -> FeaturesConfig:
    """``encoder``'s features, refused unless it is a ``level``-level encoder; ``instead``
    says what to use for an encoder of another level."""
    features = FeaturesConfig(encoder=encoder)  # checks the name
    found = encoder_level(encoder)
    if found == "patient":
        raise ValueError(f"encoder '{encoder}' is patient-level: no preset covers patient-level encoders yet.")
    if found != level:
        raise ValueError(
            f"encoder '{encoder}' is {found}-level, but this preset needs a {level}-level encoder: {instead}."
        )
    return features


def _tile_encoder(encoder: str, slide_preset: str) -> FeaturesConfig:
    """A tile encoder's features; a slide encoder is sent to ``slide_preset``."""
    return _features(encoder, "tile", f"use {slide_preset}, which trains on a slide encoder's embedding")


def _tissue_tiles(features: FeaturesConfig, spacing: float | None, tile_size: int | None) -> FeaturesConfig:
    """``features`` over the tiles holding at least 10% tissue (hs2p's binary tissue
    vocabulary). The tiling is left unset, for the run to take the encoder's recommended
    tile size and spacing, unless ``spacing`` or ``tile_size`` is given: the one not given
    then comes from the encoder, and the tolerance and overlap are slide2vec's defaults."""
    masks = MaskSamplingConfig(pixel_mapping={"background": 0, "tissue": 1}, min_coverage={"tissue": 0.1})
    features.extraction = PreprocessingConfig(masks=masks)
    if spacing is None and tile_size is None:
        return features
    encoder = features.encoder.name
    try:
        fields = resolve_preprocessing_fields(encoder, requested_spacing_um=spacing, requested_tile_size_px=tile_size)
    except ValueError:
        if spacing is not None:
            raise
        # An encoder of several spacings names no default; slide2vec's message names its
        # own settings, so name the preset's.
        raise ValueError(f"encoder '{encoder}' has no default spacing: pass spacing as well as tile_size") from None
    tiling = TilingConfig(
        requested_spacing_um=fields["spacing_um"] if spacing is None else spacing,
        requested_tile_size_px=fields["tile_size_px"] if tile_size is None else tile_size,
        tolerance=0.05,
        overlap=0.0,
    )
    features.extraction = PreprocessingConfig(tiling=tiling, masks=masks)
    return features


def _tile_probe(encoder: str, head: Component, slide_preset: str) -> ExperimentConfig:
    """``head`` on tile encoder ``encoder``'s embedding of each image: batches of 256, AdamW
    (lr 3e-4) without a schedule. ``slide_preset`` is the preset for a slide encoder."""
    return ExperimentConfig(
        cohort=_cohort(),
        features=_tile_encoder(encoder, slide_preset),
        loaders=LoadersConfig(batch_size=256),
        learner=LearnerConfig("mil", {"head": head, "optimizer": Component("adamw", {"lr": 3e-4})}),
    )


@_register
def tile_classification(encoder: str) -> ExperimentConfig:
    """Classify tile images: a linear probe on a frozen tile encoder's embeddings.

    Each row of the cohort is one image, embedded whole by ``encoder`` (no tiling, no
    aggregator), and a ``multiclass_classification`` head learns the cohort's ``label``
    column. Batches of 256 embeddings, AdamW (lr 3e-4) without a schedule, 50 epochs,
    the best tune-loss checkpoint and early stopping after 10 epochs without improvement.
    """
    return _tile_probe(encoder, Component("multiclass_classification"), "slide_classification")


@_register
def tile_regression(encoder: str) -> ExperimentConfig:
    """Regress a value from tile images: a linear probe on a frozen tile encoder's embeddings.

    Each row of the cohort is one image, embedded whole by ``encoder`` (no tiling, no
    aggregator), and a ``regression`` head learns the cohort's ``value`` column with a mean
    squared error loss. To regress several values, set the head's ``num_outputs`` param to
    their count ``G``: the head then reads the columns ``value_0`` to ``value_{G-1}``.
    Batches of 256 embeddings, AdamW (lr 3e-4) without a schedule, 50 epochs, the best
    tune-loss checkpoint and early stopping after 10 epochs without improvement.
    """
    return _tile_probe(encoder, Component("regression"), "slide_regression")


def _mil(
    encoder: str,
    aggregator: str,
    spacing: float | None,
    tile_size: int | None,
    head: Component,
    slide_preset: str,
) -> ExperimentConfig:
    """``head`` over ``aggregator``'s pooling of each slide's tissue tiles, trained with
    AdamW (lr 1e-4) and a cosine schedule. ``slide_preset`` is the preset for a slide encoder."""
    return ExperimentConfig(
        cohort=_cohort(),
        features=_tissue_tiles(_tile_encoder(encoder, slide_preset), spacing, tile_size),
        learner=LearnerConfig(
            "mil",
            {
                "aggregator": Component(aggregator),
                "head": head,
                "optimizer": Component("adamw", {"lr": 1e-4}),
                "scheduler": [Component("cosine")],
            },
        ),
    )


def _survival_head(loss: str, num_bins: int | None, *, cox: Component) -> Component:
    """The head ``loss`` picks: ``discrete_survival`` over ``num_bins`` bins derived from
    each fold's training split (``nll``), or ``cox`` (``cox``, which takes no bins)."""
    if loss == "nll":
        if num_bins is None:
            raise ValueError(
                "loss 'nll' needs num_bins: the number of time bins the run derives from each "
                "fold's training split."
            )
        return Component("discrete_survival", {"binning": "train_quantiles", "num_bins": num_bins})
    if loss == "cox":
        if num_bins is not None:
            raise ValueError("loss 'cox' takes no num_bins: a Cox head has no time bins.")
        return cox
    raise ValueError(f"loss must be one of 'nll', 'cox', got {loss!r}.")


@_register
def mil_classification(
    encoder: str, aggregator: str, spacing: float | None = None, tile_size: int | None = None
) -> ExperimentConfig:
    """Classify whole slides: a frozen tile encoder, a MIL aggregator and a classifier.

    Each slide is tiled at ``spacing`` (µm per pixel) and ``tile_size`` (pixels), keeping
    the tiles with at least 10% tissue. An argument left out takes the encoder's
    recommended value; ``virchow2``, which supports several spacings, recommends none, so
    give it ``spacing``. :func:`soma.plan.check` refuses a spacing or tile size the
    encoder does not support. The tiles are embedded once and cached, and
    ``aggregator`` (``abmil``, ``transmil``, ``mean_pool``, ...; ``soma list aggregators``)
    pools the bag for a ``multiclass_classification`` head (two classes or more; the
    classes come from the cohort's ``label`` column). AdamW (lr 1e-4) with a cosine
    schedule, 50 epochs, the best tune-loss checkpoint and early stopping after 10 epochs
    without improvement.
    """
    head = Component("multiclass_classification")
    return _mil(encoder, aggregator, spacing, tile_size, head, "slide_classification")


@_register
def mil_regression(
    encoder: str, aggregator: str, spacing: float | None = None, tile_size: int | None = None
) -> ExperimentConfig:
    """Regress a value from whole slides: a frozen tile encoder, a MIL aggregator and a regressor.

    Slides are tiled at ``spacing`` and ``tile_size`` and embedded as in
    :func:`mil_classification`; ``aggregator`` pools the bag for a ``regression`` head,
    which learns the cohort's ``value`` column with a mean squared error loss. To regress
    several values, set the head's ``num_outputs`` param to their count ``G``: the head
    then reads the columns ``value_0`` to ``value_{G-1}``. AdamW (lr 1e-4) with a cosine
    schedule, 50 epochs, the best tune-loss checkpoint and early stopping after 10 epochs
    without improvement.
    """
    return _mil(encoder, aggregator, spacing, tile_size, Component("regression"), "slide_regression")


@_register
def mil_survival(
    encoder: str,
    aggregator: str,
    loss: Literal["nll", "cox"],
    num_bins: int | None = None,
    spacing: float | None = None,
    tile_size: int | None = None,
) -> ExperimentConfig:
    """Predict survival from whole slides: a MIL bag and a discrete-time or Cox head.

    Slides are tiled at ``spacing`` and ``tile_size`` and embedded as in
    :func:`mil_classification`; ``aggregator`` pools the bag for the head ``loss`` picks.
    Both heads read the cohort's ``time`` and ``event`` columns:

    * ``nll``: the ``discrete_survival`` head, a discrete-time model trained with the hazard
      negative log-likelihood over ``num_bins`` time bins (required for ``nll``). The run
      derives the bins from each fold's training split (``binning: train_quantiles``):
      the cut points are quantiles of its uncensored times, so the tune and test times
      never shape them. The cohort carries no ``bin`` column.
    * ``cox``: the ``cox_survival`` head, a Cox proportional-hazards model trained with
      the Breslow partial likelihood; it has no bins, so it takes no ``num_bins``. Bags
      are forwarded one at a time and each loss is computed over a window of 32 of them,
      its risk set (the head's ``cox_window``): one optimizer step per 32 bags, where
      ``nll`` takes one per bag.

    Both losses train the same way: AdamW (lr 1e-4) with a cosine schedule, 50 epochs,
    selection and early stopping on the tune loss.
    """
    head = _survival_head(loss, num_bins, cox=Component("cox_survival", {"cox_window": 32}))
    return _mil(encoder, aggregator, spacing, tile_size, head, "slide_survival")


def _slide_probe(encoder: str, head: Component, instead: str) -> ExperimentConfig:
    """``head`` on slide encoder ``encoder``'s embedding of each slide's tissue tiles:
    batches of 32, AdamW (lr 3e-4) without a schedule. ``instead`` names the presets for a
    tile encoder."""
    return ExperimentConfig(
        cohort=_cohort(),
        features=_tissue_tiles(_features(encoder, "slide", instead), None, None),
        loaders=LoadersConfig(batch_size=32),
        learner=LearnerConfig("mil", {"head": head, "optimizer": Component("adamw", {"lr": 3e-4})}),
    )


@_register
def slide_classification(encoder: str) -> ExperimentConfig:
    """Classify whole slides: a linear probe on a frozen slide encoder's embeddings.

    Each slide is tiled at the encoder's own tile size and spacing, keeping the tiles with
    at least 10% tissue (the tissue of the ``mil_*`` presets). ``encoder`` (``titan``,
    ``prism``, ...; ``soma list encoders --level slide``) embeds each slide's tiles with
    its tile encoder and pools them into one embedding, cached once. A
    ``multiclass_classification`` head (two classes or more; the classes come from the
    cohort's ``label`` column) learns from that embedding: there is no aggregator to
    train. Batches of 32 slides, AdamW (lr 3e-4) without a schedule, 50 epochs, the best
    tune-loss checkpoint and early stopping after 10 epochs without improvement.
    """
    instead = (
        "use mil_classification to pool its tiles with an aggregator, or tile_classification to classify tile images"
    )
    return _slide_probe(encoder, Component("multiclass_classification"), instead)


@_register
def slide_regression(encoder: str) -> ExperimentConfig:
    """Regress a value from whole slides: a linear probe on a frozen slide encoder's embeddings.

    Slides are tiled and embedded as in :func:`slide_classification`, and a ``regression``
    head learns the cohort's ``value`` column from each slide's embedding with a mean
    squared error loss. To regress several values, set the head's ``num_outputs`` param to
    their count ``G``: the head then reads the columns ``value_0`` to ``value_{G-1}``.
    Batches of 32 slides, AdamW (lr 3e-4) without a schedule, 50 epochs, the best
    tune-loss checkpoint and early stopping after 10 epochs without improvement.
    """
    instead = "use mil_regression to pool its tiles with an aggregator, or tile_regression to regress from tile images"
    return _slide_probe(encoder, Component("regression"), instead)


@_register
def slide_survival(encoder: str, loss: Literal["nll", "cox"], num_bins: int | None = None) -> ExperimentConfig:
    """Predict survival from whole slides: a discrete-time or Cox head on a slide encoder's embeddings.

    Slides are tiled and embedded as in :func:`slide_classification`, and the head ``loss``
    picks learns from each slide's embedding. Both heads read the cohort's ``time`` and
    ``event`` columns:

    * ``nll``: the ``discrete_survival`` head, a discrete-time model trained with the hazard
      negative log-likelihood over ``num_bins`` time bins (required for ``nll``). The run
      derives the bins from each fold's training split (``binning: train_quantiles``), as
      in :func:`mil_survival`. The cohort carries no ``bin`` column.
    * ``cox``: the ``cox_survival`` head, a Cox proportional-hazards model trained with
      the Breslow partial likelihood; it has no bins, so it takes no ``num_bins``. Each
      slide is one embedding, so the head computes its loss over each batch of 32 slides,
      its risk set (padded mode, with no ``cox_window``).

    Both losses train the same way: batches of 32 slides, AdamW (lr 3e-4) without a
    schedule, 50 epochs, selection and early stopping on the tune loss.
    """
    head = _survival_head(loss, num_bins, cox=Component("cox_survival"))
    return _slide_probe(encoder, head, "use mil_survival to pool its tiles with an aggregator")


@_register
def segmentation(encoder: str, decoder: str, num_classes: int) -> ExperimentConfig:
    """Segment tissue: a decoder over a frozen encoder's dense token grid.

    Each tile (at the encoder's recommended size and spacing) is encoded once into a
    token grid and cached; ``decoder`` upsamples it to per-pixel ``segmentation`` logits
    for ``num_classes`` classes, whose indices the label masks hold (``2`` for background
    0 and one foreground class 1). For masks holding raw values, replace the head's params
    with a ``classes`` mapping (class name -> raw mask values) and ``ignore``. Batches of 8,
    AdamW (lr 1e-4) with a cosine schedule, 50 epochs, selection and early stopping on the
    tune loss.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=_features(encoder, "tile", _DENSE),
        loaders=LoadersConfig(batch_size=8),
        learner=LearnerConfig(
            "dense",
            {
                "decoder": Component(decoder),
                "head": Component("segmentation", {"num_classes": num_classes}),
                "optimizer": Component("adamw", {"lr": 1e-4}),
                "scheduler": [Component("cosine")],
            },
        ),
    )


@_register
def detection(encoder: str, decoder: str, num_classes: int, match_distance: float = 3.0) -> ExperimentConfig:
    """Detect cells: a decoder over a frozen encoder's dense token grid, one heatmap per class.

    Each image is encoded once into a token grid and cached; ``decoder`` predicts a peak
    heatmap for each of ``num_classes`` classes, whose indices the point annotations hold
    (``1`` for a single cell class), scored by F1 at ``match_distance`` µm (3 µm, about a
    nucleus radius, by default). For points holding raw classes, replace the head's params
    with a ``classes`` mapping (class name -> raw point classes) and ``drop``. One image
    per batch, AdamW (lr 1e-4) with a cosine schedule, 50 epochs, selection and early
    stopping on the tune loss.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=_features(encoder, "tile", _DENSE),
        loaders=LoadersConfig(batch_size=1),
        learner=LearnerConfig(
            "dense",
            {
                "decoder": Component(decoder),
                "head": Component("detection", {"num_classes": num_classes, "match_distance": match_distance}),
                "optimizer": Component("adamw", {"lr": 1e-4}),
                "scheduler": [Component("cosine")],
            },
        ),
    )
