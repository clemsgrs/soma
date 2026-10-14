"""Presets: the beginner path into soma (design §8.4).

Each preset is a plain function that returns a complete
:class:`~soma.config.ExperimentConfig` from a few typed arguments. There is no override
mechanism: change the returned config by assigning its fields
(``cfg.trainer.max_epochs = 20``), or print it as YAML with ``soma init <preset>``, edit
the file and run it with ``soma run``. ``soma run --preset`` builds the config from a
preset and the cohort, splits, output directory and seed given on the command line.

The cohort paths of a preset config are placeholders (``dataset.csv``, ``splits.csv``):
point ``cfg.cohort`` at your files. The registered benchmarks build their configs from
these presets and then write their protocol values, so every benchmark run exercises them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from hs2p import MaskSamplingConfig
from slide2vec import PreprocessingConfig

from soma.config import (
    CohortConfig,
    Component,
    ExperimentConfig,
    FeaturesConfig,
    LearnerConfig,
    LoadersConfig,
)
from soma.registry import Registry, signature_params

__all__ = [
    "detection",
    "mil_classification",
    "preset_registry",
    "segmentation",
    "survival",
    "tile_classification",
]

#: The ``presets`` domain of :func:`soma.components`. An entry is the preset function; its
#: params are the function's arguments (``None`` for a required one, listed in ``required``).
preset_registry = Registry("presets")


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


def _cohort() -> CohortConfig:
    return CohortConfig(records="dataset.csv", splits="splits.csv")


def _tissue_tiles() -> PreprocessingConfig:
    """Tiles holding at least 10% tissue (hs2p's binary tissue vocabulary); the tile size
    and spacing come from the encoder's recommended settings."""
    return PreprocessingConfig(
        masks=MaskSamplingConfig(pixel_mapping={"background": 0, "tissue": 1}, min_coverage={"tissue": 0.1})
    )


@_register
def mil_classification(encoder: str, aggregator: str = "abmil") -> ExperimentConfig:
    """Classify whole slides: a frozen tile encoder, a MIL aggregator and a classifier.

    Each slide is tiled at the encoder's recommended size and spacing (tiles with at
    least 10% tissue), its tiles are embedded once and cached, and ``aggregator`` pools
    the bag for a ``multiclass_classification`` head (two classes or more; the classes
    come from the cohort's ``label`` column). AdamW (lr 1e-4) with a cosine schedule,
    50 epochs, the best tune-loss checkpoint and early stopping after 10 epochs without
    improvement.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=FeaturesConfig(encoder=encoder, extraction=_tissue_tiles()),
        learner=LearnerConfig(
            "mil",
            {
                "aggregator": Component(aggregator),
                "head": Component("multiclass_classification"),
                "optimizer": Component("adamw", {"lr": 1e-4}),
                "scheduler": [Component("cosine")],
            },
        ),
    )


@_register
def tile_classification(encoder: str) -> ExperimentConfig:
    """Classify tile images: a linear probe on a frozen tile encoder's embeddings.

    Each row of the cohort is one image, embedded whole by ``encoder`` (no tiling, no
    aggregator), and a ``multiclass_classification`` head learns the cohort's ``label``
    column. Batches of 256 embeddings, AdamW (lr 3e-4) without a schedule, 50 epochs,
    the best tune-loss checkpoint and early stopping after 10 epochs without improvement.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=FeaturesConfig(encoder=encoder),
        loaders=LoadersConfig(batch_size=256),
        learner=LearnerConfig(
            "mil",
            {"head": Component("multiclass_classification"), "optimizer": Component("adamw", {"lr": 3e-4})},
        ),
    )


@_register
def survival(encoder: str, aggregator: str) -> ExperimentConfig:
    """Predict survival from whole slides: a discrete-time hazard head over a MIL bag.

    Slides are tiled and embedded as in :func:`mil_classification`; ``aggregator`` pools
    the bag for the ``survival`` head, which reads the cohort's ``time``, ``event`` and
    ``bin`` columns (the number of bins comes from the cohort). AdamW (lr 1e-4) with a
    cosine schedule, 50 epochs, selection and early stopping on the tune loss.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=FeaturesConfig(encoder=encoder, extraction=_tissue_tiles()),
        learner=LearnerConfig(
            "mil",
            {
                "aggregator": Component(aggregator),
                "head": Component("survival"),
                "optimizer": Component("adamw", {"lr": 1e-4}),
                "scheduler": [Component("cosine")],
            },
        ),
    )


@_register
def segmentation(encoder: str, decoder: str, num_classes: int = 2) -> ExperimentConfig:
    """Segment tissue: a decoder over a frozen encoder's dense token grid.

    Each tile (at the encoder's recommended size and spacing) is encoded once into a
    token grid and cached; ``decoder`` upsamples it to per-pixel ``segmentation`` logits
    for ``num_classes`` classes, whose indices the label masks hold (background 0 and one
    foreground class by default). For masks holding raw values, replace the head's params
    with a ``classes`` mapping (class name -> raw mask values) and ``ignore``. Batches of 8,
    AdamW (lr 1e-4) with a cosine schedule, 50 epochs, selection and early stopping on the
    tune loss.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=FeaturesConfig(encoder=encoder),
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
def detection(encoder: str, decoder: str, num_classes: int = 1, match_distance: float = 3.0) -> ExperimentConfig:
    """Detect cells: a decoder over a frozen encoder's dense token grid, one heatmap per class.

    Each image is encoded once into a token grid and cached; ``decoder`` predicts a peak
    heatmap for each of ``num_classes`` classes, whose indices the point annotations hold
    (one cell class by default), scored by F1 at ``match_distance`` µm (3 µm, about a
    nucleus radius, by default). For points holding raw classes, replace the head's params
    with a ``classes`` mapping (class name -> raw point classes) and ``drop``. One image
    per batch, AdamW (lr 1e-4) with a cosine schedule, 50 epochs, selection and early
    stopping on the tune loss.
    """
    return ExperimentConfig(
        cohort=_cohort(),
        features=FeaturesConfig(encoder=encoder),
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
