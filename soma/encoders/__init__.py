"""Encoder module — tile-level feature extraction with foundation models."""

from soma.encoders.base import Encoder, TimmEncoder
from soma.encoders.distributed import ExtractionSummary, SlideTask, extract_dataset
from soma.encoders.extraction import (
    SuperTileBatchSampler,
    TileBatchCollator,
    TileIndexDataset,
    extract_features,
    save_features,
)
from soma.encoders.progress import (
    JsonlProgressReporter,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
)
from soma.encoders.registry import encoder_registry, register_encoder
from soma.encoders.tile_reader import SuperTile, SuperTileIndex, build_supertile_index
from soma.encoders.validation import validate_encoder_config

__all__ = [
    # Base classes
    "Encoder",
    "TimmEncoder",
    # Registry
    "encoder_registry",
    "register_encoder",
    # Tile reading
    "SuperTile",
    "SuperTileIndex",
    "build_supertile_index",
    # Extraction (single slide)
    "TileIndexDataset",
    "TileBatchCollator",
    "SuperTileBatchSampler",
    "extract_features",
    "save_features",
    # Distributed extraction (multi-slide, multi-GPU)
    "SlideTask",
    "ExtractionSummary",
    "extract_dataset",
    # Progress
    "ProgressEvent",
    "ProgressReporter",
    "JsonlProgressReporter",
    "NullProgressReporter",
    # Validation
    "validate_encoder_config",
]
