"""Encoder registry, base classes, and validation for soma encoders."""

from soma.encoders import models as _models  # noqa: F401
from soma.encoders.base import (
    Encoder,
    SlideEncoder,
    TileEncoder,
    TimmTileEncoder,
)
from soma.encoders.progress import (
    JsonlProgressReporter,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
)
from soma.encoders.registry import encoder_registry, register_encoder
from soma.encoders.validation import validate_encoder_config

__all__ = [
    # Base classes
    "Encoder",
    "TileEncoder",
    "SlideEncoder",
    "TimmTileEncoder",
    # Registry
    "encoder_registry",
    "register_encoder",
    # Progress
    "ProgressEvent",
    "ProgressReporter",
    "JsonlProgressReporter",
    "NullProgressReporter",
    # Validation
    "validate_encoder_config",
]
