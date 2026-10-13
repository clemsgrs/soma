"""Decoder module — dense feature grid -> per-pixel class logits.

First-class swappable component parallel to ``aggregators`` (segmentation-design §6).
"""

from soma.models.decoders.base import Decoder
from soma.models.decoders.registry import decoder_registry

# Import to trigger registration.
from soma.models.decoders.heavy_conv import HeavyConvDecoder
from soma.models.decoders.lightweight_conv import LightweightConvDecoder, LinearDecoder


__all__ = [
    "Decoder",
    "decoder_registry",
    "LinearDecoder",
    "LightweightConvDecoder",
    "HeavyConvDecoder",
]
