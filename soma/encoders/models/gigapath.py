"""Prov-GigaPath encoder implementation."""

from __future__ import annotations

from soma.encoders.base import TimmEncoder
from soma.encoders.registry import register_encoder


@register_encoder(
    "gigapath",
    encode_dim=1536,
    input_size=256,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="prov-gigapath/prov-gigapath",
)
class GigaPath(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__("hf_hub:prov-gigapath/prov-gigapath", token=token)
