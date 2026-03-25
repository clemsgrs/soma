"""Encoder registry with enforced metadata schema."""

from __future__ import annotations

from typing import Any

from soma.registry import Registry

encoder_registry = Registry("encoders")


def register_encoder(
    name: str,
    *,
    encode_dim: int,
    input_size: int | None = None,
    level: str = "tile",
    tile_encoder: str | None = None,
    recommended_tile_size_px: int | None = None,
    recommended_spacing_um: float | list[float],
    precision: str = "fp16",
    source: str = "",
):
    """Decorator that registers an encoder class with required metadata.

    Args:
        name: Unique encoder name (e.g. "uni2", "virchow2").
        encode_dim: Output feature dimensionality.
        input_size: Recommended encoder input image size in pixels.
        level: Encoder output level ("tile" or "slide").
        tile_encoder: Registered tile encoder dependency for slide-level models.
        recommended_tile_size_px: Recommended preprocessing tile size in pixels.
        recommended_spacing_um: Recommended spacing(s) in µm/px.
        precision: Recommended inference precision ("fp16" or "fp32").
        source: Model source identifier (e.g. HuggingFace hub path).
    """
    metadata: dict[str, Any] = {
        "encode_dim": encode_dim,
        "level": level,
        "input_size": input_size,
        "tile_encoder": tile_encoder,
        "recommended_tile_size_px": (
            recommended_tile_size_px
            if recommended_tile_size_px is not None
            else input_size
        ),
        "recommended_spacing_um": recommended_spacing_um,
        "precision": precision,
        "source": source,
    }
    return encoder_registry.register_decorator(name, metadata=metadata)
