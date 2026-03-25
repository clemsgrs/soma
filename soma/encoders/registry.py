"""Encoder registry with enforced metadata schema."""

from __future__ import annotations

from typing import Any

from soma.registry import Registry

encoder_registry = Registry("encoders")


def register_encoder(
    name: str,
    *,
    encode_dim: int,
    input_size: int,
    recommended_spacing_um: float | list[float],
    precision: str = "fp16",
    source: str = "",
):
    """Decorator that registers an encoder class with required metadata.

    Args:
        name: Unique encoder name (e.g. "uni2", "virchow2").
        encode_dim: Output feature dimensionality.
        input_size: Recommended input image size in pixels.
        recommended_spacing_um: Recommended spacing(s) in µm/px.
        precision: Recommended inference precision ("fp16" or "fp32").
        source: Model source identifier (e.g. HuggingFace hub path).
    """
    metadata: dict[str, Any] = {
        "encode_dim": encode_dim,
        "input_size": input_size,
        "recommended_spacing_um": recommended_spacing_um,
        "precision": precision,
        "source": source,
    }
    return encoder_registry.register_decorator(name, metadata=metadata)
