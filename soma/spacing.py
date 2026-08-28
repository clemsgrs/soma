"""Explicit physical-spacing policy shared by image and supervision reads."""

from __future__ import annotations


def resolve_effective_spacing_um(
    *,
    requested_spacing_um: float,
    spacing_at_level_0: float | None,
    tolerance: float,
    policy: str,
) -> float:
    """Return the requested spacing, or native spacing under an explicit fallback."""
    requested = float(requested_spacing_um)
    if policy != "native_if_coarser" or spacing_at_level_0 is None:
        return requested
    native = float(spacing_at_level_0)
    if native > requested * (1.0 + float(tolerance)):
        return native
    return requested


__all__ = ["resolve_effective_spacing_um"]
