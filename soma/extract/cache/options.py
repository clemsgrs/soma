"""``CacheOptions``: the feature and tiling cache policy the ``soma.extract`` verb takes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soma._internal.editable import Editable

__all__ = ["CacheOptions"]


@dataclass
class CacheOptions(Editable):
    """``features.cache``: the feature and tiling cache policy. ``dtype`` (``None``
    follows the compute precision) enters every cache key; ``on_identity_mismatch``
    (``error`` or ``reextract``) handles a feature cache whose recorded feature
    identity, encoder code included, differs from the current one or cannot be
    verified."""

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    validate_payloads: bool = False
    dtype: str | None = None
    on_identity_mismatch: str = "error"

    def __post_init__(self) -> None:
        choices = ("error", "reextract")
        if self.on_identity_mismatch not in choices:
            raise ValueError(
                f"cache.on_identity_mismatch must be one of {list(choices)}, got {self.on_identity_mismatch!r}"
            )
