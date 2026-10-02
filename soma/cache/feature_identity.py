"""The feature identity a pooled cache records, read from the sidecars slide2vec writes.

A pooled cache key covers the encoder name, output variant, precision, registry input
size and spacing, but not the image transform. slide2vec records the full feature
identity, transform included, in each sidecar's ``compatibility`` block. soma copies that
block into ``cache_metadata.json`` with the slide2vec version that wrote it, so a later
run can tell whether the installed slide2vec would still extract the same features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import slide2vec

from soma.cache._types import FeatureCacheResolution
from soma.cache.io import _write_metadata, recorded_feature_identity

logger = logging.getLogger(__name__)

FEATURE_IDENTITY_METADATA_KEY = "feature_identity"

SLIDE2VEC_RELEASE_NOTES_URL = "https://github.com/clemsgrs/slide2vec/releases"

# Caches already reported as unverifiable: a run resolves a cache it completes twice.
_WARNED_UNRECORDED: set[Path] = set()

# Cache kinds whose payloads are pooled embeddings. Dense caches record no identity.
_POOLED_CACHE_KINDS = frozenset({"tile", "image", "slide", "patient", "hierarchical"})


class CacheFeatureIdentityMismatch(ValueError):
    """A cache was extracted with a different feature identity than the current one."""


@dataclass(frozen=True)
class FeatureIdentityCheck:
    """How a pooled cache's recorded feature identity is verified when it is resolved.

    ``differing`` compares a recorded identity with the one the installed slide2vec
    would produce for this run, and returns ``{field: (recorded, current)}`` for the
    fields that differ. It loads the encoder to read its transform, so it is called
    only when the cache was last verified with another slide2vec version.
    """

    differing: Callable[[dict[str, Any]], dict[str, tuple[Any, Any]]]
    # ``cache.on_identity_mismatch``: 'error' or 'reextract'.
    on_mismatch: str = "error"
    # ``cache.on_unrecorded_identity``: 'warn' or 'reextract'.
    on_unrecorded: str = "warn"


def discard_reason(
    *,
    cache_dir: Path,
    metadata_path: Path,
    existing: dict[str, Any],
    check: FeatureIdentityCheck,
) -> str | None:
    """Why the features already in a cache must be extracted again, or None to reuse them.

    Raises :class:`CacheFeatureIdentityMismatch` when the recorded identity differs from
    the current one and the check does not ask to extract again.
    """
    if not holds_features(existing):
        return None
    recorded = existing.get(FEATURE_IDENTITY_METADATA_KEY)
    if recorded is None:
        message = (
            f"Feature cache at {cache_dir} records no feature identity, so soma cannot "
            f"verify that slide2vec {slide2vec.__version__} would extract the same "
            "features."
        )
        if check.on_unrecorded == "reextract":
            logger.warning(
                "%s Extracting the cache again (cache.on_unrecorded_identity).", message
            )
            return "no recorded feature identity"
        if cache_dir not in _WARNED_UNRECORDED:
            _WARNED_UNRECORDED.add(cache_dir)
            logger.warning(
                "%s It is reused as is. If it was written with an older slide2vec, check "
                "the slide2vec release notes (%s) for changes to the preprocessing of "
                "'%s'. To extract it again, delete the cache directory or set "
                "cache.on_unrecorded_identity: reextract.",
                message,
                SLIDE2VEC_RELEASE_NOTES_URL,
                existing.get("encoder_name"),
            )
        return None
    recorded_version = recorded.get("slide2vec_version")
    if recorded_version == slide2vec.__version__:
        return None
    differing = check.differing(dict(recorded.get("identity") or {}))
    if not differing:
        # Verified against the installed slide2vec: record that, so the next hit skips
        # the comparison. The identity stays the one slide2vec wrote with the features.
        existing[FEATURE_IDENTITY_METADATA_KEY] = {
            **recorded,
            "slide2vec_version": slide2vec.__version__,
        }
        _write_metadata(metadata_path, existing)
        return None
    details = "; ".join(
        f"{leaf} (recorded {old!r}, current {new!r})"
        for field, (old, new) in sorted(differing.items())
        for leaf, old, new in _leaf_differences(field, old, new)
    )
    message = (
        f"Feature cache at {cache_dir} was extracted with slide2vec {recorded_version}, "
        f"and slide2vec {slide2vec.__version__} now extracts different features: {details}."
    )
    if check.on_mismatch == "reextract":
        logger.warning("%s Extracting the cache again (cache.on_identity_mismatch).", message)
        return "feature identity changed"
    raise CacheFeatureIdentityMismatch(
        f"{message} Reusing or completing the cache would mix the two. Delete the cache "
        "directory, choose another cache.root_dir, or set "
        "cache.on_identity_mismatch: reextract."
    )


def _leaf_differences(field: str, recorded: Any, current: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(dotted field, recorded, current)`` for each differing leaf of a field."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(recorded.keys() | current.keys()):
            if recorded.get(key) != current.get(key):
                yield from _leaf_differences(f"{field}.{key}", recorded.get(key), current.get(key))
    else:
        yield field, recorded, current


def holds_features(metadata: dict[str, Any]) -> bool:
    """True when the cache has at least one signed sample with a feature payload."""
    signed = set(metadata.get("sample_identity_signature_by_id", {}))
    return bool(signed - {str(s) for s in metadata.get("empty_sample_ids", [])})


def written_identity_record(
    resolution: FeatureCacheResolution, cache_ids: Sequence[str]
) -> dict[str, Any] | None:
    """The identity record of the samples slide2vec just wrote, or None if they hold none.

    The identity is the ``compatibility`` block of the first sidecar that has one: every
    sample of one extraction shares it.
    """
    if resolution.cache_kind not in _POOLED_CACHE_KINDS:
        return None
    for cache_id in cache_ids:
        if str(cache_id) not in resolution.cache_stem_by_id:
            continue
        identity = recorded_feature_identity(resolution.feature_path_for_id(cache_id))
        if identity is not None:
            return {"slide2vec_version": slide2vec.__version__, "identity": identity}
    return None
