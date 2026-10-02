"""The feature identity a pooled cache records, and its verification when the cache is reused.

A pooled cache key covers the encoder name, output variant, precision, registry input
size and spacing, but not the image transform. slide2vec records the full feature
identity, transform included, in each sidecar's ``compatibility`` block. soma copies that
block into ``cache_metadata.json`` with the slide2vec version, so a later run can tell
whether the installed slide2vec would still extract the same features.

The record lives under ``feature_identity`` and goes through three states:

* absent: the cache was written before soma recorded identities, and cannot be verified;
* pending (``identity`` is null): the cache was created, and no feature is committed yet;
* recorded: ``identity`` is the block slide2vec wrote with the first committed features,
  and ``slide2vec_version`` the version it was last verified with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import slide2vec

from soma.cache._types import FeatureCacheResolution
from soma.cache.io import _load_metadata, _write_metadata

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


def pending_identity_record(cache_kind: str) -> dict[str, Any] | None:
    """The record a new cache starts with, or None for a kind that records no identity."""
    if cache_kind not in _POOLED_CACHE_KINDS:
        return None
    return {"slide2vec_version": slide2vec.__version__, "identity": None}


def recorded_identity(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """The feature identity a cache records for its features, if it records one."""
    return (metadata.get(FEATURE_IDENTITY_METADATA_KEY) or {}).get("identity")


def record_committed_identity(
    metadata: dict[str, Any],
    resolution: FeatureCacheResolution,
    cache_ids: Sequence[str],
) -> None:
    """Complete a pending record with the identity of the features being committed.

    The identity is the ``compatibility`` block slide2vec wrote in the sidecar of the
    first committed sample: every sample of one extraction shares it. Features whose
    sidecar holds no transform cannot be verified later, so the cache then drops its
    record and counts as one that never had any.
    """
    record = metadata.get(FEATURE_IDENTITY_METADATA_KEY)
    if record is None or record.get("identity") is not None:
        return
    committed = [cache_id for cache_id in cache_ids if str(cache_id) in resolution.cache_stem_by_id]
    if not committed:
        return
    payload_path = resolution.feature_path_for_id(committed[0])
    identity = _sidecar_identity(payload_path.with_name(f"{payload_path.stem}.meta.json"))
    if _is_verifiable(identity):
        metadata[FEATURE_IDENTITY_METADATA_KEY] = {
            "slide2vec_version": slide2vec.__version__,
            "identity": identity,
        }
    else:
        del metadata[FEATURE_IDENTITY_METADATA_KEY]


def stale_features_reason(
    *,
    cache_dir: Path,
    features_dir: Path,
    metadata_path: Path,
    existing: dict[str, Any],
    check: FeatureIdentityCheck,
) -> str | None:
    """Verify the recorded identity of an existing cache against the current one.

    Returns None when the features in the cache can be reused, and otherwise the reason
    why the whole cache must be extracted again. Raises
    :class:`CacheFeatureIdentityMismatch` when the recorded identity differs from the
    current one and ``check`` does not ask to extract again.
    """
    installed = slide2vec.__version__
    populated = any(features_dir.glob("*.pt"))
    record = existing.get(FEATURE_IDENTITY_METADATA_KEY)
    if record is not None and record.get("identity") is None:
        # An extraction stopped before its first commit. What it left on disk is reused
        # when the installed slide2vec wrote it, and takes its identity from a sidecar.
        # What another version wrote was never committed, and is not kept.
        if record.get("slide2vec_version") != installed:
            return f"extraction was interrupted under slide2vec {record.get('slide2vec_version')}"
        sidecar_path = next(features_dir.glob("*.meta.json"), None)
        identity = None if sidecar_path is None else _sidecar_identity(sidecar_path)
        if _is_verifiable(identity):
            _write_record(
                metadata_path, existing, {"slide2vec_version": installed, "identity": identity}
            )
            return None
        if not populated:
            return None
        # Features with no identity to recover, as aggregated from tiles with no record:
        # the cache drops its record and counts as one that never had any.
        _write_record(metadata_path, existing, None)
        record = None
    if record is None:
        if not populated:
            # Nothing to verify: an empty cache starts a record like a new one.
            _write_record(metadata_path, existing, {"slide2vec_version": installed, "identity": None})
            return None
        message = (
            f"Feature cache at {cache_dir} records no feature identity, so soma cannot "
            f"verify that slide2vec {installed} would extract the same features."
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
    verified_version = record.get("slide2vec_version")
    if verified_version == installed:
        return None
    differing = check.differing(dict(record["identity"]))
    if not differing:
        # Verified against the installed slide2vec: record that, so the next hit skips
        # the comparison. The identity stays the one slide2vec wrote with the features.
        _write_record(metadata_path, existing, {**record, "slide2vec_version": installed})
        return None
    details = "; ".join(
        f"{leaf} (recorded {old!r}, current {new!r})"
        for field, (old, new) in sorted(differing.items())
        for leaf, old, new in _leaf_differences(field, old, new)
    )
    message = (
        f"Feature cache at {cache_dir} was last verified with slide2vec {verified_version}, "
        f"and slide2vec {installed} now extracts different features: {details}."
    )
    if check.on_mismatch == "reextract":
        logger.warning("%s Extracting the cache again (cache.on_identity_mismatch).", message)
        return "feature identity changed"
    raise CacheFeatureIdentityMismatch(
        f"{message} Reusing or completing the cache would mix the two. Delete the cache "
        "directory, choose another cache.root_dir, or set "
        "cache.on_identity_mismatch: reextract."
    )


def _write_record(
    metadata_path: Path, existing: dict[str, Any], record: dict[str, Any] | None
) -> None:
    """Set the identity record in ``existing`` and on disk, keeping what else is on disk.

    ``record=None`` removes it. The file is read again because the comparison that
    precedes a write loads an encoder, long enough for another job to commit samples to
    the same cache.
    """
    on_disk = _load_metadata(metadata_path)
    for metadata in (existing, on_disk):
        if record is None:
            metadata.pop(FEATURE_IDENTITY_METADATA_KEY, None)
        else:
            metadata[FEATURE_IDENTITY_METADATA_KEY] = record
    _write_metadata(metadata_path, on_disk)


def _is_verifiable(identity: dict[str, Any] | None) -> bool:
    """True for an identity that holds the transform, the field the cache key lacks."""
    return identity is not None and "transform" in identity


def _sidecar_identity(sidecar_path: Path) -> dict[str, Any] | None:
    """The ``compatibility`` block slide2vec wrote in a feature sidecar, if any."""
    try:
        identity = json.loads(sidecar_path.read_text(encoding="utf-8")).get("compatibility")
    except (OSError, ValueError):
        logger.debug("Could not read feature sidecar at %s", sidecar_path, exc_info=True)
        return None
    return identity if isinstance(identity, dict) else None


def _leaf_differences(field: str, recorded: Any, current: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(dotted field, recorded, current)`` for each differing leaf of a field."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(recorded.keys() | current.keys()):
            if recorded.get(key) != current.get(key):
                yield from _leaf_differences(f"{field}.{key}", recorded.get(key), current.get(key))
    else:
        yield field, recorded, current
