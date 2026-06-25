"""Shared dataclasses and constants for the cache layer."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_METADATA_NAME = "cache_metadata.json"
MANIFEST_NAME = "manifest.csv"
PROCESS_LIST_NAME = "process_list.csv"
SCHEMA_VERSION = "v1"

_FEATURE_TYPE_TO_RANK = {
    "tile": 1,
    "bag": 2,
    "slide": 1,
    "patient": 1,
    "hierarchical": 3,
    # dense_grid is (channels, grid_h, grid_w); same rank as hierarchical but the
    # feature dim is the channel axis (0), not the last axis. The dense validator
    # and DenseFeatureStore read the channel axis from metadata, never from rank.
    "dense_grid": 3,
}

_CACHE_KIND_TO_FEATURES_SUBDIR = {
    "tile": "tile_embeddings",
    "slide": "slide_embeddings",
    "patient": "patient_embeddings",
    "hierarchical": "hierarchical_embeddings",
    "dense": "dense_embeddings",
}


def _features_subdir_for_kind(cache_kind: str) -> str:
    try:
        return _CACHE_KIND_TO_FEATURES_SUBDIR[cache_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache_kind '{cache_kind}' for features_dir resolution") from exc


def _list_feature_filenames(features_dir: Path) -> set[str]:
    """List ``features_dir`` once, returning the set of existing filenames.

    Single source of truth for cache-content existence: one ``readdir`` answers
    both the ``<id>.pt`` and ``<id>.meta.json`` questions for *every* expected id,
    so the validator and :meth:`FeatureCacheResolution.missing_sample_ids` decide
    presence by set membership instead of a per-id ``stat`` — the per-launch tax
    that, on a slow/near-full network mount, dwarfs the useful work and scales with
    cache size. Both consumers route through this helper so they cannot drift.
    """
    return {entry.name for entry in features_dir.glob("*")}


@dataclass(frozen=True)
class BaseCacheResolution(ABC):
    key: str
    cache_dir: Path
    metadata_path: Path
    manifest_path: Path
    reused: bool
    complete: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CacheValidationResult:
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class FeatureCacheResolution(BaseCacheResolution):
    cache_kind: str
    features_dir: Path
    cache_ids: tuple[str, ...]
    cache_stem_by_id: dict[str, str]
    validation: CacheValidationResult

    @property
    def empty_sample_ids(self) -> set[str]:
        return {str(s) for s in self.metadata.get("empty_sample_ids", [])}

    def feature_path_for_id(self, cache_id: str) -> Path:
        return self.features_dir / f"{cache_id}.pt"

    def missing_sample_ids(self) -> list[str]:
        expected = self.cache_ids
        empty = self.empty_sample_ids
        # One directory listing decides ``.pt`` (and, for dense, ``.meta.json``)
        # existence for every id, mirroring the validator — no per-id stat.
        existing = _list_feature_filenames(self.features_dir)
        # Dense (``dense_grid``) caches additionally require the shape sidecar to
        # exist before a sample counts as present, matching the validator (which
        # gates the sidecar requirement on ``dense_grid``). The ``.pt`` and sidecar
        # are written non-atomically (``.pt`` first), so a crash between them leaves
        # a ``.pt`` with no sidecar; that grid is not loadable and must be
        # re-encoded, never silently skipped. Non-dense caches stay sidecar-agnostic
        # (their ``.pt`` is self-describing); the asymmetry is intentional.
        dense_sidecar_suffix: str | None = None
        if str(self.metadata.get("feature_type", "")) == "dense_grid":
            from soma.dense.store import DENSE_SIDECAR_SUFFIX

            dense_sidecar_suffix = DENSE_SIDECAR_SUFFIX
        cached_signature_by_id = {
            str(cache_id): str(signature)
            for cache_id, signature in self.metadata.get("sample_identity_signature_by_id", {}).items()
        }
        missing: list[str] = []
        for cache_id in expected:
            cache_id = str(cache_id)
            if cache_id in empty:
                continue
            expected_signature = str(self.cache_stem_by_id[cache_id])
            cached_signature = cached_signature_by_id.get(cache_id)
            if cached_signature is None or cached_signature != expected_signature:
                missing.append(cache_id)
                continue
            if f"{cache_id}.pt" not in existing:
                missing.append(cache_id)
                continue
            if dense_sidecar_suffix is not None and f"{cache_id}{dense_sidecar_suffix}" not in existing:
                missing.append(cache_id)
        return missing


@dataclass(frozen=True)
class TilingCacheResolution(BaseCacheResolution):
    process_list_path: Path
    artifacts_dir: Path
    cache_ids: tuple[str, ...]
    cache_stem_by_id: dict[str, str]

    @property
    def previews_dir(self) -> Path:
        return self.cache_dir / "previews"
