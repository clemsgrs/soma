"""Re-key legacy dense caches, whose key folded the registry default ``input_size``.

Dense caches used to key on ``encoder_registry.info(name)["input_size"]``. That value does
not decide what a dense extraction computes, so the key no longer carries it. A cache
written before the change keeps valid features under a folder name that no longer
resolves; this module renames it and brings its metadata to the current signature.

Everything is derived from what the cache recorded (``cache_metadata.json`` and
``manifest.csv``): no encoder is loaded and no registry is consulted.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from soma.cache._types import CACHE_METADATA_NAME, MANIFEST_NAME
from soma.cache.features import dense_static_identity
from soma.cache.io import _load_metadata, _write_metadata
from soma.cache.keys import (
    _sample_stem_for_kind,
    dense_cache_key_from_metadata,
    sample_identity_signature,
)

MIGRATION_NOTE_NAME = "MIGRATION.json"
DENSE_CACHE_KINDS = ("dense", "dense_image")

# The execution field the dense key stopped carrying.
_LEGACY_EXECUTION_FIELD = "input_size"


@dataclass(frozen=True)
class DenseCacheMigration:
    """What the migration did, or would do, to one legacy dense cache folder.

    ``status`` is ``"planned"`` (dry run), ``"migrated"`` or ``"refused"``; ``reason`` is
    set when refused. ``unverified_sample_ids`` lists cached samples whose identity could
    not be re-derived from the cache's ``manifest.csv``: their identity is left as
    recorded, so they read as missing under the new key and are extracted again.
    """

    source: Path
    target: Path
    old_key: str
    new_key: str
    status: str
    reason: str | None = None
    unverified_sample_ids: tuple[str, ...] = ()


def migrate_legacy_dense_caches(
    cache_root: Path | str,
    *,
    apply: bool = False,
    today: date | None = None,
) -> list[DenseCacheMigration]:
    """Re-key every legacy dense cache under ``cache_root``; a dry run unless ``apply``.

    Caches already on the current key are skipped and not reported. A cache is refused,
    and left untouched, when its target folder exists or when its recorded key cannot be
    reproduced from its recorded metadata.
    """
    cache_root = Path(cache_root)
    results: list[DenseCacheMigration] = []
    for cache_kind in DENSE_CACHE_KINDS:
        kind_dir = cache_root / cache_kind
        if not kind_dir.is_dir():
            continue
        for cache_dir in sorted(path for path in kind_dir.iterdir() if path.is_dir()):
            result = _migrate_one(
                cache_dir, cache_kind=cache_kind, apply=apply, today=today or date.today()
            )
            if result is not None:
                results.append(result)
    return results


def _migrate_one(
    cache_dir: Path,
    *,
    cache_kind: str,
    apply: bool,
    today: date,
) -> DenseCacheMigration | None:
    metadata_path = cache_dir / CACHE_METADATA_NAME
    if not metadata_path.is_file():
        return None
    metadata = _load_metadata(metadata_path)
    if _LEGACY_EXECUTION_FIELD not in metadata.get("execution", {}):
        return None

    old_key = str(metadata.get("cache_key"))
    migrated = dict(metadata)
    migrated["execution"] = {
        field: value
        for field, value in metadata["execution"].items()
        if field != _LEGACY_EXECUTION_FIELD
    }
    new_key = dense_cache_key_from_metadata(migrated)
    target = cache_dir.with_name(new_key)

    def outcome(status: str, reason: str | None = None, unverified: tuple[str, ...] = ()):
        return DenseCacheMigration(
            source=cache_dir,
            target=target,
            old_key=old_key,
            new_key=new_key,
            status=status,
            reason=reason,
            unverified_sample_ids=unverified,
        )

    if cache_dir.name != old_key:
        return outcome(
            "refused", f"folder name does not match the recorded cache_key {old_key!r}"
        )
    if dense_cache_key_from_metadata(metadata) != old_key:
        return outcome(
            "refused",
            f"recorded metadata does not reproduce the recorded cache_key {old_key!r}",
        )
    if target.exists():
        return outcome("refused", f"target folder already exists: {target}")

    migrated["cache_key"] = new_key
    identities, unverified = _migrated_sample_identities(
        metadata,
        manifest_path=cache_dir / MANIFEST_NAME,
        cache_kind=cache_kind,
        old_key=old_key,
        new_key=new_key,
    )
    migrated["sample_identity_signature_by_id"] = identities
    if not apply:
        return outcome("planned", unverified=unverified)

    # Rename first: it is atomic, and it is the step the no-overwrite guard protects.
    cache_dir.rename(target)
    _write_metadata(target / CACHE_METADATA_NAME, migrated)
    _write_metadata(
        target / MIGRATION_NOTE_NAME,
        {
            "migration": "dense cache key no longer includes the registry default input_size",
            "old_key": old_key,
            "new_key": new_key,
            "migrated_on": today.isoformat(),
            "dropped_execution_fields": {
                _LEGACY_EXECUTION_FIELD: metadata["execution"][_LEGACY_EXECUTION_FIELD]
            },
            "unverified_sample_ids": list(unverified),
        },
    )
    return outcome("migrated", unverified=unverified)


def _migrated_sample_identities(
    metadata: dict[str, Any],
    *,
    manifest_path: Path,
    cache_kind: str,
    old_key: str,
    new_key: str,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Re-derive each recorded sample identity under the new key.

    A sample identity hashes the cache key, so it moves with it. It is rewritten only
    when the identity recomputed from ``manifest.csv`` under the *old* key equals the
    recorded one; anything else is left as recorded and reported as unverified.
    """
    sample_signature_by_id = _manifest_sample_signatures(manifest_path)
    identities: dict[str, str] = {}
    unverified: list[str] = []
    for sample_id, recorded in metadata.get("sample_identity_signature_by_id", {}).items():
        sample_signature = sample_signature_by_id.get(str(sample_id))

        def identity(cache_key: str) -> str:
            return _sample_stem_for_kind(
                sample_signature=sample_signature,
                cache_kind=cache_kind,
                static_identity_payload=dense_static_identity(
                    cache_kind=cache_kind, cache_key=cache_key
                ),
            )

        if sample_signature is not None and identity(old_key) == str(recorded):
            identities[str(sample_id)] = identity(new_key)
        else:
            identities[str(sample_id)] = str(recorded)
            unverified.append(str(sample_id))
    return identities, tuple(unverified)


def _manifest_sample_signatures(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["sample_id"]: sample_identity_signature(
            sample_id=row["sample_id"],
            image_path=row["image_path"],
            mask_path=row.get("mask_path") or None,
            spacing_at_level_0=(
                float(row["spacing_at_level_0"]) if row.get("spacing_at_level_0") else None
            ),
        )
        for row in rows
    }
