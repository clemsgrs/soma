"""Extraction geometry recorded in the feature cache (ADR 0008).

The cache records the geometry triple for a run — the tile size that was **requested**,
the tile size actually **read** off the slide, and the **effective encoder input**: the
geometry of the tensor handed to the encoder. The third is the only member worth
validating on reuse, and soma derives its expected value from config plus slide2vec's
registry without loading the model.

What this catches is a *regime shift*. A 512 px request on a variable-input encoder records
an encoder input of 224 under the encoder's shipped transform and 512 under a
normalization-only one; reusing features across that change would train on grids registered
to a different extent. What it deliberately does not catch is a change in *how* pixels are
produced at unchanged sizes — a different interpolation kernel, a resize moving stage, a
corrected photometric recipe. Those are not sizes, so no geometry record can see them; the
accepted mitigation is deleting caches by hand on a slide2vec upgrade.

The record applies to **declared** runs only. A Given-geometry run (pre-cropped images)
never requested a size, so there is nothing to validate a later run against — the encoder
input is observed after the fact, and slide2vec records it per artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slide2vec.api import EncoderInputContract

__all__ = [
    "GEOMETRY_METADATA_KEY",
    "CacheGeometryMismatch",
    "pooled_extraction_geometry",
    "dense_extraction_geometry",
    "validate_recorded_geometry",
]

GEOMETRY_METADATA_KEY = "extraction_geometry"


class CacheGeometryMismatch(ValueError):
    """A cache's recorded effective encoder input disagrees with this run's.

    Deliberately distinct from ordinary incompleteness, which returns
    ``CacheValidationResult(complete=False, …)`` and recomputes: silently recomputing a
    400 GB feature set is the surprise worth preventing, so this raises instead.
    """


def _size_pair(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(value[0]), int(value[1])]
    return [int(value), int(value)]


def _geometry(
    *,
    requested_tile_size_px: Any,
    encoder_input_size_px: list[int] | None,
    read_tile_size_px_by_id: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "requested_tile_size_px": _size_pair(requested_tile_size_px),
        "encoder_input_size_px": encoder_input_size_px,
        # Per slide, because the read size is a property of that slide's pyramid: the same
        # 224 px request at 0.5 µm reads 448 px off a 0.25 µm slide and 224 off a 0.5 one.
        # Provenance only — it is not validated, since a changed read size already changes
        # the tiling the cache key is derived from.
        "read_tile_size_px_by_id": (
            None
            if read_tile_size_px_by_id is None
            else {str(k): int(v) for k, v in sorted(read_tile_size_px_by_id.items())}
        ),
    }


def pooled_extraction_geometry(
    *,
    encoder_name: str,
    requested_tile_size_px: int | None,
    allow_non_recommended_settings: bool = False,
    read_tile_size_px_by_id: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """The geometry record for a pooled (tiled-slide) run, or ``None`` if undeclarable.

    ``requested_tile_size_px`` is ``None`` before the pipeline resolves it from the
    encoder's own preset; there is no declaration to record until it is.
    """
    if requested_tile_size_px is None:
        return None
    plan = EncoderInputContract.declared_pooled(
        encoder_name,
        requested_tile_size_px=int(requested_tile_size_px),
        allow_non_recommended_settings=bool(allow_non_recommended_settings),
    ).plan
    # A ``None`` effective size means the encoder's shipped transform decides, and it
    # resizes to the encoder's own preset — so that preset *is* what the encoder sees.
    effective = getattr(plan, "expected_encoder_input_size_px", None)
    if effective is None:
        effective = plan.preset_input_size_px
    return _geometry(
        requested_tile_size_px=requested_tile_size_px,
        encoder_input_size_px=_size_pair(effective),
        read_tile_size_px_by_id=read_tile_size_px_by_id,
    )


def dense_extraction_geometry(
    *,
    encoder_name: str,
    target_size_px: int | tuple[int, int],
    window_size: int | None,
    read_tile_size_px_by_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The geometry record for a dense run.

    Dense states a *supervision* geometry, so the effective encoder input is derived from
    it: the padded tile for a whole-tile run, one patch-aligned window for a sliding one.
    """
    plan = EncoderInputContract.declared_dense(
        encoder_name, target_size_px=target_size_px, window_size=window_size
    ).plan
    return _geometry(
        requested_tile_size_px=target_size_px,
        encoder_input_size_px=_size_pair(plan.effective_encoder_input_size_px),
        read_tile_size_px_by_id=read_tile_size_px_by_id,
    )


def validate_recorded_geometry(
    *,
    cache_dir: Path,
    existing: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    """Raise if the cache was written under a different effective encoder input.

    Silent on either side being absent: a cache predating the record has nothing to
    compare, and a run that cannot declare its geometry has nothing to compare with.
    """
    if expected is None:
        return
    recorded = existing.get(GEOMETRY_METADATA_KEY)
    if not recorded:
        return
    recorded_input = _size_pair(recorded.get("encoder_input_size_px"))
    expected_input = _size_pair(expected.get("encoder_input_size_px"))
    if recorded_input is None or expected_input is None or recorded_input == expected_input:
        return
    raise CacheGeometryMismatch(
        f"Feature cache at {cache_dir} was written with an effective encoder input of "
        f"{recorded_input[0]}x{recorded_input[1]}px, but this run resolves to "
        f"{expected_input[0]}x{expected_input[1]}px. The cached features are registered to "
        "a different extent, so reusing them would train on the wrong geometry. This is not "
        "ordinary incompleteness and is not recomputed automatically: delete the cache "
        "directory to re-extract, or point the run at a different cache root."
    )
