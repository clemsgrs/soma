"""Task-neutral manifests: what exists on disk for each sample (design §4.3).

``ImageManifest`` is read by extraction and curators; ``AnnotationManifest`` by target
sources and curators. The same CSV may feed both and a :class:`~soma.data.Cohort`; each
reader validates only its own columns.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from soma.data.validation import (
    optional_path,
    optional_text,
    parse_spacing_at_level_0,
    require_columns,
    validate_patient_ids,
    validate_sample_ids,
    validate_spacing_declaration_columns,
)

__all__ = ["AnnotationEntry", "AnnotationManifest", "ImageEntry", "ImageManifest"]


@dataclass(frozen=True)
class ImageEntry:
    """Where one sample's pixels live, plus the optional extraction aids."""

    sample_id: str
    image_path: Path
    # Optional precomputed tissue mask (a tile-sampling aid, never trained on).
    mask_path: Path | None = None
    # Optional caller declaration of ``image_path``'s level-0 µm/px.
    spacing_at_level_0: float | None = None
    # User-supplied hs2p tile coordinates (``.coordinates.npz``); set, soma does not tile.
    coordinates_path: Path | None = None
    patient_id: str | None = None


@dataclass(frozen=True)
class AnnotationEntry:
    """Dense supervision files of one sample, in the image's level-0 pixel frame."""

    sample_id: str
    label_mask_path: Path | None = None
    points_path: Path | None = None
    # Optional don't-care raster: 255 = ignored, 0 = supervised.
    ignore_mask_path: Path | None = None


class _Manifest:
    _entries: dict

    @property
    def sample_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def entries(self) -> dict:
        return dict(self._entries)

    def __getitem__(self, sample_id: str):
        try:
            return self._entries[sample_id]
        except KeyError:
            raise KeyError(f"Unknown sample_id {sample_id!r}.") from None

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._entries

    def __iter__(self) -> Iterator:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


class ImageManifest(_Manifest):
    """``sample_id, image_path`` plus optional ``mask_path``, ``spacing_at_level_0``,
    ``coordinates_path``, ``patient_id``. Tile-vs-WSI is an extraction setting, not a
    manifest kind."""

    def __init__(self, entries: Mapping[str, ImageEntry] | list[ImageEntry]) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, ImageEntry] = dict(entries)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "ImageManifest":
        validate_sample_ids(df, what="image manifest")
        validate_patient_ids(df)
        validate_spacing_declaration_columns(df)
        if "tissue_mask_path" in df.columns:
            raise ValueError(
                "Use 'mask_path' (the tissue mask column) instead of 'tissue_mask_path'."
            )
        require_columns(df, {"image_path"}, what="image manifest")
        blank = df.loc[df["image_path"].isna(), "sample_id"].astype(str).tolist()
        if blank:
            raise ValueError(f"image_path is blank for sample(s): {blank}")
        if "coordinates_path" in df.columns:
            missing = df.loc[df["coordinates_path"].isna(), "sample_id"].astype(str).tolist()
            if missing and len(missing) < len(df):
                raise ValueError(
                    "Manifest column 'coordinates_path' must be set for every row or for "
                    f"none; missing for {len(missing)} sample(s): {missing[:20]}"
                    f"{' ...' if len(missing) > 20 else ''}."
                )
        entries: list[ImageEntry] = []
        for _, row in df.iterrows():
            entries.append(
                ImageEntry(
                    sample_id=str(row["sample_id"]),
                    image_path=Path(str(row["image_path"])),
                    mask_path=optional_path(row, "mask_path"),
                    spacing_at_level_0=parse_spacing_at_level_0(row.get("spacing_at_level_0")),
                    coordinates_path=optional_path(row, "coordinates_path"),
                    patient_id=optional_text(row, "patient_id"),
                )
            )
        return cls(entries)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ImageManifest":
        return cls.from_frame(pd.read_csv(path))

    @property
    def supplies_coordinates(self) -> bool:
        """True when every row names its hs2p tile coordinates (no soma tiling)."""
        return bool(self._entries) and all(
            entry.coordinates_path is not None for entry in self._entries.values()
        )

    def with_coordinates(self, coordinates_paths: Mapping[str, Path]) -> "ImageManifest":
        """A copy whose rows name ``coordinates_paths`` (a run's snapshot of the artifacts)."""
        clone = copy.copy(self)
        clone._entries = {
            sample_id: replace(entry, coordinates_path=Path(coordinates_paths[sample_id]))
            for sample_id, entry in self._entries.items()
        }
        return clone


_ANNOTATION_COLUMNS = ("label_mask_path", "points_path", "ignore_mask_path")


class AnnotationManifest(_Manifest):
    """``sample_id`` plus whichever of ``label_mask_path`` / ``points_path`` /
    ``ignore_mask_path`` exist; ``pixel_mapping`` (raw mask value per class name) is
    manifest metadata."""

    def __init__(
        self,
        entries: Mapping[str, AnnotationEntry] | list[AnnotationEntry],
        *,
        pixel_mapping: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, AnnotationEntry] = dict(entries)
        self._pixel_mapping = None if pixel_mapping is None else dict(pixel_mapping)

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, *, pixel_mapping: Mapping[str, int] | None = None
    ) -> "AnnotationManifest":
        validate_sample_ids(df, what="annotation manifest")
        present = [column for column in _ANNOTATION_COLUMNS if column in df.columns]
        if "label_mask_path" not in df.columns and "mask_path" in df.columns and not present:
            raise ValueError(
                "Table has 'mask_path' but no 'label_mask_path': this looks like a pre-rename "
                "segmentation manifest (soma < 1.11 used 'mask_path' for the supervision "
                "raster; it now means an optional tissue mask). Regenerate it with its "
                "curator, or rename the column to 'label_mask_path'."
            )
        if not present:
            raise ValueError(
                "An annotation manifest needs at least one of the columns "
                f"{list(_ANNOTATION_COLUMNS)} ('label_mask_path' or 'points_path' for "
                f"supervision). Available: {list(df.columns)}"
            )
        for column in ("label_mask_path", "points_path"):
            if column in df.columns:
                blank = df.loc[df[column].isna(), "sample_id"].astype(str).tolist()
                if blank:
                    raise ValueError(f"{column} is blank for sample(s): {blank}")
        entries = [
            AnnotationEntry(
                sample_id=str(row["sample_id"]),
                label_mask_path=optional_path(row, "label_mask_path"),
                points_path=optional_path(row, "points_path"),
                ignore_mask_path=optional_path(row, "ignore_mask_path"),
            )
            for _, row in df.iterrows()
        ]
        return cls(entries, pixel_mapping=pixel_mapping)

    @classmethod
    def from_csv(
        cls, path: str | Path, *, pixel_mapping: Mapping[str, int] | None = None
    ) -> "AnnotationManifest":
        return cls.from_frame(pd.read_csv(path), pixel_mapping=pixel_mapping)

    @property
    def pixel_mapping(self) -> dict[str, int] | None:
        return None if self._pixel_mapping is None else dict(self._pixel_mapping)

    @property
    def has_label_masks(self) -> bool:
        return any(entry.label_mask_path is not None for entry in self._entries.values())

    @property
    def has_points(self) -> bool:
        return any(entry.points_path is not None for entry in self._entries.values())
