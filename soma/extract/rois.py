"""Annotation-sampled ROIs, private to extraction (ADR 0021).

Annotation sampling turns each whole slide of an image manifest into the ROIs its
annotation raster selects. What one ROI is — its parent slide's manifest rows, its level-0
origin in that slide, the mask crop sampling stored for it — lives on a :class:`RoiEntry`
that only extraction reads: the grid extractor encodes each ROI's window of its slide, and
:class:`RoiSet` serves the label maps and pixels keyed by ROI id. No public type carries a
path or ROI geometry: what leaves extraction is a :class:`~soma.data.GridSource` keyed by
ROI id whose ``rois`` name each ROI's parent sample (:attr:`RoiSet.children`), from which
:meth:`soma.data.Cohort.derived` builds the ROI cohort.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soma._internal.atomic_io import atomic_write_text
from soma.data.manifests import AnnotationManifest, ImageEntry, ImageManifest

__all__ = ["RoiSet"]

#: Columns of the ROI table an annotation-sampled extraction writes beside its grids.
ROI_TABLE_COLUMNS = (
    "sample_id",
    "slide_id",
    "image_path",
    "mask_path",
    "label_mask_path",
    "patient_id",
    "spacing_at_level_0",
    "region_x",
    "region_y",
    "label_mask_crop_path",
)


@dataclass(frozen=True)
class RoiEntry:
    """One annotation-sampled ROI.

    ``slide`` is the parent slide's image row and ``label_mask_path`` its annotation
    raster; ``region`` is the ROI's level-0 ``(x, y)`` origin in the slide and
    ``label_mask_crop_path`` the raw-value mask crop ROI sampling stores for it.
    """

    sample_id: str
    slide: ImageEntry
    label_mask_path: Path
    region: tuple[int, int]
    label_mask_crop_path: Path | None = None

    @property
    def slide_id(self) -> str:
        return self.slide.sample_id

    @property
    def image(self) -> ImageEntry:
        """The ROI's image row: its id over its slide's image fields (its cache identity)."""
        return ImageEntry(
            sample_id=self.sample_id,
            image_path=self.slide.image_path,
            mask_path=self.slide.mask_path,
            spacing_at_level_0=self.slide.spacing_at_level_0,
            patient_id=self.slide.patient_id,
        )


def roi_id(slide_id: str, x: int, y: int) -> str:
    """The id of the ROI at level-0 ``(x, y)`` of slide ``slide_id``."""
    return f"{slide_id}__x{int(x)}_y{int(y)}"


def roi_entries(
    images: ImageManifest,
    annotations: AnnotationManifest,
    coords_by_slide: Mapping[str, Sequence[tuple[int, int]]],
    *,
    mask_crop_dirs: Mapping[str, Path] | None = None,
) -> list[RoiEntry]:
    """The ROIs of ``coords_by_slide``, slide by slide in manifest order.

    ``mask_crop_dirs`` gives each slide's mask crop directory; each ROI's crop is then
    ``<dir>/<x>_<y>.png`` there (``None`` without it).
    """
    from soma.extract.cache._types import mask_crop_name

    entries: list[RoiEntry] = []
    for slide_id in images.sample_ids:
        slide = images[slide_id]
        for x, y in coords_by_slide.get(slide_id, []):
            crop = None if mask_crop_dirs is None else Path(mask_crop_dirs[slide_id]) / mask_crop_name(x, y)
            entries.append(
                RoiEntry(
                    sample_id=roi_id(slide_id, x, y),
                    slide=slide,
                    label_mask_path=annotations[slide_id].label_mask_path,
                    region=(int(x), int(y)),
                    label_mask_crop_path=crop,
                )
            )
    return entries


def write_roi_table(entries: Sequence[RoiEntry], *, out_dir: Path) -> Path:
    """Record the ROIs as ``<out_dir>/dataset.csv``, one row per ROI (provenance only:
    nothing in soma reads it back)."""
    rows = [
        {
            "sample_id": entry.sample_id,
            "slide_id": entry.slide_id,
            "image_path": str(entry.slide.image_path),
            "mask_path": "" if entry.slide.mask_path is None else str(entry.slide.mask_path),
            "label_mask_path": str(entry.label_mask_path),
            "patient_id": "" if entry.slide.patient_id is None else entry.slide.patient_id,
            "spacing_at_level_0": entry.slide.spacing_at_level_0,
            "region_x": int(entry.region[0]),
            "region_y": int(entry.region[1]),
            "label_mask_crop_path": "" if entry.label_mask_crop_path is None else str(entry.label_mask_crop_path),
        }
        for entry in entries
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dataset.csv"
    # Launches that split one run's folds all rewrite this file at startup while
    # siblings read it, so it must never be visible half-written.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(ROI_TABLE_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), newline="")
    return path


class RoiSet:
    """The ROIs annotation sampling drew from a cohort's whole slides.

    An annotation-sampled :func:`soma.extract` returns a grid source keyed by ROI id whose
    ``rois`` is this set. It names each ROI's parent sample (:attr:`children`, for
    :meth:`soma.data.Cohort.derived`) and serves what is read per ROI: the label maps of
    the mask crops sampling stored (:meth:`label_maps`) and each ROI's window of its slide
    (:meth:`pixels`). Pass it back as ``soma.extract(..., rois=...)`` to encode the same
    ROIs again (another encoder or recipe) without sampling.

    ``sampling`` records the extraction the ROIs were sampled (and their mask crops
    stored) at, as ``soma.extract.cache.roi_sampling_signature`` gives it; a replay at another one
    is refused. A set built without it is not checked.
    """

    def __init__(self, entries: Sequence[RoiEntry], *, sampling: Mapping[str, Any] | None = None) -> None:
        self._entries: dict[str, RoiEntry] = {}
        for entry in entries:
            if entry.sample_id in self._entries:
                raise ValueError(f"Duplicate ROI id {entry.sample_id!r}.")
            self._entries[entry.sample_id] = entry
        self.sampling = sampling

    @property
    def sample_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def children(self) -> dict[str, str]:
        """ROI id -> the ``sample_id`` of the slide it was sampled from."""
        return {roi: entry.slide_id for roi, entry in self._entries.items()}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._entries

    def __iter__(self) -> Iterator[RoiEntry]:
        return iter(self._entries.values())

    def __getitem__(self, sample_id: str) -> RoiEntry:
        try:
            return self._entries[sample_id]
        except KeyError:
            raise KeyError(f"Unknown ROI id {sample_id!r}.") from None

    def subset(self, sample_ids: Sequence[str]) -> "RoiSet":
        """The same set restricted to ``sample_ids``, in their order."""
        return RoiSet([self[str(sample_id)] for sample_id in sample_ids], sampling=self.sampling)

    def label_maps(self, *, size: tuple[int, int], label_remap: Any = None, ignore_index: int = 255):
        """The ROIs' label maps, read from the mask crops sampling stored (a
        :class:`~soma.data.CachedLabelMapSource`)."""
        from soma.data import CachedLabelMapSource

        missing = [roi for roi, entry in self._entries.items() if entry.label_mask_crop_path is None]
        if missing:
            raise ValueError(
                f"segmentation ROI(s) {missing[:10]} have no stored mask crop; ROI sampling "
                "writes one per ROI."
            )
        return CachedLabelMapSource(
            {roi: entry.label_mask_crop_path for roi, entry in self._entries.items()},
            size=size,
            label_remap=label_remap,
            ignore_index=ignore_index,
        )

    def pixels(
        self,
        *,
        size: int | tuple[int, int],
        spacing_um: float,
        spacing_policy: str = "strict",
        backend: str = "auto",
        tolerance: float = 0.05,
    ):
        """Each ROI's window of its slide, read at ``spacing_um`` as the grid extractor
        reads it (an :class:`~soma.data.ImageSource`)."""
        from soma.data.images import RegionImageSource

        return RegionImageSource(
            [entry.image for entry in self._entries.values()],
            regions={roi: entry.region for roi, entry in self._entries.items()},
            size=size,
            spacing_um=spacing_um,
            spacing_policy=spacing_policy,
            backend=backend,
            tolerance=tolerance,
        )


def label_map_identity(
    sample_ids: Sequence[str],
    *,
    images: ImageManifest,
    annotations: AnnotationManifest | None,
    rois: RoiSet | None = None,
) -> list[dict[str, Any]]:
    """What each sample's label map is read from, as the ROI population cache keys it: an
    ROI's stored crop (and its slide, origin and spacing), any other sample's raster."""
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        if rois is not None and sample_id in rois:
            entry = rois[sample_id]
            rows.append(
                {
                    "sample_id": entry.sample_id,
                    "slide_id": entry.slide_id,
                    "label_mask_path": str(entry.label_mask_path),
                    "label_mask_crop_path": (
                        None if entry.label_mask_crop_path is None else str(entry.label_mask_crop_path)
                    ),
                    "region": [int(entry.region[0]), int(entry.region[1])],
                    "spacing_at_level_0": entry.slide.spacing_at_level_0,
                }
            )
            continue
        annotation = annotations[sample_id] if annotations is not None and sample_id in annotations else None
        label_mask_path = None if annotation is None else annotation.label_mask_path
        rows.append(
            {
                "sample_id": sample_id,
                "slide_id": None,
                "label_mask_path": None if label_mask_path is None else str(label_mask_path),
                "label_mask_crop_path": None,
                "region": None,
                "spacing_at_level_0": images[sample_id].spacing_at_level_0 if sample_id in images else None,
            }
        )
    return rows
