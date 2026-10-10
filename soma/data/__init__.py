"""soma.data: the data contracts features and targets enter soma through (design §4)."""

from __future__ import annotations

from soma.data.adapters import from_arrays, from_directory
from soma.data.cohort import Cohort, FoldSplit
from soma.data.geometry import GridGeometry
from soma.data.manifests import AnnotationEntry, AnnotationManifest, ImageEntry, ImageManifest
from soma.data.records import SampleRecord
from soma.data.sources import (
    GridSource,
    GroupedSetSource,
    ImageSource,
    Points,
    SetSource,
    TargetSource,
    covers,
    group_by,
    require_coverage,
)
from soma.data.validation import ensure_filename_safe_id, is_filename_safe_id

__all__ = [
    "AnnotationEntry",
    "AnnotationManifest",
    "Cohort",
    "FoldSplit",
    "GridGeometry",
    "GridSource",
    "GroupedSetSource",
    "ImageEntry",
    "ImageManifest",
    "ImageSource",
    "Points",
    "SampleRecord",
    "SetSource",
    "TargetSource",
    "covers",
    "ensure_filename_safe_id",
    "from_arrays",
    "from_directory",
    "group_by",
    "is_filename_safe_id",
    "require_coverage",
]
