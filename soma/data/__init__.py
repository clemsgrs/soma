"""soma.data: the data contracts features and targets enter soma through (design §4)."""

from __future__ import annotations

from soma.data.adapters import from_arrays, from_directory
from soma.data.cached import CachedSetSource
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
from soma.data.targets import (
    CachedLabelMapSource,
    LabelMapEntry,
    LabelMapSource,
    PointEntry,
    PointSource,
    apply_label_remap,
    build_label_remap,
    resolve_class_scheme,
)
from soma.data.validation import ensure_filename_safe_id, is_filename_safe_id

__all__ = [
    "AnnotationEntry",
    "AnnotationManifest",
    "CachedGridSource",
    "CachedLabelMapSource",
    "CachedSetSource",
    "Cohort",
    "FoldSplit",
    "GridGeometry",
    "GridSource",
    "GroupedSetSource",
    "ImageEntry",
    "ImageManifest",
    "ImageSource",
    "LabelMapEntry",
    "LabelMapSource",
    "PointEntry",
    "PointSource",
    "Points",
    "SampleRecord",
    "apply_label_remap",
    "build_label_remap",
    "resolve_class_scheme",
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


def __getattr__(name: str):
    # CachedGridSource lives in soma.dense (its cache sidecars and writers are there) and
    # would import soma.data circularly; serve it lazily so the three cached sources share
    # one import path.
    if name == "CachedGridSource":
        from soma.dense.store import CachedGridSource

        return CachedGridSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
