"""Manifest curation interfaces and legacy packaged providers.

Every curator emits the same unified Manifest through
:func:`~soma.curation.manifest.write_manifest`. Curators are Protocol-typed,
deterministic functions with **no base class** (ADR 0004). New concrete dataset
providers live outside the installable package (ADR 0010); the remaining named exports
are compatibility debt pending staged migration.
"""

from soma.curation.eva import (
    curate_eva_patch_dataset,
    curate_eva_patch_datasets,
)
from soma.curation.hest import curate_hest
from soma.curation.manifest import (
    SUPERVISION_COLUMN,
    CuratedManifest,
    Curator,
    write_manifest,
)
from soma.curation.midog import curate_midog_detection
from soma.curation.monkey import curate_monkey_detection
from soma.curation.ocelot import curate_ocelot_detection
from soma.curation.croma import (
    curate_croma_view,
    curate_croma_views,
)
from soma.curation.segmentation_coverage import (
    summarize_coverage,
    write_coverage_csv,
)
from soma.curation.tile_detection import tile_detection_manifest

__all__ = [
    "SUPERVISION_COLUMN",
    "CuratedManifest",
    "Curator",
    "curate_eva_patch_dataset",
    "curate_eva_patch_datasets",
    "curate_hest",
    "curate_midog_detection",
    "curate_monkey_detection",
    "curate_ocelot_detection",
    "curate_croma_view",
    "curate_croma_views",
    "summarize_coverage",
    "tile_detection_manifest",
    "write_coverage_csv",
    "write_manifest",
]
