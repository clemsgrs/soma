"""Dataset curation utilities.

Every curator emits the same unified Manifest (``dataset.csv`` + ``splits.csv`` +
``summary.json``) via the shared :func:`~soma.curation.manifest.write_manifest`. Curators
are Protocol-typed, deterministic functions with **no base class** (ADR 0004); the shared
contract lives in :mod:`soma.curation.manifest`.
"""

from soma.curation.beetle import curate_beetle_slide_manifest
from soma.curation.eva import (
    curate_eva_patch_dataset,
    curate_eva_patch_datasets,
)
from soma.curation.manifest import (
    SUPERVISION_COLUMN,
    CuratedManifest,
    Curator,
    write_manifest,
)
from soma.curation.midog import curate_midog_detection
from soma.curation.monkey import curate_monkey_detection
from soma.curation.ocelot import curate_ocelot_detection
from soma.curation.segmentation_coverage import (
    summarize_coverage,
    write_coverage_csv,
)

__all__ = [
    "SUPERVISION_COLUMN",
    "CuratedManifest",
    "Curator",
    "curate_beetle_slide_manifest",
    "curate_eva_patch_dataset",
    "curate_eva_patch_datasets",
    "curate_midog_detection",
    "curate_monkey_detection",
    "curate_ocelot_detection",
    "summarize_coverage",
    "write_coverage_csv",
    "write_manifest",
]
