"""Dense (grid) feature infrastructure: the cached grid source, its readers and stores.

Grid geometry lives in :mod:`soma.data.geometry`.
"""

from soma.extract.dense.store import (
    DENSE_ARTIFACT_TYPE,
    DENSE_IMAGE_PAYLOAD_SUBDIR,
    DENSE_PAYLOAD_SUBDIR,
    DENSE_SIDECAR_SUFFIX,
    CachedGridSource,
    DenseSampleSpacing,
    DenseSourceProvenance,
    dense_grid_metadata,
    dense_sample_spacing_from_metadata,
    require_declared_spacing,
    resolve_dense_payload_dir,
    write_dense_grid,
)

__all__ = [
    "CachedGridSource",
    "DenseSampleSpacing",
    "DenseSourceProvenance",
    "dense_sample_spacing_from_metadata",
    "DENSE_ARTIFACT_TYPE",
    "DENSE_IMAGE_PAYLOAD_SUBDIR",
    "DENSE_PAYLOAD_SUBDIR",
    "DENSE_SIDECAR_SUFFIX",
    "dense_grid_metadata",
    "require_declared_spacing",
    "resolve_dense_payload_dir",
    "write_dense_grid",
]
