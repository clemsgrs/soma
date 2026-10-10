"""Dense (segmentation) feature infrastructure: geometry + the cached grid source."""

from soma.data.geometry import (
    DenseGridGeometry,
    GridGeometry,
    compute_dense_geometry,
    normalize_hw,
)
from soma.dense.store import (
    DENSE_ARTIFACT_TYPE,
    DENSE_IMAGE_PAYLOAD_SUBDIR,
    DENSE_PAYLOAD_SUBDIR,
    DENSE_SIDECAR_SUFFIX,
    CachedGridSource,
    DenseSampleSpacing,
    DenseSourceProvenance,
    dense_grid_metadata,
    dense_sample_spacing_from_metadata,
    resolve_dense_payload_dir,
    write_dense_grid,
)

__all__ = [
    "DenseGridGeometry",
    "GridGeometry",
    "compute_dense_geometry",
    "normalize_hw",
    "CachedGridSource",
    "DenseSampleSpacing",
    "DenseSourceProvenance",
    "dense_sample_spacing_from_metadata",
    "DENSE_ARTIFACT_TYPE",
    "DENSE_IMAGE_PAYLOAD_SUBDIR",
    "DENSE_PAYLOAD_SUBDIR",
    "DENSE_SIDECAR_SUFFIX",
    "dense_grid_metadata",
    "resolve_dense_payload_dir",
    "write_dense_grid",
]
