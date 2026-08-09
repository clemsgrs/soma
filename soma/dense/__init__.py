"""Dense (segmentation) feature infrastructure: geometry + dense grid cache store."""

from soma.dense.geometry import (
    DenseGridGeometry,
    compute_dense_geometry,
    normalize_hw,
)
from soma.dense.source import (
    CacheBackedDenseSource,
    DenseFeatureSource,
    DenseSampleSpacing,
    DenseSourceProvenance,
)
from soma.dense.store import (
    DENSE_ARTIFACT_TYPE,
    DENSE_IMAGE_PAYLOAD_SUBDIR,
    DENSE_PAYLOAD_SUBDIR,
    DENSE_SIDECAR_SUFFIX,
    DenseFeatureStore,
    dense_grid_metadata,
    resolve_dense_payload_dir,
    write_dense_grid,
)

__all__ = [
    "DenseGridGeometry",
    "compute_dense_geometry",
    "normalize_hw",
    "CacheBackedDenseSource",
    "DenseFeatureSource",
    "DenseSampleSpacing",
    "DenseSourceProvenance",
    "DENSE_ARTIFACT_TYPE",
    "DENSE_IMAGE_PAYLOAD_SUBDIR",
    "DENSE_PAYLOAD_SUBDIR",
    "DENSE_SIDECAR_SUFFIX",
    "DenseFeatureStore",
    "dense_grid_metadata",
    "resolve_dense_payload_dir",
    "write_dense_grid",
]
