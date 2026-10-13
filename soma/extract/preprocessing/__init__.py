from typing import Any

from soma.extract.preprocessing.hierarchy import derive_preprocessing_for_aggregator
from hs2p.wsi import (
    overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)

__all__ = [
    "derive_preprocessing_for_aggregator",
    "is_annotation_sampling",
    "overlay_mask_on_slide",
    "save_overlay_preview",
    "write_coordinate_preview",
]


def is_annotation_sampling(masks: Any) -> bool:
    """Whether an hs2p ``MaskSamplingConfig`` samples annotations rather than binary
    tissue (hs2p's own test)."""
    from hs2p.configs.resolvers import is_binary_tissue_tiling

    return not is_binary_tissue_tiling(masks)
