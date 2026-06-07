"""Feature extraction package: tiling, encoding, and aggregation."""

import gc  # noqa: F401 — exposed so tests can patch soma.extraction.gc.collect
from slide2vec import Model  # noqa: F401 — exposed so tests can patch soma.extraction.Model.from_preset

from soma.extraction.extractor import (
    FeatureExtractor,
    _validate_runtime,
    # modules imported by extractor (exposed for patching in tests)
    torch,
    os,
    Pipeline,
    validate_slide2vec_encoder_config,
    probe_resolved_backends,
    resolve_tiling_cache,
    resolve_slide_cache,
    load_tilings,
    # orchestration helpers (imported into extractor namespace, exposed here)
    _aggregate_patients,
    _aggregate_tiles,
    _embed_tiles,
    _load_model,
    _release_parent_cuda_state,
    _run_with_coordinates,
)

__all__ = [
    "FeatureExtractor",
    "_validate_runtime",
    "_aggregate_patients",
    "_aggregate_tiles",
    "_embed_tiles",
    "_load_model",
    "_release_parent_cuda_state",
    "_run_with_coordinates",
]
