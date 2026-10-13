"""``soma.extract``: the verb features enter soma through, and its namespace.

``soma.extract(images, *, encoder, cache, ...)`` reads an image manifest and returns the
source the extraction produced (:func:`soma.extract.extract`, ADR 0021). The namespace
also holds :func:`live_encoder`, the only place the live grid path speaks slide2vec: it
prepares slide2vec's ``DenseEncodeKit`` and wraps it in a :class:`FrozenEncoder`, the
parameter-free encoder stage of :class:`~soma.models.DenseModel`.

Everything extraction owns lives here: the pooled extractor (``extractor``, ``facade``,
``orchestration``), the tile, dense and annotation-sampled extractions, the slide2vec
adapter, spacing resolution, preprocessing, the feature caches (``cache``) and the dense
grid stores and readers (``dense``).
"""

import sys
from types import ModuleType

from soma.extract.contracts import FeatureSource, PooledFeatureSource
from soma.extract.facade import extract
from soma.extract.live import FrozenEncoder, live_encoder
from soma.extract.rois import RoiSet

__all__ = ["FeatureSource", "FrozenEncoder", "PooledFeatureSource", "RoiSet", "extract", "live_encoder"]


class _ExtractNamespace(ModuleType):
    """The namespace is the verb too: ``soma.extract(...)`` calls :func:`extract`."""

    def __call__(self, *args, **kwargs):
        return extract(*args, **kwargs)


sys.modules[__name__].__class__ = _ExtractNamespace
