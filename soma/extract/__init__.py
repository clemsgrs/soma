"""``soma.extract``: the verb features enter soma through, and its namespace.

``soma.extract(images, *, encoder, cache, ...)`` reads an image manifest and returns the
source the extraction produced (:func:`soma.extraction.extract`, ADR 0021). The namespace
also holds :func:`live_encoder`, the only place the live grid path speaks slide2vec: it
prepares slide2vec's ``DenseEncodeKit`` and wraps it in a :class:`FrozenEncoder`, the
parameter-free encoder stage of :class:`~soma.training.model.DenseModel`.
"""

import sys
from types import ModuleType

from soma.extract.live import FrozenEncoder, live_encoder
from soma.extraction.facade import extract
from soma.extraction.rois import RoiSet

__all__ = ["FrozenEncoder", "RoiSet", "extract", "live_encoder"]


class _ExtractNamespace(ModuleType):
    """The namespace is the verb too: ``soma.extract(...)`` calls :func:`extract`."""

    def __call__(self, *args, **kwargs):
        return extract(*args, **kwargs)


sys.modules[__name__].__class__ = _ExtractNamespace
