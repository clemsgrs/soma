"""Concrete WSI backend adapters."""

from soma.wsi.backends.cucim import CuCIMReader
from soma.wsi.backends.openslide import OpenSlideReader

__all__ = ["OpenSlideReader", "CuCIMReader"]
