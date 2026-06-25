"""Per-pixel classifiers for the decoder-free segmentation path.

Importing this package registers the built-in classifiers (the registry is the
discovery surface used by the config validator and the pipeline).
"""

from soma.pixel_classifiers.base import PixelClassifier
from soma.pixel_classifiers.registry import pixel_classifier_registry

# Register built-ins on import (side-effecting, like aggregators/decoders).
from soma.pixel_classifiers import mlp_clf  # noqa: F401,E402
from soma.pixel_classifiers import sklearn_clf  # noqa: F401,E402
from soma.pixel_classifiers import xgboost_clf  # noqa: F401,E402


def list_pixel_classifiers() -> list[str]:
    """Return registered per-pixel classifier names in a stable order."""
    return sorted(pixel_classifier_registry.list())


__all__ = ["PixelClassifier", "pixel_classifier_registry", "list_pixel_classifiers"]
