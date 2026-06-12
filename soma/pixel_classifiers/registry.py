"""Pixel-classifier registry.

Parallel to ``decoder_registry`` / ``aggregator_registry`` — the decoder-free
segmentation path selects a registered per-pixel classifier by name.
"""

from soma.registry import Registry

pixel_classifier_registry = Registry("pixel_classifiers")
