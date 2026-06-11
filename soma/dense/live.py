"""LiveSegmentationSource — the carrier for the live re-encode segmentation path.

Where the cached path returns a :class:`~soma.dense.DenseFeatureStore` (grids on
disk), the live path returns a :class:`LiveSegmentationSource`: a passive struct that
holds the **single loaded frozen encoder** plus everything the fold needs to build a
:class:`~soma.training.model.LiveSegmentationModel` and a
:class:`~soma.training.segmentation_dataset.LiveSegmentationDataset` —
``{encoder, device, precision, geometry, feature_dim, dense_transform, augmentation,
spacing, pad}``.

It is built **once**, before the fold loop, so the (large) backbone loads a single
time and every fold's model shares the same frozen encoder (safe: it has no trainable
state — each fold gets a fresh decoder+head and its own optimizer). It is not a
behavioral protocol; the segmentation fold reads its fields directly in an inline
branch (design §13.B-3/§13.B-8). It deliberately does *not* subclass
``DenseFeatureStore`` — the two share the ``validate_coverage(ids)`` name, not an
inheritance relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from soma.config import AugmentationConfig
from soma.dense.geometry import DenseGridGeometry


@dataclass
class LiveSegmentationSource:
    """Frozen encoder + geometry/transform bundle for the live re-encode path.

    Attributes:
        encoder: The loaded dense-capable slide2vec tile encoder (shared across folds).
        device: Device the encoder lives on.
        precision: Encoder autocast precision (resolved as extraction does).
        geometry: Dense geometry from ``patch_size + target_size`` (no cache sidecar).
        feature_dim: Encoder output channels ``d`` (from a probe forward — the same
            source of truth as the cached extractor's ``grids.shape[1]``).
        dense_transform: The encoder's normalization-only transform.
        augmentation: The run's augmentation config (applied on the train split only).
        spacing_um: µm/px to read image+mask at (``None`` = flat PIL read).
        backend / tolerance: hs2p reader settings.
        pad_mode / image_pad_value: image pad-to-encoded contract (mirrors extraction).
        window_size / overlap: dense encoder-window knobs (design §5, window-as-knob).
            ``window_size=None`` ⇒ ``whole`` (one padded forward, the live default and
            the cached-parity anchor); a smaller window slides the encoder over
            patch-aligned windows and blends the token grids — identical mechanism to
            the cached extractor, so cached/live agree under any window setting.
    """

    encoder: object
    device: object
    precision: str
    geometry: DenseGridGeometry
    feature_dim: int
    dense_transform: Callable
    augmentation: AugmentationConfig
    spacing_um: float | None
    backend: str = "auto"
    tolerance: float = 0.05
    pad_mode: str = "reflect"
    image_pad_value: float | None = None
    window_size: int | None = None
    overlap: float = 0.0

    def validate_coverage(self, sample_ids) -> None:
        """No-op coverage hook (name-compatible with ``DenseFeatureStore``).

        There is nothing cached to cover — the live path re-encodes from each record's
        ``image_path``/``mask_path``, which the fold validates against the records
        directly (it needs the records, not just the ids).
        """
        return None
