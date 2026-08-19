"""LiveSegmentationSource — the carrier for the live re-encode segmentation path.

Where the cached path returns a :class:`~soma.dense.DenseFeatureSource` (grids on
disk plus provenance), the live path returns a :class:`LiveSegmentationSource`: a
passive struct that holds the **single public dense encode kit** plus everything the
fold needs to build a :class:`~soma.training.model.LiveSegmentationModel` and a
:class:`~soma.training.segmentation_dataset.LiveSegmentationDataset` —
``{kit, device, geometry, feature_dim, preprocessor, augmentation, spacing}``.

It is built **once**, before the fold loop, so the (large) backbone loads a single
time and every fold's model shares the same frozen encoder (safe: it has no trainable
state — each fold gets a fresh decoder+head and its own optimizer). It is not a
behavioral protocol; the segmentation fold reads its fields directly in an inline
branch (design §13.B-3/§13.B-8). It deliberately stays outside the cache-backed
``DenseFeatureSource`` interface — the live and cached paths share the
``validate_coverage(ids)`` name, not an inheritance relationship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from soma.config import AugmentationConfig
from soma.dense.geometry import DenseGridGeometry


@dataclass
class LiveSegmentationSource:
    """Public DenseEncodeKit + Soma data settings for the live re-encode path.

    Attributes:
        kit: The public slide2vec DenseEncodeKit shared across folds.
        device: Device on which the kit returns grids and trainable modules run.
        geometry: Soma's crop-convention adapter of the authoritative ``kit.geometry``.
        feature_dim: Encoder output channels ``d`` from public ``Model.feature_dim``.
        preprocessor: Serializable callable returned by ``kit.preprocessor()``.
        augmentation: The run's augmentation config (applied on the train split only).
        spacing_um: µm/px to read image+mask at (``None`` = flat PIL read).
        backend / tolerance: hs2p reader settings.
        Dense mode, padding, precision, output variant, feature kind, windowing, and
        attention selection are resolved and owned by the kit rather than restated here.
    """

    kit: object
    device: object
    feature_dim: int
    augmentation: AugmentationConfig
    spacing_um: float | None
    backend: str = "auto"
    tolerance: float = 0.05
    geometry: DenseGridGeometry = field(init=False)
    preprocessor: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved = self.kit.geometry
        left, top, right, bottom = (int(value) for value in resolved.crop_box)
        # Consume the kit's resolved geometry directly. Only crop-box notation differs:
        # slide2vec exposes (left, top, right, bottom), while Soma's heads use
        # (top, left, height, width).
        self.geometry = DenseGridGeometry(
            target_size=tuple(int(v) for v in resolved.target_size),
            patch_size=tuple(int(v) for v in resolved.patch_size),
            encoded_size=tuple(int(v) for v in resolved.encoded_size),
            grid_shape=tuple(int(v) for v in resolved.grid_shape),
            pad=tuple(int(v) for v in resolved.pad),
            crop_box=(top, left, bottom - top, right - left),
        )
        self.preprocessor = self.kit.preprocessor()

    def validate_coverage(self, sample_ids) -> None:
        """No-op coverage hook (name-compatible with ``DenseFeatureStore``).

        There is nothing cached to cover — the live path re-encodes from each record's
        ``image_path``/``label_mask_path``, which the fold validates against the records
        directly (it needs the records, not just the ids).
        """
        return None
