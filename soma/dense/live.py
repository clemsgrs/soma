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

from soma.config import AugmentationConfig, PipelineConfig
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


def build_live_segmentation_source(config: PipelineConfig) -> LiveSegmentationSource:
    """Prepare the public dense encoder kit from a resolved pipeline config.

    This config-only seam is shared by training and post-hoc checkpoint inference.  It
    deliberately does not construct :class:`~soma.pipeline.Pipeline`, because rebuilding
    an encoder for an external image cohort must not require loading the development
    Dataset or Splits referenced by the saved run config.
    """
    import torch
    from slide2vec import DenseImageOptions, ExecutionOptions, Model

    from soma.dense.sliding import describe_dense_mode
    from soma.encoders.validation import resolve_encoder_precision
    from soma.preprocessing.resolution import resolve_pipeline_preprocessing

    if config.encoder is None:
        raise ValueError(
            "PipelineConfig.encoder is required for feature_mode='live' segmentation."
        )
    preprocessing = resolve_pipeline_preprocessing(config)
    target_size = preprocessing.requested_tile_size_px
    if target_size is None:
        raise ValueError(
            "feature_mode='live' segmentation requires preprocessing.requested_tile_size_px "
            "(the mask/tile supervision size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "feature_mode='live' segmentation requires a spacing — set "
            "preprocessing.requested_spacing_um or use an encoder that advertises a "
            "single supported_spacing_um."
        )

    model = Model.from_preset(
        config.encoder.name,
        output_variant=config.encoder.output_variant,
        allow_non_recommended_settings=config.encoder.allow_non_recommended_settings,
    )
    precision = resolve_encoder_precision(
        config.encoder, encoder_name=config.encoder.name
    )
    window_size = preprocessing.dense_window_size
    overlap = float(preprocessing.dense_window_overlap)
    print(f"Live segmentation dense mode: {describe_dense_mode(window_size, overlap)}")
    feature_kind = preprocessing.feature_kind or "patch_features"
    dense = DenseImageOptions(
        target_size=int(target_size),
        spacing_um=float(preprocessing.requested_spacing_um),
        tolerance=float(preprocessing.tolerance),
        backend=preprocessing.backend,
        pad_mode="reflect",
        image_pad_value=None,
        window_size=window_size,
        overlap=overlap,
        feature_kind=feature_kind,
        attention_blocks=(
            tuple(preprocessing.attention.blocks)
            if feature_kind == "cls_attention"
            else (-1,)
        ),
        attention_include_registers=(
            bool(preprocessing.attention.include_registers)
            if feature_kind == "cls_attention"
            else False
        ),
    )
    kit = model.prepare_dense_encoder(
        dense=dense,
        execution=ExecutionOptions(
            num_gpus=config.execution.num_gpus,
            precision=precision,
            output_dtype="fp32",
        ),
    )
    # Probe the public tensor boundary because alternate dense outputs need not have the
    # backbone's pooled feature width.
    probe_pixels = torch.zeros(
        (3, *kit.geometry.target_size), dtype=torch.uint8, device="cpu"
    )
    probe_batch = torch.stack([kit.preprocessor()(probe_pixels)])
    probe_grid = kit.encode(probe_batch)
    if probe_grid.ndim != 4 or tuple(int(v) for v in probe_grid.shape[-2:]) != tuple(
        int(v) for v in kit.geometry.grid_shape
    ):
        raise ValueError(
            "DenseEncodeKit returned an invalid probe grid: expected "
            f"(B, d, {kit.geometry.grid_shape[0]}, {kit.geometry.grid_shape[1]}), "
            f"got {tuple(probe_grid.shape)}."
        )
    return LiveSegmentationSource(
        kit=kit,
        device=model.device,
        feature_dim=int(probe_grid.shape[1]),
        augmentation=config.augmentation,
        spacing_um=float(preprocessing.requested_spacing_um),
        backend=preprocessing.backend,
        tolerance=float(preprocessing.tolerance),
    )


__all__ = ["LiveSegmentationSource", "build_live_segmentation_source"]
