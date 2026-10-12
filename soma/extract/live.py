"""The frozen encoder stage of the live grid path (design §6, §7.2)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from soma.data.geometry import DenseGridGeometry, GridGeometry

__all__ = ["FrozenEncoder", "live_encoder"]

logger = logging.getLogger(__name__)


def _soma_geometry(resolved) -> GridGeometry:
    """slide2vec's crop box is ``(left, top, right, bottom)``; soma's heads use
    ``(top, left, height, width)``. Everything else is taken as resolved."""
    left, top, right, bottom = (int(value) for value in resolved.crop_box)
    return GridGeometry.from_dense(
        DenseGridGeometry(
            target_size=tuple(int(v) for v in resolved.target_size),
            patch_size=tuple(int(v) for v in resolved.patch_size),
            encoded_size=tuple(int(v) for v in resolved.encoded_size),
            grid_shape=tuple(int(v) for v in resolved.grid_shape),
            pad=tuple(int(v) for v in resolved.pad),
            crop_box=(top, left, bottom - top, right - left),
        )
    )


class FrozenEncoder(nn.Module):
    """A parameter-free ``nn.Module`` over slide2vec's ``DenseEncodeKit``.

    The kit holds the foundation model privately and is deliberately not an
    ``nn.Module``, so the backbone never enters a checkpoint, an optimizer or a
    ``state_dict``; the kit owns device transfer, precision and eval locking.
    ``input_device`` is ``"cpu"``: the learner leaves pixels on the host and the kit
    moves them.

    ``forward`` takes either a preprocessed float batch at ``encoded_size`` (the
    loaders ran :meth:`preprocessor` in their workers) or raw uint8 pixels at
    ``target_size`` (preprocessed here, item by item) and returns the ``(B, D, h, w)``
    grid on the encoder's device as float32, as cached grids are.

    Built once per run by :func:`live_encoder` and shared across folds.
    """

    input_device = "cpu"

    def __init__(self, kit: object, *, feature_dim: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        self._kit = kit
        if "_kit" in self._modules:  # pragma: no cover - a kit is a plain object
            raise TypeError("the DenseEncodeKit must stay a plain object, never a submodule")
        self.geometry: GridGeometry = _soma_geometry(kit.geometry)
        self.feature_dim = int(feature_dim)
        self.device = None if device is None else torch.device(device)
        self._preprocess = kit.preprocessor()

    def preprocessor(self) -> Callable[[Tensor], Tensor]:
        """The kit's pickle-safe itemwise CPU preprocessor: ``(3, H, W)`` uint8 at
        ``target_size`` -> ``(3, Henc, Wenc)`` normalised and padded."""
        return self._kit.preprocessor()

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.dtype == torch.uint8:
            pixels = torch.stack([self._preprocess(item) for item in pixels.cpu()])
        with torch.no_grad():
            return self._kit.encode(pixels).float()


def live_encoder(
    encoder: str,
    *,
    target_size: int,
    spacing_um: float,
    window: int | None = None,
    overlap: float = 0.0,
    feature_kind: str = "patch_features",
    precision: str | None = None,
    output_variant: str | None = None,
    allow_non_recommended_settings: bool = False,
    tolerance: float = 0.05,
    backend: str = "auto",
    attention_blocks: Sequence[int] = (-1,),
    attention_include_registers: bool = False,
    num_gpus: int | None = None,
    device: str | None = None,
) -> FrozenEncoder:
    """Prepare a slide2vec dense encoder for live re-encoding and wrap it as a stage.

    Args:
        encoder: slide2vec preset name (e.g. ``"uni2"``).
        target_size: Supervision tile side in pixels; the stage pads it to the encoder's
            patch multiple.
        spacing_um: µm/px the pixels are read at.
        window / overlap: Sliding-window encoding of a tile larger than the encoder's
            input (``None`` = one forward over the whole tile).
        feature_kind: slide2vec dense feature kind (``patch_features``,
            ``patch_features_prenorm``, ``cls_attention``).
        precision: Encoder precision; ``None`` resolves the preset's recommendation.
        attention_blocks / attention_include_registers: The ``cls_attention`` recipe.

    The stage probe-encodes one blank tile to learn its feature width and to check that
    the kit's grid matches its declared geometry.
    """
    from slide2vec import DenseImageOptions, ExecutionOptions, Model

    from soma.dense.sliding import describe_dense_mode
    from soma.encoders.validation import resolve_encoder_precision

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    model = Model.from_preset(
        encoder,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
        device=device,
    )
    resolved_precision = resolve_encoder_precision(precision, encoder_name=encoder)
    logger.info("Live dense mode: %s", describe_dense_mode(window, float(overlap)))
    attention = feature_kind == "cls_attention"
    dense = DenseImageOptions(
        target_size=int(target_size),
        spacing_um=float(spacing_um),
        tolerance=float(tolerance),
        backend=backend,
        pad_mode="reflect",
        image_pad_value=None,
        window_size=window,
        overlap=float(overlap),
        feature_kind=feature_kind,
        attention_blocks=tuple(int(b) for b in attention_blocks) if attention else (-1,),
        attention_include_registers=bool(attention_include_registers) if attention else False,
    )
    kit = model.prepare_dense_encoder(
        dense=dense,
        execution=ExecutionOptions(num_gpus=num_gpus, precision=resolved_precision, output_dtype="fp32"),
    )
    probe = torch.stack([kit.preprocessor()(torch.zeros((3, *kit.geometry.target_size), dtype=torch.uint8))])
    grid = kit.encode(probe)
    if grid.ndim != 4 or tuple(int(v) for v in grid.shape[-2:]) != tuple(int(v) for v in kit.geometry.grid_shape):
        raise ValueError(
            "DenseEncodeKit returned an invalid probe grid: expected (B, d, "
            f"{kit.geometry.grid_shape[0]}, {kit.geometry.grid_shape[1]}), got {tuple(grid.shape)}."
        )
    return FrozenEncoder(kit, feature_dim=int(grid.shape[1]), device=model.device)
