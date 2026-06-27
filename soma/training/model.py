"""Training models: EmbeddingModel (direct task head) and MILModel (aggregator + task head)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from typing import TYPE_CHECKING

from soma.aggregators.base import Aggregator
from soma.decoders.base import Decoder
from soma.tasks.base import TaskHead

if TYPE_CHECKING:
    from soma.dense.geometry import DenseGridGeometry


@dataclass
class EmbeddingModelOutput:
    """Output of EmbeddingModel.forward."""

    logits: Tensor  # (B, num_classes)


class EmbeddingModel(nn.Module):
    """Task head applied directly to a pre-computed feature embedding (B, D).

    Used for slide-level, patient-level, and tile-level pipelines where each
    sample is already represented by a single feature vector — no aggregation.
    """

    def __init__(self, task_head: TaskHead) -> None:
        super().__init__()
        self.task_head = task_head

    def forward(self, X: Tensor) -> EmbeddingModelOutput:
        return EmbeddingModelOutput(logits=self.task_head(X))


@dataclass
class MILModelOutput:
    """Structured output from a MIL model.

    Attributes:
        logits: Task predictions, shape (B, num_classes) or (B, 1).
        tile_attention: Per-tile attention weights from the aggregator,
            shape (B, N). None if the aggregator has no attention.
        auxiliary: Optional dict of auxiliary tensors from the aggregator.
    """

    logits: Tensor
    tile_attention: Tensor | None = None
    auxiliary: dict[str, Tensor] | None = None


class MILModel(nn.Module):
    """Composes an aggregator and a task head into a full MIL model.

    The aggregator maps (B, N, D) → AggregatorOutput, and the task head
    maps the bag representation to predictions. Tile attention (if available)
    is passed through for interpretability and heatmap generation.

    Args:
        aggregator: MIL aggregator (e.g. ABMIL, MeanPool).
        task_head: Task head (e.g. ClassificationHead).
    """

    def __init__(self, aggregator: Aggregator, task_head: TaskHead) -> None:
        super().__init__()
        aggregator.configure_for_task(task_head)
        self.aggregator = aggregator
        self.task_head = task_head

    def forward(self, X: Tensor, mask: Tensor | None = None) -> MILModelOutput:
        agg_out = self.aggregator(X, mask=mask)
        if (
            agg_out.bag_representation.ndim == 3
            and not getattr(self.task_head, "supports_branch_representation", False)
        ):
            msg = (
                f"{self.task_head.__class__.__name__} does not support branch-aware "
                "bag representations; use a compatible classification head."
            )
            raise ValueError(msg)
        logits = self.task_head(agg_out.bag_representation)
        return MILModelOutput(
            logits=logits,
            tile_attention=agg_out.tile_attention,
            auxiliary=agg_out.auxiliary,
        )


@dataclass
class SegmentationModelOutput:
    """Output of SegmentationModel.forward."""

    logits: Tensor  # (B, C, H, W) at the mask target resolution


class SegmentationModel(nn.Module):
    """Composes a decoder and a segmentation head into a dense prediction model.

    The decoder maps a dense feature grid ``(B, d, h, w)`` to logits at its own
    (upsampled) grid, and the head resizes+crops those to the mask's target size
    (``logits = task_head(decoder(X))`` — mirroring EmbeddingModel/MILModel, so
    ``out.logits`` is exactly the target-res tensor the trainer feeds to
    ``task_head.compute_loss``/``compute_metrics``). The encoder is frozen/absent at
    train time, like the aggregator+head path.

    Args:
        decoder: Decoder mapping ``(B, d, h, w) -> (B, C, h', w')``.
        task_head: SegmentationHead owning targets/loss/metric/postprocess and the
            resize-to-encoded + crop-to-target geometry.
    """

    def __init__(self, decoder: Decoder, task_head: TaskHead) -> None:
        super().__init__()
        self.decoder = decoder
        self.task_head = task_head

    def forward(self, X: Tensor) -> SegmentationModelOutput:
        return SegmentationModelOutput(logits=self.task_head(self.decoder(X)))


def _set_encoder_eval(encoder: object) -> None:
    """Put a slide2vec tile encoder (and its inner module) into eval mode.

    The encoder is slide2vec's ``TileEncoder`` wrapper — not an ``nn.Module`` — so it
    has no ``.eval()``; the trainable backbone lives on its inner ``_model``. Eval-lock
    that so frozen BN/dropout never drift during training.
    """
    if hasattr(encoder, "eval"):
        encoder.eval()  # future-proof: a Module-style encoder
    inner = getattr(encoder, "_model", None)
    if isinstance(inner, nn.Module):
        inner.eval()


class LiveSegmentationModel(nn.Module):
    """Frozen encoder + decoder + segmentation head — re-encodes tiles each step.

    The live counterpart of :class:`SegmentationModel`: instead of consuming cached
    dense grids, it runs the **frozen** foundation-model encoder on the (augmented)
    image batch every forward, so the trainer's ``out = model(features)`` →
    ``out.logits`` → ``task_head.compute_loss`` contract is unchanged (the streaming
    ``accumulate_dense_stats`` eval works as-is too).

    The encoder is slide2vec's ``TileEncoder`` wrapper, which is *not* an ``nn.Module``;
    assigning it as an attribute therefore does **not** register it as a submodule, so
    it is automatically excluded from ``state_dict()`` (checkpoints carry only
    decoder+head — the backbone is reconstructed by ``load_model``) and from
    ``parameters()`` (the optimizer only ever sees the trainable decoder+head). We
    assert that non-registration so a future ``nn.Module`` encoder fails loud here
    rather than silently bloating checkpoints / entering the optimizer.

    Gradient/precision (design §13.B-9): the encoder runs under ``torch.no_grad()``
    (NOT ``inference_mode``, which taints outputs for the downstream autograd graph)
    and the extraction autocast context, then the grid is cast to ``float()`` to mirror
    the cached extractor exactly — so a live-no-aug run is numerically identical to the
    cached path (a real regression anchor). The decoder/head run in fp32 outside
    ``no_grad`` so backprop flows to them and stops at the frozen grid.

    Args:
        encoder: A dense-capable slide2vec tile encoder (``encode_tiles_dense``).
        decoder: Decoder mapping ``(B, d, h, w) -> (B, C, h', w')``.
        task_head: SegmentationHead (geometry + targets/loss/metric/postprocess).
        device: Device the encoder runs on (the trainer moves inputs here too).
        precision: Encoder autocast precision (resolved exactly as extraction does).
    """

    def __init__(
        self,
        *,
        encoder: object,
        decoder: Decoder,
        task_head: TaskHead,
        device: torch.device | str,
        precision: str,
        geometry: "DenseGridGeometry",
        window_size: int | None = None,
        overlap: float = 0.0,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.task_head = task_head
        self._geometry = geometry
        self._window_size = None if window_size is None else int(window_size)
        self._overlap = float(overlap)
        # NOT a registered submodule (encoder is slide2vec's non-Module wrapper) —
        # see the class docstring. Assert the invariant the checkpoint/optimizer rely on.
        self.encoder = encoder
        if "encoder" in self._modules:
            raise TypeError(
                "LiveSegmentationModel.encoder was registered as a submodule (the encoder "
                "is an nn.Module). That would write the frozen backbone into every checkpoint "
                "and feed it to the optimizer — revisit the state_dict / freeze handling."
            )
        self._device = torch.device(device)
        self._precision = str(precision)
        _set_encoder_eval(self.encoder)

    def train(self, mode: bool = True) -> "LiveSegmentationModel":
        """Toggle decoder/head train mode but keep the frozen encoder in eval."""
        super().train(mode)
        _set_encoder_eval(self.encoder)
        return self

    @property
    def window_size(self) -> int | None:
        """Resolved dense encoder-window size (``None`` ⇒ whole). Read-only."""
        return self._window_size

    @property
    def overlap(self) -> float:
        """Resolved dense encoder-window overlap. Read-only."""
        return self._overlap

    def encode(self, X: Tensor) -> Tensor:
        """Frozen encoder forward ``(B, 3, Henc, Wenc) -> (B, d, grid_h, grid_w)``.

        The expensive half of :meth:`forward`, split out so an *ensemble of models that
        share this encoder* (the multi-fold inference case) can encode each tile **once**
        and run every fold's :meth:`forward_from_grid` on the shared grid — see
        :class:`soma.dense.predict.SlidingWindowSegmentationPredictor`. Runs under
        ``no_grad`` + the extraction autocast and casts to ``float()`` to mirror the cached
        extractor exactly (the cached-parity anchor), so splitting the forward changes no
        numerics: ``forward`` is still ``forward_from_grid(encode(X))``.
        """
        from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
        from slide2vec.runtime.dense_sliding import encode_dense_sliding

        with torch.no_grad(), slide_encode_autocast_ctx(self._device, self._precision):
            # window_size=None ⇒ the whole single forward (byte-identical to the cached
            # extractor — the parity anchor); a smaller window slides + blends identically.
            return encode_dense_sliding(
                self.encoder,
                X,
                geometry=self._geometry,
                window_size=self._window_size,
                overlap=self._overlap,
            ).float()  # mirror extraction

    def forward_from_grid(self, grid: Tensor) -> SegmentationModelOutput:
        """Decoder + head on a precomputed dense grid — the trainable half of the forward."""
        return SegmentationModelOutput(logits=self.task_head(self.decoder(grid)))

    def forward(self, X: Tensor) -> SegmentationModelOutput:
        return self.forward_from_grid(self.encode(X))
