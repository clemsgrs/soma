"""Training models: EmbeddingModel and MILModel (set shape), DenseModel (grid shape)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator
from soma.decoders.base import Decoder
from soma.tasks.base import TaskHead

@dataclass
class EmbeddingModelOutput:
    """Output of EmbeddingModel.forward."""

    logits: Tensor  # (B, num_classes)


class EmbeddingModel(nn.Module):
    """Task head applied directly to a pre-computed feature embedding (B, D).

    Used for slide-level, patient-level, and tile-level pipelines where each
    sample is already represented by a single feature vector — no aggregation.

    An optional :class:`~soma.training.feature_adaptor.FeatureAdaptor` sits in front of
    the task head and transforms the frozen embeddings before anything trainable sees
    them (issue #285 on the slide-encoder path). It is ``None`` unless the run asks for
    one, and ``None`` is not registered as a submodule, so a run without one has exactly
    the ``state_dict``/``parameters`` of a model built before the adaptor existed.

    When the adaptor projects it also *changes the width*: ``task_head`` must already
    have been constructed against the adaptor's ``output_dim`` rather than the encoder's
    native dim. The caller owns that wiring (see
    :func:`~soma.training.feature_adaptor.feature_adaptor_output_dim`).

    Args:
        task_head: Task head (e.g. ClassificationHead), built against the adaptor's
            output width.
        feature_adaptor: Optional front module applied to ``X`` before the head.
    """

    def __init__(
        self, task_head: TaskHead, feature_adaptor: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.task_head = task_head
        self.feature_adaptor = feature_adaptor

    def forward(self, X: Tensor) -> EmbeddingModelOutput:
        if self.feature_adaptor is not None:
            X = self.feature_adaptor(X)
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

    An optional :class:`~soma.training.feature_adaptor.FeatureAdaptor` sits in front of
    the aggregator and transforms the frozen features before anything trainable sees
    them. It is ``None`` unless the run asks for one (``normalization`` — issue #283 — or
    ``projection`` — issue #284), and ``None`` is not registered as a submodule, so a run
    without one has exactly the ``state_dict``/``parameters`` of a model built before the
    adaptor existed.

    When the adaptor projects, it also *changes the width*: ``aggregator`` must already
    have been constructed against the adaptor's ``output_dim`` rather than the encoder's
    native dim. The caller owns that wiring (see
    :func:`~soma.training.feature_adaptor.feature_adaptor_output_dim`).

    Args:
        aggregator: MIL aggregator (e.g. ABMIL, MeanPool), built against the adaptor's
            output width.
        task_head: Task head (e.g. ClassificationHead).
        feature_adaptor: Optional front module applied to ``X`` before aggregation.
    """

    def __init__(
        self,
        aggregator: Aggregator,
        task_head: TaskHead,
        feature_adaptor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        aggregator.configure_for_task(task_head)
        self.aggregator = aggregator
        self.task_head = task_head
        self.feature_adaptor = feature_adaptor

    def forward(self, X: Tensor, mask: Tensor | None = None) -> MILModelOutput:
        if self.feature_adaptor is not None:
            X = self.feature_adaptor(X)
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
class DenseModelOutput:
    """Output of :class:`DenseModel`: logits at the supervision frame ``(B, C, H, W)``."""

    logits: Tensor


class DenseModel(nn.Module):
    """The grid-shape model: ``pixels -> encoder -> adaptor -> decoder -> head``.

    Every stage but the decoder and the head is optional. Without an ``encoder`` the
    model reads cached feature grids ``(B, d, h, w)``; with one (a parameter-free
    :class:`~soma.extract.FrozenEncoder`, or any stage with the same contract) it
    re-encodes ``pixels`` every forward, so augmentation reaches the features. The
    ``adaptor`` (a fitted :class:`~soma.training.feature_adaptor.FeatureAdaptor`:
    normalisation, projection) transforms the grid channel-axis before the decoder. The
    head resizes and crops the decoder's logits to the supervision frame.

    An image-prior decoder (``consumes_image``) also receives the batch's raw uint8
    ``pixels`` at the mask size, as ``[0, 1]`` floats. On the live path those pixels must
    reach the model unpreprocessed (the encoder stage then preprocesses them itself).

    Args:
        decoder: Maps ``(B, d, h, w)`` to ``(B, C, h', w')`` logits, built against the
            adaptor's output width.
        head: A grid head (segmentation or detection) bound to the run's geometry.
        encoder: Optional frozen encoder stage over pixels.
        adaptor: Optional channel-axis feature adaptor.
    """

    def __init__(
        self,
        decoder: Decoder,
        head: TaskHead,
        *,
        encoder: nn.Module | None = None,
        adaptor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if int(decoder.num_classes) != int(head.num_classes):
            raise ValueError(
                f"decoder num_classes ({decoder.num_classes}) != head num_classes "
                f"({head.num_classes}); the logits would be misregistered"
            )
        self.decoder = decoder
        self.head = head
        self.encoder = encoder
        self.adaptor = adaptor

    @property
    def consumes_image(self) -> bool:
        return bool(getattr(self.decoder, "consumes_image", False))

    def _device(self, fallback: Tensor) -> torch.device:
        parameter = next(self.decoder.parameters(), None)
        return fallback.device if parameter is None else parameter.device

    def encode(self, pixels: Tensor) -> Tensor:
        """The encoder stage alone: pixels to a ``(B, d, h, w)`` grid."""
        if self.encoder is None:
            raise ValueError("this DenseModel has no encoder stage; it reads cached features")
        return self.encoder(pixels)

    def forward_from_grid(self, grid: Tensor, pixels: Tensor | None = None) -> DenseModelOutput:
        """Adaptor, decoder and head over an already encoded grid (the trainable half)."""
        grid = grid.to(self._device(grid)).float()
        if self.adaptor is not None:
            grid = self.adaptor.forward_grid(grid)
        if self.consumes_image:
            if pixels is None:
                raise ValueError(
                    f"decoder {type(self.decoder).__name__} consumes the image but the batch "
                    "carries no pixels; pass images= (cached grids) or an ImageSource"
                )
            if pixels.dtype != torch.uint8:
                raise ValueError(
                    f"decoder {type(self.decoder).__name__} consumes the raw image, but the "
                    f"batch pixels are {pixels.dtype}: build the loaders without preprocess= "
                    "so the encoder stage preprocesses the pixels itself"
                )
            image = pixels.to(grid.device).float() / 255.0
            decoded = self.decoder(grid, image)
        else:
            decoded = self.decoder(grid)
        return DenseModelOutput(logits=self.head(decoded))

    def forward(self, features: Tensor | None = None, pixels: Tensor | None = None) -> DenseModelOutput:
        if self.encoder is not None:
            if pixels is None:
                raise ValueError("a DenseModel with an encoder stage reads pixels")
            grid = self.encoder(pixels)
        else:
            if features is None:
                raise ValueError("a DenseModel without an encoder stage reads cached features")
            grid = features
        return self.forward_from_grid(grid, pixels)
