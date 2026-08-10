"""Training models: EmbeddingModel (direct task head) and MILModel (aggregator + task head)."""

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

    An optional :class:`~soma.training.feature_adaptor.FeatureAdaptor` sits in front of
    the decoder and transforms the frozen grid **channel-axis** before anything trainable
    sees it (issue #286 on the single-encoder dense path). It is ``None`` unless the run
    asks for one, and ``None`` is not registered as a submodule, so a run without one has
    exactly the ``state_dict``/``parameters`` of a model built before the adaptor existed.

    When the adaptor projects it also *changes the channel count*: the frozen ``d ->
    target_dim`` map composes **ahead of** the decoder's own learnable ``1x1`` projection
    conv, so ``decoder`` must already have been constructed against the adaptor's
    ``output_dim``. No decoder change is needed — and because the decoder's ``1x1``
    projection is its only ``d``-dependent module, the whole decoder becomes
    encoder-dim-independent under an active projection.

    Args:
        decoder: Decoder mapping ``(B, d, h, w) -> (B, C, h', w')``, built against the
            adaptor's output width.
        task_head: SegmentationHead owning targets/loss/metric/postprocess and the
            resize-to-encoded + crop-to-target geometry.
        feature_adaptor: Optional front module applied channel-axis to ``X`` before the
            decoder.
    """

    def __init__(
        self,
        decoder: Decoder,
        task_head: TaskHead,
        feature_adaptor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.task_head = task_head
        self.feature_adaptor = feature_adaptor

    def forward(self, X: Tensor) -> SegmentationModelOutput:
        if self.feature_adaptor is not None:
            X = self.feature_adaptor.forward_grid(X)
        return SegmentationModelOutput(logits=self.task_head(self.decoder(X)))


class LiveSegmentationModel(nn.Module):
    """Frozen encoder + decoder + segmentation head — re-encodes tiles each step.

    The live counterpart of :class:`SegmentationModel`: instead of consuming cached
    dense grids, it runs the **frozen** foundation-model encoder on the (augmented)
    image batch every forward, so the trainer's ``out = model(features)`` →
    ``out.logits`` → ``task_head.compute_loss`` contract is unchanged (the streaming
    ``accumulate_dense_stats`` eval works as-is too).

    The encoder is held privately by slide2vec's public ``DenseEncodeKit``. The kit is
    deliberately not an ``nn.Module``, so assigning it here cannot register the frozen
    foundation model in checkpoints or optimizers.

    Gradient, precision, eval-locking, and device transfer belong to the public kit. Its
    detached grid is cast to ``float()`` at the decoder boundary to mirror cached grids;
    the decoder/head remain ordinary trainable fp32 modules.

    A ``feature_adaptor`` may be present at **inference** time even though live *training*
    refuses one (issue #286): whole-slide sliding-window prediction rebuilds these models
    from checkpoints trained on the cached path, and such a checkpoint can carry a fitted
    adaptor whose buffers must be loaded and re-applied. What live training refuses is
    *fitting* one against an augmented stream — here the transform is already fit and
    frozen, so applying it to the re-encoded grid is exactly right.

    Args:
        kit: Public slide2vec ``DenseEncodeKit`` shared by all folds.
        decoder: Decoder mapping ``(B, d, h, w) -> (B, C, h', w')``, built against the
            adaptor's output width.
        task_head: SegmentationHead (geometry + targets/loss/metric/postprocess).
        feature_adaptor: Optional front module applied channel-axis to the encoded grid.
    """

    def __init__(
        self,
        *,
        kit: object,
        decoder: Decoder,
        task_head: TaskHead,
        feature_adaptor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.task_head = task_head
        self.feature_adaptor = feature_adaptor
        self.kit = kit
        if "kit" in self._modules:
            raise TypeError(
                "LiveSegmentationModel.kit was registered as a submodule. DenseEncodeKit "
                "must remain a plain object so the frozen backbone stays out of checkpoints "
                "and optimizers."
            )

    def train(self, mode: bool = True) -> "LiveSegmentationModel":
        """Toggle trainable modules; the kit itself enforces frozen eval encoding."""
        super().train(mode)
        return self

    @property
    def input_device(self) -> torch.device:
        """Live pixel batches stay on CPU; the public kit owns device transfer."""
        return torch.device("cpu")

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
        return self.kit.encode(X).float()  # mirror persisted dense feature dtype at the decoder

    def forward_from_grid(self, grid: Tensor) -> SegmentationModelOutput:
        """Decoder + head on a precomputed dense grid — the trainable half of the forward."""
        if self.feature_adaptor is not None:
            grid = self.feature_adaptor.forward_grid(grid)
        return SegmentationModelOutput(logits=self.task_head(self.decoder(grid)))

    def forward(self, X: Tensor) -> SegmentationModelOutput:
        return self.forward_from_grid(self.encode(X))
