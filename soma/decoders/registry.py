"""Decoder registry, and the one way to construct a decoder for a dense grid."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any

from soma.registry import Registry

if TYPE_CHECKING:
    from soma.dense.geometry import DenseGridGeometry

decoder_registry = Registry("decoders")

__all__ = ["decoder_registry", "build_decoder_for_grid"]


def build_decoder_for_grid(
    decoder_name: str,
    decoder_params: dict[str, Any] | None,
    *,
    geometry: "DenseGridGeometry",
    input_dim: int,
    num_classes: int,
):
    """Construct a registered decoder for a dense grid of ``geometry``.

    Every place that builds a dense decoder — the segmentation and detection fold trainers,
    and each of the sites that *reconstruct* a trained model from a checkpoint (eval-only
    re-scoring, the OCELOT greedy re-scorer, live sliding-window prediction) — must agree
    on two things, or a checkpoint silently fails to load back into the model that wrote
    it:

    * ``num_upsample_blocks``, derived from the geometry (``encoded_size / grid_shape`` per
      axis) unless the decoder does not accept it (``LinearDecoder``) or the user pinned it;
    * ``input_dim``, which under an active feature-adaptor projection is the adaptor's
      ``output_dim``, **not** the encoder's native dim (issue #286).

    Those agreements used to live as copy-pasted blocks at each site, and drifted. This is
    the single construction path; callers supply ``input_dim`` via
    :func:`~soma.training.feature_adaptor.feature_adaptor_output_dim`.
    """
    decoder_cls = decoder_registry.get(decoder_name)
    params = dict(decoder_params or {})
    ctor_params = inspect.signature(decoder_cls.__init__).parameters
    if "num_upsample_blocks" in ctor_params and "num_upsample_blocks" not in params:
        ratio = max(
            geometry.encoded_size[0] / geometry.grid_shape[0],
            geometry.encoded_size[1] / geometry.grid_shape[1],
        )
        params["num_upsample_blocks"] = max(0, math.ceil(math.log2(ratio)))
    return decoder_cls(input_dim=input_dim, num_classes=num_classes, **params)
