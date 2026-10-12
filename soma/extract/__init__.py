"""``soma.extract``: the slide2vec facade for features soma computes on the fly.

:func:`live_encoder` is the only place the live grid path speaks slide2vec: it prepares
slide2vec's ``DenseEncodeKit`` and wraps it in a :class:`FrozenEncoder`, the
parameter-free encoder stage of :class:`~soma.training.model.DenseModel`.
"""

from soma.extract.live import FrozenEncoder, live_encoder

__all__ = ["FrozenEncoder", "live_encoder"]
