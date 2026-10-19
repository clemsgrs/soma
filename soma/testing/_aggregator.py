"""``check_aggregator``: the conformance suite for a :class:`soma.models.aggregators.base.Aggregator`."""

from __future__ import annotations

import torch

from soma.models.aggregators.base import AggregatorOutput
from soma.testing._errors import ConformanceError

__all__ = ["check_aggregator"]

#: Every forward runs under this seed, so an aggregator that draws random numbers in eval
#: mode gives the same output for the same bag.
_SEED = 0
#: Scale of the noise written into padded positions: large enough that a padded tile
#: reaching the output moves it far beyond any tolerance.
_PADDING_NOISE = 1e3


def _forward(aggregator, X: torch.Tensor, mask: torch.Tensor | None, output_dim: int, when: str) -> torch.Tensor:
    """``aggregator(X, mask)``'s bag representation, checked against ``output_dim``."""
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_SEED)
            output = aggregator(X, mask)
    except Exception as error:  # noqa: BLE001 - any failure is a conformance failure
        raise ConformanceError("forward", f"raised {error!r} {when}") from error
    if not isinstance(output, AggregatorOutput):
        raise ConformanceError("forward", f"must return an AggregatorOutput, got {type(output)!r}")
    z = output.bag_representation
    if not torch.is_tensor(z) or z.ndim < 2 or int(z.shape[0]) != int(X.shape[0]):
        raise ConformanceError(
            "forward",
            f"bag_representation must be a (B, output_dim) Tensor with B = {X.shape[0]}, got "
            f"{tuple(getattr(z, 'shape', ())) or z!r} {when}",
        )
    if int(z.shape[-1]) != output_dim:
        raise ConformanceError(
            "forward", f"bag_representation's last axis is {z.shape[-1]}, output_dim is {output_dim} {when}"
        )
    return z


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ConformanceError(
            "forward",
            f"bag_representation has shape {tuple(left.shape)} in one forward and {tuple(right.shape)} in "
            "another: its shape must not depend on the batch or the bag length",
        )
    return float((left - right).abs().max())


def check_aggregator(
    aggregator, *, input_dim: int, batch_size: int = 2, n_tiles: int = 16, tolerance: float = 1e-5
) -> None:
    """Check an aggregator on CPU, in eval mode, on random ``(batch_size, n_tiles, input_dim)`` bags.

    Asserts: ``output_dim`` is a positive int; ``forward(X, mask)`` returns an
    :class:`~soma.models.aggregators.base.AggregatorOutput` whose ``bag_representation``
    is ``(B, output_dim)`` (``(B, ..., output_dim)`` for a branch-aware aggregator);
    ``mask=None`` gives the same output as a mask marking every tile valid; and padded
    tiles do not change the output. For the last check, every bag but the first keeps
    its first ``n_tiles // 2`` tiles, the rest is filled with large noise and masked out,
    and each bag's representation in that padded batch must match its representation
    alone within ``tolerance``. Every forward runs under the same seed, which leaves the
    caller's random state untouched.
    """
    if batch_size < 2 or n_tiles < 2:
        raise ValueError(
            f"check_aggregator pads a bag, so it needs batch_size >= 2 and n_tiles >= 2, got "
            f"batch_size={batch_size}, n_tiles={n_tiles}"
        )
    dim = getattr(aggregator, "output_dim", None)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ConformanceError("output_dim", f"must be a positive int, got {dim!r}")
    aggregator = aggregator.to(torch.device("cpu")).eval()
    generator = torch.Generator().manual_seed(_SEED)
    X = torch.randn(batch_size, n_tiles, input_dim, generator=generator)
    every_tile = torch.ones(batch_size, n_tiles, dtype=torch.bool)
    lengths = [n_tiles] + [n_tiles // 2] * (batch_size - 1)
    mask = torch.arange(n_tiles)[None, :] < torch.tensor(lengths)[:, None]
    padded = X.clone()
    padded[~mask] = _PADDING_NOISE * torch.randn(int((~mask).sum()), input_dim, generator=generator)

    with torch.inference_mode():
        unmasked = _forward(aggregator, X, None, dim, "with mask=None")
        difference = _max_difference(unmasked, _forward(aggregator, X, every_tile, dim, "on full bags"))
        if not difference <= tolerance:
            raise ConformanceError(
                "forward",
                f"bag_representation differs by {difference:.3g} between mask=None and a mask marking "
                f"every tile valid (tolerance {tolerance}): mask=None must mean every tile is valid",
            )
        batched = _forward(aggregator, padded, mask, dim, "on a padded batch")
        for index, length in enumerate(lengths):
            bag, valid = X[index : index + 1, :length], every_tile[index : index + 1, :length]
            difference = _max_difference(batched[index], _forward(aggregator, bag, valid, dim, f"on bag {index} alone")[0])
            if not difference <= tolerance:
                raise ConformanceError(
                    "forward",
                    f"bag {index}'s bag_representation differs by {difference:.3g} between the bag alone "
                    f"and the bag in a padded batch (tolerance {tolerance}): padded tiles, which the "
                    "mask marks False, must not reach the output",
                )
