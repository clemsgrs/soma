"""``soma.testing.check_aggregator`` on every registered aggregator, and its refusal of
broken ones."""

from __future__ import annotations

import pytest
import torch

import soma.benchmark  # noqa: F401  (registers the benchmark-private aggregators, checked too)
from soma import components
from soma.models.aggregators import AggregatorOutput, MeanPool, aggregator_registry
from soma.testing import ConformanceError, check_aggregator

D = 16

#: Small widths keep the transformers CPU-sized. HIPT reads 2x2-tile regions, and a flat
#: HIPT bag is whole regions: the check's bags (16 tiles, 8 for the padded ones) are.
AGGREGATOR_PARAMS = {
    "abmil": {"hidden_dim": 8},
    "clam_sb": {"hidden_dim": 8, "attn_dim": 4, "k_sample": 2},
    "clam_mb": {"hidden_dim": 8, "attn_dim": 4, "k_sample": 2},
    "dsmil": {"att_dim": 8},
    "dtfdmil": {"embed_dim": D, "hidden_dim": 8, "n_groups": 2},
    "transmil": {"att_dim": 16, "n_layers": 2, "n_heads": 2, "n_landmarks": 4},
    "hipt": {"region_size": 8, "patch_size": 4, "embed_dim_region": 12, "embed_dim_slide": 12},
}


@pytest.mark.parametrize("name", [entry["name"] for entry in components("aggregators")])
def test_every_registered_aggregator_conforms(name):
    torch.manual_seed(0)
    aggregator = aggregator_registry.get(name)(input_dim=D, **AGGREGATOR_PARAMS.get(name, {}))
    check_aggregator(aggregator, input_dim=D)


class _MaskBlindMean(MeanPool):
    """Averages every position, padding included: the mask is never read."""

    def forward(self, X, mask=None):
        return super().forward(X, None)


def test_check_aggregator_refuses_an_aggregator_that_pools_the_padding():
    with pytest.raises(ConformanceError, match=r"^forward: .*padd"):
        check_aggregator(_MaskBlindMean(input_dim=D), input_dim=D)


class _MaskRequiredMean(MeanPool):
    """Reads ``mask`` unconditionally, so a call without one fails."""

    def forward(self, X, mask=None):
        return super().forward(X, mask.bool())


def test_check_aggregator_refuses_an_aggregator_that_needs_a_mask():
    with pytest.raises(ConformanceError, match=r"^forward: raised .* with mask=None"):
        check_aggregator(_MaskRequiredMean(input_dim=D), input_dim=D)


class _SumWithoutMask(MeanPool):
    """Sums the bag when no mask is given and averages it otherwise."""

    def forward(self, X, mask=None):
        if mask is None:
            return AggregatorOutput(bag_representation=X.sum(dim=1))
        return super().forward(X, mask)


def test_check_aggregator_refuses_an_aggregator_whose_mask_none_is_not_every_tile_valid():
    with pytest.raises(ConformanceError, match=r"^forward: .*mask=None.*every tile"):
        check_aggregator(_SumWithoutMask(input_dim=D), input_dim=D)


class _WrongWidthMean(MeanPool):
    """Declares twice the width its mean pooling returns."""

    @property
    def output_dim(self):
        return 2 * super().output_dim


def test_check_aggregator_refuses_an_output_dim_that_forward_does_not_return():
    with pytest.raises(ConformanceError, match=r"^forward: bag_representation's last axis is 16, output_dim is 32"):
        check_aggregator(_WrongWidthMean(input_dim=D), input_dim=D)


class _Unpooled(MeanPool):
    """Forgets to pool: returns the ``(B, N, D)`` tiles as the bag representation."""

    def forward(self, X, mask=None):
        return AggregatorOutput(bag_representation=X)


def test_check_aggregator_refuses_a_bag_representation_whose_shape_depends_on_the_bag():
    with pytest.raises(ConformanceError, match=r"^forward: bag_representation has shape \(16, 16\) .* \(8, 16\)"):
        check_aggregator(_Unpooled(input_dim=D), input_dim=D)
