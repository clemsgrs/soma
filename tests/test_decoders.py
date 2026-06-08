"""Tests for the decoder component — (B, d, h, w) -> (B, C, h', w')."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from soma.decoders import (  # noqa: E402
    LightweightConvDecoder,
    LinearDecoder,
    decoder_registry,
    list_decoders,
)


def test_registry_lists_both_decoders():
    assert list_decoders() == ["lightweight_conv", "linear"]


def test_linear_decoder_preserves_grid_resolution():
    dec = LinearDecoder(input_dim=384, num_classes=3)
    out = dec(torch.randn(2, 384, 4, 6))  # non-square grid
    assert tuple(out.shape) == (2, 3, 4, 6)
    assert dec.num_classes == 3


@pytest.mark.parametrize("blocks,factor", [(0, 1), (1, 2), (2, 4)])
def test_lightweight_conv_upsamples_by_2_per_block(blocks: int, factor: int):
    dec = LightweightConvDecoder(
        input_dim=192, num_classes=5, hidden_dim=64, num_upsample_blocks=blocks
    )
    out = dec(torch.randn(2, 192, 4, 6))
    assert tuple(out.shape) == (2, 5, 4 * factor, 6 * factor)
    assert dec.num_classes == 5


def test_decoder_built_via_registry_like_pipeline():
    # Mirrors how the pipeline builds components: cls(input_dim=d, num_classes=C, **params).
    cls = decoder_registry.get("lightweight_conv")
    dec = cls(input_dim=128, num_classes=2, hidden_dim=32, num_upsample_blocks=1)
    assert tuple(dec(torch.randn(1, 128, 8, 8)).shape) == (1, 2, 16, 16)


def test_lightweight_conv_handles_hidden_dim_not_divisible_by_32():
    # _group_norm must pick a valid divisor (100 -> 25 groups), not crash.
    dec = LightweightConvDecoder(
        input_dim=64, num_classes=2, hidden_dim=100, num_upsample_blocks=1
    )
    assert tuple(dec(torch.randn(1, 64, 4, 4)).shape) == (1, 2, 8, 8)


def test_decoder_rejects_non_4d_input():
    dec = LinearDecoder(input_dim=8, num_classes=2)
    with pytest.raises(ValueError, match=r"\(B, d, h, w\)"):
        dec(torch.randn(8, 4, 4))


def test_num_classes_must_be_positive():
    with pytest.raises(ValueError, match="num_classes must be >= 1"):
        LinearDecoder(input_dim=8, num_classes=0)
    with pytest.raises(ValueError, match="num_classes must be >= 1"):
        LightweightConvDecoder(input_dim=8, num_classes=0)


def test_decoder_dim_params_must_be_positive():
    # Config-supplied params must fail loud, not ZeroDivisionError inside GroupNorm.
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        LinearDecoder(input_dim=0, num_classes=2)
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        LightweightConvDecoder(input_dim=0, num_classes=2)
    with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
        LightweightConvDecoder(input_dim=8, num_classes=2, hidden_dim=0)
    with pytest.raises(ValueError, match="num_groups must be >= 1"):
        LightweightConvDecoder(input_dim=8, num_classes=2, num_groups=0)


def test_lightweight_conv_gradients_flow():
    dec = LightweightConvDecoder(input_dim=16, num_classes=3, hidden_dim=32, num_upsample_blocks=1)
    out = dec(torch.randn(2, 16, 4, 4))
    out.sum().backward()
    assert dec.classifier.weight.grad is not None
    assert dec.proj[0].weight.grad is not None
