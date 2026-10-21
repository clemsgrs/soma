"""A benchmark that scores a tile encoder refuses an encoder of another level in its own words.

Its protocol fixes what it measures, so the preset's advice (use a ``slide_*`` preset, or
no preset covers patient-level encoders) is not an option there: the benchmark names
itself, the encoder's level and where to find a tile encoder instead.
"""

from __future__ import annotations

import pytest

from soma.benchmark import get_benchmark


@pytest.mark.parametrize("benchmark", ["hest/IDC", "eva/bach", "eva/camelyon16_small", "croma/camelyon"])
@pytest.mark.parametrize(("encoder", "level"), [("titan", "slide"), ("moozy", "patient")])
def test_a_tile_encoder_benchmark_refuses_a_wrong_level_encoder_in_its_own_words(benchmark, encoder, level):
    with pytest.raises(ValueError) as refusal:
        get_benchmark(benchmark).build_config(encoder=encoder)

    message = str(refusal.value)
    assert message.startswith(f"{benchmark} ")
    assert f"encoder '{encoder}' is {level}-level" in message
    assert "soma list encoders --level tile" in message
    assert "slide_" not in message
    assert "preset" not in message


@pytest.mark.parametrize("benchmark", ["hest/IDC", "eva/bach", "eva/camelyon16_small", "croma/camelyon"])
def test_a_tile_encoder_benchmark_refuses_an_unknown_encoder_like_any_config(benchmark):
    with pytest.raises(ValueError, match="Unknown encoder name 'not-an-encoder'. Available encoders: "):
        get_benchmark(benchmark).build_config(encoder="not-an-encoder")
