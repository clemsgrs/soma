"""soma composes hs2p's ``TilingConfig`` rather than mirroring it (ADR 0009).

Composition happens at **resolve** time, not config-parse time: hs2p requires a spacing and
a tile size, and soma leaves both ``None`` until the encoder supplies them. What the ADR is
actually after is that the geometry vocabulary have one owner — so these tests pin that the
pooled adapter, the slide-manifest sampler and the cache key all read the same composition,
and that a field hs2p adds cannot go silently unreachable.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, replace

import pytest
from hs2p.configs import FilterConfig, SegmentationConfig, TilingConfig

from soma.cache.keys import preprocessing_signature
from soma.config import MasksConfig, PreprocessingConfig, SamplingConfig
from soma.dense_slide_extraction import _build_tiling_config
from soma.slide2vec_adapter import build_preprocessing_config


def _resolved(**overrides) -> PreprocessingConfig:
    return PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
        **overrides,
    )


def test_every_hs2p_tiling_field_is_mapped_or_named_provenance():
    """The drift guard: an hs2p knob soma cannot express is what ADR 0009 exists to prevent.

    ``tiling_values`` raises on an unmapped field, so this is really a check that the
    mapping is complete *today* — the failure mode it defends against is a future hs2p
    release, where the raise turns a silently unreachable setting into a loud one.
    """
    values = _resolved().tiling_values()
    provenance = {"requested_backend", "requested_mask_backend"}
    assert set(TilingConfig.__dataclass_fields__) == set(values) | provenance


def test_an_unmapped_hs2p_field_is_a_loud_error(monkeypatch):
    monkeypatch.setitem(
        TilingConfig.__dataclass_fields__, "some_future_knob", object()
    )
    with pytest.raises(RuntimeError, match="some_future_knob"):
        _resolved().tiling_values()


def test_mask_backend_reaches_hs2p_through_the_pooled_adapter():
    """The knob that was due (hs2p 4.3.0): mask decoding is authoritative, so a mask the
    slide's backend cannot decode is only fixable if soma can state a reader for it."""
    config = build_preprocessing_config(_resolved(mask_backend="asap"))
    assert config.mask_backend == "asap"
    assert config.backend == "auto"


def test_mask_backend_reaches_hs2p_through_the_slide_manifest_sampler():
    tiling = _build_tiling_config(
        _resolved(mask_backend="asap"), SamplingConfig(strategy="joint")
    )
    assert isinstance(tiling, TilingConfig)
    assert tiling.mask_backend == "asap"


def test_mask_backend_changes_the_cache_key():
    """Decoding a mask with a different reader can select different tiles, so it is
    identity, not a runtime detail."""
    auto = preprocessing_signature(_resolved())
    asap = preprocessing_signature(_resolved(mask_backend="asap"))
    assert auto != asap


def test_the_signature_keys_on_the_same_values_the_run_is_composed_from():
    """One mapping behind both, so the key cannot describe a geometry the run did not use."""
    config = _resolved(backend="asap", mask_backend="vips", overlap=0.25, tolerance=0.1)
    signature = preprocessing_signature(config)
    tiling = config.tiling_config()
    for name, value in config.tiling_values().items():
        assert signature[name] == value
        assert getattr(tiling, name) == value


def test_composition_is_refused_before_the_geometry_resolves():
    """hs2p requires both; soma fills them from the encoder, so an unresolved config has no
    ``TilingConfig`` to give — and says which knob is missing rather than raising a
    TypeError from hs2p's constructor."""
    with pytest.raises(ValueError, match="requested_tile_size_px"):
        PreprocessingConfig(requested_spacing_um=0.5).tiling_config()
    with pytest.raises(ValueError, match="requested_spacing_um"):
        PreprocessingConfig(requested_tile_size_px=224).tiling_config()


def test_an_unresolved_config_still_has_a_signature():
    """Cache keys are built for tile-dataset pipelines too, which never tile a slide."""
    signature = preprocessing_signature(PreprocessingConfig())
    assert signature["requested_spacing_um"] is None
    assert signature["requested_tile_size_px"] is None
    assert signature["mask_backend"] == "auto"


@pytest.mark.parametrize(
    ("section", "hs2p_config", "stated_by_soma"),
    [
        ("segmentation", SegmentationConfig, {"method", "downsample", "sam2_device"}),
        ("filtering", FilterConfig, {"ref_tile_size", "a_t"}),
    ],
)
def test_slide2vec_section_defaults_agree_with_hs2p_where_soma_says_nothing(
    section, hs2p_config, stated_by_soma
):
    """The tissue/filtering sections are hs2p's vocabulary reached through slide2vec.

    slide2vec 5.6.0 completes a partial ``segmentation``/``filtering`` override against its
    own shipped YAML before handing it to hs2p, where 5.5.0 passed the partial through and
    let hs2p's dataclass defaults fill the rest. soma states only a few keys in each section
    (:func:`build_preprocessing_config`), so the two routes agree only as long as slide2vec's
    defaults match hs2p's for every key soma leaves silent. They do today — and a drift there
    would change which tiles a run keeps *without* moving soma's cache key, since the key is
    ``tiling_values()`` and these sections are not in it. That is the failure this pins.
    """
    config = build_preprocessing_config(_resolved())
    merged = getattr(config, section)
    hs2p_defaults = {
        field.name: field.default
        for field in fields(hs2p_config)
        if field.default is not MISSING
    }
    silent = {
        key: value for key, value in merged.items() if key not in stated_by_soma
    }
    assert silent, "expected slide2vec to complete the section soma only partially states"
    for key, value in silent.items():
        assert key in hs2p_defaults, f"{section}.{key} has no hs2p default to agree with"
        assert value == hs2p_defaults[key], (
            f"slide2vec's default for {section}.{key} ({value!r}) has drifted from hs2p's "
            f"({hs2p_defaults[key]!r}); soma does not state this key, so the drift silently "
            "changes tile selection under an unchanged cache key."
        )


def test_independent_sampling_is_decided_once():
    """It reaches hs2p, slide2vec and the cache key by three routes; they agree."""
    tissue_only = _resolved()
    assert tissue_only.independent_sampling is True
    assert build_preprocessing_config(tissue_only).independent_sampling is True
    assert preprocessing_signature(tissue_only)["independent_sampling"] is True

    annotated = _resolved(
        masks=MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}),
        sampling=SamplingConfig(strategy="joint"),
    )
    assert annotated.independent_sampling is False
    assert build_preprocessing_config(annotated).independent_sampling is False
    assert preprocessing_signature(annotated)["independent_sampling"] is False

    assert replace(annotated, sampling=SamplingConfig(strategy="independent")).independent_sampling
