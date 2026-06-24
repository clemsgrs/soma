"""Tests for soma.cache.compute_cache_key — deterministic cache-key recovery
from explicit operator inputs.

Ground-truth hashes were seeded from on-disk caches in
output/debug/feature_cache/{kind}/{cache_key}/cache_metadata.json — the values
soma itself produced during extraction. The helper exists so that downstream
tools (e.g. pneuma) can recompute a cache_key from the same operator inputs
without running extraction or loading model weights.

NOTE: these keys were rotated by the #98 migration (PreprocessingConfig's
tissue_threshold scalar → a min_coverage map), which deliberately changed the
preprocessing signature. Pre-#98 on-disk caches are orphaned and must be
re-extracted; the hashes below are anchored to the current signature.
"""

import pytest

from soma.cache import compute_cache_key
from soma.config import EncoderConfig, PreprocessingConfig


def test_compute_cache_key_tile_virchow():
    """tile + virchow @ 0.5 µm / 224 px → 4804c91c96ec19c1."""
    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )
    execution = EncoderConfig(name="virchow")
    key = compute_cache_key(
        kind="tile",
        encoder_name="virchow",
        preprocessing=preprocessing,
        execution=execution,
    )
    assert key == "4804c91c96ec19c1"


def test_compute_cache_key_hierarchical_conch():
    """hierarchical + conch @ 0.5 µm / 448 px / region 3584 / multiple 8 →
    fa22a8df1a8da4bf. Uses precomputed masks (the PANDA-debug dataset ships
    them for all samples, so soma promotes tissue_method automatically)."""
    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=448,
        requested_region_size_px=3584,
        region_tile_multiple=8,
        tissue_method="precomputed_mask",
    )
    execution = EncoderConfig(name="conch")
    key = compute_cache_key(
        kind="hierarchical",
        encoder_name="conch",
        preprocessing=preprocessing,
        execution=execution,
    )
    assert key == "fa22a8df1a8da4bf"


def test_compute_cache_key_slide_prism_matches_soma_resolver():
    """slide + prism (tile_encoder=virchow auto from registry): the helper
    must produce the same key that soma's own slide-cache resolver does for
    identical inputs. Self-consistency test rather than golden-hash because
    soma's slide keying changed in commit 64c4dfc (2026-04-26) and the
    on-disk legacy cache (d5a9609d2ce55dfd) predates that refactor."""
    from dataclasses import replace
    from soma.cache.keys import (
        build_slide_cache_key,
        preprocessing_signature,
        execution_signature,
    )
    from soma.encoders.validation import resolve_preprocessing_config
    from soma.encoders import encoder_registry
    from slide2vec.encoders.registry import resolve_encoder_output

    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )

    helper_key = compute_cache_key(
        kind="slide",
        encoder_name="prism",
        preprocessing=preprocessing,
    )

    tile_info = encoder_registry.info("virchow")
    tile_exec = EncoderConfig(name="virchow")
    tile_ov = resolve_encoder_output("virchow", metadata=tile_info)["output_variant"]
    resolved_tile_prep = resolve_preprocessing_config(tile_exec, preprocessing, model_metadata=tile_info)
    slide_info = encoder_registry.info("prism")
    slide_exec = EncoderConfig(name="prism")
    slide_ov = resolve_encoder_output("prism", metadata=slide_info)["output_variant"]
    tile_dep = {
        "tile_encoder_name": "virchow",
        "tile_preprocessing": preprocessing_signature(resolved_tile_prep),
        "tile_execution": execution_signature(
            replace(tile_exec, output_variant=tile_ov),
            encoder_name="virchow",
            preprocessing=resolved_tile_prep,
            output_variant=tile_ov,
        ),
    }
    reference_key = build_slide_cache_key(
        slide_encoder_name="prism",
        tile_dependency_signature=tile_dep,
        execution=replace(slide_exec, output_variant=slide_ov),
        output_variant=slide_ov,
    )
    assert helper_key == reference_key


def test_compute_cache_key_tile_virchow_has_precomputed_masks():
    """tile + virchow @ 1.0 µm / 224 px with has_precomputed_masks=True →
    31991065c7ca22fe. The operator typed tissue_method='hsv' but the dataset
    has masks for every sample, so soma promotes to 'precomputed_mask'
    during extraction; has_precomputed_masks=True mirrors that promotion."""
    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=1.0,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )
    execution = EncoderConfig(name="virchow")
    key = compute_cache_key(
        kind="tile",
        encoder_name="virchow",
        preprocessing=preprocessing,
        execution=execution,
        has_precomputed_masks=True,
    )
    assert key == "31991065c7ca22fe"


def test_compute_cache_key_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported"):
        compute_cache_key(
            kind="bogus",
            encoder_name="virchow",
            preprocessing=PreprocessingConfig(tissue_method="hsv"),
            execution=EncoderConfig(name="virchow"),
        )
