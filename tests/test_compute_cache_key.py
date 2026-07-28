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
from dataclasses import replace

from soma.config import (
    AggregatorConfig,
    EncoderConfig,
    NormalizationConfig,
    PipelineConfig,
    PreprocessingConfig,
    ProjectionConfig,
    TaskConfig,
)


def test_compute_cache_key_tile_virchow():
    """tile + virchow @ 0.5 µm / 224 px → 2b0452e96803da2b."""
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
    assert key == "2b0452e96803da2b"


def test_compute_cache_key_hierarchical_conch():
    """hierarchical + conch @ 0.5 µm / 448 px / region 3584 / multiple 8 →
    b8c228c6666d9c68. Uses precomputed masks (the PANDA-debug dataset ships
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
    assert key == "b8c228c6666d9c68"


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
    dc3b9499693b0772. The operator typed tissue_method='hsv' but the dataset
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
    assert key == "dc3b9499693b0772"


def test_compute_cache_key_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported"):
        compute_cache_key(
            kind="bogus",
            encoder_name="virchow",
            preprocessing=PreprocessingConfig(tissue_method="hsv"),
            execution=EncoderConfig(name="virchow"),
        )


# --- Pooled storage-dtype cache identity (#164) -----------------------------------------
#
# cache.dtype folds into every pooled key, guarded so fp32 (the legacy default) keys stay
# byte-stable — i.e. the ground-truth hashes above never move — while an fp16 cache resolves
# to a distinct key so the two precisions can never be mixed. Mirrors the dense key's guard.


def test_pooled_cache_keys_fp32_is_byte_stable_and_fp16_distinct():
    from soma.cache.keys import (
        build_hierarchical_cache_key,
        build_patient_cache_key,
        build_slide_cache_key,
        build_tile_cache_key,
    )

    enc = EncoderConfig(name="virchow")
    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )
    tile_dep = {"tile_encoder_name": "virchow", "tile_execution": {"precision": "fp16"}}

    builders = {
        "tile": lambda **kw: build_tile_cache_key(
            tile_encoder_name="virchow", preprocessing=preprocessing, execution=enc, **kw
        ),
        "slide": lambda **kw: build_slide_cache_key(
            slide_encoder_name="prism", tile_dependency_signature=tile_dep, execution=enc, **kw
        ),
        "patient": lambda **kw: build_patient_cache_key(
            patient_encoder_name="prism", tile_dependency_signature=tile_dep, execution=enc, **kw
        ),
        "hierarchical": lambda **kw: build_hierarchical_cache_key(
            tile_encoder_name="virchow", preprocessing=preprocessing, execution=enc, **kw
        ),
    }
    for name, build in builders.items():
        # Default == explicit fp32 == the pre-#164 key (guard drops dtype from the payload).
        assert build() == build(dtype="fp32"), name
        # fp16 storage ⇒ a distinct key, so an fp16 cache never aliases the fp32 one.
        assert build(dtype="fp16") != build(dtype="fp32"), name


# --- Annotation-restricted bag cache identity (#110) ------------------------------------
#
# A tumor-restricted merged bag must never alias a full-tissue bag of the same
# slide/encoder/geometry: the selection-relevant projection of masks/sampling
# (active pixel_mapping entries, per-class min_coverage, strategy, output_mode) folds into
# the preprocessing signature that keys tiling/tile/slide caches. ``colors`` is cosmetic
# and is excluded. The no-masks case stays byte-stable (asserted by the ground-truth hashes
# above remaining unchanged).


def _masks_preprocessing(**masks_kwargs):
    from soma.config import MasksConfig, SamplingConfig

    sampling_kwargs = {
        key: masks_kwargs.pop(key)
        for key in ("strategy", "output_mode")
        if key in masks_kwargs
    }
    return PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
        masks=MasksConfig(**masks_kwargs),
        sampling=SamplingConfig(**sampling_kwargs) if sampling_kwargs else None,
    )


def test_preprocessing_signature_omits_masks_when_absent():
    """No masks block ⇒ no ``masks``/``sampling`` keys in the signature, so legacy
    tissue-only cache keys are byte-stable."""
    from soma.cache.keys import preprocessing_signature

    sig = preprocessing_signature(
        PreprocessingConfig(
            backend="asap",
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tissue_method="hsv",
        )
    )
    assert "masks" not in sig
    assert "sampling" not in sig


def test_tumor_restricted_bag_differs_from_tissue_bag():
    """AC7: a tumor-restricted bag and a full-tissue bag of the same slide/encoder produce
    different cache keys across tiling, tile, and slide kinds."""
    from soma.cache.keys import (
        build_slide_cache_key,
        build_tile_cache_key,
        build_tiling_cache_key,
        execution_signature,
        preprocessing_signature,
    )

    tissue = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )
    tumor = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
    )
    execution = EncoderConfig(name="virchow")

    assert build_tiling_cache_key(preprocessing=tissue) != build_tiling_cache_key(
        preprocessing=tumor
    )
    assert build_tile_cache_key(
        tile_encoder_name="virchow", preprocessing=tissue, execution=execution
    ) != build_tile_cache_key(
        tile_encoder_name="virchow", preprocessing=tumor, execution=execution
    )

    def _slide_key(prep):
        tile_dep = {
            "tile_encoder_name": "virchow",
            "tile_preprocessing": preprocessing_signature(prep),
            "tile_execution": execution_signature(execution, encoder_name="virchow"),
        }
        return build_slide_cache_key(
            slide_encoder_name="prism",
            tile_dependency_signature=tile_dep,
            execution=EncoderConfig(name="prism"),
        )

    assert _slide_key(tissue) != _slide_key(tumor)


def test_annotation_bag_identical_specs_same_key():
    """AC7: identical masks/sampling specs hash to the same key."""
    from soma.cache.keys import build_tiling_cache_key

    a = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}
    )
    b = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}
    )
    assert build_tiling_cache_key(preprocessing=a) == build_tiling_cache_key(preprocessing=b)


def test_annotation_bag_min_coverage_changes_key():
    """AC7: changing ``min_coverage`` changes the key."""
    from soma.cache.keys import build_tiling_cache_key

    low = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.25}
    )
    high = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.75}
    )
    assert build_tiling_cache_key(preprocessing=low) != build_tiling_cache_key(preprocessing=high)


def test_annotation_bag_strategy_changes_key():
    """AC7: changing ``sampling.strategy`` changes the key."""
    from soma.cache.keys import build_tiling_cache_key

    joint = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
        strategy="joint",
    )
    independent = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
        strategy="independent",
    )
    assert build_tiling_cache_key(preprocessing=joint) != build_tiling_cache_key(
        preprocessing=independent
    )


def test_annotation_bag_active_class_set_changes_key():
    """AC7: changing the active class set (the ``pixel_mapping``/``min_coverage`` vocabulary)
    changes the key."""
    from soma.cache.keys import build_tiling_cache_key

    tumor_only = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
    )
    tumor_and_stroma = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1, "stroma": 2},
        min_coverage={"tumor": 0.5, "stroma": 0.5},
    )
    assert build_tiling_cache_key(preprocessing=tumor_only) != build_tiling_cache_key(
        preprocessing=tumor_and_stroma
    )


def test_annotation_bag_colors_do_not_change_key():
    """AC7: ``colors`` is cosmetic and excluded from cache identity."""
    from soma.cache.keys import build_tiling_cache_key

    no_colors = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
    )
    with_colors = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
        colors={"background": None, "tumor": [255, 0, 0]},
    )
    assert build_tiling_cache_key(preprocessing=no_colors) == build_tiling_cache_key(
        preprocessing=with_colors
    )


# --- Annotation-restricted bag cache identity, patient path (#111) ----------------------
#
# A patient cache key wraps the upstream tile recipe via its ``tile_dependency_signature``
# (the tile preprocessing_signature, which folds in the annotation-sampling projection when a
# masks block is active). So a compartment-restricted patient run must never alias a
# full-tissue patient run of the same encoder/geometry — exactly as on the slide path (#110).


def test_patient_annotation_bag_differs_from_tissue_bag():
    """AC3 (#111): a tumor-restricted patient run and a full-tissue patient run of the same
    patient encoder produce different cache keys (the selection spec rides in the patient
    key's tile_dependency_signature, just like slide)."""
    tissue = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tissue_method="hsv",
    )
    tumor = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
    )
    tissue_key = compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=tissue)
    tumor_key = compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=tumor)
    assert tissue_key != tumor_key


def test_patient_annotation_bag_identical_specs_same_key():
    """AC3 (#111): identical masks/sampling specs hash to the same patient key (determinism)."""
    a = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}
    )
    b = _masks_preprocessing(
        pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5}
    )
    assert compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=a) == (
        compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=b)
    )


@pytest.mark.parametrize("method", ["zscore", "l2", "layernorm"])
def test_normalization_leaves_feature_cache_key_untouched(method):
    """Issue #283: the feature adaptor consumes the cache, it does not define it.

    Turning normalization on must never orphan an extracted feature cache, so the
    key computed from a config's extraction inputs is identical either way.
    """
    off = PipelineConfig(
        dataset_csv="d.csv",
        splits_csv="s.csv",
        output_root="runs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(
            backend="asap",
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tissue_method="hsv",
        ),
        encoder=EncoderConfig(name="virchow"),
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
    )
    on = replace(off, normalization=NormalizationConfig(method=method))

    def _key(config: PipelineConfig) -> str:
        return compute_cache_key(
            kind="tile",
            encoder_name=config.encoder.name,
            preprocessing=config.preprocessing,
            execution=config.encoder,
        )

    assert _key(on) == _key(off) == "2b0452e96803da2b"


@pytest.mark.parametrize(
    "projection",
    [
        ProjectionConfig(method="pca", target_dim=256),
        ProjectionConfig(method="random", target_dim=256, seed=3),
    ],
)
def test_projection_leaves_feature_cache_key_untouched(projection):
    """Issue #284: the projection consumes the cache, it does not define it.

    The dim-matched ablation must never orphan an extracted feature cache — the whole
    point is to re-use one shared extraction across every projection width.
    """
    off = PipelineConfig(
        dataset_csv="d.csv",
        splits_csv="s.csv",
        output_root="runs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(
            backend="asap",
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            tissue_method="hsv",
        ),
        encoder=EncoderConfig(name="virchow"),
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
    )
    on = replace(off, projection=projection)

    def _key(config: PipelineConfig) -> str:
        return compute_cache_key(
            kind="tile",
            encoder_name=config.encoder.name,
            preprocessing=config.preprocessing,
            execution=config.encoder,
        )

    assert _key(on) == _key(off) == "2b0452e96803da2b"
