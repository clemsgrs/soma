"""Tests for soma.cache.compute_cache_key and the pooled key derivation (ADR 0019).

Every pooled key derives from the slide2vec / hs2p dataclasses soma holds: the fields of
slide2vec's ``PreprocessingConfig`` and of hs2p's tiling, segmentation, filter and
mask-sampling configs it holds, plus the precision of slide2vec's ``ExecutionOptions``.
Only fields that cannot change a feature are left out, by name.

The golden hashes below are regression pins of that derivation (cache schema v2). The
1.x hashes of the same recipes are kept to show that every pooled key moved once.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
import slide2vec
from hs2p import FilterConfig, PreviewConfig, SegmentationConfig

from soma.cache import compute_cache_key
from soma.cache.keys import (
    build_hierarchical_cache_key,
    build_patient_cache_key,
    build_slide_cache_key,
    build_tile_cache_key,
    build_tiling_cache_key,
    execution_signature,
    preprocessing_signature,
)
from soma.config import (
    AggregatorConfig,
    FeaturesConfig,
    NormalizationConfig,
    PipelineConfig,
    ProjectionConfig,
    TaskConfig,
)
from tests.extraction_configs import annotation_masks, extraction, tiling

_ASAP = {"backend": "asap"}


def _tile_key(preprocessing, execution=None, **kwargs) -> str:
    return build_tile_cache_key(
        tile_encoder_name="virchow",
        preprocessing=preprocessing,
        execution=execution,
        output_variant="default",
        **kwargs,
    )


def _resolved(**fields) -> slide2vec.PreprocessingConfig:
    """A resolved extraction (tiling and filter set), as the extractor keys it."""
    return extraction(0.5, 224, filter=FilterConfig(ref_tile_size=224), **fields)


# --- Golden keys ------------------------------------------------------------------------


def test_compute_cache_key_tile_virchow():
    key = compute_cache_key(
        kind="tile",
        encoder_name="virchow",
        preprocessing=extraction(0.5, 224, tiling_fields=_ASAP),
    )
    assert key == "7c61518e6c647449"
    assert key != "2b0452e96803da2b"  # the 1.x key of the same recipe


def test_compute_cache_key_hierarchical_conch():
    key = compute_cache_key(
        kind="hierarchical",
        encoder_name="conch",
        preprocessing=extraction(
            0.5,
            448,
            tiling_fields=_ASAP,
            method="precomputed_mask",
            requested_region_size_px=3584,
            region_tile_multiple=8,
        ),
    )
    assert key == "1ae7b9247ff9139c"
    assert key != "b8c228c6666d9c68"  # the 1.x key of the same recipe


def test_compute_cache_key_tile_virchow_has_precomputed_masks():
    """The dataset has a mask for every sample, so soma promotes the segmentation method
    to ``precomputed_mask``; ``has_precomputed_masks=True`` mirrors that promotion."""
    preprocessing = extraction(1.0, 224, tiling_fields=_ASAP)
    key = compute_cache_key(
        kind="tile", encoder_name="virchow", preprocessing=preprocessing, has_precomputed_masks=True
    )
    assert key == "1c7b23341398c215"
    assert key != "dc3b9499693b0772"  # the 1.x key of the same recipe
    assert key == compute_cache_key(
        kind="tile",
        encoder_name="virchow",
        preprocessing=replace(preprocessing, segmentation=SegmentationConfig(method="precomputed_mask")),
    )


def test_compute_cache_key_slide_prism_matches_soma_resolver():
    """slide + prism (tile_encoder=virchow auto from registry): the helper produces the key
    soma's own slide-cache resolver builds for identical inputs."""
    from slide2vec.encoders.registry import encoder_registry, resolve_encoder_output

    from soma.encoders.validation import resolve_preprocessing_config

    preprocessing = extraction(0.5, 224, tiling_fields=_ASAP)
    helper_key = compute_cache_key(kind="slide", encoder_name="prism", preprocessing=preprocessing)

    tile_ov = resolve_encoder_output("virchow", metadata=encoder_registry.info("virchow"))["output_variant"]
    resolved_tile_prep = resolve_preprocessing_config(preprocessing, encoder_name="virchow")
    slide_ov = resolve_encoder_output("prism", metadata=encoder_registry.info("prism"))["output_variant"]
    tile_dep = {
        "tile_encoder_name": "virchow",
        "tile_preprocessing": preprocessing_signature(resolved_tile_prep),
        "tile_execution": execution_signature(
            None, encoder_name="virchow", preprocessing=resolved_tile_prep, output_variant=tile_ov
        ),
    }
    reference_key = build_slide_cache_key(
        slide_encoder_name="prism",
        tile_dependency_signature=tile_dep,
        execution=None,
        output_variant=slide_ov,
    )
    assert helper_key == reference_key


def test_compute_cache_key_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported"):
        compute_cache_key(kind="bogus", encoder_name="virchow", preprocessing=extraction())


def test_an_unresolved_tiling_has_no_signature():
    with pytest.raises(ValueError, match="tiling must be resolved"):
        preprocessing_signature(extraction(None, None))


# --- The key derives from the upstream dataclass fields ------------------------------


def _hs2p_variations():
    """One change per hs2p field soma keys, each a value its dataclass accepts."""
    filter_default = FilterConfig(ref_tile_size=224)
    for field in dataclasses.fields(FilterConfig):
        value = getattr(filter_default, field.name)
        changed = (not value) if isinstance(value, bool) else value * 2
        yield f"filter.{field.name}", {"filter": replace(filter_default, **{field.name: changed})}
    for name, value in (("tolerance", 0.1), ("overlap", 0.25), ("backend", "openslide"), ("mask_backend", "asap")):
        yield f"tiling.{name}", {"tiling_fields": {name: value}}
    yield "tiling.requested_spacing_um", {"spacing": 1.0}
    yield "tiling.requested_tile_size_px", {"tile_size": 256}
    yield "segmentation.method", {"method": "otsu"}
    yield "segmentation.downsample", {"seg_downsample": 32}
    yield "masks.min_coverage", {"masks": annotation_masks(min_coverage={"tumor": 0.25})}
    yield "masks.strategy", {"masks": annotation_masks(strategy="independent")}
    yield "masks.pixel_mapping", {"masks": annotation_masks(pixel_mapping={"background": 0, "tumor": 2})}
    yield "region_tile_multiple", {"region_tile_multiple": 4}
    yield "on_the_fly", {"on_the_fly": False}


@pytest.mark.parametrize(("field", "change"), list(_hs2p_variations()), ids=lambda value: value if isinstance(value, str) else "")
def test_every_feature_relevant_upstream_field_changes_the_key(field: str, change: dict):
    """Including the hs2p knobs soma's former mirror never keyed (filter.a_h, the pixel
    QC filters, ...): a field slide2vec or hs2p holds enters the key without a soma edit."""
    base = {"spacing": 0.5, "tile_size": 224, "filter": FilterConfig(ref_tile_size=224)}
    spacing = change.pop("spacing", base["spacing"])
    tile_size = change.pop("tile_size", base["tile_size"])
    changed = extraction(spacing, tile_size, **{"filter": base["filter"], **change})
    assert _tile_key(changed) != _tile_key(_resolved()), field


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("preview", {"preview": PreviewConfig(save_mask_preview=False, save_tiling_preview=False)}),
        ("adaptive_batching", {"adaptive_batching": True}),
        ("use_supertiles", {"use_supertiles": False}),
        ("num_cucim_workers", {"num_cucim_workers": 1}),
        ("save_tiles", {"save_tiles": True}),
        ("resume", {"resume": True}),
    ],
)
def test_runtime_extraction_fields_leave_the_key_alone(field: str, change: dict):
    assert _tile_key(_resolved(**change)) == _tile_key(_resolved()), field


def test_mask_colors_and_sam2_placement_leave_the_key_alone():
    plain = _resolved(masks=annotation_masks())
    colored = _resolved(masks=annotation_masks(colors={"background": None, "tumor": [255, 0, 0]}))
    sam2 = replace(plain, segmentation=replace(plain.segmentation, sam2_device="cuda", sam2_num_workers=2))
    assert _tile_key(colored) == _tile_key(plain) == _tile_key(sam2)


def test_the_order_of_merged_mask_values_leaves_the_key_alone():
    a = _resolved(masks=annotation_masks(pixel_mapping={"background": 0, "tumor": [2, 1]}))
    b = _resolved(masks=annotation_masks(pixel_mapping={"background": 0, "tumor": [1, 2]}))
    assert _tile_key(a) == _tile_key(b)


def test_only_the_execution_precision_enters_the_key():
    """Batch size, GPUs and workers are resources; the precision changes the features.
    ``precision=None`` keys on the encoder's recommendation."""
    from slide2vec.encoders.registry import encoder_registry

    recommended = encoder_registry.info("virchow").get("precision") or "fp32"
    resources = slide2vec.ExecutionOptions(batch_size=7, num_gpus=1, num_workers_per_gpu=2, prefetch_factor=9)
    assert _tile_key(_resolved(), resources) == _tile_key(_resolved(), None)
    assert _tile_key(_resolved(), slide2vec.ExecutionOptions(precision=recommended)) == _tile_key(_resolved(), None)
    other = "fp32" if recommended != "fp32" else "fp16"
    assert _tile_key(_resolved(), slide2vec.ExecutionOptions(precision=other)) != _tile_key(_resolved(), None)


def test_pooled_keys_separate_fp16_and_fp32_storage():
    preprocessing = _resolved()
    tile_dep = {"tile_encoder_name": "virchow", "tile_execution": {"precision": "fp16"}}
    builders = {
        "tile": lambda **kw: _tile_key(preprocessing, **kw),
        "slide": lambda **kw: build_slide_cache_key(
            slide_encoder_name="prism", tile_dependency_signature=tile_dep, execution=None, **kw
        ),
        "patient": lambda **kw: build_patient_cache_key(
            patient_encoder_name="prism", tile_dependency_signature=tile_dep, execution=None, **kw
        ),
        "hierarchical": lambda **kw: build_hierarchical_cache_key(
            tile_encoder_name="virchow", preprocessing=preprocessing, execution=None, **kw
        ),
    }
    for name, build in builders.items():
        assert build() == build(dtype="fp32"), name
        assert build(dtype="fp16") != build(dtype="fp32"), name


# --- Annotation-restricted bags (#110, #111) --------------------------------------------


def test_tumor_restricted_bag_differs_from_tissue_bag():
    """A tumor-restricted bag and a full-tissue bag of the same slide/encoder produce
    different tiling, tile and slide keys."""
    tissue = _resolved()
    tumor = _resolved(masks=annotation_masks())

    assert build_tiling_cache_key(preprocessing=tissue) != build_tiling_cache_key(preprocessing=tumor)
    assert _tile_key(tissue) != _tile_key(tumor)

    def _slide_key(prep):
        tile_dep = {
            "tile_encoder_name": "virchow",
            "tile_preprocessing": preprocessing_signature(prep),
            "tile_execution": execution_signature(None, encoder_name="virchow"),
        }
        return build_slide_cache_key(slide_encoder_name="prism", tile_dependency_signature=tile_dep, execution=None)

    assert _slide_key(tissue) != _slide_key(tumor)


def test_annotation_bag_identical_specs_same_key():
    a = _resolved(masks=annotation_masks())
    b = _resolved(masks=annotation_masks())
    assert build_tiling_cache_key(preprocessing=a) == build_tiling_cache_key(preprocessing=b)


def test_patient_annotation_bag_differs_from_tissue_bag():
    """A compartment-restricted patient run never aliases a full-tissue one: the selection
    rides in the patient key's tile dependency, as on the slide path."""
    tissue = extraction(0.5, 224, tiling_fields=_ASAP)
    tumor = extraction(0.5, 224, tiling_fields=_ASAP, masks=annotation_masks())
    tissue_key = compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=tissue)
    tumor_key = compute_cache_key(kind="patient", encoder_name="moozy", preprocessing=tumor)
    assert tissue_key != tumor_key
    assert tumor_key == compute_cache_key(
        kind="patient",
        encoder_name="moozy",
        preprocessing=extraction(0.5, 224, tiling_fields=_ASAP, masks=annotation_masks()),
    )


# --- The feature adaptor consumes the cache, it does not define it ---------------------


def _slide_config(**overrides) -> PipelineConfig:
    return PipelineConfig(
        dataset_csv="d.csv",
        splits_csv="s.csv",
        output_root="runs",
        dataset_type="slide",
        features=FeaturesConfig(encoder="virchow", extraction=extraction(0.5, 224, tiling_fields=_ASAP)),
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
        **overrides,
    )


def _config_key(config: PipelineConfig) -> str:
    return compute_cache_key(
        kind="tile",
        encoder_name=config.features.encoder.name,
        preprocessing=config.features.extraction,
        execution=config.features.execution,
    )


@pytest.mark.parametrize("method", ["zscore", "l2", "layernorm"])
def test_normalization_leaves_feature_cache_key_untouched(method):
    """Issue #283: turning normalization on must never orphan an extracted feature cache."""
    off = _slide_config()
    on = replace(off, normalization=NormalizationConfig(method=method))
    assert _config_key(on) == _config_key(off) == "7c61518e6c647449"


@pytest.mark.parametrize(
    "projection",
    [
        ProjectionConfig(method="pca", target_dim=256),
        ProjectionConfig(method="random", target_dim=256, seed=3),
    ],
)
def test_projection_leaves_feature_cache_key_untouched(projection):
    """Issue #284: the dim-matched ablation re-uses one shared extraction."""
    off = _slide_config()
    on = replace(off, projection=projection)
    assert _config_key(on) == _config_key(off) == "7c61518e6c647449"


def test_tiling_helper_spells_hs2p_defaults():
    assert tiling() == tiling(0.5, 224, tolerance=0.05, overlap=0.0, backend="auto", mask_backend="auto")
