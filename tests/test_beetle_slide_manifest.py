"""Config-validation tests for the BEETLE slide-manifest segmentation deliverable (#93).

These are config-only (no slide/mask I/O): they assert the tracked example config
``examples/segmentation_beetle.yaml`` loads through soma's loader and encodes the
BEETLE recipe — the masks ``pixel_mapping`` (BEETLE's raw vocabulary),
the 5%% min-coverage rule, 512 px @ 0.5 µm/px spacing-aware, phikon sliding-224 dense
window, lightweight_conv decoder, num_classes=4, and the three metrics — and that the
derived raw-pixel → class-index remap matches BEETLE's pixel→class contract exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from soma.config import load_config
from soma.dense.reader import build_label_remap

REPO_ROOT = Path(__file__).resolve().parents[1]
BEETLE_CONFIG = REPO_ROOT / "examples" / "segmentation_beetle.yaml"

# BEETLE's pixel -> soma class contract (255 = ignore).
EXPECTED_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 0: 255}


def test_beetle_config_loads_and_validates():
    cfg = load_config(BEETLE_CONFIG)
    assert cfg.dataset_type == "segmentation"
    assert cfg.task.name == "segmentation"
    assert cfg.task.params["num_classes"] == 4


def test_beetle_config_encodes_masks_contract():
    cfg = load_config(BEETLE_CONFIG)
    masks = cfg.preprocessing.masks
    assert masks is not None
    # masks.pixel_mapping is the BEETLE raw vocabulary; must include background (unannotated).
    assert masks.pixel_mapping["background"] == 0
    assert masks.pixel_mapping["other"] == 1
    assert masks.pixel_mapping["non_invasive_epithelium"] == 2
    assert masks.pixel_mapping["invasive_epithelium"] == 3
    assert masks.pixel_mapping["necrosis"] == 4
    # >=5%% min-coverage rule on the annotated (non-background) classes.
    assert all(v == 0.05 for v in masks.min_coverage.values())
    assert set(masks.min_coverage) == {"other", "non_invasive_epithelium", "invasive_epithelium", "necrosis"}


def test_beetle_config_encodes_recipe():
    cfg = load_config(BEETLE_CONFIG)
    pp = cfg.preprocessing
    assert pp.requested_tile_size_px == 512
    assert pp.requested_spacing_um == 0.5
    # phikon native-224 sliding window @ 0.5 overlap.
    assert pp.dense_window_size == 224
    assert pp.dense_window_overlap == 0.5
    assert cfg.encoder.name == "phikon"
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.evaluation.metrics == ["mean_dice", "mean_iou", "dice_per_class"]
    assert cfg.preprocessing.sampling.output_mode == "merged"


def test_beetle_remap_matches_curation_contract():
    cfg = load_config(BEETLE_CONFIG)
    lut, num_classes = build_label_remap(cfg.preprocessing.masks.pixel_mapping, ignore_index=255)
    assert num_classes == 4
    raw = np.array(sorted(EXPECTED_REMAP), dtype=np.int64)
    expected = np.array([EXPECTED_REMAP[int(v)] for v in raw], dtype=lut.dtype)
    np.testing.assert_array_equal(lut[raw], expected)
