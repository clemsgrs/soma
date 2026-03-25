"""Tests for soma.wsi.reader — select_level() and spacing override."""

import pytest

from soma.wsi.reader import LevelSelection, select_level


# --- Typical pyramid: 0.25 µm/px base, 4 levels (1x, 2x, 4x, 16x) ---

BASE_SPACING = 0.25
DOWNSAMPLES = [1.0, 2.0, 4.0, 16.0]
# Effective spacings: 0.25, 0.50, 1.00, 4.00


def test_exact_match():
    sel = select_level(0.5, DOWNSAMPLES, BASE_SPACING)
    assert sel.level == 1
    assert sel.effective_spacing_um == pytest.approx(0.5)
    assert sel.is_within_tolerance is True


def test_exact_match_level0():
    sel = select_level(0.25, DOWNSAMPLES, BASE_SPACING)
    assert sel.level == 0
    assert sel.effective_spacing_um == pytest.approx(0.25)
    assert sel.is_within_tolerance is True


def test_no_exact_match_picks_closest_below():
    # Request 0.75 µm/px: level 1 (0.5) is closest ≤ target
    sel = select_level(0.75, DOWNSAMPLES, BASE_SPACING)
    assert sel.level == 1
    assert sel.effective_spacing_um == pytest.approx(0.5)
    assert sel.is_within_tolerance is False


def test_within_tolerance():
    # Request 0.51 µm/px with 5% tolerance: 0.50 is within 2% → within tolerance
    sel = select_level(0.51, DOWNSAMPLES, BASE_SPACING, tolerance=0.05)
    assert sel.level == 1
    assert sel.is_within_tolerance is True


def test_outside_tolerance():
    # Request 0.60 µm/px with 5% tolerance: 0.50 is 16.7% off → not within tolerance
    sel = select_level(0.60, DOWNSAMPLES, BASE_SPACING, tolerance=0.05)
    assert sel.level == 1
    assert sel.is_within_tolerance is False


def test_never_upsamples():
    # Request 0.1 µm/px: nothing below, must pick level 0 (0.25)
    sel = select_level(0.1, DOWNSAMPLES, BASE_SPACING)
    assert sel.level == 0
    assert sel.effective_spacing_um == pytest.approx(0.25)
    assert sel.is_within_tolerance is False


def test_large_spacing_request():
    # Request 2.0 µm/px: level 2 (1.0) is closest ≤ 2.0
    sel = select_level(2.0, DOWNSAMPLES, BASE_SPACING)
    assert sel.level == 2
    assert sel.effective_spacing_um == pytest.approx(1.0)


def test_returns_level_selection_dataclass():
    sel = select_level(0.5, DOWNSAMPLES, BASE_SPACING)
    assert isinstance(sel, LevelSelection)


def test_single_level_pyramid():
    sel = select_level(0.5, [1.0], 0.25)
    assert sel.level == 0
    assert sel.effective_spacing_um == pytest.approx(0.25)


def test_non_standard_downsamples():
    # Real-world non-power-of-2 pyramid
    ds = [1.0, 2.003, 4.01, 15.98]
    # Request 0.51: level 1 effective = 0.25 * 2.003 = 0.50075 ≤ 0.51
    sel = select_level(0.51, ds, 0.25, tolerance=0.05)
    assert sel.level == 1
    assert sel.effective_spacing_um == pytest.approx(0.50075)
    assert sel.is_within_tolerance is True
