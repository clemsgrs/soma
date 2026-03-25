"""Tests for soma.preprocessing.preview — combined visualization."""

import numpy as np
import pytest

from soma.preprocessing.preview import render_preview, save_preview
from soma.preprocessing.tiling import TilingResult


def _make_thumbnail(h: int = 100, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _make_mask(h: int = 100, w: int = 200) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 40:160] = 255
    return mask


def _make_tiling_result() -> TilingResult:
    return TilingResult(
        coordinates=np.array([[100, 100], [200, 100], [100, 200]], dtype=np.int64),
        tissue_fractions=np.array([0.8, 0.9, 0.7], dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
    )


# --- render_preview ---


def test_render_preview_returns_rgb():
    thumb = _make_thumbnail()
    mask = _make_mask()
    preview = render_preview(thumb, mask)
    assert preview.dtype == np.uint8
    assert preview.shape == thumb.shape


def test_render_preview_with_mask_only():
    """Mask overlay should modify the thumbnail."""
    thumb = _make_thumbnail()
    mask = _make_mask()
    preview = render_preview(thumb, mask)
    # Preview should differ from original where mask is non-zero
    assert not np.array_equal(preview, thumb)


def test_render_preview_with_tiling():
    thumb = _make_thumbnail()
    mask = _make_mask()
    result = _make_tiling_result()
    preview = render_preview(thumb, mask, tiling_result=result)
    assert preview.dtype == np.uint8
    assert preview.shape == thumb.shape


def test_render_preview_custom_colors():
    thumb = _make_thumbnail()
    mask = _make_mask()
    preview = render_preview(
        thumb, mask, mask_color=(255, 0, 0), mask_alpha=0.5
    )
    assert preview.dtype == np.uint8


# --- save_preview ---


def test_save_preview(tmp_path):
    preview = np.full((100, 200, 3), 128, dtype=np.uint8)
    path = tmp_path / "preview.png"
    save_preview(preview, path)
    assert path.exists()
    assert path.stat().st_size > 0
