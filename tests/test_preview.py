"""Tests for soma.preprocessing.preview — combined visualization."""

import numpy as np
import pytest

from soma.preprocessing.preview import render_preview, save_preview
from soma.preprocessing.tiling import TilingResult
from soma.wsi.reader import SlideReader


class _PreviewReader:
    backend_name = "synthetic"
    dimensions = (1000, 800)
    spacing = 0.5
    level_count = 3
    level_dimensions = [(1000, 800), (500, 400), (250, 200)]
    level_downsamples = [1.0, 2.0, 4.0]

    def __init__(self):
        self.calls = []

    def read_region(self, location, level, size, *, pad_missing=False):
        self.calls.append(
            {
                "location": location,
                "level": level,
                "size": size,
                "pad_missing": pad_missing,
            }
        )
        width, height = size
        image = np.full((height, width, 3), 200, dtype=np.uint8)
        image[: min(20, height), : min(20, width)] = 20
        return image

    def get_thumbnail(self, size):
        raise AssertionError("render_preview should read a true level, not a thumbnail")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


assert isinstance(_PreviewReader(), SlideReader)


def _make_mask(h: int = 200, w: int = 250) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:160, 50:200] = 255
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
        slide_dimensions=[1000, 800],
        seg_level=2,
        seg_spacing_um=2.0,
    )


# --- render_preview ---


def test_render_preview_returns_rgb():
    reader = _PreviewReader()
    mask = _make_mask()
    preview = render_preview(reader, mask, _make_tiling_result())
    assert preview.dtype == np.uint8
    assert preview.shape == (200, 250, 3)
    assert reader.calls == [
        {"location": (0, 0), "level": 2, "size": (250, 200), "pad_missing": False}
    ]


def test_render_preview_with_mask_only():
    """Mask overlay should modify the thumbnail."""
    reader = _PreviewReader()
    mask = _make_mask()
    preview = render_preview(reader, mask, _make_tiling_result())
    assert not np.array_equal(preview, np.full((200, 250, 3), 200, dtype=np.uint8))


def test_render_preview_with_tiling():
    reader = _PreviewReader()
    mask = _make_mask()
    result = _make_tiling_result()
    preview = render_preview(reader, mask, tiling_result=result)
    assert preview.dtype == np.uint8
    assert preview.shape == (200, 250, 3)


def test_render_preview_custom_colors():
    reader = _PreviewReader()
    mask = _make_mask()
    preview = render_preview(
        reader, mask, _make_tiling_result(), mask_color=(255, 0, 0), mask_alpha=0.5
    )
    assert preview.dtype == np.uint8


def test_render_preview_respects_requested_preview_downsample():
    reader = _PreviewReader()
    mask = _make_mask()
    preview = render_preview(reader, mask, _make_tiling_result(), preview_downsample=2)
    assert preview.shape == (400, 500, 3)
    assert reader.calls[0]["level"] == 1


def test_render_preview_requires_seg_level():
    reader = _PreviewReader()
    mask = _make_mask()
    result = TilingResult(
        coordinates=np.empty((0, 2), dtype=np.int64),
        tissue_fractions=np.empty(0, dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=1,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=512,
        is_within_tolerance=True,
        slide_dimensions=[1000, 800],
    )

    with pytest.raises(ValueError, match="requires tiling_result.seg_level"):
        render_preview(reader, mask, result)


# --- save_preview ---


def test_save_preview(tmp_path):
    preview = np.full((100, 200, 3), 128, dtype=np.uint8)
    path = tmp_path / "preview.png"
    save_preview(preview, path)
    assert path.exists()
    assert path.stat().st_size > 0
