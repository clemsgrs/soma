"""Tests for preview entrypoints re-exported by soma."""

from hs2p.wsi import (
    overlay_mask_on_slide as hs2p_overlay_mask_on_slide,
    save_overlay_preview as hs2p_save_overlay_preview,
    write_coordinate_preview as hs2p_write_coordinate_preview,
)

from soma.extract.preprocessing import (
    overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)


def test_preview_exports_delegate_to_hs2p():
    assert overlay_mask_on_slide is hs2p_overlay_mask_on_slide
    assert save_overlay_preview is hs2p_save_overlay_preview
    assert write_coordinate_preview is hs2p_write_coordinate_preview
