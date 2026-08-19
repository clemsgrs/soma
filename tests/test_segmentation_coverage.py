"""Tests for the segmentation coverage driver (block A1) — verifies the wide-CSV assembly.

The numeric coverage math lives in hs2p (tested in hs2p/tests/test_annotation_coverage.py);
here we stub the two hs2p calls and check soma's DataFrame shaping, column ordering, and
None handling.
"""

import pandas as pd
import pytest

import soma.curation.segmentation_coverage as cov

PIXEL_MAPPING = {"background": 0, "tumor": 1, "stroma": 2}
_CANNED_SUMMARY = {
    "tumor": {"area_mm2": 0.01, "frac": 0.6, "est_tiles": 5},
    "stroma": {"area_mm2": 0.005, "frac": 0.4, "est_tiles": None},
}


@pytest.fixture
def stub_hs2p(monkeypatch):
    monkeypatch.setattr(cov, "open_slide", lambda path, backend="auto": object())
    monkeypatch.setattr(cov, "resolve_annotation_masks", lambda **kwargs: object())
    monkeypatch.setattr(
        cov,
        "summarize_annotation_coverage",
        lambda **kwargs: {k: dict(v) for k, v in _CANNED_SUMMARY.items()},
    )


def _manifest(n=2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"slide{i}" for i in range(n)],
            "image_path": [f"/fake/slide{i}.tif" for i in range(n)],
            "label_mask_path": [f"/fake/slide{i}_mask.tif" for i in range(n)],
        }
    )


def test_summarize_coverage_wide_columns_grouped_by_class(stub_hs2p):
    df = cov.summarize_coverage(
        _manifest(2),
        pixel_mapping=PIXEL_MAPPING,
        min_coverage={"tumor": 0.1, "stroma": 0.1},
        tile_size_px=256,
        spacing_um=0.5,
    )
    assert list(df.columns) == [
        "sample_id",
        "area_mm2_tumor",
        "frac_tumor",
        "est_tiles_tumor",
        "area_mm2_stroma",
        "frac_stroma",
        "est_tiles_stroma",
    ]
    assert len(df) == 2
    assert df.loc[0, "area_mm2_tumor"] == pytest.approx(0.01)
    assert df.loc[0, "frac_stroma"] == pytest.approx(0.4)
    assert df.loc[0, "est_tiles_tumor"] == 5


def test_summarize_coverage_preserves_none_est_tiles(stub_hs2p):
    df = cov.summarize_coverage(
        _manifest(1),
        pixel_mapping=PIXEL_MAPPING,
        min_coverage={"tumor": 0.1},
        tile_size_px=256,
        spacing_um=0.5,
    )
    assert pd.isna(df.loc[0, "est_tiles_stroma"])


def test_summarize_coverage_requires_manifest_columns(stub_hs2p):
    bad = pd.DataFrame({"sample_id": ["s0"], "image_path": ["/fake/s0.tif"]})
    with pytest.raises(ValueError, match="label_mask_path"):
        cov.summarize_coverage(
            bad,
            pixel_mapping=PIXEL_MAPPING,
            min_coverage=None,
            tile_size_px=256,
            spacing_um=0.5,
        )


def test_summarize_coverage_requires_background_label(stub_hs2p):
    with pytest.raises(ValueError, match="background"):
        cov.summarize_coverage(
            _manifest(1),
            pixel_mapping={"tumor": 1},
            min_coverage=None,
            tile_size_px=256,
            spacing_um=0.5,
        )


def test_write_coverage_csv_roundtrip(stub_hs2p, tmp_path):
    df = cov.summarize_coverage(
        _manifest(2),
        pixel_mapping=PIXEL_MAPPING,
        min_coverage={"tumor": 0.1, "stroma": 0.1},
        tile_size_px=256,
        spacing_um=0.5,
    )
    out = cov.write_coverage_csv(tmp_path / "coverage.csv", df)
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == list(df.columns)
    assert len(reloaded) == 2
