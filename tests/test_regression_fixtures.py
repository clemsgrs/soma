"""Fixture-backed regression tests grounded in production extraction code."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import slide2vec
from hs2p import FilterConfig

from soma.config import CacheOptions
from tests.extraction_configs import extraction
import soma
from soma.data import ImageManifest


TESTS_DIR = Path(__file__).parent
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "regression"


def _fixture_path(*parts: str) -> Path:
    path = FIXTURE_ROOT.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"Regression fixture missing: {path}")
    return path


def _build_dataset(tmp_path: Path) -> Dataset:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "\n".join(
            [
                "sample_id,image_path,label,mask_path",
                (
                    f"test-wsi,{_fixture_path('input', 'test-wsi.tif')},tumor,"
                    f"{_fixture_path('input', 'test-mask.tif')}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return ImageManifest.from_csv(dataset_csv)


def _regression_preprocessing() -> slide2vec.PreprocessingConfig:
    return extraction(
        0.5,
        224,
        tiling_fields={"backend": "openslide", "tolerance": 0.07},
        filter=FilterConfig(ref_tile_size=224, a_t=4),
    )


def _require_openslide() -> None:
    pytest.importorskip("openslide", reason="openslide is required for regression fixtures")


def _allow_prism_regression() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get(
        "SOMA_RUN_PRISM_REGRESSION"
    ) == "1"


def test_prism_slide_feature_matches_slide2vec_gt(tmp_path: Path):
    if not _allow_prism_regression():
        pytest.skip("PRISM regression runs only in GitHub CI or with SOMA_RUN_PRISM_REGRESSION=1")

    _require_openslide()
    torch = pytest.importorskip("torch", reason="torch is required for PRISM regression")
    pytest.importorskip("transformers", reason="transformers is required for PRISM regression")

    try:
        source = soma.extract(
            _build_dataset(tmp_path),
            encoder="prism",
            extraction=_regression_preprocessing(),
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(f"PRISM regression unavailable in this environment: {exc}")

    emb = source.load("test-wsi")
    gt_emb = torch.load(_fixture_path("gt", "test-wsi.pt"), map_location="cpu", weights_only=True)

    assert emb.shape == gt_emb.shape

    cos = torch.nn.functional.cosine_similarity(emb, gt_emb, dim=-1)
    mean_cos = float(cos.mean())
    atol = 1e-2
    rtol = 1e-3
    if not torch.allclose(emb, gt_emb, atol=atol, rtol=rtol):
        assert mean_cos >= 0.99, (
            f"Embedding mismatch: mean cosine similarity={mean_cos:.4f} "
            f"(atol={atol}, rtol={rtol})"
        )
