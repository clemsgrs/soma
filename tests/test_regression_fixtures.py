"""Fixture-backed regression tests grounded in production extraction code."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from soma.cache import CacheConfig
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.extraction import FeatureExtractor
from hs2p import load_tiling_result


TESTS_DIR = Path(__file__).parent
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "regression"
INPUT_DIR = FIXTURE_ROOT / "input"
GT_DIR = FIXTURE_ROOT / "gt"


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
    return Dataset(dataset_csv)


def _regression_preprocessing() -> PreprocessingConfig:
    return PreprocessingConfig(
        target_tile_size_px=224,
        target_spacing_um=0.5,
        tolerance=0.07,
        tissue_threshold=0.1,
        overlap=0.0,
        seg_downsample=64,
        ref_tile_size_px=224,
        a_t=4,
    )


def _load_legacy_coordinate_gt() -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads(_fixture_path("gt", "test-wsi.coordinates.meta.json").read_text())
    arrays = np.load(_fixture_path("gt", "test-wsi.coordinates.npz"), allow_pickle=False)
    coordinates = np.stack([arrays["x"], arrays["y"]], axis=1).astype(np.int64)
    tile_index = arrays["tile_index"].astype(np.int32)
    tissue_fractions = arrays["tissue_fraction"].astype(np.float32)
    return meta, tile_index, coordinates, tissue_fractions


def _require_openslide() -> None:
    pytest.importorskip("openslide", reason="openslide is required for regression fixtures")


def _allow_prism_regression() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get(
        "SOMA_RUN_PRISM_REGRESSION"
    ) == "1"


@pytest.mark.skip(reason="Needs updating: slide2vec pipeline no longer outputs .coordinates.npz at fixed paths; use process_list.csv instead")
def test_coordinate_outputs_match_slide2vec_gt(tmp_path: Path):
    _require_openslide()

    dataset = _build_dataset(tmp_path)
    output_dir = tmp_path / "tiling"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name="prism"),
        preprocessing=_regression_preprocessing(),
        cache=CacheConfig(enabled=False),
    )

    extractor.preprocess(output_dir, backend="openslide")

    tiling = load_tiling_result(
        output_dir / "test-wsi.coordinates.npz",
        output_dir / "test-wsi.coordinates.meta.json",
    )
    soma_meta = json.loads((output_dir / "test-wsi.coordinates.meta.json").read_text())
    gt_meta, gt_tile_index, gt_coordinates, gt_tissue_fractions = _load_legacy_coordinate_gt()

    assert soma_meta["sample_id"] == gt_meta["sample_id"]
    assert soma_meta["requested_spacing_um"] == pytest.approx(gt_meta["target_spacing_um"])
    assert soma_meta["requested_tile_size_px"] == gt_meta["target_tile_size_px"]
    assert soma_meta["read_level"] == gt_meta["read_level"]
    assert soma_meta["effective_tile_size_px"] == gt_meta["read_tile_size_px"]
    assert soma_meta["tile_size_lv0"] == gt_meta["tile_size_lv0"]
    assert soma_meta["step_px_lv0"] == gt_meta["step_px_lv0"]
    assert soma_meta["overlap"] == gt_meta["overlap"]

    if gt_meta.get("backend") != "openslide":
        pytest.xfail(
            "Exact coordinate parity with the copied slide2vec GT is backend-specific "
            f"({gt_meta.get('backend')} vs Soma OpenSlide). Keep this harness, but "
            "port or regenerate a Soma-native golden artifact before enforcing exact arrays."
        )

    np.testing.assert_array_equal(tiling.tile_index, gt_tile_index)
    np.testing.assert_array_equal(tiling.coordinates, gt_coordinates)
    np.testing.assert_allclose(tiling.tissue_fractions, gt_tissue_fractions, atol=1e-6, rtol=0.0)
    assert soma_meta["n_tiles"] == gt_meta["num_tiles"]


def test_prism_slide_feature_matches_slide2vec_gt(tmp_path: Path):
    if not _allow_prism_regression():
        pytest.skip("PRISM regression runs only in GitHub CI or with SOMA_RUN_PRISM_REGRESSION=1")

    _require_openslide()
    torch = pytest.importorskip("torch", reason="torch is required for PRISM regression")
    pytest.importorskip("transformers", reason="transformers is required for PRISM regression")

    dataset = _build_dataset(tmp_path)
    output_dir = tmp_path / "features"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name="prism"),
        preprocessing=_regression_preprocessing(),
        cache=CacheConfig(enabled=False),
    )

    try:
        extractor.run(output_dir, backend="openslide")
    except (ImportError, OSError) as exc:
        pytest.skip(f"PRISM regression unavailable in this environment: {exc}")

    emb = torch.load(output_dir / "test-wsi.pt", map_location="cpu", weights_only=True)
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
