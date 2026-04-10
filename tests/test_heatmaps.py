"""Tests for soma.heatmaps — attention extraction and heatmap rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from soma.config import HeatmapConfig
from soma.heatmaps import (
    _build_coordinate_map,
    _normalize_attention,
    _read_sample_ids_from_predictions,
    render_attention_heatmap,
    render_heatmaps,
    save_attention,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeSlide:
    """Minimal SlideReader-compatible object for testing."""

    backend_name = "synthetic"
    dimensions = (1000, 800)
    spacing = 0.5
    spacings = [0.5, 1.0, 2.0]
    level_count = 3
    level_dimensions = [(1000, 800), (500, 400), (250, 200)]
    level_downsamples = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]

    def read_region(self, location, level, size):
        w, h = size
        return np.full((h, w, 3), 180, dtype=np.uint8)

    def read_level(self, level):
        w, h = self.level_dimensions[level]
        return np.full((h, w, 3), 180, dtype=np.uint8)

    def get_thumbnail(self, size):
        w, h = size
        return np.full((h, w, 3), 180, dtype=np.uint8)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _write_process_list(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, sample_ids: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "true_label", "predicted_label"])
        writer.writeheader()
        for sid in sample_ids:
            writer.writerow({"sample_id": sid, "true_label": 0, "predicted_label": 0})


def _write_coords(npz_path: Path, meta_path: Path, n: int = 4, tile_size: int = 200) -> None:
    x = np.arange(n, dtype=np.int64) * tile_size
    y = np.zeros(n, dtype=np.int64)
    np.savez(npz_path, x=x, y=y, tile_index=np.arange(n), tissue_fraction=np.ones(n, dtype=np.float32))
    meta = {"tile_size_lv0": tile_size, "step_px_lv0": tile_size}
    meta_path.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# _normalize_attention
# ---------------------------------------------------------------------------


def test_normalize_attention_applies_softmax_for_abmil():
    attn = torch.tensor([[1.0, 2.0, 3.0]])  # (1, N)
    result = _normalize_attention(attn, "abmil")
    assert result.shape == (3,)
    assert abs(result.sum() - 1.0) < 1e-5


def test_normalize_attention_applies_softmax_for_clam_sb():
    attn = torch.tensor([[0.5, -0.5, 1.0]])  # (1, N)
    result = _normalize_attention(attn, "clam_sb")
    assert result.shape == (3,)
    assert abs(result.sum() - 1.0) < 1e-5


def test_normalize_attention_no_softmax_for_dsmil():
    raw = torch.tensor([[0.2, 0.3, 0.5]])  # already sums to 1
    result = _normalize_attention(raw, "dsmil")
    assert result.shape == (3,)
    np.testing.assert_allclose(result, np.array([0.2, 0.3, 0.5]), atol=1e-6)


def test_normalize_attention_clam_mb_shape():
    # (1, n_classes, N) input
    attn = torch.randn(1, 3, 10)
    result = _normalize_attention(attn, "clam_mb")
    assert result.shape == (3, 10)
    # Each branch should sum to ~1 after softmax
    for k in range(3):
        assert abs(result[k].sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# _read_sample_ids_from_predictions
# ---------------------------------------------------------------------------


def test_read_sample_ids_from_predictions(tmp_path):
    path = tmp_path / "predictions.csv"
    _write_predictions(path, ["s1", "s2", "s3"])
    ids = _read_sample_ids_from_predictions(path)
    assert ids == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# _build_coordinate_map
# ---------------------------------------------------------------------------


def test_build_coordinate_map_returns_valid_paths(tmp_path):
    npz = tmp_path / "s1.npz"
    meta = tmp_path / "s1.meta.json"
    _write_coords(npz, meta)

    manifest = tmp_path / "process_list.csv"
    _write_process_list(manifest, [
        {
            "sample_id": "s1",
            "tiling_status": "success",
            "feature_status": "success",
            "coordinates_npz_path": str(npz),
            "coordinates_meta_path": str(meta),
        }
    ])

    store = MagicMock()
    store.feature_manifest_path = manifest

    coord_map = _build_coordinate_map(store)
    assert "s1" in coord_map
    assert coord_map["s1"] == (npz, meta)


def test_build_coordinate_map_no_manifest():
    store = MagicMock()
    store.feature_manifest_path = None
    assert _build_coordinate_map(store) == {}


def test_build_coordinate_map_skips_missing_files(tmp_path):
    manifest = tmp_path / "process_list.csv"
    _write_process_list(manifest, [
        {
            "sample_id": "s1",
            "coordinates_npz_path": str(tmp_path / "nonexistent.npz"),
            "coordinates_meta_path": str(tmp_path / "nonexistent.meta.json"),
        }
    ])
    store = MagicMock()
    store.feature_manifest_path = manifest
    assert _build_coordinate_map(store) == {}


# ---------------------------------------------------------------------------
# render_attention_heatmap
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_slide_patch():
    """Patch open_slide to return _FakeSlide instead of opening a real file."""
    with patch("soma.heatmaps.open_slide", return_value=_FakeSlide()) as p:
        yield p


def test_render_attention_heatmap_returns_rgb(fake_slide_patch, tmp_path):
    x = np.array([0, 200, 400, 600], dtype=np.int64)
    y = np.array([0, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)

    img = render_attention_heatmap(
        tmp_path / "fake.svs", x, y, scores,
        tile_size_lv0=200, seg_downsample=4,
    )

    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_render_attention_heatmap_size_matches_vis_level(fake_slide_patch, tmp_path):
    # seg_downsample=4 → level 2 → (250, 200) in _FakeSlide
    img = render_attention_heatmap(
        tmp_path / "fake.svs",
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
        scores=np.array([1.0], dtype=np.float32),
        tile_size_lv0=200,
        seg_downsample=4,
    )
    assert img.shape == (200, 250, 3)


def test_render_attention_heatmap_no_tiles_returns_canvas(fake_slide_patch, tmp_path):
    img = render_attention_heatmap(
        tmp_path / "fake.svs",
        x=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.int64),
        scores=np.array([], dtype=np.float32),
        tile_size_lv0=200,
        seg_downsample=4,
    )
    assert img.dtype == np.uint8


def test_render_attention_heatmap_constant_scores(fake_slide_patch, tmp_path):
    """All-equal scores should not crash (mx == mn edge case)."""
    x = np.array([0, 200], dtype=np.int64)
    y = np.array([0, 0], dtype=np.int64)
    scores = np.array([0.5, 0.5], dtype=np.float32)
    img = render_attention_heatmap(
        tmp_path / "fake.svs", x, y, scores,
        tile_size_lv0=200, seg_downsample=4,
    )
    assert img.dtype == np.uint8


def test_render_attention_heatmap_blur(fake_slide_patch, tmp_path):
    x = np.array([0, 200], dtype=np.int64)
    y = np.array([0, 0], dtype=np.int64)
    scores = np.array([0.2, 0.8], dtype=np.float32)
    img = render_attention_heatmap(
        tmp_path / "fake.svs", x, y, scores,
        tile_size_lv0=200, seg_downsample=4,
        blur_sigma=2.0,
    )
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# render_heatmaps — output file naming
# ---------------------------------------------------------------------------


def test_render_heatmaps_single_branch(tmp_path):
    """Single-branch attention → one PNG per sample."""
    # Write fold structure
    fold_dir = tmp_path / "fold_0"
    attn_dir = fold_dir / "attention"
    attn_dir.mkdir(parents=True)

    # Write attention npz (1-D = single branch)
    np.savez_compressed(attn_dir / "s1.npz", attention=np.array([0.3, 0.5, 0.2]))

    # Write coordinate files
    npz = tmp_path / "s1.npz"
    meta = tmp_path / "s1.meta.json"
    _write_coords(npz, meta, n=3, tile_size=200)

    # Write process_list.csv
    manifest = tmp_path / "process_list.csv"
    _write_process_list(manifest, [
        {
            "sample_id": "s1",
            "feature_status": "success",
            "coordinates_npz_path": str(npz),
            "coordinates_meta_path": str(meta),
        }
    ])

    dataset = MagicMock()
    dataset.samples = {"s1": SimpleNamespace(image_path=tmp_path / "fake.svs")}

    feature_store = MagicMock()
    feature_store.feature_manifest_path = manifest

    with patch("soma.heatmaps.open_slide", return_value=_FakeSlide()):
        render_heatmaps(
            tmp_path, dataset, feature_store,
            HeatmapConfig(enabled=True), seg_downsample=4,
        )

    assert (fold_dir / "heatmaps" / "s1.png").is_file()


def test_render_heatmaps_clam_mb_multi_branch(tmp_path):
    """Multi-branch attention (n_classes, N) → one PNG per class."""
    fold_dir = tmp_path / "fold_0"
    attn_dir = fold_dir / "attention"
    attn_dir.mkdir(parents=True)

    n_classes = 3
    n_tiles = 4
    attention = np.random.rand(n_classes, n_tiles).astype(np.float32)
    np.savez_compressed(attn_dir / "s2.npz", attention=attention)

    npz = tmp_path / "s2.npz"
    meta = tmp_path / "s2.meta.json"
    _write_coords(npz, meta, n=n_tiles, tile_size=200)

    manifest = tmp_path / "process_list.csv"
    _write_process_list(manifest, [
        {
            "sample_id": "s2",
            "feature_status": "success",
            "coordinates_npz_path": str(npz),
            "coordinates_meta_path": str(meta),
        }
    ])

    dataset = MagicMock()
    dataset.samples = {"s2": SimpleNamespace(image_path=tmp_path / "fake.svs")}

    feature_store = MagicMock()
    feature_store.feature_manifest_path = manifest

    with patch("soma.heatmaps.open_slide", return_value=_FakeSlide()):
        render_heatmaps(
            tmp_path, dataset, feature_store,
            HeatmapConfig(enabled=True), seg_downsample=4,
        )

    heatmap_dir = fold_dir / "heatmaps"
    for k in range(n_classes):
        assert (heatmap_dir / f"s2_class_{k}.png").is_file(), \
            f"Expected s2_class_{k}.png to exist"


def test_render_heatmaps_skips_unknown_sample(tmp_path):
    """Samples not in the dataset are skipped without error."""
    fold_dir = tmp_path / "fold_0"
    attn_dir = fold_dir / "attention"
    attn_dir.mkdir(parents=True)
    np.savez_compressed(attn_dir / "unknown.npz", attention=np.array([0.5, 0.5]))

    manifest = tmp_path / "process_list.csv"
    manifest.write_text("sample_id,feature_status,coordinates_npz_path,coordinates_meta_path\n")

    dataset = MagicMock()
    dataset.samples = {}  # "unknown" not in dataset

    feature_store = MagicMock()
    feature_store.feature_manifest_path = manifest

    render_heatmaps(
        tmp_path, dataset, feature_store,
        HeatmapConfig(enabled=True), seg_downsample=4,
    )
    # No error, no output files
    assert not list((fold_dir / "heatmaps").glob("*.png"))


# ---------------------------------------------------------------------------
# save_attention — model skipping behavior
# ---------------------------------------------------------------------------


def test_save_attention_skips_no_attention_aggregator(tmp_path):
    """TransMIL/MeanPool/MaxPool are skipped without error."""
    from soma.config import (
        AggregatorConfig, CacheConfig, EncoderConfig, HeatmapConfig,
        PipelineConfig, TaskConfig, TrainingConfig, save_config,
    )

    config = PipelineConfig(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        aggregator=AggregatorConfig(name="meanpool"),
        task=TaskConfig(name="binary_classification", params={"num_classes": 2}),
    )
    save_config(config, tmp_path / "config.yaml")

    feature_store = MagicMock()
    feature_store.is_slide_level = False
    feature_store.is_hierarchical = False

    dataset = MagicMock()

    # Should return without error, writing nothing
    save_attention(tmp_path, dataset, feature_store)
    assert not list(tmp_path.glob("fold_*/attention/*.npz"))


def test_save_attention_skips_slide_level_features(tmp_path):
    """Slide-level features have no tile attention — skipped early."""
    from soma.config import AggregatorConfig, PipelineConfig, TaskConfig, save_config

    config = PipelineConfig(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification", params={"num_classes": 2}),
    )
    save_config(config, tmp_path / "config.yaml")

    feature_store = MagicMock()
    feature_store.is_slide_level = True  # <-- slide-level

    dataset = MagicMock()
    save_attention(tmp_path, dataset, feature_store)
    assert not list(tmp_path.glob("fold_*/attention/*.npz"))
