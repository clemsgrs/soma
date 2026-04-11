"""Attention heatmap generation for MIL models.

Two-phase design:
  1. save_attention()   — run inference on test samples, persist attention
                          scores as .npz files (no WSI access needed).
  2. render_heatmaps()  — read attention .npz + coordinates + WSI thumbnail,
                          produce PNG overlays (no model/GPU needed).

Use generate_heatmaps() to run both phases in sequence.

Supported aggregators: ABMIL, CLAM-SB, CLAM-MB, DSMIL.
HIPT (hierarchical features), TransMIL, MeanPool, and MaxPool are skipped.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hs2p.wsi.geometry import select_level_for_downsample
from hs2p.wsi.reader import open_slide

from soma.config import HeatmapConfig
from soma.dataset import Dataset
from soma.features import FeatureStore

logger = logging.getLogger(__name__)

# Aggregators whose tile_attention is already post-softmax (don't re-apply softmax).
_POST_SOFTMAX_AGGREGATORS = {"dsmil"}

# Aggregators with no per-tile attention — skip silently.
_NO_ATTENTION_AGGREGATORS = {"transmil", "meanpool", "maxpool"}


# ---------------------------------------------------------------------------
# Phase 1: inference → attention .npz
# ---------------------------------------------------------------------------


def save_attention(
    run_dir: Path | str,
    dataset: Dataset,
    feature_store: FeatureStore,
) -> None:
    """Run inference on test samples and persist per-tile attention scores.

    For each fold directory in *run_dir*, loads ``best_model.pt`` and the
    run's ``config.yaml``, then writes one ``.npz`` file per test sample to
    ``fold_N/attention/<sample_id>.npz``.

    The stored array has key ``"attention"`` with shape:

    * ``(N,)`` for single-branch aggregators (ABMIL, CLAM-SB, DSMIL).
    * ``(n_classes, N)`` for CLAM-MB.

    Args:
        run_dir: Root run directory containing fold sub-directories.
        dataset: Dataset used to reconstruct the task head (label map).
        feature_store: Feature store used to load tile embeddings.
    """
    from soma.config import load_config
    from soma.aggregators.registry import aggregator_registry
    from soma.tasks.classification import BranchAwareClassificationHead
    from soma.tasks.registry import task_registry
    from soma.training.model import MILModel

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")

    if feature_store.is_slide_level or feature_store.is_hierarchical:
        logger.info("save_attention: skipping — slide-level or hierarchical features have no tile attention")
        return

    aggregator_cfg = config.aggregator
    if aggregator_cfg is None:
        logger.info("save_attention: skipping — no aggregator configured")
        return

    agg_name = aggregator_cfg.name
    if agg_name in _NO_ATTENTION_AGGREGATORS:
        logger.info("save_attention: skipping — %s produces no tile attention", agg_name)
        return

    task_cfg = config.task
    eval_cfg = config.eval
    task_cls = task_registry.get(task_cfg.name)
    task_params = {**task_cls.auto_params(dataset), **task_cfg.params, "metrics": eval_cfg.metrics}
    feature_dim = feature_store.feature_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_dir in sorted(run_dir.glob("fold_*")):
        checkpoint_path = fold_dir / "best_model.pt"
        predictions_path = fold_dir / "predictions.csv"
        if not checkpoint_path.is_file():
            logger.warning("save_attention: no checkpoint found at %s, skipping fold", checkpoint_path)
            continue
        if not predictions_path.is_file():
            logger.warning("save_attention: no predictions.csv found at %s, skipping fold", predictions_path)
            continue

        # Reconstruct model
        aggregator_cls = aggregator_registry.get(agg_name)
        agg = aggregator_cls(input_dim=feature_dim, **aggregator_cfg.params)
        if agg_name == "clam_mb" and task_cfg.name == "multiclass_classification":
            head = BranchAwareClassificationHead(input_dim=agg.output_dim, **task_params)
        else:
            head = task_cls(input_dim=agg.output_dim, **task_params)
        model = MILModel(aggregator=agg, task_head=head)

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Read test sample IDs from predictions.csv
        test_sample_ids = _read_sample_ids_from_predictions(predictions_path)

        attention_dir = fold_dir / "attention"
        attention_dir.mkdir(exist_ok=True)

        with torch.inference_mode():
            for sample_id in test_sample_ids:
                if sample_id not in feature_store.available_samples:
                    logger.warning("save_attention: features not found for %s, skipping", sample_id)
                    continue

                features = feature_store.load(sample_id).unsqueeze(0).to(device)  # (1, N, D)
                out = model(features)

                if out.tile_attention is None:
                    logger.info(
                        "save_attention: %s returned no tile_attention for sample %s — skipping",
                        agg_name, sample_id,
                    )
                    continue

                attention = _normalize_attention(out.tile_attention, agg_name)  # (N,) or (n_classes, N)
                np.savez_compressed(attention_dir / f"{sample_id}.npz", attention=attention)
                logger.debug("Saved attention for %s → %s", sample_id, attention_dir / f"{sample_id}.npz")

        logger.info("Saved attention scores for fold %s to %s", fold_dir.name, attention_dir)


# ---------------------------------------------------------------------------
# Phase 2: attention .npz → heatmap PNG
# ---------------------------------------------------------------------------


def render_heatmaps(
    run_dir: Path | str,
    dataset: Dataset,
    feature_store: FeatureStore,
    heatmap_config: HeatmapConfig,
    seg_downsample: int,
) -> None:
    """Render attention heatmaps from persisted attention ``.npz`` files.

    For each fold directory in *run_dir*, reads ``fold_N/attention/*.npz``
    and writes PNG files to ``fold_N/heatmaps/``.

    Single-branch models produce ``<sample_id>.png``.
    CLAM-MB produces ``<sample_id>_class_<k>.png`` for each branch.

    Args:
        run_dir: Root run directory containing fold sub-directories.
        dataset: Dataset for resolving WSI paths per sample.
        feature_store: Feature store for resolving tile coordinate paths.
        heatmap_config: Rendering parameters (colormap, alpha, blur).
        seg_downsample: Downsample factor for the WSI thumbnail — matches
            the preprocessing preview downsample so all visual outputs are
            consistent.
    """
    run_dir = Path(run_dir)
    coord_map = _build_coordinate_map(feature_store)

    for fold_dir in sorted(run_dir.glob("fold_*")):
        attention_dir = fold_dir / "attention"
        if not attention_dir.is_dir():
            logger.debug("render_heatmaps: no attention dir in %s, skipping", fold_dir)
            continue

        heatmap_dir = fold_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)

        for npz_path in sorted(attention_dir.glob("*.npz")):
            sample_id = npz_path.stem
            if sample_id not in dataset.samples:
                logger.warning("render_heatmaps: sample_id %s not in dataset, skipping", sample_id)
                continue
            if sample_id not in coord_map:
                logger.warning("render_heatmaps: no coordinates found for %s, skipping", sample_id)
                continue

            slide_path = dataset.samples[sample_id].image_path
            coords_npz_path, coords_meta_path = coord_map[sample_id]

            coords = np.load(coords_npz_path)
            x_lv0 = coords["x"]
            y_lv0 = coords["y"]

            with open(coords_meta_path) as f:
                meta = json.load(f)
            tile_size_lv0 = int(meta["tile_size_lv0"])

            attention = np.load(npz_path)["attention"]  # (N,) or (n_classes, N)

            if attention.ndim == 1:
                # Single-branch
                img = render_attention_heatmap(
                    slide_path, x_lv0, y_lv0, attention, tile_size_lv0, seg_downsample,
                    cmap=heatmap_config.cmap,
                    alpha=heatmap_config.alpha,
                    blur_sigma=heatmap_config.blur_sigma,
                )
                _save_heatmap(img, heatmap_dir / f"{sample_id}.png")
            else:
                # Multi-branch (CLAM-MB): one heatmap per class
                n_classes = attention.shape[0]
                for k in range(n_classes):
                    img = render_attention_heatmap(
                        slide_path, x_lv0, y_lv0, attention[k], tile_size_lv0, seg_downsample,
                        cmap=heatmap_config.cmap,
                        alpha=heatmap_config.alpha,
                        blur_sigma=heatmap_config.blur_sigma,
                    )
                    _save_heatmap(img, heatmap_dir / f"{sample_id}_class_{k}.png")

        logger.info("Rendered heatmaps for fold %s to %s", fold_dir.name, heatmap_dir)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_heatmaps(
    run_dir: Path | str,
    dataset: Dataset,
    feature_store: FeatureStore,
    heatmap_config: HeatmapConfig,
    seg_downsample: int,
) -> None:
    """Run attention extraction then heatmap rendering.

    Equivalent to calling :func:`save_attention` followed by
    :func:`render_heatmaps`. Use this when both steps are needed
    in one call (e.g. from ``Pipeline.run()``).

    Args:
        run_dir: Root run directory containing fold sub-directories.
        dataset: Dataset for resolving WSI paths and label map.
        feature_store: Feature store for loading features and coordinates.
        heatmap_config: Rendering parameters.
        seg_downsample: Downsample factor for the WSI thumbnail.
    """
    save_attention(run_dir, dataset, feature_store)
    render_heatmaps(run_dir, dataset, feature_store, heatmap_config, seg_downsample)


# ---------------------------------------------------------------------------
# Core rendering primitive
# ---------------------------------------------------------------------------


def render_attention_heatmap(
    slide_path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    tile_size_lv0: int,
    seg_downsample: int,
    *,
    cmap: str = "coolwarm",
    alpha: float = 0.5,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Render a single attention heatmap overlaid on a WSI thumbnail.

    Args:
        slide_path: Path to the whole-slide image.
        x: Tile X coordinates at level-0 resolution, shape ``(N,)``.
        y: Tile Y coordinates at level-0 resolution, shape ``(N,)``.
        scores: Attention scores, shape ``(N,)``. Values are min-max
            normalized to ``[0, 1]`` before applying the colormap.
        tile_size_lv0: Tile width/height in level-0 pixels.
        seg_downsample: Target downsample factor for the visualization level.
            Uses the closest available pyramid level (same logic as
            preprocessing previews).
        cmap: Matplotlib colormap name.
        alpha: Opacity of the attention overlay (0=transparent, 1=opaque).
        blur_sigma: Standard deviation for optional Gaussian blur applied to
            the attention map before coloring. 0 disables blurring.

    Returns:
        RGB image as a ``(H, W, 3)`` uint8 array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open_slide(slide_path) as slide:
        vis_level = select_level_for_downsample(seg_downsample, slide.level_downsamples)
        vis_size = slide.level_dimensions[vis_level]  # (W, H)
        canvas = slide.read_region((0, 0), vis_level, vis_size).copy()  # (H, W, 3) uint8

    vis_w, vis_h = vis_size
    ds_x, ds_y = slide.level_downsamples[vis_level]
    scale_x = 1.0 / ds_x
    scale_y = 1.0 / ds_y
    tile_vis_w = max(1, round(tile_size_lv0 * scale_x))
    tile_vis_h = max(1, round(tile_size_lv0 * scale_y))

    # Accumulate attention scores into a float overlay
    overlay = np.zeros((vis_h, vis_w), dtype=np.float32)
    count = np.zeros((vis_h, vis_w), dtype=np.float32)

    for i in range(len(scores)):
        x1 = int(x[i] * scale_x)
        y1 = int(y[i] * scale_y)
        x2 = min(vis_w, x1 + tile_vis_w)
        y2 = min(vis_h, y1 + tile_vis_h)
        if x2 <= x1 or y2 <= y1:
            continue
        overlay[y1:y2, x1:x2] += scores[i]
        count[y1:y2, x1:x2] += 1.0

    valid = count > 0
    if not valid.any():
        return canvas

    overlay[valid] /= count[valid]

    # Optional Gaussian blur on the attention map
    if blur_sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        overlay = gaussian_filter(overlay, sigma=blur_sigma)

    # Min-max normalize to [0, 1] over tissue-covered pixels
    mn = overlay[valid].min()
    mx = overlay[valid].max()
    if mx > mn:
        overlay[valid] = (overlay[valid] - mn) / (mx - mn)
    else:
        overlay[valid] = 0.5

    # Apply colormap → (H, W, 3) float in [0, 1]
    cmap_fn = plt.get_cmap(cmap)
    colored = cmap_fn(overlay)[:, :, :3].astype(np.float32)  # drop alpha channel

    # Alpha-blend attention color onto canvas, only over tile-covered pixels
    result = canvas.astype(np.float32)
    result[valid] = (
        alpha * colored[valid] * 255.0
        + (1.0 - alpha) * result[valid]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_attention(tile_attention: torch.Tensor, agg_name: str) -> np.ndarray:
    """Convert raw model attention output to a normalized numpy array.

    Applies softmax for aggregators that return raw logits (ABMIL, CLAM).
    DSMIL already returns post-softmax values — these are passed through.

    Args:
        tile_attention: Attention tensor from the model, shape ``(1, N)``
            for single-branch or ``(1, n_classes, N)`` for CLAM-MB.
        agg_name: Aggregator name from config (e.g. ``"abmil"``).

    Returns:
        Numpy array of shape ``(N,)`` or ``(n_classes, N)``.
    """
    attn = tile_attention.squeeze(0)  # (N,) or (n_classes, N)
    if agg_name not in _POST_SOFTMAX_AGGREGATORS:
        attn = F.softmax(attn, dim=-1)
    return attn.cpu().float().numpy()


def _read_sample_ids_from_predictions(predictions_path: Path) -> list[str]:
    """Return sample IDs from a predictions.csv file."""
    sample_ids: list[str] = []
    with open(predictions_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_ids.append(row["sample_id"])
    return sample_ids


def _build_coordinate_map(
    feature_store: FeatureStore,
) -> dict[str, tuple[Path, Path]]:
    """Build a mapping from sample_id to (coordinates_npz_path, coordinates_meta_path).

    Reads from the ``process_list.csv`` tracked by the feature store.

    Returns an empty dict (with a warning) if the manifest is not available
    or lacks coordinate columns.
    """
    manifest_path = feature_store.feature_manifest_path
    if manifest_path is None:
        logger.warning(
            "render_heatmaps: feature store has no process_list.csv — cannot resolve tile coordinates"
        )
        return {}

    coord_map: dict[str, tuple[Path, Path]] = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if "coordinates_npz_path" not in (reader.fieldnames or []):
            logger.warning(
                "render_heatmaps: process_list.csv has no 'coordinates_npz_path' column — "
                "cannot resolve tile coordinates"
            )
            return {}
        for row in reader:
            sid = str(row["sample_id"])
            npz_val = row.get("coordinates_npz_path", "")
            meta_val = row.get("coordinates_meta_path", "")
            if not npz_val or npz_val.lower() in ("", "nan", "none"):
                continue
            if not meta_val or meta_val.lower() in ("", "nan", "none"):
                continue
            npz_path = Path(npz_val)
            meta_path = Path(meta_val)
            if npz_path.is_file() and meta_path.is_file():
                coord_map[sid] = (npz_path, meta_path)
            else:
                logger.debug(
                    "render_heatmaps: coordinate files missing for %s, skipping", sid
                )

    return coord_map


def _save_heatmap(img: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as a PNG file."""
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
