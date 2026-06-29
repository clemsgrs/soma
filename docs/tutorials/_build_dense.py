"""Build + execute ``walkthrough-dense.ipynb`` with committed outputs.

Authoring tooling (not part of the tutorial). See ``_build_slide_level.py`` for
the rationale; this assembles the dense-prediction walkthrough (segmentation +
detection) and executes it on CPU against the repo source.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-dense.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Dense prediction\n"
            "\n"
            "This notebook walks through soma's **dense** flow — per-pixel and per-point\n"
            "prediction on a frozen foundation-model token grid:\n"
            "\n"
            "```\n"
            "Dataset(+masks/points) -> DenseTileFeatureExtractor -> train (decoder + head) -> evaluate\n"
            "```\n"
            "\n"
            "It mirrors the [slide-level walkthrough](walkthrough-slide-level.ipynb), but\n"
            "the encoder now emits a **token grid per tile** instead of one vector per\n"
            "slide, and a **decoder** (not a MIL aggregator) upsamples that grid. We work\n"
            "**segmentation** end to end, then show that **detection** is the same flow\n"
            "with a different head + point supervision.\n"
            "\n"
            "> **Tiny synthetic data, runs on CPU, ungated encoder — the numbers are\n"
            "> meaningless; the point is the API.** We use\n"
            "> [`phikon`](https://huggingface.co/owkin/phikon) at its native **224 px**\n"
            "> window (a 14×14 token grid), which avoids position-embedding interpolation."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "Dense supervision lives in per-sample files, not a scalar `label`:\n"
            "\n"
            "* **segmentation** — `dataset.csv` has `sample_id, image_path, mask_path`;\n"
            "  the mask is an integer-class raster the same size as the ROI.\n"
            "* **detection** — `sample_id, image_path, points_path`; the points file is a\n"
            "  CSV of `x, y, class` in ROI-pixel coordinates.\n"
            "\n"
            "We fabricate small **224 px ROI tiles** (the dense flow consumes fixed-size\n"
            "tiles/ROIs, not whole WSIs) plus their masks and point files."
        ),
        code(
            "import logging, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "logging.getLogger().setLevel(logging.ERROR)\n"
            "\n"
            "import tempfile\n"
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import tifffile\n"
            "from PIL import Image\n"
            "\n"
            "WORK = Path(tempfile.mkdtemp(prefix='soma-dense-tutorial-'))\n"
            "ROIS = WORK / 'rois'; MASKS = WORK / 'masks'; POINTS = WORK / 'points'\n"
            "for d in (ROIS, MASKS, POINTS): d.mkdir()\n"
            "rng = np.random.default_rng(0)\n"
            "\n"
            "SIZE = 224          # phikon native window\n"
            "SPACING = 0.5       # microns/pixel\n"
            "NUM_CLASSES = 3     # 0 = background, 1, 2 = tissue classes\n"
            "\n"
            "def make_roi(path):\n"
            "    img = np.clip(np.stack([np.full((SIZE, SIZE), 150),\n"
            "                            np.full((SIZE, SIZE), 70),\n"
            "                            np.full((SIZE, SIZE), 160)], -1).astype(np.int16)\n"
            "                  + rng.integers(-30, 30, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)\n"
            "    tifffile.imwrite(path, img, photometric='rgb', tile=(SIZE, SIZE),\n"
            "                     resolution=(20000, 20000), resolutionunit='CENTIMETER')\n"
            "\n"
            "def make_mask(path):\n"
            "    m = np.zeros((SIZE, SIZE), np.uint8)\n"
            "    m[SIZE // 4:SIZE // 2, SIZE // 4:SIZE // 2] = 1\n"
            "    m[SIZE // 2:3 * SIZE // 4, SIZE // 2:3 * SIZE // 4] = 2\n"
            "    Image.fromarray(m).save(path)\n"
            "\n"
            "def make_points(path):\n"
            "    # a few cells per class, in ROI-pixel coordinates\n"
            "    pts = [(56, 56, 0), (112, 112, 1), (160, 160, 1)]\n"
            "    pd.DataFrame(pts, columns=['x', 'y', 'class']).to_csv(path, index=False)\n"
            "\n"
            "ids = [f'roi{i:02d}' for i in range(8)]\n"
            "for sid in ids:\n"
            "    make_roi(ROIS / f'{sid}.tif')\n"
            "    make_mask(MASKS / f'{sid}.png')\n"
            "    make_points(POINTS / f'{sid}.csv')\n"
            "\n"
            "split = ['train'] * 4 + ['tune'] * 2 + ['test'] * 2\n"
            "splits_csv = WORK / 'splits.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'split': split, 'fold': 0}).to_csv(splits_csv, index=False)\n"
            "\n"
            "img_paths = [str(ROIS / f'{s}.tif') for s in ids]\n"
            "\n"
            "# Feature extraction only needs the images; supervision lives in the\n"
            "# task-specific manifests below (masks for segmentation, points for detection).\n"
            "extract_csv = WORK / 'extract.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'image_path': img_paths,\n"
            "              'label': 0}).to_csv(extract_csv, index=False)\n"
            "\n"
            "seg_csv = WORK / 'seg.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'image_path': img_paths,\n"
            "              'mask_path': [str(MASKS / f'{s}.png') for s in ids]}).to_csv(seg_csv, index=False)\n"
            "print('segmentation manifest:')\n"
            "print(pd.read_csv(seg_csv).head(3).to_string(index=False))"
        ),
        md(
            "## 1. Extract dense token grids\n"
            "\n"
            "`DenseTileFeatureExtractor` reads each ROI at `spacing_um` and runs the frozen\n"
            "encoder to produce a `(feature_dim, gh, gw)` grid per sample, stored in a\n"
            "`DenseFeatureStore`. With phikon at 224 px and patch-16 that's a 14×14 grid."
        ),
        code(
            "from soma import (\n"
            "    Dataset, DenseTileFeatureExtractor, EncoderConfig, CacheConfig, PreprocessingConfig,\n"
            ")\n"
            "\n"
            "extractor = DenseTileFeatureExtractor(\n"
            "    Dataset(extract_csv),\n"
            "    EncoderConfig(name='phikon'),\n"
            "    target_size=SIZE,\n"
            "    spacing_um=SPACING,\n"
            "    backend='openslide',\n"
            "    cache=CacheConfig(enabled=False),\n"
            ")\n"
            "dense_store = extractor.run(str(WORK / 'dense'))\n"
            "print('dense store ready for', len(dense_store.available_samples), 'ROIs')"
        ),
        md(
            "## 2. Train segmentation (decoder + head)\n"
            "\n"
            "`train(dataset_type='segmentation', ...)` builds a **decoder**\n"
            "(`lightweight_conv` upsamples the token grid) plus a parameter-free\n"
            "segmentation head that crops to the mask and scores Dice / IoU.\n"
            "`SegmentationManifest` is the dense counterpart of `Dataset`."
        ),
        code(
            "from soma.dataset import SegmentationManifest\n"
            "from soma import (\n"
            "    Splits, DecoderConfig, TaskConfig, TrainingConfig, EvalConfig, train,\n"
            ")\n"
            "\n"
            "seg_manifest = SegmentationManifest(seg_csv)\n"
            "seg_splits = Splits(splits_csv, seg_manifest)\n"
            "\n"
            "seg_result = train(\n"
            "    feature_store=dense_store,\n"
            "    dataset=seg_manifest,\n"
            "    splits=seg_splits,\n"
            "    dataset_type='segmentation',\n"
            "    decoder=DecoderConfig(name='lightweight_conv'),\n"
            "    task=TaskConfig(name='segmentation', params={'num_classes': NUM_CLASSES}),\n"
            "    training=TrainingConfig(epochs=3, batch_size=2, learning_rate=1e-3, seed=0),\n"
            "    evaluation=EvalConfig(metrics=['mean_dice', 'mean_iou']),\n"
            "    # our PNG masks carry no spacing metadata; declare the ROI spacing so they\n"
            "    # register against the grids extracted at the same spacing.\n"
            "    preprocessing=PreprocessingConfig(requested_spacing_um=SPACING),\n"
            "    run_dir=str(WORK / 'runs' / 'segmentation'),\n"
            ")\n"
            "print('segmentation run dir:', seg_result.run_dir)"
        ),
        md(
            "## 3. Switch to detection — same grid, point supervision\n"
            "\n"
            "Detection reuses the **same dense extraction**; only the supervision and head\n"
            "change. `TaskConfig('detection')` renders each annotated point as a peak\n"
            "Gaussian, the decoder smooths the grid into a peak heatmap, and the head\n"
            "recovers points (local-maxima + NMS) scored with **F1 at a matching distance\n"
            "δ**. `match_distance` and `sigma` are given in **µm**."
        ),
        code(
            "from soma.dataset import DetectionManifest\n"
            "\n"
            "det_csv = WORK / 'det.csv'\n"
            "pd.DataFrame({'sample_id': ids,\n"
            "              'image_path': [str(ROIS / f'{s}.tif') for s in ids],\n"
            "              'points_path': [str(POINTS / f'{s}.csv') for s in ids]}).to_csv(det_csv, index=False)\n"
            "\n"
            "det_manifest = DetectionManifest(det_csv)\n"
            "det_splits = Splits(splits_csv, det_manifest)\n"
            "\n"
            "det_result = train(\n"
            "    feature_store=dense_store,\n"
            "    dataset=det_manifest,\n"
            "    splits=det_splits,\n"
            "    dataset_type='detection',\n"
            "    decoder=DecoderConfig(name='lightweight_conv'),\n"
            "    task=TaskConfig(name='detection', params={\n"
            "        'num_classes': NUM_CLASSES,\n"
            "        'match_distance': 2.0,   # microns\n"
            "        'sigma': 0.7,            # microns\n"
            "    }),\n"
            "    training=TrainingConfig(epochs=3, batch_size=2, learning_rate=1e-3, seed=0),\n"
            "    evaluation=EvalConfig(metrics=['mean_f1', 'f1_per_class']),\n"
            "    preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=SIZE),\n"
            "    run_dir=str(WORK / 'runs' / 'detection'),\n"
            ")\n"
            "print('detection run dir:', det_result.run_dir)"
        ),
        md(
            "## 4. The one-shot `Pipeline` equivalent\n"
            "\n"
            "As with the slide-level flow, `Pipeline` collapses extract + train + evaluate\n"
            "into a single config-driven call.\n"
            "\n"
            "*(Shown for reference, not executed.)*\n"
            "\n"
            "```python\n"
            "from soma import (\n"
            "    Pipeline, PipelineConfig, PreprocessingConfig, EncoderConfig,\n"
            "    DecoderConfig, TaskConfig, TrainingConfig, EvalConfig, CacheConfig,\n"
            ")\n"
            "\n"
            "config = PipelineConfig(\n"
            "    dataset_csv=str(seg_csv),\n"
            "    splits_csv=str(splits_csv),\n"
            "    output_root='output/segmentation',\n"
            "    dataset_type='segmentation',\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide', requested_tile_size_px=224, requested_spacing_um=0.5,\n"
            "    ),\n"
            "    encoder=EncoderConfig(name='phikon'),\n"
            "    decoder=DecoderConfig(name='lightweight_conv'),\n"
            "    task=TaskConfig(name='segmentation', params={'num_classes': 3}),\n"
            "    training=TrainingConfig(epochs=3, batch_size=2, learning_rate=1e-3),\n"
            "    evaluation=EvalConfig(metrics=['mean_dice', 'mean_iou']),\n"
            "    cache=CacheConfig(enabled=True),\n"
            ")\n"
            "results = Pipeline(config).run()\n"
            "```"
        ),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    # Reachable via :doc: links from the tutorial hub pages, but kept out of the
    # sidebar nav (the hub pages are the nav entries).
    nb.metadata["nbsphinx"] = {"orphan": True}
    return nb


def main() -> None:
    nb = build()
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbformat.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
