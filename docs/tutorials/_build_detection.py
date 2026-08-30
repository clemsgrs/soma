"""Build ``walkthrough-detection.ipynb``.

Authoring tooling (not part of the tutorial). Assembles the dense detection
walkthrough — point prediction on a frozen token grid via a peak-heatmap decoder
— and writes it *unexecuted*. See ``_build_tile_level.py`` for the build-only
pattern.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-detection.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Detection\n"
            "\n"
            "Point prediction — cell / nucleus centroids — on a frozen foundation-model\n"
            "token grid:\n"
            "\n"
            "```\n"
            "DetectionManifest -> FeatureExtractor -> train (decoder + head) -> evaluate\n"
            "```\n"
            "\n"
            "The encoder emits a **token grid per tile**, and a **decoder** smooths it into\n"
            "a per-class peak heatmap the head reads points back from.\n"
            "[Segmentation](walkthrough-segmentation.ipynb) is the same dense flow with mask\n"
            "supervision and a per-pixel head.\n"
            "\n"
            "> Tiny synthetic data, CPU-only, ungated encoder — the numbers are\n"
            "> meaningless; the point is the API. We use\n"
            "> [phikon](https://huggingface.co/owkin/phikon) at its native **224 px**\n"
            "> window (a 14×14 token grid), which avoids position-embedding interpolation."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "Dense supervision lives in per-sample files, not a scalar `label`: `dataset.csv`\n"
            "carries `sample_id, image_path, points_path`, where the points file is a CSV of\n"
            "`x, y, class` in ROI-pixel coordinates. We fabricate small **224 px ROI tiles**\n"
            "(the dense flow consumes fixed-size tiles/ROIs, not whole WSIs) plus their\n"
            "point files."
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
            "\n"
            "WORK = Path(tempfile.mkdtemp(prefix='soma-detection-'))\n"
            "ROIS = WORK / 'rois'; POINTS = WORK / 'points'\n"
            "for d in (ROIS, POINTS): d.mkdir()\n"
            "rng = np.random.default_rng(0)\n"
            "\n"
            "SIZE = 224          # phikon native window\n"
            "SPACING = 0.5       # microns/pixel\n"
            "NUM_CLASSES = 3     # 0 = background, 1, 2 = cell classes\n"
            "\n"
            "def make_roi(path):\n"
            "    img = np.clip(np.stack([np.full((SIZE, SIZE), 150),\n"
            "                            np.full((SIZE, SIZE), 70),\n"
            "                            np.full((SIZE, SIZE), 160)], -1).astype(np.int16)\n"
            "                  + rng.integers(-30, 30, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)\n"
            "    tifffile.imwrite(path, img, photometric='rgb', tile=(SIZE, SIZE),\n"
            "                     resolution=(20000, 20000), resolutionunit='CENTIMETER')\n"
            "\n"
            "def make_points(path):\n"
            "    # a few cells per class, in ROI-pixel coordinates\n"
            "    pts = [(56, 56, 0), (112, 112, 1), (160, 160, 1)]\n"
            "    pd.DataFrame(pts, columns=['x', 'y', 'class']).to_csv(path, index=False)\n"
            "\n"
            "ids = [f'roi{i:02d}' for i in range(8)]\n"
            "for sid in ids:\n"
            "    make_roi(ROIS / f'{sid}.tif')\n"
            "    make_points(POINTS / f'{sid}.csv')\n"
            "\n"
            "split = ['train'] * 4 + ['tune'] * 2 + ['test'] * 2\n"
            "splits_csv = WORK / 'splits.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'split': split, 'fold': 0}).to_csv(splits_csv, index=False)\n"
            "\n"
            "img_paths = [str(ROIS / f'{s}.tif') for s in ids]\n"
            "\n"
            "det_csv = WORK / 'det.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'image_path': img_paths,\n"
            "              'points_path': [str(POINTS / f'{s}.csv') for s in ids]}).to_csv(det_csv, index=False)\n"
            "print(pd.read_csv(det_csv).head(3).to_string(index=False))"
        ),
        md(
            "## 1. Extract dense token grids\n"
            "\n"
            "`FeatureExtractor` reads each ROI at the requested spacing and runs the frozen\n"
            "encoder to produce a `(feature_dim, gh, gw)` grid per sample, stored in a\n"
            "`DenseFeatureStore`. With phikon at 224 px and patch-16 that's a 14×14 grid."
        ),
        code(
            "from soma import (\n"
            "    DetectionManifest, FeatureExtractor, EncoderConfig, CacheConfig,\n"
            "    PreprocessingConfig,\n"
            ")\n"
            "\n"
            "det_manifest = DetectionManifest(det_csv)\n"
            "extractor = FeatureExtractor(\n"
            "    det_manifest,\n"
            "    EncoderConfig(name='phikon'),\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        requested_tile_size_px=SIZE, requested_spacing_um=SPACING,\n"
            "        backend='openslide',\n"
            "    ),\n"
            "    cache=CacheConfig(enabled=False),\n"
            "    output_root=str(WORK / 'dense'),\n"
            ")\n"
            "dense_store = extractor.extract().source\n"
            "print('dense grids for', len(dense_store.available_samples), 'ROIs')"
        ),
        md(
            "## 2. Train the decoder + head\n"
            "\n"
            "`TaskConfig('detection')` renders each annotated point as a peak Gaussian; the\n"
            "decoder smooths the grid into a peak heatmap, and the head recovers points\n"
            "(local-maxima + NMS) scored with **F1 at a matching distance δ**.\n"
            "`match_distance` and `sigma` are given in **µm**. `DetectionManifest` is the\n"
            "dense counterpart of `Dataset`."
        ),
        code(
            "from soma import (\n"
            "    Splits, DecoderConfig, TaskConfig, TrainingConfig, EvalConfig,\n"
            "    PreprocessingConfig, train,\n"
            ")\n"
            "\n"
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
            "## 3. The one-shot `Pipeline` equivalent\n"
            "\n"
            "`Pipeline` collapses extract + train + evaluate into a single config-driven\n"
            "call.\n"
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
            "    dataset_csv=str(det_csv),\n"
            "    splits_csv=str(splits_csv),\n"
            "    output_root='output/detection',\n"
            "    dataset_type='detection',\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide', requested_tile_size_px=224, requested_spacing_um=0.5,\n"
            "    ),\n"
            "    encoder=EncoderConfig(name='phikon'),\n"
            "    decoder=DecoderConfig(name='lightweight_conv'),\n"
            "    task=TaskConfig(name='detection', params={'num_classes': 3, 'match_distance': 2.0, 'sigma': 0.7}),\n"
            "    training=TrainingConfig(epochs=3, batch_size=2, learning_rate=1e-3),\n"
            "    evaluation=EvalConfig(metrics=['mean_f1', 'f1_per_class']),\n"
            "    cache=CacheConfig(enabled=True),\n"
            ")\n"
            "results = Pipeline(config).run()\n"
            "```"
        ),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["nbsphinx"] = {"orphan": True}
    return nb


def main() -> None:
    nbformat.write(build(), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
