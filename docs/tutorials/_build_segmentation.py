"""Build ``walkthrough-segmentation.ipynb``.

Authoring tooling (not part of the tutorial). Assembles the dense segmentation
walkthrough — per-pixel prediction on a frozen token grid via a decoder — and
writes it *unexecuted*. See ``_build_tile_level.py`` for the build-only pattern.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-segmentation.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Segmentation\n"
            "\n"
            "Per-pixel class prediction on a frozen foundation-model token grid:\n"
            "\n"
            "```\n"
            "Cohort + manifests -> FeatureExtractor(shape='grid') -> train (decoder + head) -> evaluate\n"
            "```\n"
            "\n"
            "The encoder emits a **token grid per tile** (not one vector), and a **decoder**\n"
            "upsamples that grid to a per-pixel map. [Detection](walkthrough-detection.ipynb)\n"
            "is the same dense flow with point supervision and a different head.\n"
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
            "carries `sample_id, image_path, label_mask_path`, where the mask is an integer-class\n"
            "raster the same size as the tile. We fabricate small **224 px pre-cropped tiles**\n"
            "plus their masks. To train from whole slides instead, set `preprocessing.masks`;\n"
            "see the Segmentation page."
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
            "WORK = Path(tempfile.mkdtemp(prefix='soma-segmentation-'))\n"
            "ROIS = WORK / 'rois'; MASKS = WORK / 'masks'\n"
            "for d in (ROIS, MASKS): d.mkdir()\n"
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
            "ids = [f'roi{i:02d}' for i in range(8)]\n"
            "for sid in ids:\n"
            "    make_roi(ROIS / f'{sid}.tif')\n"
            "    make_mask(MASKS / f'{sid}.png')\n"
            "\n"
            "split = ['train'] * 4 + ['tune'] * 2 + ['test'] * 2\n"
            "splits_csv = WORK / 'splits.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'split': split, 'fold': 0}).to_csv(splits_csv, index=False)\n"
            "\n"
            "img_paths = [str(ROIS / f'{s}.tif') for s in ids]\n"
            "\n"
            "seg_csv = WORK / 'seg.csv'\n"
            "pd.DataFrame({'sample_id': ids, 'image_path': img_paths,\n"
            "              'label_mask_path': [str(MASKS / f'{s}.png') for s in ids]}).to_csv(seg_csv, index=False)\n"
            "print(pd.read_csv(seg_csv).head(3).to_string(index=False))"
        ),
        md(
            "## 1. Extract dense token grids\n"
            "\n"
            "`FeatureExtractor(shape='grid', unit='tile')` reads each pre-cropped ROI at the\n"
            "requested spacing and runs the frozen encoder to produce a `(feature_dim, gh, gw)`\n"
            "grid per sample, served by a `CachedGridSource`. With phikon at 224 px and\n"
            "patch-16 that's a 14×14 grid. `records_for_pipeline` bridges the cohort and\n"
            "manifests to the current `FeatureExtractor` signature."
        ),
        code(
            "from soma import (\n"
            "    AnnotationManifest, CacheOptions, Cohort, ImageManifest, FeatureExtractor,\n"
            ")\n"
            "from slide2vec import DenseOptions\n"
            "from soma.data._legacy import records_for_pipeline\n"
            "\n"
            "# One CSV, three readers: Cohort (ids + splits; dense heads declare no scalar\n"
            "# target), ImageManifest (image paths), AnnotationManifest (supervision files).\n"
            "cohort = Cohort.from_csv(seg_csv, splits_csv)\n"
            "seg_samples = records_for_pipeline(\n"
            "    cohort, ImageManifest.from_csv(seg_csv), AnnotationManifest.from_csv(seg_csv),\n"
            ")\n"
            "extractor = FeatureExtractor(\n"
            "    seg_samples,\n"
            "    'phikon',\n"
            "    dense=DenseOptions(spacing_um=SPACING, target_size=SIZE, backend='openslide'),\n"
            "    shape='grid', unit='tile',\n"
            "    cache=CacheOptions(enabled=False),\n"
            "    output_root=str(WORK / 'dense'),\n"
            ")\n"
            "dense_store = extractor.extract().source\n"
            "print('dense grids for', len(dense_store.sample_ids), 'ROIs')"
        ),
        md(
            "## 2. Train the decoder + head\n"
            "\n"
            "A `Pipeline` given the grid store trains the `dense` learner on it: a **decoder**\n"
            "(`lightweight_conv` upsamples the token grid) plus a parameter-free head that\n"
            "crops to the mask and scores Dice / IoU. The head reads its label maps from a\n"
            "`LabelMapSource` the pipeline binds; records never carry file paths."
        ),
        code(
            "from soma import ExperimentConfig, Pipeline\n"
            "from soma.config import (\n"
            "    CohortConfig, EvaluationConfig, FeaturesConfig, LearnerConfig, LoadersConfig,\n"
            "    RunConfig, TrainerConfig,\n"
            ")\n"
            "\n"
            "config = ExperimentConfig(\n"
            "    run=RunConfig(output_dir=str(WORK / 'runs' / 'segmentation'), seed=0),\n"
            "    cohort=CohortConfig(records=seg_csv, splits=splits_csv),\n"
            "    # our PNG masks carry no spacing metadata; declare the ROI spacing so they\n"
            "    # register against the grids extracted at the same spacing.\n"
            "    features=FeaturesConfig(dense=DenseOptions(spacing_um=SPACING, target_size=SIZE)),\n"
            "    loaders=LoadersConfig(batch_size=2),\n"
            "    learner=LearnerConfig('dense', {\n"
            "        'decoder': 'lightweight_conv',\n"
            "        'head': {'name': 'segmentation', 'params': {'num_classes': NUM_CLASSES}},\n"
            "        'optimizer': {'name': 'adam', 'params': {'lr': 1e-3}},\n"
            "    }),\n"
            "    trainer=TrainerConfig(max_epochs=3),\n"
            "    evaluation=EvaluationConfig(metrics=['mean_dice', 'mean_iou']),\n"
            ")\n"
            "seg_result = Pipeline(config, source=dense_store).run()\n"
            "print('segmentation run dir:', seg_result.run_dir)"
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
            "from soma import ExperimentConfig, Pipeline\n"
            "from soma.config import (\n"
            "    CohortConfig, EvaluationConfig, FeaturesConfig, LearnerConfig, LoadersConfig,\n"
            "    RunConfig, TrainerConfig,\n"
            ")\n"
            "from slide2vec import DenseOptions\n"
            "\n"
            "config = ExperimentConfig(\n"
            "    run=RunConfig(output_dir='output/segmentation'),\n"
            "    cohort=CohortConfig(records=seg_csv, splits=splits_csv),\n"
            "    features=FeaturesConfig(\n"
            "        encoder='phikon',\n"
            "        dense=DenseOptions(spacing_um=0.5, target_size=224, backend='openslide'),\n"
            "    ),\n"
            "    loaders=LoadersConfig(batch_size=2),\n"
            "    learner=LearnerConfig('dense', {\n"
            "        'decoder': 'lightweight_conv',\n"
            "        'head': {'name': 'segmentation', 'params': {'num_classes': 3}},\n"
            "        'optimizer': {'name': 'adam', 'params': {'lr': 1e-3}},\n"
            "    }),\n"
            "    trainer=TrainerConfig(max_epochs=3),\n"
            "    evaluation=EvaluationConfig(metrics=['mean_dice', 'mean_iou']),\n"
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
