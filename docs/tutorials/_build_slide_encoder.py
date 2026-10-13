"""Build ``walkthrough-slide-encoder.ipynb``.

Authoring tooling (not part of the tutorial). Assembles the slide-encoder
walkthrough — a slide-native encoder that returns one vector per slide, no
aggregator — and writes it *unexecuted*. See ``_build_tile_level.py`` for the
build-only pattern.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-slide-encoder.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Slide-level classification · slide encoder\n"
            "\n"
            "Predict a slide-level label using a **slide-native encoder** — a model that\n"
            "runs its own tile encoder internally and returns **one vector per slide**:\n"
            "\n"
            "```\n"
            "Cohort + ImageManifest -> FeatureExtractor (slide encoder) -> train (head, no aggregator) -> evaluate\n"
            "```\n"
            "\n"
            "There is no bag to pool, so **no MIL aggregator** (`aggregator=None`). The\n"
            "alternative — a tile encoder whose per-tile bag is pooled by an aggregator —\n"
            "is the [tile-encoder + MIL walkthrough](walkthrough-slide-mil.ipynb), and the\n"
            "two differ by exactly the encoder name and that one argument.\n"
            "\n"
            "> Tiny synthetic data, CPU-only, ungated encoder — the numbers are\n"
            "> meaningless; the point is the API."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "The cell below fabricates toy slides and the two CSVs soma expects. **Replace\n"
            "this with your own slides and labels** — only the on-disk contract matters:\n"
            "\n"
            "* `dataset.csv` — one row per slide: `sample_id`, `image_path`, `label`.\n"
            "* `splits.csv` — `sample_id`, `split` (`train` / `tune` / `test*`), optional\n"
            "  `fold`."
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
            "WORK = Path(tempfile.mkdtemp(prefix='soma-slide-encoder-'))\n"
            "SLIDES = WORK / 'slides'; SLIDES.mkdir()\n"
            "rng = np.random.default_rng(0)\n"
            "\n"
            "def make_toy_slide(path, size=640):\n"
            '    """A white background with a central H&E-ish blob, saved as a tiled TIFF\n'
            '    whose resolution tags make OpenSlide report 0.5 microns/pixel."""\n'
            "    img = np.full((size, size, 3), 240, np.uint8)\n"
            "    yy, xx = np.mgrid[0:size, 0:size]\n"
            "    blob = ((xx - size // 2) ** 2 + (yy - size // 2) ** 2) < (size * 0.35) ** 2\n"
            "    tissue = np.stack([np.full((size, size), 150),\n"
            "                       np.full((size, size), 70),\n"
            "                       np.full((size, size), 160)], -1).astype(np.int16)\n"
            "    tissue += rng.integers(-30, 30, (size, size, 3))\n"
            "    img[blob] = np.clip(tissue, 0, 255).astype(np.uint8)[blob]\n"
            "    tifffile.imwrite(path, img, photometric='rgb', tile=(256, 256),\n"
            "                     resolution=(20000, 20000), resolutionunit='CENTIMETER')\n"
            "\n"
            "N = 8\n"
            "sample_ids = [f's{i:02d}' for i in range(N)]\n"
            "for sid in sample_ids:\n"
            "    make_toy_slide(SLIDES / f'{sid}.tif')\n"
            "\n"
            "# train/tune/test assignment (single fold) with both classes in every split\n"
            "split = (['train'] * 4) + (['tune'] * 2) + (['test'] * 2)\n"
            "binary = [0, 1, 0, 1,  0, 1,  0, 1]\n"
            "\n"
            "dataset_csv = WORK / 'dataset.csv'\n"
            "splits_csv = WORK / 'splits.csv'\n"
            "pd.DataFrame({'sample_id': sample_ids,\n"
            "              'image_path': [str(SLIDES / f'{s}.tif') for s in sample_ids],\n"
            "              'label': binary}).to_csv(dataset_csv, index=False)\n"
            "pd.DataFrame({'sample_id': sample_ids, 'split': split}).to_csv(splits_csv, index=False)\n"
            "\n"
            "print(pd.read_csv(dataset_csv).head().to_string(index=False))"
        ),
        md(
            "## 1. Load the cohort and the image manifest\n"
            "\n"
            "Identical to every other path — `Cohort` reads identities, the `label`\n"
            "target and the splits; `ImageManifest` reads the image paths. The\n"
            "`records_for_pipeline` bridge feeds the current `FeatureExtractor`\n"
            "signature, as in the\n"
            "[MIL walkthrough](walkthrough-slide-mil.ipynb)."
        ),
        code(
            "from soma import Cohort, ImageManifest\n"
            "from soma.data._legacy import records_for_pipeline\n"
            "\n"
            "cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=['label'])\n"
            "dataset = records_for_pipeline(cohort, ImageManifest.from_csv(dataset_csv))\n"
            "print('slides:', len(cohort.sample_ids), '| folds:', cohort.num_folds)"
        ),
        md(
            "## 2. Extract one vector per slide\n"
            "\n"
            "A **slide-level** encoder runs its own tile encoder internally and returns a\n"
            "single vector per slide, so the store holds one vector each — not a bag. We\n"
            "use [moozy-slide](https://huggingface.co/AtlasAnalyticsLab/MOOZY) (built on\n"
            "the ungated `lunit` tile encoder) so this runs on CPU with no token; swap in\n"
            "`titan`, `prism`, or `gigapath-slide` if you have access."
        ),
        code(
            "from hs2p import FilterConfig, SegmentationConfig, TilingConfig\n"
            "from slide2vec import PreprocessingConfig\n"
            "from soma import CacheOptions, EncoderLeaf, FeatureExtractor\n"
            "\n"
            "extractor = FeatureExtractor(\n"
            "    dataset,\n"
            "    EncoderLeaf('moozy-slide', {'allow_non_recommended_settings': True}),\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        tiling=TilingConfig(\n"
            "            requested_spacing_um=0.5, requested_tile_size_px=224,\n"
            "            tolerance=0.05, overlap=0.0, backend='openslide',\n"
            "        ),\n"
            "        segmentation=SegmentationConfig(method='otsu', downsample=16),\n"
            "        filter=FilterConfig(ref_tile_size=224, a_t=1),\n"
            "    ),\n"
            "    cache=CacheOptions(enabled=True, root_dir=str(WORK / 'cache')),\n"
            "    output_root=str(WORK / 'output'),\n"
            ")\n"
            "store = extractor.extract().source\n"
            "vec = store.load(store.sample_ids[0])\n"
            "print('slide-level features: rank', store.rank, '| one vector of dim', tuple(vec.shape))"
        ),
        md(
            "## 3. Train the head — no aggregator\n"
            "\n"
            "The store already holds one vector per slide, so the `mil` learner attaches the\n"
            "task head directly: its params name no aggregator. Everything else — task heads, metrics,\n"
            "and swapping to multiclass/regression/survival on the same store — is exactly\n"
            "as in the [MIL walkthrough](walkthrough-slide-mil.ipynb); only the aggregator\n"
            "is gone."
        ),
        code(
            "from soma import ExperimentConfig, Pipeline\n"
            "from soma.config import (\n"
            "    CohortConfig, EvaluationConfig, LearnerConfig, LoadersConfig, RunConfig, TrainerConfig,\n"
            ")\n"
            "\n"
            "config = ExperimentConfig(\n"
            "    run=RunConfig(output_dir=str(WORK / 'runs' / 'binary'), seed=0),\n"
            "    cohort=CohortConfig(records=dataset_csv, splits=splits_csv),\n"
            "    loaders=LoadersConfig(batch_size=4),\n"
            "    learner=LearnerConfig('mil', {\n"
            "        'head': 'binary_classification',  # <- slide-level features need no aggregator\n"
            "        'optimizer': {'name': 'adam', 'params': {'lr': 1e-3}},\n"
            "    }),\n"
            "    trainer=TrainerConfig(max_epochs=3),\n"
            "    evaluation=EvaluationConfig(metrics=['balanced_accuracy', 'auroc']),\n"
            ")\n"
            "result = Pipeline(config, source=store).run()\n"
            "print('run dir:', result.run_dir)"
        ),
        md(
            "## 4. The one-shot `Pipeline` equivalent\n"
            "\n"
            "The same run as a single config-driven call. Note the absent `aggregator` leaf —\n"
            "the only structural difference from the MIL pipeline config.\n"
            "\n"
            "*(Shown for reference, not executed.)*\n"
            "\n"
            "```python\n"
            "from hs2p import SegmentationConfig, TilingConfig\n"
            "from slide2vec import PreprocessingConfig\n"
            "from soma import CacheOptions, EncoderLeaf, ExperimentConfig, Pipeline\n"
            "from soma.config import (\n"
            "    CohortConfig, EvaluationConfig, FeaturesConfig, LearnerConfig, LoadersConfig,\n"
            "    RunConfig, TrainerConfig,\n"
            ")\n"
            "\n"
            "config = ExperimentConfig(\n"
            "    run=RunConfig(output_dir='output/binary'),\n"
            "    cohort=CohortConfig(records=dataset_csv, splits=splits_csv),\n"
            "    features=FeaturesConfig(\n"
            "        encoder=EncoderLeaf('moozy-slide', {'allow_non_recommended_settings': True}),\n"
            "        extraction=PreprocessingConfig(\n"
            "            tiling=TilingConfig(\n"
            "                requested_spacing_um=0.5, requested_tile_size_px=224,\n"
            "                tolerance=0.05, overlap=0.0, backend='openslide',\n"
            "            ),\n"
            "            segmentation=SegmentationConfig(method='otsu'),\n"
            "        ),\n"
            "        cache=CacheOptions(enabled=True),\n"
            "    ),\n"
            "    loaders=LoadersConfig(batch_size=4),\n"
            "    learner=LearnerConfig('mil', {\n"
            "        'head': 'binary_classification',  # <- slide encoder: no bag to pool\n"
            "        'optimizer': {'name': 'adam', 'params': {'lr': 1e-3}},\n"
            "    }),\n"
            "    trainer=TrainerConfig(max_epochs=3),\n"
            "    evaluation=EvaluationConfig(metrics=['balanced_accuracy', 'auroc']),\n"
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
