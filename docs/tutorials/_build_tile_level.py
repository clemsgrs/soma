"""Build ``walkthrough-tile-level.ipynb``.

This is *authoring* tooling, not part of the tutorial. It assembles the notebook
cell-by-cell and writes it next to this file. Unlike the slide-level builder it
does **not** execute the notebook — the committed outputs are refreshed by
``scripts/execute_tutorials.sh`` (which runs the ``.ipynb`` in place against the
repo source with the ungated ``phikon`` encoder). This builder only needs to run
when the cell *content* changes.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-tile-level.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Tile-level classification\n"
            "\n"
            "This notebook walks through soma's **tile-level dataset path** — the one\n"
            "behind patch-classification benchmarks like EVA — end to end:\n"
            "\n"
            "```\n"
            "Dataset  ->  TileFeatureExtractor  ->  train (head)  ->  evaluate\n"
            "```\n"
            "\n"
            "Here every sample is **one already-cropped tile image with a class label**,\n"
            "not a whole-slide image. That changes the shape of the pipeline versus the\n"
            "[slide-level MIL walkthrough](walkthrough-slide-mil.ipynb):\n"
            "\n"
            "* **No tissue masking or tiling.** Your inputs are already tiles, so there is\n"
            "  no `PreprocessingConfig` — soma reads each `image_path` directly.\n"
            "* **No MIL aggregator.** A tile encoder maps each tile to a single vector, so\n"
            "  the task head consumes that vector as-is (`aggregator=None`). There is no\n"
            "  bag to pool.\n"
            "\n"
            "What stays the same is soma's core property: features don't depend on the\n"
            "labels, so a **binary** and a **multiclass** head are just a different\n"
            "`TaskConfig` on the **same extracted features**. The last cell collapses the\n"
            "whole thing into a single `Pipeline` call.\n"
            "\n"
            "> **This uses a tiny synthetic dataset so it runs anywhere on CPU with no\n"
            "> gated model and no real slides.** The numbers are therefore meaningless —\n"
            "> the point is the API, not the result."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "The cell below fabricates a toy dataset: a handful of small RGB tile images\n"
            "plus the two CSVs soma expects. **You would replace this with your own tiles\n"
            "and labels** — the only thing that matters downstream is the on-disk\n"
            "contract, which is identical to the slide-level one:\n"
            "\n"
            "* `dataset.csv` — one row per **tile** with `sample_id`, `image_path`, `label`\n"
            "  (extra columns are kept as free metadata; you pick which one is `label`).\n"
            "  The only difference from the slide path is that `image_path` points at an\n"
            "  individual tile image, not a whole slide.\n"
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
            "from PIL import Image\n"
            "\n"
            "WORK = Path(tempfile.mkdtemp(prefix='soma-tile-tutorial-'))\n"
            "TILES = WORK / 'tiles'; TILES.mkdir()\n"
            "rng = np.random.default_rng(0)\n"
            "\n"
            "def make_toy_tile(path, label, size=224):\n"
            "    \"\"\"A 224x224 RGB tile whose color tint is a weak, noisy signal for the\n"
            "    label — enough for the API to run, not enough to mean anything.\"\"\"\n"
            "    base = np.array([[150, 70, 160], [70, 150, 90]][label % 2], np.int16)\n"
            "    img = base + rng.integers(-40, 40, (size, size, 3))\n"
            "    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)\n"
            "\n"
            "N = 8\n"
            "sample_ids = [f't{i:02d}' for i in range(N)]\n"
            "\n"
            "# train/tune/test assignment (single fold) with both classes in every split\n"
            "split = (['train'] * 4) + (['tune'] * 2) + (['test'] * 2)\n"
            "binary = [0, 1, 0, 1,  0, 1,  0, 1]\n"
            "\n"
            "for sid, y in zip(sample_ids, binary):\n"
            "    make_toy_tile(TILES / f'{sid}.png', y)\n"
            "\n"
            "dataset_csv = WORK / 'dataset.csv'\n"
            "splits_csv = WORK / 'splits.csv'\n"
            "pd.DataFrame({'sample_id': sample_ids,\n"
            "              'image_path': [str(TILES / f'{s}.png') for s in sample_ids],\n"
            "              'label': binary}).to_csv(dataset_csv, index=False)\n"
            "pd.DataFrame({'sample_id': sample_ids, 'split': split}).to_csv(splits_csv, index=False)\n"
            "\n"
            "print('dataset.csv'); print(pd.read_csv(dataset_csv).head().to_string(index=False))\n"
            "print('\\nsplits.csv'); print(pd.read_csv(splits_csv)['split'].value_counts().to_string())"
        ),
        md(
            "## 1. Load the dataset and splits\n"
            "\n"
            "`Dataset` reads `dataset.csv` and infers the label space; `Splits` reads\n"
            "`splits.csv` and pairs it with the dataset. This is identical to every other\n"
            "path — soma never partitions your data, so the splits you provide are the\n"
            "splits it uses, which is what keeps evaluation reproducible and leakage-free."
        ),
        code(
            "from soma import Dataset, Splits\n"
            "\n"
            "dataset = Dataset(dataset_csv)\n"
            "splits = Splits(splits_csv, dataset)\n"
            "print('tiles:   ', len(dataset.sample_ids))\n"
            "print('classes: ', sorted(pd.read_csv(dataset_csv)['label'].unique()))\n"
            "print('folds:   ', splits.num_folds)"
        ),
        md(
            "## 2. Encode each tile to a vector\n"
            "\n"
            "This is where the tile path diverges from slide-level. There is no tissue\n"
            "masking and no tiling — the inputs are already tiles — so we use\n"
            "**`TileFeatureExtractor`** (not `FeatureExtractor`) and pass **no**\n"
            "`PreprocessingConfig`. It loads each `image_path`, applies the encoder's\n"
            "transform, and writes **one 1-D vector per tile**.\n"
            "\n"
            "We use [phikon](https://huggingface.co/owkin/phikon) because it is\n"
            "**ungated** (no HF token) and small enough to run on CPU. With\n"
            "`CacheConfig(enabled=True)` the vectors are cached so the second task below\n"
            "reuses this one extraction. `run()` returns the `FeatureStore` directly."
        ),
        code(
            "from soma import TileFeatureExtractor, EncoderConfig, CacheConfig\n"
            "\n"
            "extractor = TileFeatureExtractor(\n"
            "    dataset,\n"
            "    EncoderConfig(name='phikon'),\n"
            "    cache=CacheConfig(enabled=True, root_dir=str(WORK / 'cache')),\n"
            ")\n"
            "store = extractor.run(str(WORK / 'output' / 'features'))\n"
            "\n"
            "vec = store.load(store.available_samples[0])\n"
            "print('encoded', len(store.available_samples), 'tiles')\n"
            "print('one vector per tile:', store.is_slide_level, '| shape', tuple(vec.shape))"
        ),
        md(
            "## 3. Train a classifier head directly (no aggregator)\n"
            "\n"
            "`train()` consumes the `FeatureStore` and trains a **task head** straight on\n"
            "the per-tile vectors. Because each sample is already one vector, we pass\n"
            "`aggregator=None` and `dataset_type='tile'` — there is no bag to pool.\n"
            "`TaskConfig(name='binary_classification')` picks the head, loss, and metrics."
        ),
        code(
            "from soma import TaskConfig, TrainingConfig, EvalConfig, train\n"
            "\n"
            "training = TrainingConfig(epochs=3, learning_rate=1e-3, batch_size=4, seed=0)\n"
            "\n"
            "result = train(\n"
            "    feature_store=store,\n"
            "    dataset=dataset,\n"
            "    splits=splits,\n"
            "    aggregator=None,          # <- tile features need no aggregator\n"
            "    dataset_type='tile',\n"
            "    task=TaskConfig(name='binary_classification'),\n"
            "    training=training,\n"
            "    evaluation=EvalConfig(metrics=['balanced_accuracy', 'auroc']),\n"
            "    run_dir=str(WORK / 'runs' / 'binary'),\n"
            ")\n"
            "print('run dir:', result.run_dir)"
        ),
        md(
            "## 4. Swap the task — same features, new head\n"
            "\n"
            "Features don't depend on the labels, so we **reuse the same `FeatureStore`**\n"
            "and only change the `Dataset` labels + `TaskConfig` + metrics. No\n"
            "re-extraction — the cached vectors from step 2 are reused as-is."
        ),
        code(
            "def relabel(values):\n"
            "    df = pd.DataFrame({'sample_id': sample_ids,\n"
            "                       'image_path': [str(TILES / f'{s}.png') for s in sample_ids],\n"
            "                       'label': values})\n"
            "    path = WORK / f'dataset_{abs(hash(tuple(map(str, values)))) % 10**6}.csv'\n"
            "    df.to_csv(path, index=False)\n"
            "    return Dataset(path)\n"
            "\n"
            "multiclass = [0, 1, 2, 0,  1, 2,  0, 1]\n"
            "ds_mc = relabel(multiclass)\n"
            "result_mc = train(\n"
            "    feature_store=store, dataset=ds_mc, splits=Splits(splits_csv, ds_mc),\n"
            "    aggregator=None, dataset_type='tile',\n"
            "    task=TaskConfig(name='multiclass_classification'),\n"
            "    training=training, evaluation=EvalConfig(metrics=['balanced_accuracy', 'accuracy']),\n"
            "    run_dir=str(WORK / 'runs' / 'multiclass'),\n"
            ")\n"
            "print('multiclass run dir:', result_mc.run_dir)"
        ),
        md(
            "## 5. The one-shot `Pipeline` equivalent\n"
            "\n"
            "Everything above — encode, train, evaluate — is what `Pipeline` does in a\n"
            "single call from one config. Note what is **absent** versus the slide-level\n"
            "config: no `preprocessing` block and no `aggregator`. Setting\n"
            "`dataset_type='tile'` is what routes the pipeline through\n"
            "`TileFeatureExtractor` and the single-vector head.\n"
            "\n"
            "*(Shown for reference, not executed — it would repeat the work above.)*\n"
            "\n"
            "```python\n"
            "from soma import (\n"
            "    Pipeline, PipelineConfig, EncoderConfig,\n"
            "    TaskConfig, TrainingConfig, EvalConfig, CacheConfig,\n"
            ")\n"
            "\n"
            "config = PipelineConfig(\n"
            "    dataset_csv=str(dataset_csv),\n"
            "    splits_csv=str(splits_csv),\n"
            "    output_root='output/tile-binary',\n"
            "    dataset_type='tile',       # <- routes through TileFeatureExtractor, no aggregator\n"
            "    encoder=EncoderConfig(name='phikon'),\n"
            "    task=TaskConfig(name='binary_classification'),\n"
            "    training=TrainingConfig(epochs=3, learning_rate=1e-3, batch_size=4),\n"
            "    evaluation=EvalConfig(metrics=['balanced_accuracy', 'auroc']),\n"
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
    # Reachable via :doc: links from the tutorial hub pages, but kept out of the
    # sidebar nav (the hub pages are the nav entries).
    nb.metadata["nbsphinx"] = {"orphan": True}
    return nb


def main() -> None:
    # Unlike the slide-level builder, we do not execute here: outputs are trimmed
    # and refreshed via scripts/execute_tutorials.sh against the repo source.
    nbformat.write(build(), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
