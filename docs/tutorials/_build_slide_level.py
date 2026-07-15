"""Build + execute ``walkthrough-slide-level.ipynb`` with committed outputs.

This is *authoring* tooling, not part of the tutorial. It assembles the notebook
cell-by-cell, executes it on CPU against the repo source, and writes the result
next to this file. Refresh via ``scripts/execute_tutorials.sh`` (which re-executes
the committed ``.ipynb`` directly); this builder only needs to run when the cell
*content* changes.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-slide-level.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Slide-level\n"
            "\n"
            "This notebook walks through soma's **modular building-block API** for a\n"
            "slide-level prediction task, end to end:\n"
            "\n"
            "```\n"
            "Dataset  ->  FeatureExtractor  ->  train (aggregator + head)  ->  evaluate\n"
            "```\n"
            "\n"
            "We build a slide-level representation two ways — a **tile encoder + MIL\n"
            "aggregator** (Path A) and a **slide-level encoder with no aggregator**\n"
            "(Path B) — then show that switching the task to multiclass, regression, or\n"
            "survival is just a different `TaskConfig` on the **same extracted features**,\n"
            "the property soma is built around. The last cell shows how the whole thing\n"
            "collapses into a single `Pipeline` call.\n"
            "\n"
            "> **This uses a tiny synthetic dataset so it runs anywhere on CPU with no\n"
            "> gated model and no real slides.** The numbers are therefore meaningless —\n"
            "> the point is the API, not the result."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "The cell below fabricates a toy dataset: a handful of small tissue-like\n"
            "`.tif` slides plus the two CSVs soma expects. **You would replace this with\n"
            "your own slides and labels** — the only thing that matters downstream is the\n"
            "on-disk contract:\n"
            "\n"
            "* `dataset.csv` — one row per slide with `sample_id`, `image_path`, `label`\n"
            "  (extra label columns are fine; you pick which one is `label`).\n"
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
            "WORK = Path(tempfile.mkdtemp(prefix='soma-slide-tutorial-'))\n"
            "SLIDES = WORK / 'slides'; SLIDES.mkdir()\n"
            "rng = np.random.default_rng(0)\n"
            "\n"
            "def make_toy_slide(path, size=640):\n"
            "    \"\"\"A white background with a central H&E-ish blob, saved as a tiled\n"
            "    TIFF whose resolution tags make OpenSlide report 0.5 microns/pixel.\"\"\"\n"
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
            "print('dataset.csv'); print(pd.read_csv(dataset_csv).head().to_string(index=False))\n"
            "print('\\nsplits.csv'); print(pd.read_csv(splits_csv)['split'].value_counts().to_string())"
        ),
        md(
            "## 1. Load the dataset and splits\n"
            "\n"
            "`Dataset` reads `dataset.csv` and infers the label space; `Splits` reads\n"
            "`splits.csv` and pairs it with the dataset. soma never partitions your data —\n"
            "the splits you provide are the splits it uses, which is what keeps evaluation\n"
            "reproducible and leakage-free."
        ),
        code(
            "from soma import Dataset, Splits\n"
            "\n"
            "dataset = Dataset(dataset_csv)\n"
            "splits = Splits(splits_csv, dataset)\n"
            "print('slides:  ', len(dataset.sample_ids))\n"
            "print('classes: ', sorted(pd.read_csv(dataset_csv)['label'].unique()))\n"
            "print('folds:   ', splits.num_folds)"
        ),
        md(
            "## 2. Two ways to a slide-level representation\n"
            "\n"
            "A prediction needs **one vector per slide**. soma supports two routes, and\n"
            "you choose by the encoder you name:\n"
            "\n"
            "* **Path A — a tile encoder + an aggregator.** A tile-level foundation model\n"
            "  embeds each tile into a *bag* of vectors, then a trainable MIL\n"
            "  **aggregator** (e.g. attention-MIL) pools the bag into a slide vector.\n"
            "* **Path B — a slide-level encoder.** A slide-level model emits the slide\n"
            "  vector directly, so there is **no aggregator** (`aggregator=None`).\n"
            "\n"
            "We run both below on the same dataset and splits.\n"
            "\n"
            "### Path A — tile encoder (`phikon`) + MIL aggregator\n"
            "\n"
            "`FeatureExtractor` runs preprocessing (tissue masking + tiling) and then a\n"
            "frozen tile encoder over each tile. We use\n"
            "[`phikon`](https://huggingface.co/owkin/phikon) because it is **ungated** (no\n"
            "HF token) and small enough to run on CPU. `extract()` writes a bag of tile\n"
            "embeddings per slide; with `CacheConfig(enabled=True)` they are cached so\n"
            "every experiment below reuses this one extraction."
        ),
        code(
            "from soma import FeatureExtractor, FeatureStore, EncoderConfig, PreprocessingConfig, CacheConfig\n"
            "\n"
            "feature_dir = WORK / 'output' / 'features'\n"
            "extractor = FeatureExtractor(\n"
            "    dataset,\n"
            "    EncoderConfig(name='phikon'),\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide',\n"
            "        requested_tile_size_px=224,\n"
            "        requested_spacing_um=0.5,\n"
            "        tissue_method='otsu',   # otsu | hsv | threshold | sam2\n"
            "        tolerance=0.07,\n"
            "        # Our toy slides are small; a fine segmentation downsample + low\n"
            "        # area threshold keep the tissue mask usable (defaults assume WSIs).\n"
            "        seg_downsample=16,\n"
            "        a_t=1,\n"
            "    ),\n"
            "    cache=CacheConfig(enabled=True, root_dir=str(WORK / 'cache')),\n"
            "    output_root=str(WORK / 'output'),\n"
            ")\n"
            "\n"
            "extractor.extract(feature_dir='features')\n"
            "store = FeatureStore(str(feature_dir))   # the embeddings written above\n"
            "print('extracted feature bags for', len(store.available_samples), 'slides')"
        ),
        md(
            "Now `train()` consumes the bag `FeatureStore` and trains a MIL **aggregator**\n"
            "(here attention-MIL, `abmil`) plus a **task head**. The aggregator turns each\n"
            "slide's bag of tile features into one slide-level vector; the head maps that\n"
            "to a prediction. `TaskConfig(name='binary_classification')` picks the head,\n"
            "loss, and default metrics."
        ),
        code(
            "from soma import AggregatorConfig, TaskConfig, TrainingConfig, EvalConfig, train\n"
            "\n"
            "training = TrainingConfig(epochs=3, learning_rate=1e-3, batch_size=4, seed=0)\n"
            "\n"
            "result = train(\n"
            "    feature_store=store,\n"
            "    dataset=dataset,\n"
            "    splits=splits,\n"
            "    aggregator=AggregatorConfig(name='abmil'),\n"
            "    task=TaskConfig(name='binary_classification'),\n"
            "    training=training,\n"
            "    evaluation=EvalConfig(metrics=['balanced_accuracy', 'auroc']),\n"
            "    run_dir=str(WORK / 'runs' / 'binary_tile_abmil'),\n"
            ")\n"
            "print('run dir:', result.run_dir)"
        ),
        md(
            "## 3. Path B — a slide-level encoder (no aggregator)\n"
            "\n"
            "Some foundation models are **slide-level**: they run their own tile encoder\n"
            "internally and return one vector per slide, so you skip the aggregator\n"
            "entirely. We use [`moozy-slide`](https://huggingface.co/AtlasAnalyticsLab/MOOZY)\n"
            "(built on the ungated `lunit` tile encoder) so this runs on CPU with no token;\n"
            "swap in `titan`, `prism`, or `gigapath-slide` if you have access.\n"
            "\n"
            "The only API differences from Path A: the encoder name, and\n"
            "`aggregator=None` in `train()` (there is no bag to pool — the store already\n"
            "holds one vector per slide)."
        ),
        code(
            "slide_feature_dir = WORK / 'output' / 'features_slide'\n"
            "slide_extractor = FeatureExtractor(\n"
            "    dataset,\n"
            "    EncoderConfig(name='moozy-slide', allow_non_recommended_settings=True),\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide', requested_tile_size_px=224, requested_spacing_um=0.5,\n"
            "        tissue_method='otsu', tolerance=0.07, seg_downsample=16, a_t=1,\n"
            "    ),\n"
            "    cache=CacheConfig(enabled=True, root_dir=str(WORK / 'cache')),\n"
            "    output_root=str(WORK / 'output'),\n"
            ")\n"
            "slide_extractor.extract(feature_dir='features_slide')\n"
            "slide_store = FeatureStore(str(slide_feature_dir))\n"
            "print('slide-level features:', slide_store.is_slide_level,\n"
            "      '| one vector per slide of dim',\n"
            "      tuple(slide_store.load(slide_store.available_samples[0]).shape))\n"
            "\n"
            "slide_result = train(\n"
            "    feature_store=slide_store,\n"
            "    dataset=dataset,\n"
            "    splits=splits,\n"
            "    aggregator=None,   # <- slide-level features need no aggregator\n"
            "    task=TaskConfig(name='binary_classification'),\n"
            "    training=training,\n"
            "    evaluation=EvalConfig(metrics=['balanced_accuracy', 'auroc']),\n"
            "    run_dir=str(WORK / 'runs' / 'binary_slide_noagg'),\n"
            ")\n"
            "print('run dir:', slide_result.run_dir)"
        ),
        md(
            "## 4. Swap the task — same features, new head\n"
            "\n"
            "This is the payoff of the modular design. Features don't depend on the\n"
            "labels, so we **reuse Path A's tile `FeatureStore`** (`store`) and only change\n"
            "the `Dataset` labels + `TaskConfig` + metrics. No re-extraction. (Each of\n"
            "these works identically on Path B's `slide_store` with `aggregator=None`.)\n"
            "\n"
            "### Multiclass classification"
        ),
        code(
            "def relabel(values, **extra_cols):\n"
            "    df = pd.DataFrame({'sample_id': sample_ids,\n"
            "                       'image_path': [str(SLIDES / f'{s}.tif') for s in sample_ids],\n"
            "                       'label': values, **extra_cols})\n"
            "    path = WORK / f'dataset_{abs(hash(tuple(map(str, values)))) % 10**6}.csv'\n"
            "    df.to_csv(path, index=False)\n"
            "    return Dataset(path)\n"
            "\n"
            "multiclass = [0, 1, 2, 3,  0, 1,  2, 3]\n"
            "ds_mc = relabel(multiclass)\n"
            "result_mc = train(\n"
            "    feature_store=store, dataset=ds_mc, splits=Splits(splits_csv, ds_mc),\n"
            "    aggregator=AggregatorConfig(name='abmil'),\n"
            "    task=TaskConfig(name='multiclass_classification'),\n"
            "    training=training, evaluation=EvalConfig(metrics=['balanced_accuracy']),\n"
            "    run_dir=str(WORK / 'runs' / 'multiclass'),\n"
            ")\n"
            "print('multiclass run dir:', result_mc.run_dir)"
        ),
        md("### Regression"),
        code(
            "targets = rng.uniform(0, 10, size=N).round(2)\n"
            "ds_reg = relabel(targets)\n"
            "result_reg = train(\n"
            "    feature_store=store, dataset=ds_reg, splits=Splits(splits_csv, ds_reg),\n"
            "    aggregator=AggregatorConfig(name='abmil'),\n"
            "    task=TaskConfig(name='regression'),\n"
            "    training=training, evaluation=EvalConfig(metrics=['mae', 'r2']),\n"
            "    run_dir=str(WORK / 'runs' / 'regression'),\n"
            ")\n"
            "print('regression run dir:', result_reg.run_dir)"
        ),
        md(
            "### Survival (Cox)\n"
            "\n"
            "Time-to-event reuses `label` as the event/censoring **time** and adds an\n"
            "`event` column (1 = event observed, 0 = censored). `TaskConfig('survival')`\n"
            "with `loss='cox'` selects continuous-time CoxPH, where the risk set is the\n"
            "batch — so `batch_size >= 2`."
        ),
        code(
            "times = rng.uniform(1, 100, size=N).round(1)\n"
            "events = [1, 0, 1, 1,  1, 0,  1, 1]\n"
            "ds_surv = relabel(times, event=events)\n"
            "result_surv = train(\n"
            "    feature_store=store, dataset=ds_surv, splits=Splits(splits_csv, ds_surv),\n"
            "    aggregator=AggregatorConfig(name='abmil'),\n"
            "    task=TaskConfig(name='survival', params={'loss': 'cox', 'min_events_per_window': 1}),\n"
            "    training=TrainingConfig(epochs=3, learning_rate=1e-3, batch_size=4, seed=0),\n"
            "    evaluation=EvalConfig(metrics=['c_index']),\n"
            "    run_dir=str(WORK / 'runs' / 'survival'),\n"
            ")\n"
            "print('survival run dir:', result_surv.run_dir)"
        ),
        md(
            "## 5. The one-shot `Pipeline` equivalent\n"
            "\n"
            "Everything above — preprocess, extract, train, evaluate — is what\n"
            "`Pipeline` does in a single call from one config. The building blocks are for\n"
            "when you want to reuse features across many experiments (as we just did); the\n"
            "`Pipeline` is for when you just want the result.\n"
            "\n"
            "*(Shown for reference, not executed — it would repeat all the work above.)*\n"
            "\n"
            "```python\n"
            "from soma import (\n"
            "    Pipeline, PipelineConfig, PreprocessingConfig, EncoderConfig,\n"
            "    AggregatorConfig, TaskConfig, TrainingConfig, EvalConfig, CacheConfig,\n"
            ")\n"
            "\n"
            "config = PipelineConfig(\n"
            "    dataset_csv=str(dataset_csv),\n"
            "    splits_csv=str(splits_csv),\n"
            "    output_root='output/binary',\n"
            "    dataset_type='slide',\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide', requested_tile_size_px=224,\n"
            "        requested_spacing_um=0.5, tissue_method='otsu', tolerance=0.07,\n"
            "    ),\n"
            "    encoder=EncoderConfig(name='phikon'),\n"
            "    aggregator=AggregatorConfig(name='abmil'),\n"
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
    nb = build()
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    nbformat.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
