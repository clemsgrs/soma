"""Build ``walkthrough-attention-segmentation.ipynb`` (decoder-free segmentation).

Authoring tooling (not part of the tutorial). Mirrors ``_build_dense.py``: it
assembles the **decoder-free attention-probing** segmentation walkthrough
cell-by-cell and writes the result next to this file. This is the per-head
CLS-attention variant of segmentation — the encoder emits a per-head attention
grid (``feature_kind='cls_attention'``) and a swappable per-pixel classifier
replaces the neural decoder.

Outputs are populated **out-of-band on the HPC** (issue #200), not here: the
``NotebookClient(...).execute()`` call is gated behind ``SOMA_EXECUTE_NOTEBOOKS``
and defaults OFF, so running this locally emits the notebook with **empty cell
outputs**. The docs build never re-executes notebooks (``nbsphinx_execute =
"never"``), so empty-output notebooks render and build clean. Flipping the env
flag on the HPC is the one change needed to populate real outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough-attention-segmentation.ipynb"

md = new_markdown_cell
code = new_code_cell


def build() -> nbformat.NotebookNode:
    cells = [
        md(
            "# Attention-based segmentation\n"
            "\n"
            "This notebook walks through soma's **decoder-free** segmentation method —\n"
            "the per-head **CLS-attention** variant of the dense flow:\n"
            "\n"
            "```\n"
            "SegmentationManifest -> FeatureExtractor(cls_attention) -> train (pixel classifier) -> evaluate\n"
            "```\n"
            "\n"
            "It is the alternative to the neural-decoder path in the\n"
            "[segmentation walkthrough](walkthrough-segmentation.ipynb). Same\n"
            "`dataset_type='segmentation'` contract — same manifest, splits, spacing-aware\n"
            "mask reader, dense metrics, and prediction artifacts — but **only the\n"
            "trainable component changes**:\n"
            "\n"
            "* the encoder emits the ViT's **per-head CLS-token self-attention** as a dense\n"
            "  `(K, grid_h, grid_w)` grid (`feature_kind='cls_attention'`) instead of the\n"
            "  patch-feature grid, and\n"
            "* a lightweight **per-pixel classifier** `(K,) → class` (logistic / random\n"
            "  forest / XGBoost / MLP) replaces the neural decoder — **no decoder, no\n"
            "  Trainer, no `.pt` checkpoints**.\n"
            "\n"
            "This re-implements Ramchandani et al., *Benchmarking Computational Pathology\n"
            "Foundation Models for Semantic Segmentation*\n"
            "([arXiv:2602.18747](https://arxiv.org/abs/2602.18747)). The **Pixel-classifier**\n"
            "page in the docs covers the method and the one deliberate divergence\n"
            "(native-window vs resize-to-native extraction).\n"
            "\n"
            "> **Tiny synthetic data, runs on CPU, ungated encoder — the numbers are\n"
            "> meaningless; the point is the API.** We use\n"
            "> [phikon](https://huggingface.co/owkin/phikon) at its native **224 px**\n"
            "> window (a 14×14 token grid), which avoids position-embedding interpolation."
        ),
        md(
            "## ⚠️ Scaffolding (not soma API)\n"
            "\n"
            "Decoder-free segmentation consumes the **same** dense supervision as the\n"
            "neural-decoder path — a per-sample mask, not a scalar `label`:\n"
            "\n"
            "* `dataset.csv` has `sample_id, image_path, label_mask_path`; the mask is an\n"
            "  integer-class raster the same size as the ROI.\n"
            "\n"
            "We fabricate small **224 px ROI tiles** (the dense flow consumes fixed-size\n"
            "tiles/ROIs, not whole WSIs) plus their integer-class masks."
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
            "WORK = Path(tempfile.mkdtemp(prefix='soma-attn-seg-tutorial-'))\n"
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
            "print('segmentation manifest:')\n"
            "print(pd.read_csv(seg_csv).head(3).to_string(index=False))"
        ),
        md(
            "## 1. Extract per-head CLS-attention grids\n"
            "\n"
            "This is the one extraction difference from the neural-decoder path. We point\n"
            "`FeatureExtractor` at `feature_kind='cls_attention'`, so instead of\n"
            "the ViT **patch-feature** grid it captures the **CLS-token self-attention of\n"
            "the chosen block(s)** — one `(grid_h × grid_w)` map **per head**. With phikon\n"
            "at 224 px and patch-16 that's a 14×14 grid; the channel count `K` is\n"
            "`num_blocks × num_heads` (per-head is preserved — head specialization is the\n"
            "signal the classifier exploits).\n"
            "\n"
            "`AttentionConfig(blocks=[-1])` takes the **last** block (the paper's choice);\n"
            "`include_registers=False` keeps only the CLS query row. Everything else — the\n"
            "sliding/stitching, cache, and `DenseFeatureStore` — is shared with the decoder\n"
            "path, because an attention grid is structurally just another `(K, gh, gw)`\n"
            "dense grid."
        ),
        code(
            "from soma import (\n"
            "    SegmentationManifest, FeatureExtractor, EncoderConfig, AttentionConfig,\n"
            "    CacheConfig, PreprocessingConfig,\n"
            ")\n"
            "\n"
            "seg_manifest = SegmentationManifest(seg_csv)\n"
            "extractor = FeatureExtractor(\n"
            "    seg_manifest,\n"
            "    EncoderConfig(name='phikon'),\n"
            "    cache=CacheConfig(enabled=False),\n"
            "    # feature_kind='cls_attention' is what makes this the decoder-free path:\n"
            "    # the encoder emits per-head CLS-attention maps, not the patch grid.\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        feature_kind='cls_attention',\n"
            "        attention=AttentionConfig(blocks=[-1], include_registers=False),\n"
            "        requested_tile_size_px=SIZE,\n"
            "        requested_spacing_um=SPACING,\n"
            "        backend='openslide',\n"
            "    ),\n"
            "    output_root=str(WORK / 'attention'),\n"
            ")\n"
            "attn_store = extractor.extract().source\n"
            "print('attention store ready for', len(attn_store.available_samples), 'ROIs')"
        ),
        md(
            "## 2. Train a per-pixel classifier (no decoder)\n"
            "\n"
            "`train(dataset_type='segmentation', pixel_classifier=...)` selects the\n"
            "**decoder-free** path: it upsamples the per-head attention grid to the mask's\n"
            "resolution and fits a per-pixel classifier `(K,) → class` on\n"
            "class-stratified sampled pixels, then predicts **every** pixel at evaluation.\n"
            "`pixel_classifier` and `decoder` are mutually exclusive under\n"
            "`dataset_type='segmentation'`.\n"
            "\n"
            "We use `logistic` (multinomial logistic regression) because it is the lightest\n"
            "and has no extra dependency; swap in `random_forest`, `xgboost`, or `mlp` by\n"
            "name. The classifiers own their own training loop (no torch `Trainer`, no\n"
            "`.pt` fold checkpoints on this path). `TrainingConfig.max_train_pixels` is the\n"
            "class-stratified pixel budget for the fit."
        ),
        code(
            "from soma import (\n"
            "    Splits, PixelClassifierConfig, TaskConfig, TrainingConfig, EvalConfig, train,\n"
            ")\n"
            "\n"
            "seg_splits = Splits(splits_csv, seg_manifest)\n"
            "\n"
            "seg_result = train(\n"
            "    feature_store=attn_store,\n"
            "    dataset=seg_manifest,\n"
            "    splits=seg_splits,\n"
            "    dataset_type='segmentation',\n"
            "    # the decoder-free trainable component — XOR `decoder=...`\n"
            "    pixel_classifier=PixelClassifierConfig(\n"
            "        name='logistic',                # logistic | random_forest | xgboost | mlp\n"
            "        params={'max_iter': 200},\n"
            "    ),\n"
            "    task=TaskConfig(name='segmentation', params={'num_classes': NUM_CLASSES}),\n"
            "    training=TrainingConfig(epochs=1, batch_size=2, seed=0, max_train_pixels=50_000),\n"
            "    evaluation=EvalConfig(metrics=['mean_dice', 'mean_iou']),\n"
            "    # our PNG masks carry no spacing metadata; declare the ROI spacing so they\n"
            "    # register against the attention grids extracted at the same spacing. The\n"
            "    # cross-default also resolves feature_kind=cls_attention from the classifier.\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        feature_kind='cls_attention', requested_spacing_um=SPACING,\n"
            "    ),\n"
            "    run_dir=str(WORK / 'runs' / 'attention_segmentation'),\n"
            ")\n"
            "print('attention-segmentation run dir:', seg_result.run_dir)"
        ),
        md(
            "## 3. The one-shot `Pipeline` equivalent\n"
            "\n"
            "As with the other walkthroughs, `Pipeline` collapses extract + train +\n"
            "evaluate into a single config-driven call. The only things that select the\n"
            "decoder-free path are `preprocessing.feature_kind='cls_attention'` (auto when a\n"
            "`pixel_classifier` is set) and naming a `pixel_classifier` instead of a\n"
            "`decoder`.\n"
            "\n"
            "*(Shown for reference, not executed.)*\n"
            "\n"
            "```python\n"
            "from soma import (\n"
            "    Pipeline, PipelineConfig, PreprocessingConfig, EncoderConfig, AttentionConfig,\n"
            "    PixelClassifierConfig, TaskConfig, TrainingConfig, EvalConfig, CacheConfig,\n"
            ")\n"
            "\n"
            "config = PipelineConfig(\n"
            "    dataset_csv=str(seg_csv),\n"
            "    splits_csv=str(splits_csv),\n"
            "    output_root='output/attention-segmentation',\n"
            "    dataset_type='segmentation',\n"
            "    preprocessing=PreprocessingConfig(\n"
            "        backend='openslide', requested_tile_size_px=224, requested_spacing_um=0.5,\n"
            "        feature_kind='cls_attention',         # auto when pixel_classifier is set\n"
            "        attention=AttentionConfig(blocks=[-1], include_registers=False),\n"
            "        dense_window_size=None,               # null=whole; =native input for native-window mode\n"
            "    ),\n"
            "    encoder=EncoderConfig(name='phikon'),\n"
            "    pixel_classifier=PixelClassifierConfig(name='logistic'),   # XOR decoder=...\n"
            "    task=TaskConfig(name='segmentation', params={'num_classes': 3}),\n"
            "    training=TrainingConfig(max_train_pixels=2_000_000),\n"
            "    evaluation=EvalConfig(metrics=['mean_dice', 'mean_iou']),\n"
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
    # Outputs are populated out-of-band on the HPC (issue #200): execution is gated
    # behind SOMA_EXECUTE_NOTEBOOKS and defaults OFF, so locally this writes the
    # notebook with EMPTY cell outputs. The docs build never re-executes notebooks
    # (nbsphinx_execute = "never"), so empty-output notebooks render and build clean.
    if os.environ.get("SOMA_EXECUTE_NOTEBOOKS"):
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
