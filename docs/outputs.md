# Outputs

This project writes two different kinds of outputs:

- reusable feature-cache artifacts
- training run outputs

## Feature Extraction Outputs

With caching enabled, embeddings are written under the shared cache root rather than the run-local feature directory.

If caching is disabled, `extract()` writes embeddings directly into the requested output directory.

When caching is enabled, the requested run-local feature directory is left with a small `README.txt` marker so it is clear that the real payload lives in the shared cache.

The run-local feature directory also gets a `process_list.csv` with one row per sample.
It records the resolved `feature_path` plus canonical feature provenance columns:
`encoder_name`, `output_variant`, and `feature_kind`.

For uncached multi-GPU extraction, `process_list.csv` is synchronized back from the tiling directory into the feature directory after embedding so the feature output tree remains self-contained.

Typical locations:

```text
output/
├── tiling/                 # preprocessing / tiling artifacts
├── features/               # run-local feature directory when cache is disabled
└── feature_cache/          # shared cache when cache is enabled
```

When you use the `FeatureExtractor.run(feature_dir="output/features")` convenience form, `tiling/` is created as a sibling of `features/`, not nested inside it.

## Training Outputs

`Pipeline.run()` resolves a managed run directory from a user-provided `output_root`.

```text
<output_root>/
├── experiments/
│   └── <dataset>-<encoder>-<aggregator>-<task_type>_<short_hash>/
│       ├── experiment.yaml
│       ├── experiment.json
│       ├── runs/
│       │   └── <run_id>/
│       │       ├── config.yaml
│       │       ├── run.yaml
│       │       ├── fold_0/
│       │       │   ├── best_model.pt
│       │       │   ├── metrics.json
│       │       │   ├── predictions.csv
│       │       │   ├── attention/          # present when heatmaps.enabled = true
│       │       │   │   ├── <sample_id>.npz
│       │       │   │   └── ...
│       │       │   └── heatmaps/           # present when heatmaps.enabled = true
│       │       │       ├── <sample_id>.png
│       │       │       └── ...
│       │       ├── fold_1/
│       │       │   └── ...
│       │       └── summary.json
│       └── latest
└── indexes/
    ├── experiments.csv
    └── runs.csv
```

## Attention Heatmaps

When `heatmaps.enabled = true`, two additional subdirectories are written inside each fold directory.

**`attention/`** — per-tile attention scores extracted from the best checkpoint, one `.npz` file per test sample. These are saved before rendering so heatmaps can be re-generated later with different visual settings (colormap, alpha, blur) without re-running inference:

| Aggregator | Array shape |
|---|---|
| ABMIL, CLAM-SB, DSMIL | `(N,)` |
| CLAM-MB | `(n_classes, N)` |

**`heatmaps/`** — PNG images composited over a WSI thumbnail at the `seg_downsample` resolution (same pyramid level used for preprocessing previews):

| Aggregator | Files |
|---|---|
| ABMIL, CLAM-SB, DSMIL | `<sample_id>.png` |
| CLAM-MB | `<sample_id>_class_0.png`, `<sample_id>_class_1.png`, … |

HIPT, TransMIL, MeanPool, and MaxPool produce no attention scores and are skipped silently.

To re-render heatmaps from saved attention without re-running inference:

```python
from soma.heatmaps import render_heatmaps
from soma.config import HeatmapConfig

render_heatmaps(
    run_dir="experiments/.../runs/<run_id>",
    dataset=dataset,
    feature_store=store,
    heatmap_config=HeatmapConfig(cmap="viridis", alpha=0.4),
    seg_downsample=32,
)
```

`predictions.csv` format depends on the task head:

- **Classification**: columns `sample_id`, `true_label`, `predicted_label`, `prob_0`, `prob_1`, ...
- **Regression**: columns `sample_id`, `true_label`, `predicted_value`

`metrics.json` contains tune and test metrics for each fold. `summary.json` aggregates per-metric mean and std across folds.

`run.yaml` records run provenance such as seed, timestamps, git metadata, and summary metrics. `indexes/runs.csv` and `indexes/experiments.csv` provide convenience views over the per-run and per-experiment metadata.

## FeatureStore Paths

`FeatureStore` accepts either:

- a plain directory of `.pt` files,
- a cache directory containing `cache_metadata.json` and `features/`,
- or a native slide2vec artifact root such as `tile_embeddings/`.

This is why `FeatureStore("output/feature_cache/tile/<hash>")` works without pointing at the nested `features/` directory directly.
