# Outputs

This project writes two different kinds of outputs.

## Feature Extraction Outputs

With caching enabled, embeddings are written under the shared cache root rather than the run-local feature directory.

If caching is disabled, `extract()` writes embeddings directly into the requested output directory.

When caching is enabled, the requested run-local feature directory is left with a small `README.txt` marker so it is clear that the real payload lives in the shared cache.

The run-local feature directory also gets a `process_list.csv` with one row per sample and a `feature_path` column that points at the resolved cached `.pt` file.

For uncached multi-GPU extraction, `process_list.csv` is synchronized back from the tiling directory into the feature directory after embedding so the feature output tree remains self-contained.

Typical locations:

```text
output/
├── tiling/                 # preprocessing / tiling artifacts
├── features/               # run-local feature directory when cache is disabled
└── feature_cache/          # shared cache when cache is enabled
```

## Training Outputs

`Pipeline.run()` writes experiment artifacts to `output_dir`.

```text
experiments/run1/
├── config.yaml
├── fold_0/
│   ├── best_model.pt
│   ├── metrics.json
│   └── predictions.csv
├── fold_1/
│   └── ...
└── summary.json
```

`predictions.csv` format depends on the task head:

- **Classification**: columns `sample_id`, `true_label`, `predicted_label`, `prob_0`, `prob_1`, ...
- **Regression**: columns `sample_id`, `true_label`, `predicted_value`

`metrics.json` contains tune and test metrics for each fold. `summary.json` aggregates per-metric mean and std across folds.

## FeatureStore Paths

`FeatureStore` accepts either:

- a plain directory of `.pt` files,
- a cache directory containing `cache_metadata.json` and `features/`,
- or a native slide2vec artifact root such as `tile_embeddings/`.

This is why `FeatureStore("output/feature_cache/tile/<hash>")` works without pointing at the nested `features/` directory directly.
