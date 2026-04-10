# Workflow

`soma` is organized around two stages:

1. Feature extraction from whole-slide images.
2. Training and evaluation from precomputed features.

## Extraction

Use `FeatureExtractor` when you want to preprocess slides and write embeddings to disk.

```python
from soma import Dataset, FeatureExtractor, EncoderConfig, PreprocessingConfig

dataset = Dataset("dataset.csv")
extractor = FeatureExtractor(
    dataset=dataset,
    encoder=EncoderConfig(name="uni2"),
    output_root="output",
    preprocessing=PreprocessingConfig(
        backend="openslide",
        target_tile_size_px=224,
        target_spacing_um=0.5,
    ),
)

extractor.preprocess(tiling_dir="output/tiling")
store = extractor.extract(feature_dir="output/features", tiling_dir="output/tiling")
```

`extractor.run(feature_dir="output/features")` is the convenience form for the same workflow.
It writes tiling artifacts to the sibling `output/tiling/` directory rather than nesting them under `features/`.
If you do not pass a backend argument to `preprocess()`, `extract()`, or `run()`,
the value from `PreprocessingConfig.backend` is used.
That field is the requested backend; the actual backend selected during tiling is
recorded on the loaded tiling result.

## Training

Use `Pipeline` when you want extraction and training coordinated from a single config.

```python
from soma import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    Pipeline,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_root="experiments",
    cache=CacheConfig(root_dir="shared/feature_cache"),
    encoder=EncoderConfig(name="uni2"),
    preprocessing=PreprocessingConfig(backend="openslide", target_tile_size_px=224, target_spacing_um=0.5),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="classification"),
    training=TrainingConfig(epochs=50),
)

result = Pipeline(config).run()
```

If you already have a feature store on disk, you can bypass extraction and call `train()` directly.

## Task Configuration

`task` is a required field in `PipelineConfig`. Set it to `classification` or `regression`:

```python
task=TaskConfig(name="classification")
task=TaskConfig(name="regression")
```

Or in YAML:
```yaml
task:
  name: classification   # or: regression
```

For regression, use float values in the `label` column of `dataset.csv`. For classification, `num_classes` is auto-inferred from the dataset.

## Attention Heatmaps

Set `heatmaps.enabled = true` in the pipeline config to automatically generate tile-level attention heatmaps after training. Heatmaps are produced for the test split of each fold and saved inside the run directory.

```python
from soma import Pipeline, PipelineConfig, HeatmapConfig

config = PipelineConfig(
    ...
    heatmaps=HeatmapConfig(enabled=True, cmap="coolwarm", alpha=0.5),
)

Pipeline(config).run()
```

Or in YAML:

```yaml
heatmaps:
  enabled: true
  cmap: coolwarm
  alpha: 0.5
  blur_sigma: 0.0
```

Generation is split into two phases. First, attention scores are extracted from `best_model.pt` and saved as `fold_N/attention/<sample_id>.npz`. Then heatmaps are rendered by overlaying attention-colored tiles on a WSI thumbnail at the `seg_downsample` pyramid level. Because the two steps are decoupled, you can re-render with different visual settings (colormap, alpha, blur) without re-running inference:

```python
from soma.heatmaps import render_heatmaps
from soma.config import HeatmapConfig

render_heatmaps(
    run_dir="experiments/.../runs/<run_id>",
    dataset=dataset,
    feature_store=store,
    heatmap_config=HeatmapConfig(cmap="viridis", alpha=0.4, blur_sigma=1.5),
    seg_downsample=32,
)
```

Supported aggregators: ABMIL, CLAM-SB, CLAM-MB, DSMIL. HIPT, TransMIL, MeanPool, and MaxPool are skipped automatically.

## Common Paths

- `preprocess()` writes tiling artifacts.
- `extract()` writes embeddings and returns a `FeatureStore`.
- Lower-level APIs use specific destination names like `tiling_dir`, `feature_dir`, `fold_dir`, and `run_dir` instead of a generic `output_dir`.
- `Pipeline.run()` writes training outputs under a managed experiment/run directory inside `output_root`.
- The shared `feature_cache/` remains separate from run-local training artifacts.
