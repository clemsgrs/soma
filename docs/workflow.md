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

## Common Paths

- `preprocess()` writes tiling artifacts.
- `extract()` writes embeddings and returns a `FeatureStore`.
- Lower-level APIs use specific destination names like `tiling_dir`, `feature_dir`, `fold_dir`, and `run_dir` instead of a generic `output_dir`.
- `Pipeline.run()` writes training outputs under a managed experiment/run directory inside `output_root`.
- The shared `feature_cache/` remains separate from run-local training artifacts.
