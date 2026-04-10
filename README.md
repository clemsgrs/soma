# soma

`soma` is a fast, scalable Python package for building and testing deep learning models in computational pathology.

It provides a unified API to go from a dataset of slides and labels to a full, reproducible result report. Along the way, it makes it easy to sweep core design choices such as preprocessing, encoding (foundation models), and aggregation (MIL) so you can quickly find the strongest configuration for your data.

You can use it either as a full end-to-end pipeline or as a set of composable building blocks for custom experiment orchestration.

## Install

```bash
pip install soma
```

## API Overview

The package root exports the main entry points:

- `Dataset` and `Splits` for loading data
- `FeatureExtractor` for preprocessing slides and extracting embeddings
- `train()` and `train_one_fold()` for training directly from features
- `Pipeline` for the full preprocessing + feature extraction + training workflow

## Quick Start

### 1. Prepare dataset and splits

`dataset.csv` should contain one row per slide with at least `sample_id`, `image_path`, and `label`. `sample_id` must be unique, `image_path` should point to the slide file, and `label` can be either a string class name or an integer target.

`splits.csv` should assign each `sample_id` to `train`, `tune`, or `test` for every fold. This is what keeps evaluation reproducible and prevents leakage.

More details on the dataset and split contract are in [docs/reference.md](docs/reference.md).

```python
from soma import Dataset, Splits

dataset = Dataset("dataset.csv")
splits = Splits("splits.csv", dataset)

print(dataset.num_classes)
print(splits.num_folds)
```

### 2. Extract once, cache, and reuse features across experiments

`FeatureExtractor` handles preprocessing and embedding extraction, and the cache lets you reuse the same extracted features across multiple training runs instead of recomputing them every time. That is especially useful when you want to compare several MIL aggregators or heads against the same encoder output.

When you run the full pipeline, the same cache system also handles tiling:

- live tiling runs write a local `tiling/` directory first
- a complete tiling-cache hit replaces that directory with a run-local stub
- a complete feature-cache hit reuses the shared embeddings directly

More details about the caching mechanism are in [docs/cache.md](docs/cache.md).

```python
from soma import Dataset, Splits
from soma import FeatureExtractor, train
from soma import CacheConfig, EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig

## Extract features once

dataset = Dataset("dataset.csv")
extractor = FeatureExtractor(
    dataset=dataset,
    encoder=EncoderConfig(name="uni2"),
    output_root="output",
    cache=CacheConfig(enabled=True, root_dir="shared/feature_cache"),
)

store = extractor.extract(feature_dir="output/features/uni2")

## Build on top of these features

splits = Splits("splits.csv", dataset)
task = TaskConfig(name="classification")

abmil_result = train(
    feature_store=store,
    dataset=dataset,
    splits=splits,
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=task,
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
    run_dir="output/abmil/uni2",
)

clam_result = train(
    feature_store=store,
    dataset=dataset,
    splits=splits,
    aggregator=AggregatorConfig(name="clam_sb", params={"hidden_dim": 256, "attn_dim": 128}),
    task=task,
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
    run_dir="output/clam_sb/uni2",
)
```

This is the sweet spot for sweep-style workflows: one feature set, many model
variants.  
More details on the training and pipeline APIs are in [docs/reference.md](docs/reference.md).

### 3. Run a full pipeline in one call

`Pipeline(config).run()` is the single-call path from `(dataset, split)` to a
reproducible result bundle. It handles preprocessing, feature extraction,
training across folds, and metric aggregation for you.

```python
from soma import Pipeline, PipelineConfig
from soma import EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_root="output",
    encoder=EncoderConfig(name="uni2"),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="classification"),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
)

result = Pipeline(config).run()
```

The returned `PipelineResult` includes:

- `fold_results`: one entry per fold, each with training, tune, and test reports
- `summary`: aggregated metrics across folds
- `run_dir`: the resolved run directory containing the saved artifacts

More details about the generated artifacts are in [docs/outputs.md](docs/outputs.md).

## Docs

- [Documentation](docs/documentation.md)
- [Workflow](docs/workflow.md)
- [Cache](docs/cache.md)
- [Outputs](docs/outputs.md)
- [Reference](docs/reference.md)

## License

This repository is available under [AGPL-3.0](LICENSE).
