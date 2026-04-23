# soma

`soma` is a modular framework to streamline computational pathology research.

It provides a unified API to go from a dataset of slides and labels to a full, reproducible result report. Along the way, it makes it easy to sweep core design choices such as preprocessing (spacing, field-of-view), encoding (foundation models), and aggregation (MIL) so you can quickly find the strongest configuration for your data.

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

```python
from soma import Dataset, Splits

dataset = Dataset("dataset.csv")
splits = Splits("splits.csv", dataset)

print(dataset.num_classes)
print(splits.num_folds)
```

### 2. Extract once, cache, and reuse features across experiments

`FeatureExtractor` handles preprocessing and embedding extraction. The cache lets you reuse the same extracted features across multiple training runs, which is especially useful when comparing several MIL aggregators or heads against the same encoder output.

```python
from soma import Dataset, Splits, FeatureExtractor, train
from soma import CacheConfig, EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig

# Extract features once

dataset = Dataset("dataset.csv")
extractor = FeatureExtractor(
    dataset=dataset,
    encoder=EncoderConfig(name="uni2"),
    output_root="output",
    cache=CacheConfig(enabled=True, root_dir="shared/feature_cache"),
)

store = extractor.extract(feature_dir="output/features/uni2")

# Train multiple model variants on the same features

splits = Splits("splits.csv", dataset)
task = TaskConfig(name="binary_classification")

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

### 3. Run a full pipeline in one call

`Pipeline(config).run()` handles preprocessing, feature extraction, training across folds, and metric aggregation in a single call.

```python
from soma import Pipeline, PipelineConfig
from soma import EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_root="output",
    dataset_type="slide",
    encoder=EncoderConfig(name="uni2"),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="binary_classification"),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
)

result = Pipeline(config).run()
```

The returned `PipelineResult` includes:

- `fold_results`: one entry per fold, each with training, tune, and test reports
- `summary`: aggregated metrics across folds
- `run_dir`: the resolved run directory containing the saved artifacts

## CLI

`soma` ships a command-line interface that runs a full pipeline from a YAML config file:

```bash
soma /path/to/config.yaml
python -m soma /path/to/config.yaml
```

The YAML layout is grouped by concern: `run`, `data`, `preprocessing`,
`encoder`, `aggregation`, `task`, `evaluation`, `training`, `execution`,
`cache`, and `reports`. `soma` merges your file on top of the bundled
`soma/configs/default.yaml`, so you usually only need to edit the blocks you
want to change.

You can also inspect the available presets directly from the terminal:

```bash
soma list encoders --level tile
soma list aggregators
soma list tasks
```

`examples/` contains a `reference.yaml` documenting every available field, and focused per-task starting points (`slide_binary_classification.yaml`, `slide_ordinal_classification.yaml`, `slide_regression.yaml`, `tile_classification.yaml`).

## Docs

- [Getting Started](docs/getting-started.rst)
- [Pipeline](docs/pipeline.rst)
- [Preprocessing](docs/preprocessing.rst)
- [Encoders](docs/encoders.rst)
- [Aggregators](docs/aggregators.rst)
- [Tasks](docs/tasks.rst)
- [Training and Evaluation](docs/training.rst)
- [Caching](docs/caching.rst)
- [Run Outputs](docs/outputs.rst)
- [CLI Guide](docs/cli.rst)

## License

This repository is available under [AGPL-3.0](LICENSE).
