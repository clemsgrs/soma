# soma

Modular experimentation framework for computational pathology.

## Install

```bash
pip install -e ".[all]"
```

## Quick Start

Define `dataset.csv` and `splits.csv`, then run a pipeline:

```python
from soma import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    Pipeline,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_dir="experiments/run1",
    cache=CacheConfig(root_dir="shared/feature_cache"),
    encoder=EncoderConfig(name="uni2", batch_size=64),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="classification"),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
)

result = Pipeline(config).run()
```

## Core API

- `Dataset` and `Splits` load the CSV inputs.
- `FeatureExtractor` preprocesses slides and writes embeddings.
- `FeatureStore` reads cached or plain feature directories.
- `Pipeline`, `train()`, and `train_one_fold()` run experiments.
- `TaskConfig` selects the task head: `classification` or `regression`.

## Docs

- [Workflow](docs/workflow.md)
- [Cache](docs/cache.md)
- [Outputs](docs/outputs.md)
- [Reference](docs/reference.md)

## License

Apache 2.0
