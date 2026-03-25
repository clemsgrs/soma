# soma

Modular experimentation framework for computational pathology.

WSIs in, metrics out. Extract features with foundation models, train MIL classifiers, evaluate across folds — all from two CSVs.

## Install

```bash
pip install -e ".[all]"
```

## Quick Start

Define your data in two CSV files:

**dataset.csv**
```
sample_id,image_path,label
TCGA-A1,/slides/A1.svs,tumor
TCGA-A2,/slides/A2.svs,normal
```

**splits.csv**
```
fold,sample_id,split
0,TCGA-A1,train
0,TCGA-A2,test
```

Run a full experiment:

```python
from soma import Pipeline, PipelineConfig, CacheConfig, EncoderConfig, AggregatorConfig, TrainingConfig

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_dir="experiments/run1",
    cache=CacheConfig(),  # shared cache enabled by default
    encoder=EncoderConfig(name="uni2", batch_size=64),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
)
results = Pipeline(config).run()
```

## Modular API

Every step is independently accessible. Use what you need, plug in your own code for the rest.

### Feature Extraction

```python
from soma import Dataset, FeatureExtractor, EncoderConfig

dataset = Dataset("dataset.csv")
extractor = FeatureExtractor(dataset, EncoderConfig(name="uni2"))

# Full pipeline: tissue segmentation → tiling → encoding
store = extractor.run("features/uni2")

# Or step by step
extractor.preprocess("output/tiling")
store = extractor.extract("features/uni2", tiling_dir="output/tiling")
```

When the shared cache is enabled, `soma` stores canonical reusable artifacts under a stable cache root instead of treating each run-local feature directory as the only source of truth. This means a slide-model run can populate a shared tile-feature cache, and a later tile-model run with matching preprocessing and encoder settings can reuse those tile features directly.

### Training

```python
from soma import Dataset, Splits, FeatureStore, train
from soma import AggregatorConfig, TaskConfig, TrainingConfig

dataset = Dataset("dataset.csv")
splits = Splits("splits.csv", dataset)
store = FeatureStore("features/uni2")

# Train all folds + summarize
result = train(
    feature_store=store,
    dataset=dataset,
    splits=splits,
    aggregator=AggregatorConfig(name="abmil"),
    task=TaskConfig(name="classification"),
    training=TrainingConfig(epochs=50),
    output_dir="experiments/run1",
)
```

`train_one_fold()` is also available for single-fold control (e.g., in agent sweeps):

```python
from soma import train_one_fold

for agg in ["abmil", "mean_pool", "max_pool"]:
    for i, fold in enumerate(splits.folds):
        train_one_fold(
            store, dataset, fold,
            aggregator=AggregatorConfig(name=agg),
            task=TaskConfig(),
            training=TrainingConfig(),
            output_dir=f"experiments/{agg}/fold_{i}",
        )
```

## Supported Encoders

| Model | Dim | Spacing |
|-------|-----|---------|
| uni / uni2 | 1024 / 1536 | 0.5 µm/px |
| virchow / virchow2 | 2560 | 0.5 µm/px |
| conch / conchv15 | 512 / 768 | 0.5 µm/px |
| gigapath | 1536 | 0.5 µm/px |
| h-optimus-0 / h-optimus-1 / h0-mini | 1536 | 0.5 µm/px |
| phikon / phikonv2 | 768 / 1024 | 0.5 µm/px |
| hibou-b / hibou-l | 768 / 1024 | 0.5 µm/px |
| midnight | 3072 | 0.25–2.0 µm/px |

## Aggregators

- **ABMIL** — Attention-Based MIL with gated attention
- **MeanPool** / **MaxPool** — Simple baselines

## Output

```
experiments/run1/
├── config.yaml           # reproducible config snapshot
├── fold_0/
│   ├── best_model.pt     # checkpoint
│   ├── metrics.json      # tune + test metrics
│   └── predictions.csv   # per-sample predictions
├── fold_1/
│   └── ...
└── summary.json          # mean ± std across folds
```

## License

Apache 2.0
