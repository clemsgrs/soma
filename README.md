# soma

`soma` runs computational pathology experiments from images and labels to predictions, metrics, and reports. It combines preprocessing, frozen foundation-model features, and downstream training through a modular Python API or a YAML pipeline.

<p align="center">
  <img src="docs/_static/figures/pipeline-overview.png" alt="The soma pipeline — data, a frozen encoder, a trained decoder, and evaluation." width="640">
</p>

**[Documentation](https://clemsgrs.github.io/soma)** · **[PyPI](https://pypi.org/project/soma-pathology/)**

## Install

Requires Python 3.11 or later:

```bash
pip install soma-pathology
```

The distribution is `soma-pathology`; the Python package and CLI are `soma`.

## Run an experiment

For slide classification, prepare `dataset.csv` (sample ID, image path, label) and `splits.csv` (sample ID, split, fold). The [getting started guide](https://clemsgrs.github.io/soma/getting-started.html) shows both files and a complete walkthrough.

```python
from soma import (
    AggregatorConfig,
    EncoderConfig,
    EvalConfig,
    Pipeline,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_root="output",
    dataset_type="slide",
    encoder=EncoderConfig(name="phikon"),
    aggregator=AggregatorConfig(name="abmil"),
    task=TaskConfig(name="binary_classification"),
    training=TrainingConfig(epochs=5, learning_rate=1e-4, seed=0),
    evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
)
result = Pipeline(config).run()
print(result.summary)
print(result.run_dir)
```

`phikon` weights are public and download on first use. To extract features once and reuse them across experiments, see the [API guide](https://clemsgrs.github.io/soma/api.html).

## Use the CLI

Run the same pipeline from a YAML config:

```bash
soma config.yaml
# Equivalent:
python -m soma config.yaml
```

soma merges the file with its bundled defaults. Start with the [task examples](examples/README.md) or the [CLI reference](https://clemsgrs.github.io/soma/cli.html). `soma list encoders` (and `aggregators`, `decoders`, `tasks`, `benchmarks`) prints the registered components.

## Explore

- [How soma works](https://clemsgrs.github.io/soma/how-soma-works.html): pipeline components and experiment design.
- [Modeling paths](https://clemsgrs.github.io/soma/modeling.html): tile, slide, patient, and dense prediction workflows.
- [Tutorials](https://clemsgrs.github.io/soma/tutorials/index.html): task-specific walkthroughs.
- [Benchmarking](https://clemsgrs.github.io/soma/benchmarking.html): fixed protocols and encoder comparisons, including [task-free CRoMa evaluation](https://clemsgrs.github.io/soma/api.html#task-free-representation-evaluation).
- [Caching](https://clemsgrs.github.io/soma/caching.html) and [run outputs](https://clemsgrs.github.io/soma/outputs.html): feature reuse and experiment provenance.

## License

[Apache-2.0](LICENSE).
