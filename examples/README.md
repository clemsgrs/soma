# Examples

## Config starting points

`reference.yaml` documents every available config field. The other YAML files
are focused per-task starting points (for example
`slide_binary_classification.yaml`, `tile_classification.yaml`,
`slide_survival.yaml`).

## External benchmark API scripts

Four self-contained scripts demonstrate the external benchmark API introduced
in soma 1.12.0. Each runs with no arguments on synthetic data, from a clean
`pip install soma-pathology`:

| Script | Shows |
| --- | --- |
| [`external_benchmark_spec.py`](external_benchmark_spec.py) | Build a `BenchmarkSpec` (config builder, canonical seeds, scorer) and execute it with `run_benchmark_spec`: per-seed evidence roots, the shared feature cache root, and the aggregated `BenchmarkRunResult`. |
| [`fixed_step_training.py`](fixed_step_training.py) | Train with `TrainingConfig(max_steps=N, epochs=None)`: the exact optimizer-update count, including a partial final epoch and its tune evaluation. |
| [`aggregator_resolution.py`](aggregator_resolution.py) | Resolve one fixed MIL recipe with `soma.encoders.resolve_aggregator` for a tile encoder and a slide encoder; both resulting `PipelineConfig`s validate without loading encoder weights. |
| [`portable_identity.py`](portable_identity.py) | The same manifest under two storage roots keeps `experiment_id` and the leaderboard triple unchanged, while an added `image_path_sha256` column changes identity. |

`external_benchmark_spec.py` downloads the small ungated
[phikon](https://huggingface.co/owkin/phikon) tile encoder and runs a real
(tiny) extraction and training loop; the other three run in seconds with no
model download.

## Benchmark curation

`eva/`, `ocelot/`, `beetle/`, and `detection_benchmark/` hold the curation and
protocol assets of the built-in benchmarks, and `make_beetle_manifest.py`
builds the BEETLE slide manifest.
