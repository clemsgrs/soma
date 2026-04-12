# Documentation

## Training UX

- The training progress live panel now shows epoch, loss, learning rate, tune metrics, patience, and status without rendering the checkpoint file path.
- The live panel also shows the number of trainable parameters for the model as `# params`.
- The live panel now also shows average epoch time below the status row, and total elapsed training time with ETA inline on the next row, so long runs have at-a-glance duration readouts.
- Saved `fold_N/training_history.json` entries persist elapsed and average epoch time, and the HTML report shows those values in a dedicated training timing section. ETA remains live-only.
- Checkpoint files are still written and returned via `TrainResult.checkpoint_path`; the UX change only removes the path from the live progress display.

## Metrics

- `multiclass_classification` now accepts `qwk` as an opt-in metric for ordered class labels while keeping cross-entropy loss.

## Extraction

- Distributed slide-level cache refresh now receives the resolved preprocessing and backend provenance explicitly, which keeps the post-embedding refresh path from depending on branch-local variables.
- Tiling cache metadata now labels the original user-supplied preprocessing snapshot as `requested_preprocessing`, which reads more clearly than the previous internal name.
- Hierarchical tiling cache validation treats the resolved region size as the effective requested tile size when the tiling artifact records region geometry in `requested_tile_size_px`.
- Feature-cache and tiling-cache metadata mismatches now include the cache directory and group differences into `missing`, `extra`, and `changed` sections to make stale or incompatible cache entries easier to diagnose.

## Configuration

- `AggregatorConfig.name` is now required, and `PipelineConfig.aggregator` defaults to `None`. Slide-level feature runs must keep the aggregator explicit as `None`, while MIL runs must pass a named aggregator config.

## Splits and Evaluation

- `FoldSplit.tests` is now a dict mapping split names to sample ID tuples. Any split name that starts with `"test"` is valid (e.g., `"test"`, `"test_external"`, `"test_prospective"`).
- `FoldResult.test_reports` is now a dict `{split_name: EvaluationReport}` — one report per test split.
- `summary.json` keys are always prefixed by split name: `"test/auroc_mean"`, `"test_external/auroc_mean"`.
- Per-fold predictions are saved as `predictions_{split_name}.csv` (e.g., `predictions_test.csv`, `predictions_test_external.csv`).
- Subgroup metrics are saved as `subgroup_metrics_{split_name}.json`.
- Attention maps (when enabled) go to `attention/{split_name}/` per split.
- The HTML report renders a separate results table, prediction analysis section, and subgroup analysis section for each test split.

## CLI

- `soma run config.yaml` launches a full pipeline run from a YAML config file (registered via `[project.scripts]` in `pyproject.toml`).
- Example configs live in `examples/`: `reference.yaml` documents every field and valid option; `slide_binary_classification.yaml`, `slide_ordinal_classification.yaml`, `slide_regression.yaml`, and `tile_classification.yaml` are minimal task-specific starting points.
