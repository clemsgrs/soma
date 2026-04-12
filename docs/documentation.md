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
