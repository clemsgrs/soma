# Documentation

## Training UX

- The training progress live panel now shows epoch, loss, learning rate, tune metrics, patience, and status without rendering the checkpoint file path.
- The live panel also shows the number of trainable parameters for the model.
- The live panel now also shows total elapsed training time with ETA inline on the same row, plus average epoch time on the next row, so long runs have at-a-glance duration readouts.
- Saved `fold_N/training_history.json` entries persist elapsed and average epoch time, and the HTML report shows those values in a dedicated training timing section. ETA remains live-only.
- Checkpoint files are still written and returned via `TrainResult.checkpoint_path`; the UX change only removes the path from the live progress display.

## Metrics

- `multiclass_classification` now accepts `qwk` as an opt-in metric for ordered class labels while keeping cross-entropy loss.

## Extraction

- Distributed slide-level cache refresh now receives the resolved preprocessing and backend provenance explicitly, which keeps the post-embedding refresh path from depending on branch-local variables.

## Configuration

- `AggregatorConfig.name` is now required, and `PipelineConfig.aggregator` defaults to `None`. Slide-level feature runs must keep the aggregator explicit as `None`, while MIL runs must pass a named aggregator config.
