# One model format, the bundle; resume is per fold

A trained model persists in exactly one format, the bundle, written by `soma.save(fit, path, *, preprocessing=None, provenance=None)` and read by `soma.load(path) -> ModelBundle(model, preprocessing, provenance)`. Notebooks and `Pipeline` share it: the pipeline saves one bundle per fold, under `bundle/` in the fold directory. `best_model.pt` is no longer written by the pipeline. A fold is complete when its `metrics.json` and its bundle exist; a relaunch skips complete folds and trains the others from epoch 0.

## Why

`best_model.pt` held a bare `state_dict`. Rebuilding the model it fit took the run's `config.yaml`, the cohort (for the label map and class count `auto_params` derive) and a copy of the pipeline's construction logic, which the attention heatmaps and the dense slide predictor each reimplemented. Nothing recorded which soma wrote it, which epoch was selected, or which features it was trained on.

## Decisions

- **Four files.** `weights.safetensors` is the full `model.state_dict()`, parameters and buffers, so the fitted feature adaptor (center, scale, projection) rides in the weights rather than in JSON. `model.json` records the shape, the builder's dotted path and the builder's arguments: the aggregator or decoder section, the task head with its resolved params, the adaptor sections and, for a live run, the `soma.extract.live_encoder` arguments. `preprocessing.json` holds what scoring needs beyond the module (task family, class scheme, target schema, metrics, grid geometry, read spacing, tile size, the detection threshold chosen on tune and the matching distance); it holds no tensors. `provenance.json` holds the soma, slide2vec and hs2p versions, the selected epoch and monitor, the fold, the run id and the feature-source identity as the cache records it (cache key, feature identity), never a path.
- **The builders rebuild.** The pipeline builds every model through `soma.pipeline.build.build_set_model` / `build_grid_model`, which record themselves on the model; `soma.load` calls the recorded builder with the recorded arguments. A bundle is rebuilt by the code that built the model, not by a second copy of it. In slice 6 the arguments become the config's `{name, params}` tree.
- **Eager load.** `soma.load` builds the live frozen encoder when it loads the bundle, so missing gated weights fail at load time, not halfway through a prediction. Bundles loaded together share one encoder stage while any of them is alive.
- **Atomic write.** `soma.save` stages the bundle in a sibling directory and renames it into place; a reader sees the old bundle or the new one.
- **Versioning.** `soma.load` refuses a bundle from another soma major version and warns on another minor or patch version. The format may still change before 2.0.0; nothing reads 1.x `best_model.pt`.
- **Fold-level resume only.** The bundle carries no optimizer state, so a crashed fold restarts from epoch 0, as it did with `best_model.pt`. Epoch-level resume is a non-goal.
- **Ensembles stay a predict-time policy.** The dense slide predictor takes a list of bundles (`SlidingWindowSegmentationPredictor.from_bundles`). There is no general ensemble API.

## Consequences

- `Checkpoint` keeps its in-memory selection and its optional `dirpath` for notebook use; the pipeline passes none.
- Attention heatmaps are computed after every fold is done, by loading each fold's bundle. `build_dense_models` is gone: `from_bundles` takes the loaded bundles and, for folds trained on cached grids, the encoder stage that extracted them.
- A model built by hand saves, but `soma.load` refuses it: its `model.json` records no builder.
- The artifact mirror publishes a fold only when it is complete under the same rule, so a crash between `metrics.json` and the bundle is never restored as finished.
