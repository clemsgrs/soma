# The pipeline is the public verbs in a canonical order, gated by parity

`soma.pipeline` is a package whose `Pipeline.run()` only orchestrates: open the run layout and its resume guards, attach the supplied feature source or extract one, run every fold not yet complete, aggregate the summary from disk, render attention heatmaps from the bundles, and record provenance. Each fold is `run_fold(fold_dir, setup)`, which applies the public verbs in one fixed order. The former `train`, `train_one_fold` and `train_one_grid_fold` entry points are retired.

## Why

`pipeline.py` held three fold functions, a run loop and the orchestrator in one 3300-line module. Each fold function interleaved config parsing, loader building, training, scoring and file writing, so the order in which random state was consumed was implicit in three places and nothing checked that the verbs, composed by hand, reproduced a run. The layer-1 functions were a second entry point that bypassed the run layout and the resume drift guard.

## Decisions

- **The canonical fold order.** 1. `soma.seed_everything(seed, fold=fold)`; 2. fit the feature adaptor on the train split, as an explicit train-only step outside `soma.fit`; 3. build the model; 4. build the loaders; 5. `Baseline.fit` on the train targets (set shape); 6. `soma.fit` with `Checkpoint` (no file) and `EarlyStopping`; 7. the tune-time step (detection sweeps its per-class threshold on tune); 8. per split, `soma.predict` (plus baseline rows), `soma.evaluate`, then write `predictions_<split>` and `metrics.json`; 9. `soma.save` the bundle. It is written once, in `run_fold`.
- **Shape differences live in two builders.** `build_set_fold` and `build_grid_fold` turn the config into a `FoldSetup`: deferred callables for each step, so they run in the canonical order. The `dataset_type` branches stay there until slice 6.
- **Parity gates the order.** Three CPU e2e scenarios run one fold through the CLI and again by hand from plain constructors with the YAML's values, never importing `soma.pipeline`: set MIL classification over cached bags with a tile-less sample (exercising `Baseline`), cached grid segmentation, and cached grid detection (exercising the threshold sweep). Every predictions file must have the same sha256, the metrics must be equal and `weights.safetensors` must be byte-identical. A parity failure is fixed in the pipeline, never by loosening the comparison. Byte identity is a same-machine claim; EVA rows remain the cross-machine check.
- **The source is an input.** `Pipeline(config, *, source=None)` attaches a supplied source; otherwise `run()` extracts one or builds the live source. The CLI builds the cached source from `--feature-dir`. Slice 5 keeps `PipelineConfig` and its drift guard; slice 6 swaps in `resolve()` at the one place the config is consumed.
- **`soma.fit` takes the run's policy.** A `Checkpoint`, `EarlyStopping` or `Progress` passed in `callbacks` replaces the one `fit` would build, so the pipeline and a hand composition state selection with the same constructors.
- **Public run-management pieces.** `RunRecorder`, the resume drift guards, the fold layout (`layout.py`) and the per-fold writers and from-disk aggregation (`aggregate.py`) are public, so a benchmark composing the verbs itself (HEST) writes the same files without private imports.

## Consequences

- `soma.output_layout`, `soma.provenance` and `soma.artifact_mirror` move to `soma.pipeline.layout` and `soma.pipeline.mirror`. Representation evaluation moves to `soma.pipeline.representation`, dispatched in one line.
- The segmentation grid-spacing check is a data check on the grid source (`soma.dense.require_declared_spacing`), called by the grid builder.
- Unit tests that drove the retired fold functions were deleted or now drive `Pipeline`; tests pinning `train`, `train_one_*` or `best_model.pt` are gone.
