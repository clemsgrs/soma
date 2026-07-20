# Project Documentation Notes

- 2026-07-20: Added the top-level `projection` section (`none` | `pca` | `random`, plus
  `target_dim` and `seed`) as the feature adaptor's second stage, applied after
  `normalization` (order: normalize → project). It is the dim-matched ablation for the
  capacity confound — a wider encoder otherwise buys a larger aggregator — so when a
  projection is active the aggregator/head is built against `target_dim` rather than the
  encoder's native dim, equalizing trainable capacity across a roster. `pca` is fitted per
  fold on the Support split only, centers intrinsically, and pins a sign convention so
  repeated fits are byte-identical; `random` is a fixed Gaussian matrix seeded from `seed`
  + encoder identity + dims, scaled to preserve inner products and constant across
  trajectories. Both are frozen buffers, never learned. A preflight requires
  `n_fit_rows >= target_dim` and `target_dim <= D` for PCA. Provenance mirrors
  `normalization`: out of the feature-extraction cache key, folded into experiment
  identity only when non-default, always serialized in the saved config, and summarized in
  the per-fold `feature_adapter.json` sidecar (now with the PCA explained-variance ratio).
- 2026-07-20: Added the top-level `normalization` section (`none` | `zscore` | `l2` |
  `layernorm`, plus `eps`) and the *feature adaptor* it drives — a buffer-carrying front
  module inserted ahead of the aggregator/head on the tile-encoder MIL path. `zscore` is
  fitted on the Support (train) split only, so the transform is leak-free; its center and
  scale live in buffers (never in `model.parameters()`) and ride in the checkpoint, so the
  final-checkpoint test pass re-applies the exact transform. `none` (the default) builds no
  adaptor at all, leaving the model structurally identical to before. The section does not
  enter the feature-extraction cache key, folds into experiment identity only when
  non-default, is always serialized in the saved run config, and is summarized in a
  per-fold `feature_adapter.json` QC sidecar.
- 2026-07-20: Added `TrainingConfig.checkpoint_selection` (`best` | `last`). `last`
  evaluates the final-epoch weights, takes model selection off the tune metric and
  disables early stopping (it requires `patience: null`), while still computing and
  logging per-epoch tune metrics as diagnostics. `patience` is now `int | None`, with
  `None` meaning "no early stopping". The default `best` keeps every existing run —
  and every `experiment_id` — unchanged: the setting folds into experiment identity
  only when non-default, though the saved run config always serializes it.
- 2026-05-31: Added `TrainingConfig.tune_is_test` for benchmark protocols that
  use the single test split as the checkpoint-selection split. This keeps
  train/test-only split files explicit while warning that tune and test share
  samples.
- 2026-04-22: Added `TrainingConfig.allow_missing_tune` as an explicit escape hatch for datasets that only provide train/test splits. The pipeline still fails by default when no tune samples are available, but now emits a warning and reuses the train split for tuning when the flag is enabled.
- 2026-04-22: Patched slide-level `encode_slide(...)` calls in `slide2vec` to use CUDA autocast when the requested execution precision is `fp16` or `bf16`, which prevents Titan from entering FlashAttention in `fp32`.
- 2026-04-22: Hardened cross-run comparison so `compare_run_predictions(...)` only uses shared prediction columns. Reports now handle runs that omit `prob_*` columns for label-based metrics instead of crashing with a `KeyError`.
- 2026-04-22: Updated cross-run comparison to preserve stored `predicted_label` values for ordinal and classification runs instead of reconstructing labels from `raw_score` or probabilities when the label column is already present.
- 2026-04-22: Cross-run comparison now treats missing `predicted_label` / `predicted_value` columns as an explicit error instead of silently reconstructing those fields.
- 2026-04-22: Fold aggregation preserves label-only `predicted_label` values when no probability or raw-score signal is available, instead of dropping the label column.
- 2026-04-22: Fold aggregation now raises on conflicting label-only duplicates instead of arbitrarily choosing a `predicted_label`.
- 2026-04-22: Fixed the completed-run console summary to render single-fold coverage from plain `summary.json` keys (`coverage`, `num_samples`, etc.) as well as multi-fold aggregated keys (`*_mean`/`*_std`).
- 2026-04-22: Moved cross-run comparison reports into dedicated bundle directories under the shared `output_root`, with `index.html` as the entry point instead of a shared `comparison.html` file beside one run.
- 2026-04-22: Updated the `slide2vec` torchrun launcher to use standalone rendezvous for single-node GPU jobs, avoiding collisions on the default `29500` port.
- 2026-04-23: Feature manifest generation now reuses existing rank and dimensionality metadata when available, so cached runs no longer need to reopen a `.pt` file just to infer feature shape.
- 2026-04-23: Slide-level cache population now writes slide embeddings directly into `feature_cache/.../features/` as each slide is aggregated, instead of staging everything through a temp directory first.
- 2026-04-23: Patient-level, tile-level, and hierarchical cache population now also target the shared cache directory directly, so their intermediate outputs appear in the live cache instead of a temp staging directory.
- 2026-04-23: Fixed the `unicorn-task1.py` post-embedding training failure by materializing cache-backed run-local feature files as independent atomic copies instead of hardlinks. This avoids a CIFS fresh-write/hardlink handoff race before training starts.
