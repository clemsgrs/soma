# Project Documentation Notes

- 2026-09-02: Extraction commits its cache in chunks. Identity signatures were recorded
  only after every image/slide had been encoded, and unsigned payloads are deleted on the
  next run, so an interrupted extraction restarted from zero. The tile-image path now
  commits every ``cache.commit_every`` images (default 1024) and the WSI tile/hierarchical
  paths every ``commit_every`` slides (default 8 per GPU; each chunk is one slide2vec
  pipeline call, hence not one slide). The knob is not part of any cache key. The
  tile-image cache key now resolves ``encoder.output_variant`` before keying (as the
  pooled WSI path already did), so a null variant and the encoder's explicit default
  share one cache; pre-cropped tile caches keyed under a null variant re-key, and the
  old/new keys are logged so the directory can be renamed. WSI tile/slide/dense caches are
  unaffected. Dense grid storage dtype now goes through the same resolver as the pooled
  caches (``cache.dtype``, else the encoder's override, else the registry recommendation):
  dense caches with ``cache.dtype: null`` on an encoder whose registry precision is fp16
  re-key from fp32 to fp16; pin ``cache.dtype`` to keep an existing cache.
- 2026-09-02: Fixed DTFD-MIL feature distillation. The top-k slice selected every
  instance of each pseudo-bag (and ``maxmin`` duplicated them), so tier 2 saw the whole
  bag instead of a distilled set. ``DTFDMIL`` now takes ``instances_per_group`` (default
  1, the reference ``total_instance // numGroup``; clamped to the pseudo-bag size), draws
  the training-time pseudo-bag permutation from torch's global RNG (so it follows the run
  seed) and uses a deterministic contiguous partition in eval mode. Existing DTFD-MIL
  runs are not comparable with new ones. ``cox_breslow_loss`` now builds each event's
  risk set from an explicit ``time_j >= time_i`` mask instead of a sort plus cumulative
  log-sum-exp, so tied times share one denominator (true Breslow) and the loss is
  invariant to sample order. ``clam_mb`` is refused for every task other than
  ``multiclass_classification`` (the guard previously only caught binary).
- 2026-09-02: Manifest validation fails loudly instead of dropping rows. A blank
  ``fold`` cell in ``splits.csv`` and a blank ``label`` cell in ``dataset.csv`` (slide,
  patient and tile datasets) are now hard errors listing the offending sample ids;
  previously the blank-fold row vanished from every fold and the blank label became its
  own ``nan`` class. ``Splits`` also logs a warning naming any fold/split that lacks a
  class present elsewhere in the fold, since threshold-free metrics are undefined there.
  Resume hardening: the cross-validation summary reads only the folds the current split
  file declares, a run dir holding ``fold_*`` dirs beyond that count is refused, and the
  resume drift guard now also recomputes the train/tune experiment identity from the
  manifest *content* so an edited label or reshuffled fold under unchanged paths is caught.
- 2026-09-02: Degenerate tune/test splits no longer masquerade as chance-level
  performance. AUROC, macro AUROC and the C-index return ``nan`` (instead of ``0.5``)
  when a split holds a single class or no comparable pairs; the trainer raises at the
  first epoch, naming the fold and metric, when the monitored value is non-finite
  (previously the run finished silently without ever saving a checkpoint). The
  cross-validation summary averages over the finite folds and reports
  ``<split>/<metric>_nan_folds`` when any fold was excluded. All reported spreads
  (``*_std`` in ``summary.json``, the leaderboard ``std`` column) are now the sample
  standard deviation (``ddof=1``) via one shared helper; a single fold/seed yields
  ``nan`` / blank. ``peak_per_metric`` honours lower-is-better metrics, and checkpoints,
  ``metrics.json``, ``training_history.json`` and ``summary.json`` are written through a
  staging file so an interrupted write never leaves a truncated artifact.
- 2026-07-20: Extended the feature adaptor (`normalization` + `projection`) to the
  **single-encoder dense path** (`segmentation` and `detection` over one encoder's cached
  grids), completing the protocol across all three paths. The adaptor operates
  **channel-axis** on `(B, d, h, w)` grids and is fit over **all positions in the Support
  ROIs** — so at a 2×2 token grid, two Support ROIs give 8 fit rows, not 2. The frozen
  projection composes **ahead of** the decoder's own learnable 1×1 projection conv (frozen
  `d → target_dim`, then learnable `target_dim → hidden`), so no decoder change was needed
  and, because that 1×1 is the decoder's only `d`-dependent module, the whole decoder
  becomes encoder-dim-independent under an active projection. This path **requires
  `feature_mode: cached`**: `live` re-encodes *augmented* tiles every step, so a transform
  fit on the cached Support grids would not match what it transforms — the combination is
  refused at config validation and again in the fold. Composite (multi-encoder) dense
  streams, the decoder-free `pixel_classifier` path, and `spatial_expression` are still
  refused; composites keep their per-member `member_norm` unchanged. Checkpoint
  reconstruction sites on this path (detection eval-only re-scoring, the OCELOT greedy
  re-scorer, and `build_live_segmentation_models` for whole-slide sliding-window
  inference) now rebuild the adaptor and size the decoder from its `output_dim`, so a
  cached-trained projected checkpoint replays correctly.
- 2026-07-20: Extended the feature adaptor (`normalization` + `projection`) to the
  **slide-encoder embedding path**, so the same protocol now covers both the tile-encoder
  MIL path and the embedding path. There the fit is over the Support split's *embeddings*
  — one vector per slide — so the fit sample count is exactly `K`, which makes the PCA
  preflight load-bearing: at `K = 12` no PCA wider than 12 components exists, and the
  request raises rather than producing a degenerate basis. `EmbeddingModel` now carries an
  optional adaptor as a front module, the task head is built against the adaptor's
  `output_dim` under an active projection (the dim rewire), and everything else is
  unchanged from the MIL path — leak-free Support-only fit, buffers not parameters riding
  in the checkpoint, cache key untouched, identity folded only when non-default, config
  always serialized, `feature_adapter.json` written. With both blocks off no adaptor is
  built, so existing slide-encoder runs stay byte-identical.
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
