# Project Documentation Notes

- 2026-04-22: Added `TrainingConfig.allow_missing_tune` as an explicit escape hatch for datasets that only provide train/test splits. The pipeline still fails by default when no tune samples are available, but now emits a warning and reuses the train split for tuning when the flag is enabled.
- 2026-04-22: Patched slide-level `encode_slide(...)` calls in `slide2vec` to use CUDA autocast when the requested execution precision is `fp16` or `bf16`, which prevents Titan from entering FlashAttention in `fp32`.
- 2026-04-22: Hardened cross-run comparison so `compare_run_predictions(...)` only uses shared prediction columns. Reports now handle runs that omit `prob_*` columns for label-based metrics instead of crashing with a `KeyError`.
- 2026-04-22: Updated cross-run comparison to preserve stored `predicted_label` values for ordinal and classification runs instead of reconstructing labels from `raw_score` or probabilities when the label column is already present.
- 2026-04-22: Cross-run comparison now treats missing `predicted_label` / `predicted_value` columns as an explicit error instead of silently reconstructing those fields.
- 2026-04-22: Fold aggregation preserves label-only `predicted_label` values when no probability or raw-score signal is available, instead of dropping the label column.
- 2026-04-22: Fold aggregation now raises on conflicting label-only duplicates instead of arbitrarily choosing a `predicted_label`.
- 2026-04-22: Fixed the completed-run console summary to render single-fold coverage from plain `summary.json` keys (`coverage`, `num_samples`, etc.) as well as multi-fold aggregated keys (`*_mean`/`*_std`).
- 2026-04-22: Moved cross-run comparison reports into dedicated bundle directories under the shared `output_root`, with `index.html` as the entry point instead of a shared `comparison.html` file beside one run.
