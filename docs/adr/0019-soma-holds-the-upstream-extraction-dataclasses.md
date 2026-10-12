# soma holds slide2vec's and hs2p's extraction dataclasses as given

soma no longer declares its own copies of the extraction configuration. A run's features are a `FeaturesConfig` with four soma fields, `kind`, `encoder`, `cache` and `spacing_policy`, and three upstream ones held as given: `extraction` is slide2vec's `PreprocessingConfig` (which composes hs2p's `TilingConfig`, `SegmentationConfig`, `FilterConfig`, `MaskSamplingConfig` and `PreviewConfig`), `execution` is slide2vec's `ExecutionOptions`, and `dense` is slide2vec's `DenseOptions`. soma's `PreprocessingConfig`, `EncoderConfig`, `EncoderMemberConfig`, `CompositeConfig`, `ExecutionConfig`, `MasksConfig`, `SamplingConfig`, `AttentionConfig` and `PreviewConfig` are deleted, with the helpers that translated them into slide2vec's types.

## Why

ADR 0009 made soma compose hs2p's `TilingConfig`, but kept a soma `PreprocessingConfig` around it and mirrored the rest: segmentation, filtering, masks, sampling, execution and the dense knobs. Every slide2vec or hs2p release meant a soma change before a user could reach a new field, and a translation layer between the two that had to stay exact. The rule is now: tiling belongs to hs2p, extraction to slide2vec, and soma owns only what neither has, which is the run's feature kind, the encoder leaf, the cache and the spacing policy.

## Decisions

- **The encoder is a registry leaf.** `EncoderLeaf(name, params)` names a slide2vec encoder; its params are those of `slide2vec.Model.from_preset` (`output_variant`, `allow_non_recommended_settings`). Batch size and precision are execution fields, so `ExecutionOptions.precision` is the run's one precision.
- **The composite is an encoder.** `composite` is the one encoder soma registers itself. Its params list the members; a member names a slide2vec encoder and may override the run's `DenseOptions` and `ExecutionOptions` fields for itself (not the shared read geometry), so a member extracts exactly as a single-encoder run of it does.
- **Cache keys derive from the upstream fields.** The pooled, tiling, ROI-sampling and dense keys read the dataclasses' own fields, minus the runtime ones that cannot change a feature (worker counts, preview rendering, resume flags, device choices). A field added upstream is keyed without a soma change. Every key changes once; the schema version is `v2`, and dtype is always in the key.
- **The slide-manifest input mode is inferred.** A segmentation run samples ROIs from whole slides when its cohort has a `label_mask_path` column and its extraction carries an annotation vocabulary (any hs2p `pixel_mapping` other than binary tissue). The column alone cannot decide, because pre-cropped segmentation datasets carry it too.
- **`spacing_policy` stays a soma field** until slice 6c gives the source its own params: hs2p and slide2vec have no field for reading a coarser slide at its native spacing.
- **Machine-resolved execution fields are not configuration.** slide2vec resolves `num_gpus` and `num_preprocessing_workers` against the machine when `ExecutionOptions` is built. The resume drift guard and the mirrored recipe ignore them, and neither enters the experiment identity.
- **The YAML surface is translated, not changed.** The `preprocessing:`, `encoder:`, `composite:`, `execution:` and `cache:` blocks are read into the held dataclasses by one internal module (`soma._legacy_layout`), and a saved `config.yaml` is written back in that layout. Unset spacing and tile size resolve from the encoder preset when the config loads. Slice 6c replaces the YAML layout and deletes the module.

## Considered and rejected

- **Keep the mirrors and forward new fields.** This is the drift ADR 0009 named, extended to every block.
- **Write a `features:` section into `config.yaml` now.** It would be the new YAML layout, which belongs to slice 6c; until then saved configs stay loadable by the same translation.

## Consequences

- Every pooled, tiling, ROI-sampling and dense cache key, and every experiment id, changes. Old caches are not migrated; the #497 dense-key migration script is deleted with the derivation it targeted.
- `config.yaml` cannot express upstream fields the old vocabulary has no key for (for example `FilterConfig.a_h` or `DenseOptions.pad_mode`). The experiment identity in `run.yaml` still records them, so the drift guard catches a change.
- In Python, unset `ExecutionOptions` and `PreprocessingConfig` fields take slide2vec's and hs2p's defaults, which can differ from soma's YAML defaults (for example the tissue threshold: 0.01 upstream, 0.1 in soma's YAML).
- soma keeps a CPU worker budget of `min(16, cpu_count // num_gpus)` when `num_workers_per_gpu` is unset, because slide2vec gives image routes no workers by default.
