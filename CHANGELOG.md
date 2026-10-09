# Changelog

- 2026-10-09: Training progress is now visible when stdout is not a terminal (a SLURM
  `.out` file, `nohup`, redirected output). Rich redraws the training panel only on a
  terminal, so before this change the log stayed empty for the whole fit and showed
  the panel once, at the end. On a non-terminal console the trainer now prints plain,
  flushed lines instead: one when the fit starts (fold, epoch or step budget, trainable
  parameters), one per epoch (train loss, tune loss, monitor value with the selected
  value, patience, learning rate, elapsed time and ETA, status), one in-epoch line
  after 30 s without output (items processed so far), and a final summary (epochs run,
  early stopping, the selected checkpoint and its tune metrics). The panel on a
  terminal or in Jupyter does not change.

- 2026-10-09: Detection manifests accept an optional `ignore_mask_path` column: a
  flat uint8 PNG in the image's pixel frame, 255 where nothing is annotated and 0
  where it is (any other value fails and names the file and the sample). Ignored
  pixels carry no loss, points on them leave the ground truth, and predicted peaks
  on them are dropped before matching, in the threshold sweep, in evaluation and in
  the detection benchmark. The ignore mask path is a storage path, so it does not
  change the dataset checksum or the experiment id. `DetectionHead.extract_targets` returns a new `valid`
  map (all True without a mask) and the loss averages over valid pixels only; runs
  without masks are numerically unchanged. The detection tiler pads and crops the
  mask per tile (`ignore_masks/<tile_id>.png`), skips tiles with no valid pixel,
  counts `points_in_ignored_region` in `summary.json` and writes the ROI's valid
  pixel count as `roi_valid_area_px`, which the stitched ROI area for FROC per mm²
  now uses. A manifest without the column tiles byte-identically to before.

- 2026-10-09: The Monkey FROC scorer now removes cross-class duplicates before it
  pools lymphocytes and monocytes into the merged inflammatory-cells (Task 1) score.
  A 2-class model can fire on one cell in both channels. Before this change, the
  second copy counted as a false positive, so the Task 1 score was too low. A
  prediction is now dropped when a higher-scoring prediction of the other class lies
  within the 5 µm MNL margin, as in a single-class submission. The per-class scores
  and `mean_froc` (Task 2) do not change.

- 2026-10-09: The `dev` and `croma` extras now cap pyarrow below 26. pyarrow 26.0.0
  requires NumPy 2, but the rest of the dependency set still resolves NumPy 1.26, so
  `import pyarrow` failed and every test module that imports it failed to collect.

- 2026-10-08: New benchmarks `eva/camelyon16_small` and `eva/panda_small` reproduce
  EVA's offline slide-level classification configs. Their curators port EVA's tile
  sampler: a shuffled non-overlapping level-0 grid, EVA's saturation foreground mask
  and 0.35 foreground ratio, at most 1000 tiles at 0.25 µm/px (Camelyon16Small) or
  200 at 0.5 µm/px (PANDASmall), with the slide spacing read by EVA's OpenSlide rule.
  They write one hs2p tiling artifact per slide and a slide manifest with
  `coordinates_path`, so soma embeds exactly EVA's tiles. Camelyon16Small splits the
  399 official slides 216 / 54 / 129 (EVA's validation slides are `tune`); PANDASmall
  keeps EVA's 9555 slides and its stratified split, 952 / 475 / 475. The runs train
  the benchmark-private `eva_abmil` aggregator (EVA's `ABMIL`: a 128-d projection and
  gated attention) and the `eva_mil_binary` / `eva_mil_multiclass` heads (EVA's MLP;
  one logit with `BCEWithLogitsLoss` for binary) with AdamW at lr 1e-3, EVA's
  `ConstantLR` warm-up, batch 32, 100 epochs, patience 20, and 20 seeds. The
  reference rows are EVA's test balanced accuracy: 0.849 / 0.861 (Camelyon16Small,
  uni2 / virchow2) and 0.657 / 0.646 (PANDASmall). The aggregator registry now
  imports `soma.benchmarks` on a missed lookup, as the task and decoder registries
  do, so a saved config naming `eva_abmil` loads in a fresh process.
  A new `training.hold_patience_until_monitor_moves` setting (default `true`, the
  current behaviour) can turn off the hold on early stopping while the monitor sits
  at its first value. Both benchmarks set it to `false`, so patience counts as in
  EVA's Lightning `EarlyStopping`: a validation balanced accuracy flat at its first
  value for 20 more epochs stops the run and keeps the first epoch. The new field
  joins the experiment identity, now v6, so existing runs get new experiment ids.

- 2026-10-08: `soma reproduce` now frees each seed's pipeline before it starts
  the next seed. A live segmentation run keeps its frozen encoder in a reference
  cycle, so without this every seed left one encoder on the GPU, and a
  five-seed `eva/monusac` run on an 11 GB GPU ran out of memory at seed 2.
  Results are unchanged.

- 2026-10-07: A slide-level `dataset.csv` may name each slide's tiles with a new
  `coordinates_path` column: one hs2p tiling artifact per slide (the `.coordinates.npz`
  written by `hs2p.artifacts.save_tiling_result`). soma then skips tissue segmentation
  and tiling and slide2vec embeds exactly those tiles. Each artifact is checked
  against its manifest row (`sample_id`, `image_path`, `spacing_at_level_0`) and
  against `preprocessing` (requested spacing and tile size); the column must be set
  for every row or none, and tile, segmentation and detection manifests reject it.
  The artifact's content joins the slide's feature-cache identity, so another tile set is extracted again, and its
  features are cached beside the first set's rather than over them. Each run keeps a
  copy of the artifacts in its `tiling` directory and reuses it on resume; a resume
  stops if an artifact changed since. Like the other path columns, `coordinates_path`
  is left out of the dataset checksum: add a column such as `coordinates_path_sha256`
  to give different tile sets different experiment identities. Only tile encoders
  accept supplied coordinates. Datasets without the column keep their cache keys.

- 2026-10-08: A feature-cache entry recorded as empty (a slide with no tiles) no
  longer hides the slide once its identity changes, for example a new `image_path`:
  the slide is extracted again, and the empty marker is cleared when its features
  are committed.

- 2026-10-07: The `eva/consep` and `eva/monusac` benchmarks follow EVA's online
  segmentation protocol, which produces the leaderboard numbers. The decoder is
  `eva_conv_with_image` (EVA's `ConvDecoderWithImage`): it concatenates the
  normalised input tile to the upsampled feature grid. To feed it, a decoder can
  now set `Decoder.consumes_image`; soma then passes the tile's pixels (float
  RGB in [0, 1], at the mask size) to `forward(X, image)` on the cached and live
  training paths and in sliding-window prediction. Existing decoders are unchanged. MoNuSAC trains on a new random
  resized crop of each whole train image every step, so its curator keeps train
  images whole and the benchmark runs live with the new
  `augmentation.random_resized_crop_scale` (a `RandomResizedCrop` to the target
  size, applied before the other ops). Re-curate MoNuSAC. The augmentation block
  is part of the experiment identity, so `identity_version` is 5 and experiment
  ids change for every run.

- 2026-10-07: `TrainingConfig` gains `lr_warmup_epochs` and `lr_warmup_factor`, a
  `ConstantLR`-style warm-up that scales the learning rate for the first epochs
  under both the epoch and the step budget and composes with the cosine
  scheduler. The EVA segmentation recipe sets EVA's default (lr/3 for five
  epochs); without it two of five `eva/consep` seeds collapsed to a foreground
  Dice of 0 because the first AdamW steps on the unnormalised pre-norm grid
  saturated the softmax. The canonical experiment payload now carries the two
  fields, so `identity_version` is 4 and experiment ids change for every run.

- 2026-10-07: A detection run without a tune split keeps the configured score
  threshold. With `training.allow_missing_tune`, train stands in for tune, and
  soma swept the per-class score threshold on the training samples. The decoder
  fits those closely, so the sweep picked an in-sample cut that does not hold on
  test. soma now keeps `task.params.score_threshold` (default 0.5) in that case
  and sweeps only on a real tune split. `detection_thresholds.json` gains
  `source`: `tune_sweep` or `configured`. Runs with a tune split are unchanged.

- 2026-10-07: `lightweight_conv` takes `decoder.params.dropout` (default 0), a
  channel dropout before the class-output convolution. At 0 the decoder trains
  exactly as before and its checkpoints keep the same keys.

- 2026-10-04: CRoMa representation runs work with croma 1.0. soma read
  `CRoMaResult.undefined_frac`, which croma 1.0 removed, so every CRoMa run on a
  fresh install failed with `AttributeError` after feature extraction. croma 1.0
  scores every selected sample or raises `RuntimeError` naming the samples it
  could not score, so soma no longer checks for undefined samples itself. soma
  now requires croma 1.0.0. Metric values are unchanged: a CRoMa run of
  `dinov2-vitb14` on the PathoROB camelyon cohort gives byte-identical summary
  and per-sample values under croma 0.3.0 and 1.0.0.

- 2026-10-03: `Pipeline.run()` and `run_benchmark_spec()` return the manifest
  checksums of the run. `PipelineResult` and `BenchmarkRunResult` gain
  `dataset_checksum`, `splits_checksum` and `test_checksum`, the same values
  `experiment.json` and `run.yaml` record, so a caller that pins a benchmark's
  data can read them from the result instead of soma's run directories.
  `run_benchmark_spec()` raises `ValueError` if its seeds ran against different
  manifests. `run_benchmark()` leaves the three fields empty.

- 2026-10-02: A run with a patient-level encoder no longer fails during tiling
  with "Patient-level models require a 'patient_id' for every slide" when the
  tiling cache is empty or incomplete. Before, such a run only worked after a
  run with a tile- or slide-level encoder had filled the same tiling cache.
  soma now requires slide2vec 6.3.3, which does not ask for patient ids when it
  only tiles.

- 2026-10-02: Pooled feature caches are checked against the encoder's
  preprocessing after a slide2vec upgrade. A pooled cache key does not cover the
  encoder's image transform, so a cache written before slide2vec 6.2.0 was reused
  for `lunit`, `dinov2-vitb14` and `mstar` although their features changed, and a
  partly filled cache was completed with features of the new transform. soma now
  records slide2vec's feature identity and version in `cache_metadata.json` when
  it writes a tile, image, hierarchical, slide or patient cache. The first time
  such a cache is used with another slide2vec version, soma loads the encoder
  once and compares the two identities, before it reuses the cache or adds
  samples to it. If they differ, the run stops with `CacheFeatureIdentityMismatch`
  and names the differing fields; set `cache.on_identity_mismatch: reextract` to
  delete the cache and extract it again. Caches written by soma 1.17.0 or
  earlier record no identity: soma reuses them and warns once, or deletes and
  extracts them again with `cache.on_unrecorded_identity: reextract`. Cache keys
  do not change. Existing caches of the three encoders above still need to be
  deleted by hand, or with the second setting. Patient-level aggregation no
  longer fails with "No encoder-input contract has been declared". soma now
  requires slide2vec 6.3.2. See `caching.rst`.

- 2026-10-01: Require hs2p 5.1.0 and slide2vec 6.3.1. Flat PNG/JPEG slides
  (with `spacing_at_level_0` in `dataset.csv`) now run end to end with the default
  mask and tiling previews; before, previews failed the slide and pooled extraction
  failed with `Unknown backend: 'pil'`. `mask_backend: auto` now reads TIFF masks
  that store samples other than 8-bit unsigned (16-bit labels, for example) with
  the new lossless `tifffile` reader; the other readers decoded those values wrong,
  and now refuse such masks. Fresh tilings can differ from hs2p 5.0.2 ones: the
  tissue ring around a hole counts as tissue again, so tiles next to holes can
  pass `min_coverage`, and runs with `overlap > 0` read above level 0 place tiles
  on a slightly different stride. Tiling cache keys depend only on the
  configuration, so tilings cached with hs2p 5.0.2 are reused; delete the tiling
  cache to re-tile. The `gigapath-slide` encoder now runs in eval mode (dropout was
  active before), so its embeddings change; delete slide feature caches written
  with it and extract again.

- 2026-09-28: Require slide2vec 6.1.1, which requires hs2p 5.0.2 like soma
  already does. No behavior change in soma.

- 2026-09-24: Fix segmentation ROI masks misregistered against their features on
  slides read at their own spacing. When a slide's spacing is within `tolerance`
  of `requested_spacing_um` (say 0.486 µm/px for 0.5), its ROI features cover 512
  px at 0.486 µm/px, but the mask was read at 0.5 µm/px, so it covered about 3%
  more tissue per side and drifted up to 15 px from the image across the ROI.
  Masks are now read at the spacing each ROI's feature grid recorded. A ROI that
  extends past the slide's right or bottom edge now trains with the part beyond
  the slide set to `ignore_index`, where it previously failed the run. Requires
  hs2p 5.0.2. Slide-manifest segmentation runs since the hs2p 5 upgrade should be
  retrained; cached ROI class counts are recomputed once. ROI feature grids must
  record the spacing they were read at, which slide2vec does since 5.7: grids
  cached by slide2vec 5.4–5.6 are re-encoded on the next run, and an up-to-date
  ROI feature cache costs one encoder load, once, before it hits again. Older
  flat-layout caches are not reused. Pre-extracted features passed as
  `Pipeline(feature_dir=...)` that lack the spacing fail the run and must be
  re-extracted. The coverage summary gains `tolerance` (`--tolerance`); pass
  your `preprocessing.tolerance` so its tile estimate matches tiling.

- 2026-09-23: Fix live segmentation (`feature_mode: live`) ignoring
  `task.params.classes` / `ignore`. The live path used raw mask values as class
  indices, so a class scheme that is not the identity trained against the wrong
  classes (or failed with a misleading out-of-range error), and ignored values
  were never excluded. Live and cached runs now remap masks identically, and a
  raw value declared in neither fails the run. Live runs with a class scheme
  should be retrained; cached runs and feature caches are unaffected.

- 2026-09-23: A `preprocessing.masks.pixel_mapping` label may list several raw
  mask values (`tumor: [1, 2]`). They are sampled as one label whose coverage is
  their sum, so a tile 30% value `1` and 30% value `2` passes `min_coverage:
  {tumor: 0.5}`. This is the sampling layer only; `task.params.classes` still
  decides what a segmentation model predicts. List order does not change cache
  keys, and configs with scalar values only keep theirs. Config load now rejects
  a value repeated within or across labels, an empty list, and a value that is
  not an integer in `[0, 255]`. See `preprocessing.rst`.

- 2026-09-23: **Breaking.** Require hs2p 5.0.0 and slide2vec 6.1.0. Source
  masks are now aligned to their slide: a mask may have a lower resolution than
  its slide but must cover it at one scale, within one mask pixel per axis, and a
  spacing tag more than 5% off the dimension-derived spacing fails. This fixes
  segmentation ROI targets read at the wrong place when the annotation mask was
  coarser than its slide. Masks may hold only declared values: tissue masks `0`
  and `1` (a `{1, 255}` mask now fails), annotation masks the values of
  `preprocessing.masks.pixel_mapping`. For whole-slide segmentation, every value
  in `task.params.classes` and `ignore` must also be in `pixel_mapping`; the
  config is rejected otherwise. Slides without resolution metadata, including
  untagged TIFFs read with VIPS, need `spacing_at_level_0`. For flat PNG/JPEG
  slides, disable mask and tiling previews. Tiling and feature caches written
  with hs2p 4 are reused; cached ROI class counts are recomputed once.
  The coverage summary (`python -m soma.curation.segmentation_coverage`) no
  longer requires a `background` label, reports every `pixel_mapping` label
  (including `background`) and gains `--mask-backend`. See `preprocessing.rst`.

- 2026-09-21: Detection takes the same class scheme as segmentation.
  `task.params.classes` maps each class name to the annotated point id(s) that
  form it (several ids merge into one class; ids need not be 0-based) and
  `task.params.drop` lists the ids to discard. A dropped point is removed from
  the targets and from the matching ground truth, so a prediction there is a
  false positive. An id declared in neither fails the run. Metric keys stay
  `f1_class_{c}`; `metrics_<split>.csv` gains a `class_name` column. Configs
  with `num_classes` alone are unchanged, and feature caches are unaffected. See
  `detection.rst`.

- 2026-09-21: **Breaking (segmentation).** Classes are declared on the task:
  `task.params.classes` maps each class name to the raw mask value(s) that form
  it (several values merge into one class; class index = declaration order) and
  `task.params.ignore` lists the raw values excluded from loss and metrics.
  `preprocessing.masks.pixel_mapping` now only selects which ROIs are sampled,
  and `background` is an ordinary label name. Whole-slide segmentation (a
  `masks` block) requires `classes`; to migrate, copy each `pixel_mapping` label
  into `classes` as `name: [value]`, and drop `num_classes` (it is derived). If
  `background` was ignored before (`num_classes` was one less than the label
  count), list its value under `ignore` instead. A mask value declared in
  neither fails the run. Pre-cropped tiles with `num_classes` alone are
  unchanged. Feature caches are unaffected; cached ROI targets re-key
  automatically. See `segmentation.rst`.

- 2026-09-20: Require slide2vec 6.0.1. With `execution.num_gpus > 1`, each
  extraction worker now sees only its own GPU and the launching process no
  longer opens an idle CUDA context (~520 MiB) on every device. No soma API,
  config or cache change; features are unaffected.

- 2026-09-12: Require slide2vec 6.0.0. Pooled cache geometry records the exact
  declared tile size for both preset and off-preset requests. Every tile encoder
  plugin must provide a geometry-preserving normalization transform. Registry
  defaults now use final encoder sizes (GigaPath 224 px, DINOv2 518 px), and
  DINOv3 ViT-B/16 is available at 256 px. Spacing-agnostic DINO presets resolve
  their declared defaults in single and composite pipelines while retaining
  explicit spacing overrides. Regenerate older pooled feature caches
  and GPFM pre-cropped image caches; dependency upgrades do not automatically
  invalidate features. See `caching.rst` and `preprocessing.rst`.

- 2026-09-07: Pooled tile, hierarchical, slide, and patient feature stores now use
  run-local manifests with absolute shared payload paths on cache hits and population.
  Concurrent runs with overlapping evaluation samples and different support sets keep
  their own membership and empty IDs without rewriting shared canonical manifests.

- 2026-09-02: **Version 1.13.0.** Closes out the codebase review (issue #443): several
  behaviour breaks land together — experiment identity v2, EVA curation without the tune
  fraction knob, the DTFD-MIL distillation fix, the resolved tile-image cache key and the
  sample-std / nan-metric reporting described in the entries above. Reporting: the
  comparison ``manifest.json`` records ``soma.__version__`` (``version("soma")`` never
  resolved the ``soma-pathology`` distribution and always wrote null) and report data
  loading tolerates ``task: null`` (task-free representation runs). Heatmaps: the
  no-attention skip list uses the registry names ``mean_pool`` / ``max_pool`` (the old
  ``meanpool`` / ``maxpool`` never matched, so those runs fell through to a failing model
  rebuild), the ``loss`` task param is stripped before rebuilding the head, and the
  slide's level downsamples are read while the reader is open. Packaging: ``torchvision``
  (imported by the dense augmentation and segmentation loaders) is a core dependency and
  ``xgboost`` lives in a new ``pixel`` extra with a lazy-import error naming it.
- 2026-09-02: EVA curation follows the official protocol only. The ``tune_fraction``
  knob (API and ``--tune-fraction`` CLI flag) and the stratified tune carve-out are gone:
  every EVA train sample is soma ``train``; a dataset without an EVA test split reports on
  EVA validation as soma ``test`` (select with ``training.tune_is_test: true``), one with
  a real test split keeps EVA val as ``tune``. BACH curation refuses a photo set that is
  not exactly the official 400 files, since the split index ranges address positions in
  the sorted list. Curation guards: ``write_manifest`` rejects duplicated ``sample_id``
  values; MIDOG derives the sample id through one helper and falls back to it when
  ``patient_id`` is present but null (previously every image collapsed onto patient
  ``"None"``); tiling names ROI rows with a blank ``spacing_at_level_0`` instead of
  listing ``nan``; the Gleason patch writer refuses non-square cores (its enumeration
  is a transposed walk that is only harmless on square TMA cores); the BEETLE split
  rotation picks the next *existing* fold and rejects fewer than two folds.
- 2026-09-02: Benchmark registry hardening. Reference/results tables now treat every
  column left of ``metric`` as a key column (one rule for both tables), a relative
  tolerance band uses ``|expected|`` so a negative expected value cannot yield a negative
  band, and ``append_result`` extends an existing ledger header in place (new key columns
  left of ``metric``, blanks for old rows) instead of silently dropping cells the header
  lacked. Gate rows must be fully keyed and non-zero: ``load_reference`` rejects an
  expected-0 placeholder or a blank key cell on a gate row (external anchors are exempt).
  The MIDOG/MONKEY placeholder gate rows are gone (their external anchors stay) and the
  OCELOT gate row is pinned to its measured anchor (``encoder=virchow2``; spacing is fixed
  by the facet and no longer a key column), so other encoders are no longer compared to
  the Virchow2 band as if it were universal. The multi-dataset ``detection-benchmark``
  object is no longer a registry entry (its curate signature never fit the protocol);
  ``detection/midog`` (primary ``f1``) and ``detection/monkey`` (primary ``mean_froc``)
  are registered as EVA-style per-dataset views with ``encoder`` as the only varied axis.
  The OCELOT greedy re-scorer now honours ``task.params.nms_distance``; CRoMa validates
  only the requested panel encoder when building a config; HEST docstrings no longer claim
  IDC is the only registered task.
- 2026-09-02: **Experiment identity v2 (breaking: pre-1.13 experiment ids differ).**
  The canonical payload now emits every key unconditionally — the "only when
  non-default" guards for ``checkpoint_selection``, ``max_steps``, ROI batch sampling,
  ``normalization``, ``projection`` and ``spacing_policy`` are gone — and excludes an
  explicit list of non-identity knobs: ``training.seed`` (as before),
  ``training.num_workers`` / ``pin_memory`` / ``persistent_workers`` (loader plumbing) and
  the evaluation artifact toggles (``save_segmentation_overlays``,
  ``save_segmentation_probabilities``, ``save_segmentation_confusion_evidence``,
  ``save_detection_overlays``, ``save_detection_heatmaps``, ``overwrite_test``).
  ``identity_version: 2`` is stamped into the payload, ``experiment.json`` and ``run.yaml``.
  Existing run dirs keep their old ids; new runs of the same config land in a new
  experiment dir. Feature, tiling and dense caches are keyed separately and are unaffected.
  The per-root ``indexes/runs.csv`` is now append-only (one line per status change,
  no rewrite, so concurrent runs cannot drop each other's rows); ``read_run_index``
  dedupes by ``run_id`` (last wins) and ``soma compact-index OUTPUT_ROOT`` rewrites the
  file on demand. Git provenance (``git_sha`` / ``git_dirty`` on ``run.yaml`` and the
  ``soma_commit`` column of recorded benchmark rows) is now computed from the soma
  *package* checkout via one helper (``soma.provenance.soma_git_state``), never from the
  working directory, and is null for a wheel install.
- 2026-09-07: WSI tile and hierarchical feature caches now commit each persisted
  slide through slide2vec's `on_slide_persisted` callback (requires slide2vec 5.9.0).
  Each cache population uses one pipeline call, retaining completed slides after
  interruption without repeatedly loading encoders. `cache.commit_every` applies
  only to pre-cropped images; WSI extraction ignores it.
- 2026-09-02: Extraction introduced chunked cache commits (WSI behavior superseded
  by per-slide callbacks on 2026-09-07). Identity signatures were recorded
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
