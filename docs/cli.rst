CLI
===

``soma`` exposes a compact command-line interface for running
experiments from YAML config files and for listing the available model
presets.

.. figure:: /_static/figures/run-flow.svg
   :figclass: soma-figure
   :alt: Three input files flow into one soma command that schedules tiling, feature extraction, training, and metrics.

   You provide three files — a dataset, splits, and a config. ``soma`` then
   schedules every step: tiling, feature extraction, training, and metrics.

Basic usage
-----------

The main entrypoint takes a config path directly::

    soma /path/to/config.yaml

You can also invoke it through Python if you prefer::

    python -m soma /path/to/config.yaml

Available commands
------------------

``soma CONFIG``
   Run a full pipeline from the given YAML config file.

``soma list encoders [--level {tile,slide,patient}]``
   List all registered encoder presets. ``--level`` narrows results to
   ``tile``, ``slide``, or ``patient`` encoders.

``soma list aggregators``
   List all registered MIL aggregator presets.

``soma list decoders``
   List all registered dense decoder presets.

``soma list pixel-classifiers``
   List all registered per-pixel classifier presets.

``soma list tasks``
   List all registered task-head presets.

``soma list benchmarks``
   List all registered foundation-model benchmarks — the names
   ``soma reproduce`` and ``soma leaderboard`` accept.

Benchmarking commands
---------------------

These drive the registered benchmarks (see :doc:`benchmarking` for the
end-to-end curate → configure → run → leaderboard → reproduce story).

``soma reproduce NAME [--raw-root DIR | --curated-dir DIR | --from-run-dir DIR] [--seeds N]``
   Curate → run → score a registered benchmark. When a matching packaged
   reference exists, report its delta and highlight potential drift;
   otherwise, explicitly skip the comparison. Reference comparisons are
   informational and never determine command success. ``NAME`` is a
   registered benchmark (``ocelot``, ``eva/bach``) or a family prefix
   (``eva``) that fans out over every ``eva/<dataset>``. Three manifest
   sources: ``--raw-root`` curates from raw data; ``--curated-dir`` reuses an
   already-curated manifest dir (``dataset.csv`` + ``splits.csv``), skipping
   curation; ``--from-run-dir`` re-scores an existing run without retraining.
   ``--seeds 1`` is the quickest smoke.

``soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]``
   Render a faceted leaderboard over the completed run dirs under an
   output root. A benchmark ``NAME`` supplies the canonical facet and
   reference band; ``--vary`` / ``--fix`` / ``--like`` shape the facet on
   top of it.

What the CLI expects
--------------------

The config file follows the canonical nested schema below. This block
is generated from ``soma/configs/default.yaml``, the bundled defaults
merged by :func:`soma.config.load_config`. Copy it when you want the
baseline public YAML shape, then replace neutral defaults such as
``encoder: null`` and ``aggregation: null`` for your run.

Full config reference
---------------------

.. code-block:: yaml

   run:
     output_root: runs
     seed: 0
     tags:
       - baseline

   data:
     dataset_csv: data/dataset.csv
     splits_csv: data/splits.csv
     dataset_type: slide
     # cached: read pre-extracted dense grids. live: re-encode (augmented) tiles through
     # the frozen encoder every step (segmentation only — enables augmentation).
     feature_mode: cached

   preprocessing:
     backend: auto
     requested_tile_size_px: null
     requested_spacing_um: null
     requested_region_size_px: null
     region_tile_multiple: null
     # Tissue segmentation method. Options: sam2 | hsv | otsu | threshold.
     # Leave empty/unused when pre-computed tissue masks are provided.
     tissue_method: hsv
     # Tissue coverage threshold as a masks-shaped map (min_coverage.tissue is the minimum
     # tissue fraction to keep a tile). Single source of truth; no separate scalar.
     min_coverage:
       tissue: 0.1
     overlap: 0.0
     # Dense (segmentation) encoder-window knobs — how the padded supervision tile reaches
     # the frozen encoder (NOT the tiling `overlap` above). dense_window_size: null => the
     # `whole` path (one padded forward; the default and cached-parity anchor). A smaller
     # window (e.g. 224 or 512) slides the encoder over patch-aligned windows and blends the
     # token grids over dense_window_overlap (raised-cosine), useful at large scale-ups.
     dense_window_size: null
     dense_window_overlap: 0.0
     seg_downsample: 64
     sam2_device: cpu
     sam2_num_workers: null
     tolerance: 0.05
     ref_tile_size_px: null
     a_t: 4
     tissue_mask_tissue_value: 1
     preview:
       save_mask_preview: true
       save_tiling_preview: true
       downsample: 32
       tissue_contour_color: [37, 94, 59]

   execution:
     num_gpus: null
     num_preprocessing_workers: null
     prefetch_factor: null
     precision: null

   cache:
     enabled: true
     root_dir: null
     reuse_policy: strict
     fingerprint_files: false
     validate_payloads: false

   # No default encoder — the framework stays neutral on model choice (you must set
   # `encoder:` for a single encoder, or `composite:` for a multi-encoder composite).
   # A baked-in default here would also collide with `composite:` via the encoder/composite
   # XOR check (the merged default encoder would make both present).
   encoder: null

   # No default aggregator — stay neutral on the trainable component. For slide MIL set
   # `aggregation:` explicitly; omitting it means slide-level features with no MIL. A baked-in
   # default would also leak into the tile/patient/segmentation paths (which forbid an
   # aggregator) when a config is hand-written without nulling it.
   aggregation: null

   task:
     name: binary_classification
     params: {}

   # No default metrics — stay neutral (the slide-classification metrics would otherwise leak
   # into segmentation/regression/survival configs and fail metric validation). Set
   # `evaluation.metrics:` for the task at hand.
   evaluation:
     metrics: []
     subgroups:
       columns: []
     # Skip ALL test-split evaluation and report tune only (no test inference, no
     # predictions_test.csv, no `test` entries in metrics.json/summary.json). Use for
     # model-selection sweeps: rank candidates by tune score, then re-run the winner
     # with this off. The test split may stay in splits.csv; it is simply not touched.
     holdout_test: false
     # Allow re-scoring a test set that was already scored for a run, overwriting its
     # prior result. Off by default: because experiment identity is test-invariant, a
     # checkpoint may be scored against several test sets — each result is namespaced by
     # test identity and a re-score of an already-scored test set is refused (loud skip)
     # unless this is set.
     overwrite_test: false

   training:
     # Per-fold trainer: 'gradient' (torch head/decoder loop) or 'ridge_pca_probe'
     # (closed-form Ridge+PCA probe for dataset_type='spatial_expression').
     method: gradient
     epochs: 50
     learning_rate: 1.0e-4
     weight_decay: 1.0e-5
     optimizer: adam
     scheduler: cosine
     # Which epoch's weights are evaluated: 'best' selects by the monitored tune metric
     # (with early stopping); 'last' evaluates the final-epoch weights, takes model
     # selection off the tune metric, and requires `patience: null`.
     checkpoint_selection: best
     # Early-stopping patience in epochs; null disables early stopping (required by
     # checkpoint_selection: last).
     patience: 10
     monitor: tune_loss
     monitor_mode: min
     batch_size: 1
     gradient_accumulation: 1
     tune_is_test: false
     allow_missing_tune: false
     num_workers: 0
     pin_memory: true
     persistent_workers: true

   # Image/mask augmentation — only applied when data.feature_mode is 'live'
   # (segmentation). Geometric ops transform image + mask jointly (mask nearest); the
   # photometric ops transform the image only. All-zero = no augmentation (live-no-aug).
   augmentation:
     horizontal_flip: 0.0
     vertical_flip: 0.0
     rotation_degrees: 0.0
     translate: 0.0
     scale: 0.0
     brightness: 0.0
     contrast: 0.0
     saturation: 0.0
     hue: 0.0

   # Per-feature normalization of frozen encoder features, applied by the feature adaptor
   # (a buffer-carrying front module) ahead of the aggregator/head. 'none' (default) means
   # no adaptor at all. 'zscore' is FITTED on the train split only (leak-free) and its
   # center/scale ride in the checkpoint; 'l2' and 'layernorm' are stateless. `eps` floors
   # the scale so a constant channel cannot blow up. Orthogonal to composite member_norm.
   normalization:
     method: none  # none | zscore | l2 | layernorm
     eps: 1.0e-6  # finite > 0; must stay at this default when method=none

   # Label-free projection of frozen encoder features to a common width — the dim-matched
   # ablation that removes the capacity confound (a wider encoder otherwise buys a larger
   # aggregator). Applied AFTER normalization. When active the aggregator is built against
   # `target_dim`, not the encoder's native dim. 'pca' is FITTED per fold on the train
   # split only (leak-free), centers intrinsically and pins a sign convention so repeated
   # fits are identical; 'random' is a fixed Gaussian matrix seeded from `seed` + the
   # encoder identity, scaled to preserve inner products. Both are frozen buffers, never
   # learned. PCA requires n_fit_rows >= target_dim and target_dim <= D; random is
   # unconstrained.
   projection:
     method: none  # none | pca | random
     target_dim: null  # required when method != none
     seed: 0  # configurable only for random; none/pca require 0

   reports:
     heatmaps:
       enabled: false
       cmap: coolwarm
       alpha: 0.5
       blur_sigma: 0.0

See also
--------

* :doc:`getting-started` – Python API equivalent of each config section
* :doc:`getting-started` – end-to-end walkthrough
