CLI
===

Use ``soma`` to run YAML experiments, discover components, score
predictions, reproduce benchmarks, and compare completed runs.

.. figure:: /_static/figures/run-flow.svg
   :figclass: soma-figure
   :alt: Three input files flow into one soma command that schedules tiling, feature extraction, training, and metrics.

   You provide three files — a dataset, splits, and a config. ``soma`` then
   schedules every step: tiling, feature extraction, training, and metrics.

Basic usage
-----------

The main entrypoint takes a config path directly::

    soma /path/to/config.yaml

Override individual settings without editing the file::

    soma config.yaml --set run.output_root=runs/local --set training.epochs=5

Repeat ``--set KEY=VALUE`` for multiple overrides. Keys are dotted YAML
paths and values are parsed as YAML, preserving numbers and booleans.
Use ``soma --help`` or ``soma COMMAND --help`` for command options.

Train on features you already have with ``--feature-dir``: the directory of a
pooled feature cache (slide, patient and tile runs) or of dense grids
(segmentation and detection). The run attaches it and extracts nothing::

    soma config.yaml --feature-dir runs/features/uni2

``soma`` prints soma's INFO log messages to stderr, and warnings and errors
from other libraries. See :ref:`log-output` for the same behaviour in a Python
script.

Available commands
------------------

``soma CONFIG [--set KEY=VALUE ...] [--feature-dir DIR]``
   Run a full pipeline from the given YAML config file, optionally on
   pre-extracted features.

``soma list [KIND] [--level {tile,slide,patient}]``
   Without ``KIND``, list the component kinds: ``aggregators``, ``augmentations``, ``benchmarks``, ``callbacks``, ``decoders``, ``encoders``, ``heads``, ``learners``, ``optimizers``, ``samplers``, ``schedulers``.
   With a ``KIND``, list its registered components and the default of
   each of their params; ``soma list benchmarks`` prints the names
   ``soma reproduce`` and ``soma leaderboard`` accept. ``--level``
   narrows the encoders to ``tile``, ``slide``, or ``patient``.
   :func:`soma.components` returns the same listing in Python.

``soma evaluate PREDICTIONS --cohort CSV [--splits CSV --split NAME --fold N] [--metrics M ...] [--subgroups COL ...] [--out JSON]``
   Score saved predictions against the targets in ``dataset.csv``,
   joined by ``sample_id``; no run directory is needed. ``PREDICTIONS``
   is a predictions parquet, a points parquet (detection), or a
   label-map directory (segmentation). A predictions parquet is scored
   against the cohort's target columns, points against the files in its
   ``points_path`` column, and label maps against the class-index masks
   in its ``label_mask_path`` column. ``--splits``
   selects one split (default ``test``) of one fold (default ``0``);
   without it every row of the cohort is a target. Every prediction
   needs a target and every target a prediction. ``--metrics`` takes
   metric names or dotted import paths (default: the task family's
   defaults); ``--subgroups`` names metadata columns to break the
   metrics down by. The metrics table is printed; ``--out`` also writes
   the evaluation as JSON. See :doc:`evaluation`.

``soma compact-index OUTPUT_ROOT``
   Compact ``indexes/runs.csv`` to the latest row per run. Readers
   already deduplicate the append-only index; compaction saves space.
   A path to the CSV itself is also accepted. See :doc:`outputs`.

Benchmarking commands
---------------------

``soma prepare-croma RAW_ROOT [--rebuild]``
   Download and decode the pinned PathoROB tile sources for
   :doc:`croma-robustness-benchmark`. ``--rebuild`` replaces a partial
   or revision-mismatched destination.

``soma reproduce NAME [--encoder NAME | --encoders NAME [NAME ...]] [--raw-root DIR | --curated-dir DIR | --from-run-dir DIR] [--seeds N]``
   Curate, run, and score a registered benchmark. ``NAME`` is a benchmark
   (``ocelot``, ``eva/bach``) or a family prefix (``eva``). ``--raw-root``
   curates from raw data, ``--curated-dir`` reuses prepared manifests, and
   ``--from-run-dir`` rescores one existing run. ``--encoders`` runs an
   ordered panel and writes one leaderboard per benchmark. ``--seeds N``
   runs seeds 0 through N−1. ``--output-root``, ``--cache-root``, and
   ``--out-dir`` place run artifacts, shared features, and curated
   manifests; ``--record`` appends the score to the packaged results
   ledger. See :doc:`benchmarking` for panel validation, partial
   failures, and reference comparisons.

``soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]``
   Render a faceted leaderboard over the completed run dirs under an
   output root. A benchmark ``NAME`` supplies the canonical facet and
   reference band; ``--vary`` / ``--fix`` / ``--like`` shape the facet on
   top of it.

Full config reference
---------------------

The YAML below is generated from ``soma/configs/default.yaml``, which
:func:`soma.config.load_config` merges with your file. Set the data paths,
encoder, and task-specific components for your run. The
:doc:`getting-started` guide provides a runnable configuration shape.
YAML uses ``aggregation`` for the Python ``aggregator`` argument.

.. code-block:: yaml

   run:
     output_root: runs
     mirror_root: null
     seed: 0
     tags:
       - baseline

   data:
     dataset_csv: data/dataset.csv
     splits_csv: data/splits.csv
     dataset_type: slide
     feature_mode: cached

   preprocessing:
     backend: auto
     requested_tile_size_px: null
     requested_spacing_um: null
     requested_region_size_px: null
     region_tile_multiple: null
     tissue_method: hsv
     min_coverage:
       tissue: 0.1
     overlap: 0.0
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
     validate_payloads: false

   encoder: null

   aggregation: null

   task:
     name: binary_classification
     params: {}

   evaluation:
     metrics: []
     subgroups:
       columns: []
     holdout_test: false
     overwrite_test: false

   training:
     epochs: 50
     learning_rate: 1.0e-4
     weight_decay: 1.0e-5
     optimizer: adam
     scheduler: cosine
     learner: null
     checkpoint_selection: best
     patience: 10
     # Hold the early stop while the monitor sits at its first value; false = plain patience.
     hold_patience_until_monitor_moves: true
     # A validation_summary key: `loss` or one of the task's metrics.
     monitor: loss
     monitor_mode: min
     batch_size: 1
     roi_batch_sampling: null
     class_request_ratios: null
     roi_draws_per_epoch: null
     gradient_accumulation: 1
     tune_is_test: false
     allow_missing_tune: false
     num_workers: 0
     pin_memory: true
     persistent_workers: true

   augmentation: null

   normalization:
     method: none  # none | zscore | l2 | layernorm
     eps: 1.0e-6  # finite > 0; must stay at this default when method=none

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

* :doc:`api` — Python interfaces and recipes.
* :doc:`modeling` — supported component combinations.
