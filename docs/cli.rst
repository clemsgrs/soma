CLI Guide
=========

``soma`` exposes a compact command-line interface for running
experiments from YAML config files and for listing the available model
presets.

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

   training:
     epochs: 50
     learning_rate: 1.0e-4
     weight_decay: 1.0e-5
     optimizer: adam
     scheduler: cosine
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

   reports:
     heatmaps:
       enabled: false
       cmap: coolwarm
       alpha: 0.5
       blur_sigma: 0.0

See also
--------

* :doc:`pipeline` – Python API equivalent of each config section
* :doc:`getting-started` – end-to-end walkthrough
