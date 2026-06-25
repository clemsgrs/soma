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

The config file follows the canonical nested schema below. Every key is
optional except those marked *required*. Omit a section entirely to
accept all its defaults.

Full config reference
---------------------

.. code-block:: yaml

   # ── Run ──────────────────────────────────────────────────────────
   run:
     output_root: runs          # required – directory for run artifacts
     seed: 0
     tags:
       - baseline               # free-form labels stored in metadata

   # ── Data ─────────────────────────────────────────────────────────
   data:
     dataset_csv: data/dataset.csv   # required – slide list and labels
     splits_csv: data/splits.csv     # required – train/tune/test folds
     dataset_type: slide             # required – slide | tile | patient

   # ── Preprocessing ────────────────────────────────────────────────
   preprocessing:
     backend: auto                   # auto | hs2p | sam2
     requested_spacing_um: null      # primary scale knob (µm/px)
     requested_tile_size_px: null    # tile edge length at the target spacing
     requested_region_size_px: null  # HIPT region size (hierarchical only)
     region_tile_multiple: null      # tiles-per-region (hierarchical only)
     # Tissue segmentation method. Options: sam2 | hsv | otsu | threshold.
     # Leave empty/unused when pre-computed tissue masks are provided.
     tissue_method: hsv
     min_coverage:                   # tissue coverage threshold (min tissue fraction per tile)
       tissue: 0.1
     overlap: 0.0
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

   # ── Cache ────────────────────────────────────────────────────────
   cache:
     enabled: true
     root_dir: null              # null → inside output_root
     reuse_policy: strict        # strict | relaxed
     fingerprint_files: false    # hash slide/mask contents for cache identity
     validate_payloads: false    # load cached tensors to verify shape/dim

   # ── Encoder ──────────────────────────────────────────────────────
   encoder:
     name: uni2                  # required – see `soma list encoders`
     batch_size: 32
     adaptive_batching: false
     output_variant: null        # preset-specific feature variant
     allow_non_recommended_settings: false
     save_tile_features: false

   # ── Aggregation (slide dataset_type only) ────────────────────────
   aggregation:
     name: abmil                 # see `soma list aggregators`
     params:
       hidden_dim: 256
       dropout: 0.25

   # ── Task ─────────────────────────────────────────────────────────
   task:
     name: binary_classification  # required – see `soma list tasks`
     params: {}

   # ── Evaluation ───────────────────────────────────────────────────
   evaluation:
     metrics:
       - auroc
       - balanced_accuracy
     subgroups:
       columns: []              # dataset.csv columns for metric breakdowns

   # ── Training ─────────────────────────────────────────────────────
   training:
     epochs: 50
     learning_rate: 1.0e-4
     weight_decay: 1.0e-5
     optimizer: adam            # adam | sgd | adamw
     scheduler: cosine          # cosine | step | none
     patience: 10               # early-stopping patience (epochs)
     monitor: tune_loss         # tune_loss or a tune metric name
     monitor_mode: min          # min | max
     batch_size: 1
     gradient_accumulation: 1
     tune_is_test: false
     allow_missing_tune: false

   # ── Reports ──────────────────────────────────────────────────────
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
