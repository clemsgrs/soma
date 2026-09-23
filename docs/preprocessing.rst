Preprocessing
=============

Preprocessing selects tissue or annotated regions and resolves tile geometry.
Spacing controls the physical scale seen by the encoder: smaller µm/px values
show finer morphology; larger values include broader structure.

The main configuration object is :class:`soma.config.PreprocessingConfig`.

.. autoclass:: soma.config.PreprocessingConfig
   :members:

Key settings
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Typical use
   * - ``requested_tile_size_px``
     - Requested tile size and final pooled encoder input size
     - Match encoder expectations
   * - ``requested_spacing_um``
     - Microns per pixel for tiling
     - Coarse vs fine biological context
   * - ``spacing_policy``
     - How to handle a source coarser than the requested spacing
     - Use ``native_if_coarser`` to forbid synthetic upsampling
   * - ``requested_region_size_px``
     - Region size for hierarchical pipelines
     - HIPT-style runs
   * - ``region_tile_multiple``
     - Tiles per side of a hierarchical region
     - Alternative to ``requested_region_size_px``
   * - ``tissue_method``
     - Tissue segmentation method: ``sam2``, ``hsv``, ``otsu``, or ``threshold``
     - Leave empty/unused when dataset rows provide pre-computed tissue masks
   * - ``sam2_device``
     - Device used for SAM2 tissue segmentation
     - Set explicitly when running SAM2 on GPU
   * - ``sam2_num_workers``
     - Cap on concurrent SAM2 tissue-segmentation workers
     - Reduce GPU memory pressure on smaller cards
   * - ``min_coverage.tissue``
     - Minimum tissue fraction to keep a tile
     - Adjust only if tissue masks are too loose or too strict
   * - ``backend``
     - Slide reader (``auto`` resolves per slide)
     - Pin when a slide decodes correctly under only one reader
   * - ``mask_backend``
     - Reader for tissue/annotation mask rasters (``auto`` resolves from the mask
       file, ``pil`` for PNG/JPEG)
     - Set when a mask needs a different reader than its slide

Tiling configuration
--------------------

soma composes hs2p's ``TilingConfig`` for ``requested_spacing_um``,
``requested_tile_size_px``, ``tolerance``, ``overlap``, ``min_coverage``,
``backend``, and ``mask_backend``. Unset spacing and tile size are resolved from
the encoder. Extraction, ROI sampling, and cache keys use that resolved config.
Read-size fields such as ``read_tile_size_px`` are derived internally.

Declared pooled extraction encodes exactly ``requested_tile_size_px``, applying
only the encoder's photometric transform after tiling. Defaults describe the
final model input: GigaPath uses 224 px, DINOv2 518 px, and DINOv3 256 px. At
fixed spacing, changing this size changes the sampled physical extent.

An explicit off-preset pooled size requires
``encoder.allow_non_recommended_settings: true`` and an encoder that supports
variable input sizes; the flag cannot bypass that capability check.
Pre-cropped classification images (``dataset_type: tile``) instead use the
encoder's shipped image transform, including its resizing and cropping.

Coarser source spacing
----------------------

``spacing_policy`` controls what happens when a manifest's declared
``spacing_at_level_0`` is coarser than ``requested_spacing_um`` beyond the configured
relative ``tolerance``. The default, ``strict``, preserves the requested spacing and lets
the reader reject a request that would require forbidden upsampling.
``native_if_coarser`` instead uses the declared level-0 spacing for that sample. Sources
within tolerance still use the requested spacing, and samples without a declared native
spacing retain the requested value.

The effective per-sample spacing is applied consistently to image extraction and
annotation-mask reads so targets remain registered to dense feature grids. The non-default
policy participates in ROI and feature-cache identities, and the effective spacing is
recorded in cache sidecars and run provenance.

.. _preprocessing-source-masks:

Source masks
------------

Tissue masks (``mask_path``) and annotation masks (``label_mask_path``) may have a
lower resolution than their slide, but must cover the same field of view at one
scale, within one mask pixel per axis. soma reads them on the slide's grid. A
mask of another shape fails, as does one whose spacing tag is more than 5% off
the spacing its dimensions imply; a tag 1–5% off logs a warning. PNG/JPEG and
untagged TIFF masks need no spacing metadata.

A mask may hold only the values it declares. A tissue mask holds ``0``
(background) and ``1`` (tissue): a ``{1, 255}`` or ``{0, 255}`` mask fails and
must be rewritten. An annotation mask holds only the values of ``pixel_mapping``
(see `Annotation labels`_).

A slide without resolution metadata needs ``spacing_at_level_0`` in
``dataset.csv``, including an untagged TIFF read with ``vips``.

Tiling artifacts record the mask level and spacing actually read
(``mask_level``, ``mask_spacing_um``). Since hs2p 5 selects the mask level with
a fixed 1% tolerance, these can differ from runs made with hs2p 4. Cache keys
depend only on the configuration, so tiling and feature caches written with
hs2p 4 are reused.

Segmentation slide-manifest sampling
------------------------------------

For segmentation from whole slides and annotation masks, place ``masks`` and
ROI ``sampling`` under ``preprocessing``:

.. code-block:: yaml

   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     masks:
       pixel_mapping: {background: 0, tumor: 1}
       min_coverage: {tumor: 0.5}
     sampling:
       output_mode: merged
       strategy: joint

For ``dataset_type: segmentation``, ``preprocessing.masks`` selects the
slide-manifest input mode: soma samples ROIs from slides and their
``label_mask_path`` annotations, extracts dense grids, then fits a segmentation
head. Without it, segmentation uses pre-cropped tiles. :doc:`segmentation`
compares the two modes and explains how to choose the tile and window sizes.
See :class:`soma.config.MasksConfig` and :class:`soma.config.SamplingConfig`.

Annotation labels
-----------------

``pixel_mapping`` maps class names to raw mask values. It must be non-empty,
and each value is an integer in ``[0, 255]`` listed under one label only;
``min_coverage`` and ``colors`` may only name mapped classes. Coverage fractions
must lie in ``[0, 1]`` and colors must be valid RGB triples. No reserved label
name is required.

A label may list several raw values. They are sampled as one label whose
coverage is their sum:

.. code-block:: yaml

   masks:
     pixel_mapping: {background: 0, tumor: [1, 2], stroma: 3}
     min_coverage: {tumor: 0.5}

A tile that is 30% value ``1`` and 30% value ``2`` is 60% ``tumor`` and is
sampled; with ``tumor: 1`` and a separate label for ``2``, neither reaches 0.5.
The order of a list does not matter, and configurations without lists keep
their cache keys.

``pixel_mapping`` must declare every value the annotation masks hold: a mask
holding any other value fails. Declare a value that should never select a tile,
such as unannotated background, and leave it out of ``min_coverage``.

These labels only select tiles and ROIs. For segmentation, the classes the model
predicts are declared separately under ``task.params.classes`` and
``task.params.ignore`` (:doc:`segmentation`), so a label here may be split or
merged differently at training time: ``tumor: [1, 2]`` above may still train as
two classes. Every raw value those list must also be in
``pixel_mapping``; the config is rejected otherwise.

Annotation-restricted bags (``dataset_type: slide`` or ``patient``)
-------------------------------------------------------------------

For ``dataset_type: slide``, ``preprocessing.masks`` restricts each MIL bag to
tiles meeting the annotation coverage thresholds. The bag retains the slide's
``label`` and passes through the configured aggregator and predictor.

For ``dataset_type: patient``, each slide supplies a restricted bag to the
pretrained patient encoder, producing compartment-restricted patient features.

.. code-block:: yaml

   data:
     dataset_type: slide          # whole-slide MIL, not segmentation
   preprocessing:
     requested_tile_size_px: 224
     requested_spacing_um: 0.5
     tissue_method: otsu
     masks:
       pixel_mapping: {background: 0, tumor: 1}
       min_coverage: {tumor: 0.5}   # keep tiles ≥ 50% tumor; tissue-only tiles are excluded
     sampling:
       output_mode: merged          # one merged bag per slide (required; see below)
       strategy: joint              # joint across classes; 'independent' tiles each class separately
   aggregation:
     name: abmil                    # any existing MIL aggregator
   task:
     name: binary_classification

Each row's ``mask_path`` is the annotation raster and ``pixel_mapping`` names
its classes exactly as stored; tiles are kept by per-class ``min_coverage``, and
the tissue threshold does not apply. The default ``{background: 0, tissue: 1}``
vocabulary is plain tissue tiling; any other mapping opts into annotation
sampling. The selection (``pixel_mapping``, ``min_coverage``, ``strategy``,
``output_mode``) is part of the cache key, so a restricted bag never reuses a
full-tissue bag's features; ``colors`` is cosmetic. A ``masks`` block is rejected
on ``dataset_type: tile``.

A ready-to-run example lives at ``examples/slide_tumor_restricted_bag.yaml``.

Tissue mask preview
-------------------

Preview rendering is inherited from :mod:`hs2p`:

- :func:`soma.preprocessing.overlay_mask_on_slide` for tissue-mask overlays
- :func:`soma.preprocessing.save_overlay_preview` for writing mask preview images
- :func:`soma.preprocessing.write_coordinate_preview` for tile-grid previews

For flat PNG/JPEG slides, set ``preview.save_mask_preview`` and
``preview.save_tiling_preview`` to ``false``: hs2p 5.0.0's preview renderers
reopen the slide without its ``spacing_at_level_0``. PNG/JPEG masks on slides
with native spacing work with previews enabled.
