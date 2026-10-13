Preprocessing
=============

Preprocessing selects tissue or annotated regions and resolves tile geometry.
Spacing controls the physical scale seen by the encoder: smaller µm/px values
show finer morphology; larger values include broader structure.

Tiling is hs2p's and extraction is slide2vec's, so soma holds their configuration
objects rather than its own copies (ADR 0019). ``features.extraction`` is
slide2vec's ``PreprocessingConfig``, which composes hs2p's ``TilingConfig``
(``tiling``), ``SegmentationConfig`` (``segmentation``), ``FilterConfig``
(``filter``), ``MaskSamplingConfig`` (``masks``) and ``PreviewConfig``
(``preview``). The YAML spells the same fields as the Python objects, and a field
left out keeps hs2p's or slide2vec's default:

.. code-block:: yaml

   features:
     encoder: {name: uni2}
     extraction:
       tiling:
         requested_spacing_um: 0.5
         requested_tile_size_px: 224
         tolerance: 0.05
       segmentation:
         method: otsu
       masks:
         pixel_mapping: {background: 0, tissue: 1}
         min_coverage: {tissue: 0.1}

Key settings
------------

.. list-table::
   :header-rows: 1

   * - Field (under ``features``)
     - Meaning
     - Typical use
   * - ``extraction.tiling.requested_tile_size_px``
     - Requested tile size and final pooled encoder input size
     - Match encoder expectations
   * - ``extraction.tiling.requested_spacing_um``
     - Microns per pixel for tiling
     - Coarse vs fine biological context
   * - ``spacing_policy``
     - How to handle a source coarser than the requested spacing
     - Use ``native_if_coarser`` to forbid synthetic upsampling
   * - ``extraction.requested_region_size_px``
     - Region size for hierarchical pipelines
     - HIPT-style runs
   * - ``extraction.region_tile_multiple``
     - Tiles per side of a hierarchical region
     - Alternative to ``requested_region_size_px``
   * - ``extraction.segmentation.method``
     - Tissue segmentation method: ``sam2``, ``hsv``, ``otsu``, ``threshold``, or
       ``precomputed_mask``
     - Set ``precomputed_mask`` when dataset rows provide tissue masks; soma
       selects it automatically when every row has one
   * - ``extraction.segmentation.sam2_device``
     - Device used for SAM2 tissue segmentation
     - Set explicitly when running SAM2 on GPU
   * - ``extraction.segmentation.sam2_num_workers``
     - Cap on concurrent SAM2 tissue-segmentation workers
     - Reduce GPU memory pressure on smaller cards
   * - ``extraction.masks.min_coverage.tissue``
     - Minimum tissue fraction to keep a tile
     - Adjust only if tissue masks are too loose or too strict
   * - ``extraction.tiling.backend``
     - Slide reader (``auto`` resolves per slide)
     - Pin when a slide decodes correctly under only one reader
   * - ``extraction.tiling.mask_backend``
     - Reader for tissue/annotation mask rasters (``auto`` resolves from the mask
       file: ``pil`` for PNG/JPEG, ``tifffile`` for TIFF masks that store
       samples other than 8-bit unsigned, such as 16-bit labels)
     - Set when a mask needs a different reader than its slide

Tiling configuration
--------------------

``features.extraction`` holds the upstream objects as given:

- ``tiling`` is hs2p's ``TilingConfig`` (``requested_spacing_um``,
  ``requested_tile_size_px``, ``tolerance``, ``overlap``, ``backend``,
  ``mask_backend``);
- ``segmentation`` is hs2p's ``SegmentationConfig`` (``method``, ``downsample``,
  ``sam2_device``, ``sam2_num_workers``, ...);
- ``filter`` is hs2p's ``FilterConfig`` (``a_t``, ``ref_tile_size``, ...);
- ``masks`` is hs2p's ``MaskSamplingConfig`` (``pixel_mapping``,
  ``min_coverage``, ``strategy``, ``output_mode``, ``colors``);
- ``requested_region_size_px`` and ``region_tile_multiple`` are slide2vec's
  ``PreprocessingConfig`` fields.

An unset tiling (spacing and tile size) and filter are resolved from the encoder
preset when the run resolves; a grid run takes them from ``features.dense``.
Extraction, ROI sampling, and cache keys use that resolved config. Read sizes
are derived by hs2p and slide2vec. Cache keys are built from the fields of these
upstream objects, so a field added upstream is keyed without a change in soma;
caches written by soma releases that keyed its own copies are not reused.

An unset filter takes hs2p's defaults, with ``ref_tile_size_px`` at the tile
size. Tissue tiling then drops tissue regions with an area of at most ``a_t: 4``
tiles. Annotation sampling keeps annotated regions of any size (``a_t: 0``). In
YAML, ``a_t: 4`` without ``ref_tile_size_px`` counts as unset: also set
``ref_tile_size_px`` to drop small annotated regions.

Declared pooled extraction encodes exactly ``requested_tile_size_px``, applying
only the encoder's photometric transform after tiling. Defaults describe the
final model input: GigaPath uses 224 px, DINOv2 224 px, and DINOv3 256 px. At
fixed spacing, changing this size changes the sampled physical extent.

An explicit off-preset pooled size requires
``features.encoder.params.allow_non_recommended_settings: true`` and an encoder that supports
variable input sizes; the flag cannot bypass that capability check.
Pre-cropped classification images (a tile-level run) instead use the
encoder's shipped image transform, including its resizing and cropping.

Coarser source spacing
----------------------

``spacing_policy`` controls what happens when a manifest's declared
``spacing_at_level_0`` is coarser than the requested spacing beyond the configured
relative ``tolerance``. The default, ``strict``, preserves the requested spacing and lets
the reader reject a request that would require forbidden upsampling.
``native_if_coarser`` instead uses the declared level-0 spacing for that sample. Sources
within tolerance still use the requested spacing, and samples without a declared native
spacing retain the requested value.

The effective per-sample spacing is applied consistently to image extraction and
annotation-mask reads so targets remain registered to dense feature grids. The policy is
part of the ROI-sampling cache identity, and so of the dense caches built on its ROIs, and
the effective spacing is recorded in cache sidecars and run provenance. It is a soma
setting (``features.spacing_policy``) because hs2p and slide2vec have no field for
it.

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
depend only on the configuration, not on the hs2p version, so a cache written
under an older hs2p is reused. This holds for tilings cached with hs2p 5.0.2, which
hs2p 5.1.0 would tile differently next to tissue holes and, with ``overlap > 0``
read above level 0, on a slightly different stride. Delete the tiling cache to
re-tile. hs2p 6 records ``null`` segmentation thresholds (``seg_sthresh`` and
the like) for a tiling made from a precomputed tissue mask or by annotation
sampling, where no segmentation ran; tilings cached under hs2p 5 still load.
With hs2p 6, ``backend: auto`` and ``mask_backend: auto`` skip a reader that
opens a file but cannot decode it, so a deflate-compressed label TIFF no longer
needs an explicit ``mask_backend``.

Segmentation slide-manifest sampling
------------------------------------

For segmentation from whole slides and annotation masks, declare the annotation
vocabulary and the ROI sampling in ``features.extraction.masks``:

.. code-block:: yaml

   features:
     dense:
       target_size: 512
       spacing_um: 0.5
     extraction:
       masks:
         pixel_mapping: {background: 0, tumor: 1}
         min_coverage: {tumor: 0.5}
         output_mode: merged
         strategy: joint

For a segmentation head, the input mode is inferred from the data: a
cohort whose ``dataset.csv`` has a ``label_mask_path`` column, extracted with an
annotation vocabulary (any ``pixel_mapping`` other than hs2p's
``{background: 0, tissue: 1}``), is a slide manifest. soma samples ROIs from the
slides and their ``label_mask_path`` annotations, extracts dense grids, then fits
a segmentation head. Otherwise segmentation uses pre-cropped tiles.
:doc:`segmentation` compares the two modes and explains how to choose the tile
and window sizes.

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

   features:
     extraction:
       masks:
         pixel_mapping: {background: 0, tumor: [1, 2], stroma: 3}
         min_coverage: {tumor: 0.5}

A tile that is 30% value ``1`` and 30% value ``2`` is 60% ``tumor`` and is
sampled; with ``tumor: 1`` and a separate label for ``2``, neither reaches 0.5.
The order of a list does not matter.

``pixel_mapping`` must declare every value the annotation masks hold: a mask
holding any other value fails. Declare a value that should never select a tile,
such as unannotated background, and leave it out of ``min_coverage``.

These labels only select tiles and ROIs. For segmentation, the classes the model
predicts are declared separately by the head's ``classes`` and ``ignore``
params (:doc:`segmentation`), so a label here may be split or
merged differently at training time: ``tumor: [1, 2]`` above may still train as
two classes. Every raw value those list must also be in
``pixel_mapping``; the config is rejected otherwise.

Annotation-restricted bags (slide and patient runs)
---------------------------------------------------

For a slide-level MIL run, an annotation vocabulary in
``features.extraction.masks`` restricts each bag to tiles meeting the annotation
coverage thresholds. The bag retains the slide's ``label`` and passes through the
configured aggregator and head.

For a patient run (``cohort.collapse: patient_id``), each slide supplies a
restricted bag to the pretrained patient encoder, producing
compartment-restricted patient features.

.. code-block:: yaml

   features:
     encoder: {name: uni2}
     extraction:
       tiling:
         requested_tile_size_px: 224
         requested_spacing_um: 0.5
       segmentation:
         method: otsu
       masks:
         pixel_mapping: {background: 0, tumor: 1}
         min_coverage: {tumor: 0.5}  # keep tiles ≥ 50% tumor; tissue-only tiles are excluded
         output_mode: merged         # one merged bag per slide (required; see below)
         strategy: joint             # joint across classes; 'independent' tiles each class separately
   learner:
     name: mil
     params:
       aggregator: {name: abmil}     # any existing MIL aggregator
       head: {name: binary_classification}

Each row's ``mask_path`` is the annotation raster and ``pixel_mapping`` names
its classes exactly as stored; tiles are kept by per-class ``min_coverage``, and
the tissue threshold does not apply. The default ``{background: 0, tissue: 1}``
vocabulary is plain tissue tiling; any other mapping opts into annotation
sampling. The selection (``pixel_mapping``, ``min_coverage``, ``strategy``,
``output_mode``) is part of the cache key, so a restricted bag never reuses a
full-tissue bag's features; ``colors`` is cosmetic. An annotation vocabulary is
rejected on a tile-level run.

A ready-to-run example lives at ``examples/slide_tumor_restricted_bag.yaml``.

.. _preprocessing-supplied-coordinates:

Bring your own coordinates
--------------------------

soma normally tiles each slide itself. To use a fixed tile set instead (a benchmark's
own sampler, coordinates released with a paper, or a frozen set for an ablation), add a
``coordinates_path`` column to a slide-level ``dataset.csv``. Each row names one hs2p
tiling artifact: the ``<name>.coordinates.npz`` written by
``hs2p.artifacts.save_tiling_result``, with its ``<name>.coordinates.meta.json`` beside
it. The artifact holds level-0 tile origins and the read geometry (``read_level``,
``read_tile_size_px``, ``requested_tile_size_px``, ``requested_spacing_um``).

.. code-block:: text

   sample_id,image_path,label,coordinates_path
   s01,/slides/s01.tif,1,/coords/s01.coordinates.npz
   s02,/slides/s02.tif,0,/coords/s02.coordinates.npz

soma then skips tissue segmentation and tiling, and slide2vec embeds exactly the
listed tiles. soma copies each artifact into the run's ``tiling`` directory, so the
run keeps its tiles if the original changes. It checks each artifact first and
stops with an error when:

- its ``sample_id``, ``image_path`` or ``spacing_at_level_0`` differs from the
  manifest row;
- its ``requested_spacing_um`` or ``requested_tile_size_px`` differs from
  ``features.extraction.tiling``;
- only some rows set ``coordinates_path``.

The content of each artifact is part of the slide's feature-cache identity. The same
artifact reuses its features across runs; a different tile set is extracted again, and
its features are cached next to the first set's, not over them.

A resumed run keeps the copies it made when it started. If an original artifact was
deleted since, the run uses its copy; if it was changed, the run stops rather than
train its remaining folds on other tiles.

The experiment identity does not follow the artifacts. Like the other path columns,
``coordinates_path`` is left out of the dataset checksum, and soma does not open the
artifacts to compute one (see :doc:`dataset`). Two runs that differ only in their tile
sets therefore share an experiment identity and a leaderboard row. To compare tile sets,
give each its own identity with an extra column, such as a checksum of each artifact
(``coordinates_path_sha256``) or the name and version of the sampler that made it:

.. code-block:: text

   sample_id,image_path,label,coordinates_path,coordinates_path_sha256
   s01,/slides/s01.tif,1,/coords/s01.coordinates.npz,9f2c...
   s02,/slides/s02.tif,0,/coords/s02.coordinates.npz,41ab...

Supplied coordinates work for pooled whole-slide bags (slide-level MIL runs) built
from a tile encoder; slide- and patient-level encoders reject them.
They cannot be combined with an annotation vocabulary or hierarchical regions, and
tile, segmentation, and detection manifests reject the column. The segmentation
method is not used.

Tissue mask preview
-------------------

Preview rendering is inherited from :mod:`hs2p`:

- :func:`soma.extract.preprocessing.overlay_mask_on_slide` for tissue-mask overlays
- :func:`soma.extract.preprocessing.save_overlay_preview` for writing mask preview images
- :func:`soma.extract.preprocessing.write_coordinate_preview` for tile-grid previews
