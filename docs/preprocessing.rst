Preprocessing
=============

Preprocessing covers tissue segmentation, tile extraction, and geometry
resolution. In practice, spacing is the primary scale-selection knob because
it controls the biological context seen by the encoder.

The main configuration object is :class:`soma.config.PreprocessingConfig`.

.. autoclass:: soma.config.PreprocessingConfig
   :members:

Key knobs
---------

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Typical use
   * - ``requested_tile_size_px``
     - Tile size requested from the tiler
     - Match encoder expectations
   * - ``requested_spacing_um``
     - Microns per pixel for tiling
     - Coarse vs fine biological context
   * - ``requested_region_size_px``
     - Region size for hierarchical pipelines
     - HIPT-style runs
   * - ``region_tile_multiple``
     - Tile count per hierarchical region
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
     - Minimum tissue fraction to keep a tile (masks-shaped map; ``min_coverage.tissue``)
     - Adjust only if tissue masks are too loose or too strict
   * - ``backend``
     - Slide reader (``auto`` resolves per slide)
     - Pin when a slide decodes correctly under only one reader
   * - ``mask_backend``
     - Reader for tissue/annotation mask rasters (``auto`` follows the slide)
     - Set when a mask needs a different reader than its slide

Geometry is hs2p's vocabulary
-----------------------------

The tiling half of ``PreprocessingConfig`` — ``requested_spacing_um``,
``requested_tile_size_px``, ``tolerance``, ``overlap``, ``min_coverage``,
``backend``, ``mask_backend`` — is hs2p's own ``TilingConfig`` vocabulary, and soma
**composes** that config rather than re-declaring it
(``docs/adr/0009-soma-composes-hs2p-tiling-config.md``).
Composition happens once the encoder has supplied the spacing and tile size soma
leaves unset, and the resulting ``TilingConfig`` is what the pooled path, the
slide-manifest ROI sampler and the feature-cache key all read — so a run's cache key
cannot describe a geometry the run did not use, and a knob hs2p adds does not become
a setting soma users silently cannot reach.

Segmentation slide-manifest sampling
------------------------------------

For the segmentation slide-manifest path (whole slides + annotation masks), the
annotation ``masks`` and ROI ``sampling`` blocks nest **under** ``preprocessing``
because annotation-based tile selection is a preprocessing/tiling concern (this
mirrors slide2vec's own ``PreprocessingConfig``):

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

The presence of ``preprocessing.masks`` selects the slide-manifest input mode
(slides + annotation masks → soma-sampled ROIs → dense grids → segmentation head).
See :class:`soma.config.MasksConfig` and :class:`soma.config.SamplingConfig`.

Annotation vocabulary: background-present vs background-absent
--------------------------------------------------------------

``pixel_mapping`` is the dataset's own raw-pixel vocabulary. **No reserved label name is
required** — soma keeps only structural validation (non-empty mapping, unique pixel values,
``min_coverage``/``colors`` keys ⊆ ``pixel_mapping``, fractions in ``[0, 1]``, RGB shape).
``background`` is an *opt-in* name that only changes how the segmentation-head label remap
(:func:`soma.dense.reader.build_label_remap`) treats unannotated regions.

**Background present** — ``background`` names the unannotated/ignore label. With a class
count one less than the mapping size, ``background`` maps to ``ignore_index`` and the other
labels take class index = their order. (If the class count *equals* the mapping size,
``background`` is instead a real class at index 0.) For example, the BEETLE exact-value
vocabulary ``{background: 1, tumor: 2}`` with one task class:

.. code-block:: yaml

   preprocessing:
     masks:
       pixel_mapping: {background: 1, tumor: 2}   # raw value 1 -> ignore, 2 -> class 0
       min_coverage: {tumor: 0.5}

**Background absent** — every named label is a real class (index = order) and the task's
class count must equal the mapping size. Any raw pixel value *not* in the mapping collapses
to ``ignore_index`` — that is how unannotated regions are expressed without naming them. A
background-free vocabulary like ``{tumor: 2}`` is accepted:

.. code-block:: yaml

   preprocessing:
     masks:
       pixel_mapping: {tumor: 2}   # raw value 2 -> class 0; every other value -> ignore
       min_coverage: {tumor: 0.5}

The "exactly one *named* label is the ignore label" mode (class count == mapping size − 1)
still requires a name: ``background`` stays the opt-in reserved name for that mode only.

.. note::

   **Migration (soma #109):** ``masks:`` and ``sampling:`` were previously
   top-level config sections. They now live under ``preprocessing:``. A top-level
   ``masks:`` / ``sampling:`` block is no longer accepted — move them under
   ``preprocessing:`` as shown above.

Annotation-restricted bags (``dataset_type: slide`` or ``patient``)
-------------------------------------------------------------------

The same ``preprocessing.masks`` block also restricts a **whole-slide MIL bag** to a
chosen compartment. On a ``dataset_type: slide`` dataset, declaring a ``masks`` block
produces **one merged bag per slide containing only the tiles that meet the per-class
coverage threshold** — e.g. a tumor-only bag that excludes the surrounding tissue. The
restricted bag then flows through the ordinary featurizer → aggregator → predictor with
its **ordinary slide-level label** (the dataset's ``label`` column); nothing in the MIL
path changes, only *which* tiles enter the bag.

The same block is also accepted on a ``dataset_type: patient`` dataset (#111): every
slide is tiled to its annotation-restricted merged bag (tiling and selection identical to
the slide path), and patient-level aggregation then consumes those restricted slide bags
to produce **compartment-restricted patient features**. A patient pipeline uses a
pretrained patient encoder rather than a trainable aggregator, so the masks block again
changes only *which* tiles enter each slide's bag.

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

How it works:

- Each dataset row's ``mask_path`` is the multi-class annotation raster (on a classification
  dataset ``mask_path`` is the tile-*sampling* mask, whatever its classes); ``pixel_mapping``
  names the classes (it must include ``background``). Tiles are kept by **per-class**
  ``min_coverage`` over the annotation mask — binary tissue filtering is bypassed, so the
  tissue threshold (``preprocessing.min_coverage.tissue``) does not gate annotation bags.
- The full ``masks`` block — ``pixel_mapping``, per-class ``min_coverage``, ``colors``, an
  explicit ``output_mode``, and ``independent_sampling`` (derived from ``sampling.strategy``)
  — is forwarded into slide2vec's annotation sampling. The default
  ``{background: 0, tissue: 1}`` vocabulary stays byte-for-byte plain tissue tiling; any
  customization opts into annotation sampling.
- A relabeled vocabulary is honored as-is: ``pixel_mapping: {background: 1, tumor: 2}``
  routes to annotation sampling with those exact mask values (there is no reserved
  ``tissue == 1`` value under a ``masks`` block — ``pixel_mapping`` is the single source of
  truth).
- The selection (active ``pixel_mapping`` entries, per-class ``min_coverage``,
  ``strategy``, ``output_mode``) folds into the **cache key**, so a tumor-restricted bag
  never reuses a full-tissue bag's cached tiles/features. ``colors`` is cosmetic and is
  excluded from cache identity.

.. note::

   ``output_mode`` **must be** ``merged`` for ``dataset_type: slide`` and ``patient`` (the
   default). ``output_mode: per_annotation`` (one bag per ``(slide, class)``) is deferred —
   see soma issue #86 — and raises at config load on both. A ``masks`` block is rejected on
   ``dataset_type: tile`` (patch manifests have no annotation-sampling step).

A ready-to-run example lives at ``examples/slide_tumor_restricted_bag.yaml``.

Guidance
--------

- Use smaller spacing when local morphology matters.
- Use larger spacing when the label depends on broader structure.
- Prefer a small set of meaningful spacing values instead of arbitrary
  near-duplicates.

Tissue mask preview
-------------------

Preview rendering is inherited from :mod:`hs2p`:

- :func:`soma.preprocessing.overlay_mask_on_slide` for tissue-mask overlays
- :func:`soma.preprocessing.save_overlay_preview` for writing mask preview images
- :func:`soma.preprocessing.write_coordinate_preview` for tile-grid previews

Read-size fields such as ``read_tile_size_px`` are resolved internally from the
requested tile/region size and spacing, so they are not shown in the
user-facing reference config.
