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

.. note::

   **Migration (soma #109):** ``masks:`` and ``sampling:`` were previously
   top-level config sections. They now live under ``preprocessing:``. A top-level
   ``masks:`` / ``sampling:`` block is no longer accepted — move them under
   ``preprocessing:`` as shown above.

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
