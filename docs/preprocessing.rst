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
     - Tissue segmentation method
     - Keep default unless segmentation is noisy
   * - ``tissue_threshold``
     - Segmentation threshold
     - Adjust only if tissue masks are too loose or too strict

Guidance
--------

- Use smaller spacing when local morphology matters.
- Use larger spacing when the label depends on broader structure.
- Prefer a small set of meaningful spacing values instead of arbitrary
  near-duplicates.

Tissue mask preview
-------------------

Preview rendering is delegated to :mod:`hs2p` so soma does not keep a separate
renderer in sync. Use the hs2p entrypoints directly, or import them from the
soma package root:

- :func:`hs2p.wsi.overlay_mask_on_slide` for tissue-mask overlays
- :func:`hs2p.wsi.save_overlay_preview` for writing mask preview images
- :func:`hs2p.wsi.write_coordinate_preview` for tile-grid previews

The rendering behavior itself lives upstream.
