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

:func:`soma.preprocessing.preview.render_preview` produces a combined
visualization of the tissue mask and tiling grid given a
:class:`hs2p.wsi.reader.SlideReader`, a binary tissue mask array, and a
:class:`hs2p.preprocessing.TilingResult`. It returns a NumPy image array that
can be saved with :func:`soma.preprocessing.preview.save_preview`.

Use this during data exploration to verify that the segmentation threshold and
tile size produce sensible results before committing to a full extraction run.
