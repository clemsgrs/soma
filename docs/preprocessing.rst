Preprocessing
=============

Preprocessing selects tissue, extracts tiles, and resolves image geometry.
Spacing controls biological scale; tile size should match the encoder's
expected input.

.. autoclass:: soma.config.PreprocessingConfig
   :members:

Core settings
-------------

.. list-table::
   :header-rows: 1

   * - Field
     - Purpose
   * - ``requested_tile_size_px``
     - Output tile size in pixels.
   * - ``requested_spacing_um``
     - Tiling resolution in microns per pixel.
   * - ``requested_region_size_px`` / ``region_tile_multiple``
     - Region geometry for hierarchical encoders.
   * - ``tissue_method``
     - Tissue segmentation with ``sam2``, ``hsv``, ``otsu``, or ``threshold``.
   * - ``min_coverage.tissue``
     - Minimum tissue fraction required to retain a tile.
   * - ``dense_window_size`` / ``dense_window_overlap``
     - Frozen-encoder windowing for dense tasks; see :doc:`decoders`.
   * - ``feature_kind`` / ``attention``
     - Dense feature substrate; see :doc:`decoders`.

For SAM2, ``sam2_device`` selects the device and ``sam2_num_workers`` limits
concurrent segmentation workers.

Annotation sampling
-------------------

Place ``masks`` and ``sampling`` under ``preprocessing`` when tiles should be
selected from annotation rasters:

.. code-block:: yaml

   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     masks:
       pixel_mapping: {background: 0, tumor: 1}
       min_coverage: {tumor: 0.5}
     sampling:
       strategy: joint
       output_mode: merged

``pixel_mapping`` maps class names to exact raw mask values. Values must be
unique; ``min_coverage`` and optional ``colors`` may reference only mapped
classes. Coverage fractions must lie in ``[0, 1]``.

For ``dataset_type: segmentation``, this configuration selects the
slide-manifest path: each row provides a whole slide and annotation mask, from
which soma samples dense-prediction ROIs. See
:class:`soma.config.MasksConfig`, :class:`soma.config.SamplingConfig`, and
:doc:`segmentation`.

Label remapping
---------------

Class indices follow ``pixel_mapping`` insertion order, not raw pixel value.
Unmapped values become ``ignore_index``.

The name ``background`` is optional. When present, it can represent either an
ignored unannotated region or a real class:

- With ``num_classes == len(pixel_mapping) - 1``, ``background`` maps to
  ``ignore_index`` and every other entry becomes a class.
- With ``num_classes == len(pixel_mapping)``, ``background`` is a real class.
- Without ``background``, every mapped entry is a class and ``num_classes``
  must equal the mapping size.

For example, ``{background: 1, tumor: 2}`` with one task class ignores raw
value ``1`` and maps raw value ``2`` to class ``0``. The same rules are
implemented by :func:`soma.dense.reader.build_label_remap`.

Annotation-restricted bags
--------------------------

On ``dataset_type: slide`` or ``patient``, the same block restricts each
whole-slide bag to tiles meeting a class coverage threshold. The ordinary
slide label, encoder, aggregator, and task remain unchanged; only tile
selection differs. Patient encoders consume the restricted bags produced for
each constituent slide.

Use ``output_mode: merged`` to produce one bag per slide. ``strategy: joint``
samples the union of qualifying classes; ``independent`` samples each class
separately before merging. ``per_annotation`` output is not accepted by the
feature-extraction path, and annotation sampling is not available for
``dataset_type: tile`` or ``detection``.

Annotation selection participates in cache identity, including the mapping,
coverage thresholds, strategy, and output mode. Display colors do not. A
runnable configuration is available at
``examples/slide_tumor_restricted_bag.yaml``.

Scale guidance
--------------

- Use smaller spacing for local morphology and larger spacing for broader
  tissue context.
- Compare a small set of biologically meaningful scales.
- Treat resolved read sizes as internal geometry; configure requested sizes
  and spacing instead.

Preview tissue masks and tile grids with
:func:`soma.preprocessing.overlay_mask_on_slide`,
:func:`soma.preprocessing.save_overlay_preview`, and
:func:`soma.preprocessing.write_coordinate_preview`.
