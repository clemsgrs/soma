Segmentation
============

A dense **semantic-segmentation** path: predict a per-pixel class map for each
tile. Segmentation is the canonical **dense contract** — the shared front half and
output machinery that the :doc:`detection` path and the decoder-free
:doc:`pixel-classifier <decoders/pixel-classifier>` method all build on. A **frozen**
foundation-model encoder produces a dense ``(d, grid_h, grid_w)`` token grid (cached as
``feature_type="dense_grid"``); a **decoder** turns that grid into a per-pixel class map;
and a :class:`~soma.tasks.segmentation.SegmentationHead` owns the geometry, loss, metric,
and prediction artifacts. Only the trainable component and the output representation differ
across the dense paths.

.. seealso::

   The :doc:`segmentation walkthrough <tutorials/walkthrough-segmentation>` runs the
   neural-decoder segmentation path end to end on a tiny synthetic dataset;
   :doc:`detection <tutorials/walkthrough-detection>` is the same dense flow with point
   supervision, so you can see exactly what changes between the two.

The dense contract
------------------

Every dense path shares the same front half and output machinery; this is the contract a
new dense task plugs into.

* **Manifest** — ``dataset_type: segmentation`` uses :class:`soma.dataset.SegmentationManifest`.
  The supervision is a per-sample **mask raster** (``mask_path``), not a scalar ``label``.
* **Spacing-aware mask reader** — masks are read at the run's spacing and registered
  against the token grids extracted at the same spacing (:mod:`soma.dense.reader`). The
  annotation vocabulary (``pixel_mapping``, per-class ``min_coverage``, the
  background-present vs background-absent label remap) is a preprocessing concern — see
  :doc:`preprocessing`.
* **Frozen dense extraction** — a frozen encoder emits a dense ``(d, grid)`` grid, cached
  as ``feature_type="dense_grid"``. No gradients flow through the backbone; only the
  trainable component on the grid is fit.
* **Window-as-knob extraction** — the encoder is read at its native **spacing** and a
  native-size **window** slides across the tile, stitching the token grids so every window
  stays in-distribution on both pixel size and mpp. ``dense_window_size`` is the knob
  (``null`` = whole-patch single forward; ``=`` native input = native-window sliding). The
  full mode table and the scale/context trade-off live on the
  :doc:`pixel-classifier <decoders/pixel-classifier>` page (``cls_attention`` shares the
  identical extraction).
* **Dense metrics** — evaluation streams compact per-image confusion counts (full
  ``(N, C, H, W)`` logits would OOM across a cohort); the head accumulates ``dense_stats``
  rows and finalizes ``mean_dice`` / ``mean_iou`` (:mod:`soma.tasks.dense_metrics`).
* **Prediction artifacts** — each fold writes ``metrics.json`` plus the
  prediction-raster / overlay / CSV artifacts per split.

The neural-decoder default path
-------------------------------

The **decoder** is the default trainable component on the dense grid (the dense-grid
analogue of an :doc:`aggregator <aggregators>`). For each tile:

1. Run the frozen ViT → dense patch-feature grid ``(d, grid_h, grid_w)``.
2. A **decoder** (``lightweight_conv`` by default) regresses a ``(C, grid)`` map; the head
   interpolates it to the supervision ``target_size`` and crops via ``crop_box``.
3. The head applies the per-pixel class activation and trains with **cross-entropy +
   soft-Dice** (the overlap term is a Tversky generalization — ``beta > alpha`` is
   recall-oriented for small structures, ``gamma > 1`` is focal-Tversky on hard classes).
4. At evaluation, predictions are argmaxed per pixel and scored with ``mean_dice`` /
   ``mean_iou``.

The decoder is **input-agnostic**: it consumes whatever dense ``(d, grid)`` grid the
encoder emits, set by ``preprocessing.feature_kind`` (see :doc:`decoders`). Multi-encoder
:doc:`composite <encoders/composite>` runs are supported on the decoder path and
auto-concatenate at token-grid resolution (``concat_resolution: grid``).

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512          # supervision (mask) size
     requested_spacing_um: 0.5            # read + native encoder spacing
   encoder: { name: uni }
   decoder:                               # the dense trainable component
     name: lightweight_conv
   task:
     name: segmentation
     params: { num_classes: 5 }
   evaluation:
     metrics: [mean_dice, mean_iou]

Methods
-------

The dense grid admits two **feature substrates** (what the encoder emits) and two
**trainable components** on it; the neural decoder above is the default. The
:doc:`Segmentation tutorial <tutorials/segmentation>` lists the substrate and component
alternatives with the runnable walkthrough for each.

Task head
---------

.. autoclass:: soma.tasks.segmentation.SegmentationHead
   :members:

References
----------

* The shared decoder trainable component (see :doc:`decoders`) and the decoder-free
  :doc:`pixel-classifier <decoders/pixel-classifier>` alternative.
* The detection path, which reuses this dense contract front half and differs only in
  output representation (see :doc:`detection`).
