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
  The supervision is a per-sample **mask raster** (``label_mask_path``), not a scalar
  ``label``. ``mask_path`` keeps its usual meaning (optional tissue mask).
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
  rows and finalizes ``mean_dice`` / ``mean_iou`` plus explicitly requested global
  reductions (:mod:`soma.tasks.dense_metrics`).
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

Dice reduction and checkpoint selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mean_dice`` retains soma's established per-sample macro reduction: compute class
Dice within each sample, average the classes defined in that sample, then average the
sample values. It remains the default so existing configurations, results, and
experiment identities do not change.

``dataset_global_mean_dice`` first sums the integer confusion counts over the complete
split, computes one Dice value per class, and averages those class values equally. A
class absent from both predictions and targets over the complete split is undefined and
excluded from the class average, matching the existing segmentation absent-class
convention. This reduction is independent of the number of classes and weights larger
samples more heavily through their pixel counts.

Use the dataset-global reduction for BEETLE decoder-checkpoint parity. Request it as an
evaluation metric and select it in maximum mode:

.. code-block:: yaml

   evaluation:
     metrics: [mean_dice, dataset_global_mean_dice, mean_iou, dice_per_class]
   training:
     monitor: dataset_global_mean_dice
     monitor_mode: max

Both names are written separately in per-epoch training history. The selected checkpoint
also stores the complete tune metrics and its selection monitor, mode, and value, so the
two reductions cannot be mistaken for each other after training.

The decoder is **input-agnostic**: it consumes whatever dense ``(d, grid)`` grid the
encoder emits, set by ``preprocessing.feature_kind`` (see :doc:`decoders`). Multi-encoder
:doc:`composite <encoders/composite>` runs are supported on the decoder path and
auto-concatenate at token-grid resolution (``concat_resolution: grid``).

Training-batch ROI sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cached segmentation runs can opt into an explicit training-batch contract
with ``training.roi_batch_sampling``. This is distinct from
``preprocessing.sampling``: preprocessing chooses ROI coordinates before feature
extraction, while training-batch sampling chooses among the already-cached ROI grids.

``uniform`` traverses a shuffled ROI collection. ``class_conditioned`` follows
``class_request_ratios`` over the task's arbitrary ``K`` class indices; null ratios mean
equal relative weight. Ratios need not sum to one, batch size need not be divisible by
``K``, and a zero ratio excludes a class from requests. For each request, eligible ROIs
are weighted by their annotated-pixel count for that class. A positively weighted class
with no training-fold support is an error rather than a silent policy change.
The relative-ratio convention follows MONAI's
`RandCropByLabelClasses <https://monai.readthedocs.io/en/stable/transforms.html#randcropbylabelclasses>`_;
soma deliberately fails on an unsupported requested class instead of renormalizing it.
Only the arbitrary-class relative-ratio convention is borrowed from that transform.
Despite MONAI's transform name, soma does not choose a crop centre or make a sub-crop:
it selects one already-cached ROI index, and the decoder receives that ROI's complete
feature grid and mask.

This controls **requested classes**, not pixels: cross-entropy and soft Dice still
consume every annotated pixel in each selected ROI, so the method is not pixel-balanced
training. Class requests are apportioned deterministically over the epoch's draw budget;
individual batches are only exactly proportional when their size permits it.

Set the same ``roi_draws_per_epoch``, batch size, seed, and all other loader/training
settings in both arms. A draw is one ROI index placed in one loader batch, so an explicit
sampling epoch is a fixed optimization/sampling horizon rather than a unique pass over
the dataset. Leave ``roi_draws_per_epoch`` null to use the largest whole-batch budget no
larger than the training ROI count. Each fold writes ``roi_batch_sampling.json`` with
the configured ratios, requested class, selected ROI, realized class-pixel exposure,
unique ROI coverage, and repeat counts for every epoch.

.. code-block:: yaml

   training:
     batch_size: 16
     roi_draws_per_epoch: 1024
     roi_batch_sampling: class_conditioned  # or uniform for the control arm
     class_request_ratios: [1, 1, 2, 0]

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
