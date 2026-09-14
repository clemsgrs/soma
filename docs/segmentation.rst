Segmentation
============

Segmentation predicts a class for every pixel in a tile or region. A frozen
encoder produces dense feature grids; a trainable :doc:`decoder <decoders>`
maps them to class logits. Alternatively, use a decoder-free
:doc:`pixel classifier <decoders/pixel-classifier>` on the same grids.

The :doc:`segmentation walkthrough <tutorials/walkthrough-segmentation>` runs
the decoder path on a small synthetic dataset.

Data and extraction
-------------------

``dataset_type: segmentation`` uses :class:`soma.dataset.SegmentationManifest`.
Each sample supplies a mask raster through ``label_mask_path``; ``mask_path``
is reserved for an optional tissue mask. Masks are read at the run's spacing
and aligned with the extracted grid. Annotation labels, ``pixel_mapping``, and
per-class ``min_coverage``, and background-present or background-absent label
remapping are configured through :doc:`preprocessing`.

``preprocessing.feature_kind`` selects patch features or attention maps; see
:doc:`decoders`. The default ``dense_window_size: null`` encodes the whole
padded tile in one forward pass. Set a window size to extract and stitch
smaller windows; see :ref:`native-window` for the scale and context trade-off.

Configure a decoder run
-----------------------

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
   encoder: { name: uni }
   decoder: { name: lightweight_conv }
   task:
     name: segmentation
     params: { num_classes: 5 }
   evaluation:
     metrics: [mean_dice, mean_iou]

The head trains with cross-entropy plus soft Dice; the overlap term supports
Tversky weighting (``beta > alpha`` puts more weight on false negatives) and
focal Tversky (``gamma > 1``). At evaluation, each pixel receives the class with
the largest logit. For live re-encoding and augmentation, see :doc:`training`.

Dice reduction and checkpoint selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``mean_dice`` computes class Dice within each sample, averages the
classes defined in that sample, then averages the sample values.

``dataset_global_mean_dice`` sums the confusion counts over the complete split,
computes one Dice value per class, and averages the classes; a class absent from
both predictions and targets is excluded. This weights larger samples more
heavily through their pixel counts. To select checkpoints on it, request it as
an evaluation metric and monitor it in maximum mode:

.. code-block:: yaml

   evaluation:
     metrics: [mean_dice, dataset_global_mean_dice, mean_iou, dice_per_class]
   training:
     monitor: dataset_global_mean_dice
     monitor_mode: max

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
The decoder receives the selected ROI's complete feature grid and mask; this
setting does not create a sub-crop.

This controls **requested classes**, not pixels: cross-entropy and soft Dice still
consume every annotated pixel in each selected ROI, so the method is not pixel-balanced
training. Class requests are apportioned deterministically over the epoch's draw budget;
individual batches are only exactly proportional when their size permits it.

A draw is one ROI index placed in one physical loader batch, so an epoch is a
fixed draw budget rather than a unique pass over the dataset. Leave
``roi_draws_per_epoch`` null to use the largest whole-effective-batch budget
(``batch_size * gradient_accumulation``) no larger than the training ROI count,
which keeps ROI exposure constant across batch-size and accumulation trade-offs.
An explicit budget takes precedence and must contain whole loader batches. Each
fold writes ``roi_batch_sampling.json`` with the resolved budget, configured
ratios, and the per-epoch requested classes, selected ROIs, class-pixel
exposure, and repeat counts.

.. code-block:: yaml

   training:
     batch_size: 16
     roi_draws_per_epoch: 1024
     roi_batch_sampling: class_conditioned  # or uniform for the control arm
     class_request_ratios: [1, 1, 2, 0]

Outputs
-------

Evaluation streams per-image confusion counts rather than retaining every
sample's logits. Each fold writes ``metrics.json`` and prediction rasters,
with optional overlays and probabilities; see :doc:`outputs` for the artifact
layout and :doc:`evaluation` for evaluation settings.

Task head
---------

.. autoclass:: soma.tasks.segmentation.SegmentationHead
   :members:
