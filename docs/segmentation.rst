Segmentation
============

Segmentation predicts a class for every pixel in a tile or region. A frozen
encoder produces dense feature grids; a trainable :doc:`decoder <decoders>`
maps them to class logits. Alternatively, use a decoder-free
:doc:`pixel classifier <decoders/pixel-classifier>` on the same grids.

The :doc:`segmentation walkthrough <tutorials/walkthrough-segmentation>` runs
the decoder path on a small synthetic dataset.

Input modes
-----------

Each ``dataset.csv`` row gives an image (``image_path``) and its label mask
(``label_mask_path``); ``mask_path`` is reserved for an optional tissue mask.
What a row holds depends on whether ``preprocessing.masks`` is set:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - Pre-cropped tiles
     - Whole slides
   * - ``preprocessing.masks``
     - Unset
     - Set
   * - A row is
     - One tile and its mask, both ``requested_tile_size_px`` wide at
       ``requested_spacing_um``. soma does not resize them.
     - One slide and its annotation mask
   * - A training sample is
     - The row
     - One ROI sampled from the slide
   * - Mask values
     - Class indices ``0`` to ``num_classes - 1``, plus ``ignore_index``
       (255 by default); or any values with ``task.params.classes``
     - Values declared by ``pixel_mapping``; ``task.params.classes`` is
       required

Whole slides
~~~~~~~~~~~~

.. code-block:: yaml

   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     masks:
       pixel_mapping: {background: 0, stroma: 1, tumor: 2, necrosis: 3}
       min_coverage: {stroma: 0.05, tumor: 0.05, necrosis: 0.05}

soma tiles each slide into ROIs of ``requested_tile_size_px`` at
``requested_spacing_um``. It keeps an ROI when at least one class covers its
``min_coverage`` fraction of the ROI. Only classes with a ``min_coverage`` entry
select ROIs, so give a threshold to every class you want sampled. Each ROI keeps
its complete multi-class mask and belongs to the same split as its slide.

``pixel_mapping`` and ``min_coverage`` only decide which ROIs are sampled;
:doc:`preprocessing` covers them. What the model predicts is set separately, by
the task.

The annotation mask may have a lower resolution than its slide. soma aligns it
to the slide and reads each ROI's mask on the ROI's pixel grid, so the targets
register to the features. :ref:`Source masks <preprocessing-source-masks>` gives
the alignment rules.

Classes
-------

.. code-block:: yaml

   task:
     name: segmentation
     params:
       classes: {tumor: [1, 2], stroma: [3], muscle: [4, 5, 6]}
       ignore: [0]

``classes`` names each class and the raw mask value(s) that form it; several
values merge into one class. The class index is the declaration order (``tumor``
is 0) and the names are recorded beside the confusion matrices. ``ignore``
lists the raw values excluded from the loss and the metrics. No name is
reserved.

A raw value belongs to one class or to ``ignore``; listing it twice is a config
error. A mask holding a value declared in neither fails the run and names the
sample, so a typo cannot silently drop a class. For whole slides, every value in
``classes`` and ``ignore`` must also be in ``preprocessing.masks.pixel_mapping``,
since the masks are read against that vocabulary.

The class scheme is not part of the feature cache key: regrouping classes reuses
the cached ROIs and features. ``num_classes`` is derived from ``classes``.
Pre-cropped tiles whose masks already hold class indices may set
``num_classes`` alone.

Tile size and encoder window
----------------------------

Two settings decide how much tissue the model sees.

``requested_tile_size_px`` is the size of a training sample. It sets the ROI,
its mask, and the area the decoder predicts. At 0.5 µm/px, a 512 px tile spans
256 µm.

``dense_window_size`` is the size of the image passed to the frozen encoder in
one forward pass. It changes the features, not the training sample. With
``null`` (the default), the encoder receives the whole tile, which only works
for encoders that accept that input size. With a smaller value, soma slides a
window of that size over the tile and stitches the token grids into one grid
for the tile. ``dense_window_overlap`` blends neighboring windows.

Most pathology encoders were pretrained on 224 px inputs at about 0.5 µm/px. A
window of the encoder's native input size keeps every forward pass at that
geometry, while a tile larger than the window gives the decoder context beyond
a single window. :ref:`native-window` compares the window modes.

The token grid is coarser than the mask. The head upsamples the decoder's
logits to the tile size, so the loss and the metrics are always computed per
mask pixel.

``preprocessing.feature_kind`` selects patch features or attention maps as the
grid content; see :doc:`decoders`.

A starting configuration
------------------------

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     dense_window_size: 224        # the encoder's native input size
     dense_window_overlap: 0.5
     # whole slides only:
     masks:
       pixel_mapping: {background: 0, stroma: 1, tumor: 2, necrosis: 3}
       min_coverage: {stroma: 0.05, tumor: 0.05, necrosis: 0.05}
   encoder: { name: uni }
   decoder: { name: lightweight_conv }
   task:
     name: segmentation
     params:
       classes: {stroma: [1], tumor: [2], necrosis: [3]}
       ignore: [0]
   evaluation:
     metrics: [mean_dice, mean_iou]

Treat these values as a first run. Tile size, window size, overlap, spacing, and
coverage thresholds all change accuracy and cost, and the best values depend on
the dataset and the encoder. Tuning them is part of the modeling work.

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
