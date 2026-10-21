Segmentation
============

Segmentation predicts a class for every pixel in a tile or region. A frozen
encoder produces dense feature grids; a trainable :doc:`decoder <decoders>`
maps them to class logits.

Input modes
-----------

Each ``dataset.csv`` row gives an image (``image_path``) and its label mask
(``label_mask_path``); ``mask_path`` is reserved for an optional tissue mask.
What a row holds depends on whether ``features.extraction.masks`` declares an
annotation vocabulary:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - Pre-cropped tiles
     - Whole slides
   * - ``features.extraction.masks``
     - Unset (hs2p's binary tissue vocabulary)
     - An annotation vocabulary
   * - A row is
     - One tile and its mask, both ``features.dense.target_size`` wide at
       ``features.dense.spacing_um``. soma does not resize them.
     - One slide and its annotation mask
   * - A training sample is
     - The row
     - One ROI sampled from the slide
   * - Mask values
     - Class indices ``0`` to ``num_classes - 1``, plus ``ignore_index``
       (255 by default); or any values with the head's ``classes``
     - Values declared by ``pixel_mapping``; the head's ``classes`` is
       required

Whole slides
~~~~~~~~~~~~

.. code-block:: yaml

   features:
     dense:
       target_size: 512
       spacing_um: 0.5
     extraction:
       masks:
         pixel_mapping: {background: 0, stroma: 1, tumor: 2, necrosis: 3}
         min_coverage: {stroma: 0.05, tumor: 0.05, necrosis: 0.05}
         output_mode: merged

soma tiles each slide into ROIs of ``target_size`` pixels at ``spacing_um``
(an unset ``features.extraction.tiling`` takes both from ``features.dense``). It
keeps an ROI when at least one class covers its
``min_coverage`` fraction of the ROI. Only classes with a ``min_coverage`` entry
select ROIs, so give a threshold to every class you want sampled. Each ROI keeps
its complete multi-class mask and belongs to the same split as its slide.

``pixel_mapping`` and ``min_coverage`` only decide which ROIs are sampled;
:doc:`preprocessing` covers them. What the model predicts is set separately, by
the head.

The annotation mask may have a lower resolution than its slide. soma aligns it
to the slide and reads each ROI's mask on the ROI's pixel grid, so the targets
register to the features. :ref:`Source masks <preprocessing-source-masks>` gives
the alignment rules.

A slide whose spacing is within ``tolerance`` of ``spacing_um`` is read
at its own spacing, so its ROIs cover ``target_size`` pixels at, for
example, 0.486 µm/px rather than 0.5 µm/px. Each ROI's mask is read at the spacing
its feature grid recorded, and so covers the same area. A ROI kept at the slide's
right or bottom edge can extend past the slide; the pixels beyond it get
``ignore_index`` and count in neither the loss nor the metrics.

soma reads each ROI's mask once, when it samples the ROIs, and stores it as a
lossless PNG of the raw mask values (16-bit when a value exceeds 255). For an
edge ROI the PNG holds only the part on the slide. Training and evaluation read
these crops and never open the annotation mask. The ROI table extraction
writes beside the grids (``segmentation_rois/dataset.csv``, a record, not a run
input) gives each crop's path in its ``label_mask_crop_path`` column. With the
cache enabled, the crops live in the ROI sampling cache, at
``roi_sampling/<key>/masks/<slide>/<x>_<y>.png``. A slide reuses its cached ROIs
only when all of its crops are present. Editing an annotation mask in place
changes its size or modification time, so soma samples that slide again.

Classes
-------

.. code-block:: yaml

   learner:
     params:
       head:
         name: segmentation
         params:
           classes: {tumor: [1, 2], stroma: [3], muscle: [4, 5, 6]}
           ignore: [0]

``classes`` names each class and the raw mask value(s) that form it; several
values merge into one class. The class index is the declaration order (``tumor``
is 0) and the names are stored with the predictions. ``ignore``
lists the raw values excluded from the loss and the metrics. No name is
reserved.

A raw value belongs to one class or to ``ignore``; listing it twice is a config
error. A mask holding a value declared in neither fails the run and names the
sample, so a typo cannot silently drop a class. For whole slides, every value in
``classes`` and ``ignore`` must also be in ``features.extraction.masks.pixel_mapping``,
since the masks are read against that vocabulary.

The class scheme is not part of the feature cache key: regrouping classes reuses
the cached ROIs and features. ``num_classes`` is derived from ``classes``.
Pre-cropped tiles whose masks already hold class indices may set
``num_classes`` alone.

Tile size and encoder window
----------------------------

Two settings decide how much tissue the model sees.

``features.dense.target_size`` is the size of a training sample. It sets the ROI,
its mask, and the area the decoder predicts. At 0.5 µm/px, a 512 px tile spans
256 µm.

``features.dense.window_size`` is the size of the image passed to the frozen encoder in
one forward pass. It changes the features, not the training sample. With
``null`` (the default), the encoder receives the whole tile, which only works
for encoders that accept that input size. With a smaller value, soma slides a
window of that size over the tile and stitches the token grids into one grid
for the tile. ``features.dense.overlap`` blends neighboring windows.

Most pathology encoders were pretrained on 224 px inputs at about 0.5 µm/px. A
window of the encoder's native input size keeps every forward pass at that
geometry, while a tile larger than the window gives the decoder context beyond
a single window. :ref:`native-window` compares the window modes.

The token grid is coarser than the mask. The head upsamples the decoder's
logits to the tile size, so the loss and the metrics are always computed per
mask pixel.

``features.dense.feature_kind`` selects patch features or attention maps as the
grid content; see :doc:`decoders`.

.. _native-window:

Choose the extraction window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read spacing and encoder-window size control different aspects of the input:
spacing sets physical scale, while window size sets the context visible in one
forward pass. Select the spacing through ``features.dense.spacing_um``; soma does
not force every run to an encoder's recommended spacing.

.. list-table::
   :header-rows: 1

   * - ``window_size``
     - Behavior
     - Context
   * - ``null`` (default)
     - Encode the whole padded tile in one forward pass.
     - All positions in the tile can attend to one another.
   * - Encoder's native input size
     - Slide native-size windows and stitch their token grids.
     - Attention is local to each window.
   * - Larger than the native input size
     - Use larger windows where supported by the encoder.
     - More context per window, with a different input geometry from pretraining.

Whole-tile and larger-window extraction require encoder support for the input
size. Native-size windows at a recommended spacing preserve those two
pretraining input settings, but each window's tokens only see that window. Set
``overlap`` to overlap windows and blend their grids.

soma does not resize a complete image to the encoder's native pixel size (the
method of `Ramchandani et al., 2026 <https://arxiv.org/abs/2602.18747>`_),
because that also changes the physical scale.

A starting configuration
------------------------

.. code-block:: yaml

   features:
     encoder: { name: uni }
     dense:
       target_size: 512
       spacing_um: 0.5
       window_size: 224             # the encoder's native input size
       overlap: 0.5
     # whole slides only:
     extraction:
       masks:
         pixel_mapping: {background: 0, stroma: 1, tumor: 2, necrosis: 3}
         min_coverage: {stroma: 0.05, tumor: 0.05, necrosis: 0.05}
         output_mode: merged
   learner:
     name: dense
     params:
       decoder: { name: lightweight_conv }
       head:
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
the largest logit.

By default the decoder trains on cached feature grids. To augment the tiles,
set ``features.kind: live`` and a ``loaders.augment`` leaf; the encoder then
re-encodes each augmented tile at every step. See :ref:`live-mode`.

Dice reduction and checkpoint selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``mean_dice`` computes class Dice within each sample, averages the
classes defined in that sample, then averages the sample values.

``dataset_global_mean_dice`` sums the confusion counts over the complete split,
computes one Dice value per class, and averages the classes; a class absent from
both predictions and targets is excluded. This weights larger samples more
heavily through their pixel counts.

``foreground_mean_dice`` leaves out the background class and, within each
sample, skips a class that has no target pixels even when it was predicted.
It is the convention of the kaiko-ai/eva segmentation leaderboard; see the
:doc:`EVA benchmark <eva-patch-classification-benchmark>`.

To select checkpoints on one of these, request it as an evaluation metric and
monitor it in maximum mode:

.. code-block:: yaml

   evaluation:
     metrics: [mean_dice, dataset_global_mean_dice, mean_iou, dice_per_class]
   trainer:
     callbacks:
       - {name: checkpoint, params: {monitor: dataset_global_mean_dice, mode: max}}
       - {name: early_stopping, params: {monitor: dataset_global_mean_dice, mode: max}}

Training-batch ROI sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cached segmentation runs can opt into an explicit training-batch contract
with a ``loaders.sampler`` leaf. This is distinct from
``features.extraction.masks.strategy``: extraction chooses ROI coordinates before feature
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
the sampler's ``draws_per_epoch`` null to use the largest whole-effective-batch
budget (``batch_size * accumulate_grad_batches``) no larger than the training ROI count,
which keeps ROI exposure constant across batch-size and accumulation trade-offs.
An explicit budget takes precedence and must contain whole loader batches. Each
fold writes ``roi_batch_sampling.json`` with the resolved budget, configured
ratios, and the per-epoch requested classes, selected ROIs, class-pixel
exposure, and repeat counts.

.. code-block:: yaml

   loaders:
     batch_size: 16
     sampler:
       name: class_conditioned            # or uniform for the control arm
       params:
         draws_per_epoch: 1024
         class_request_ratios: [1, 1, 2, 0]

Outputs
-------

Each fold predicts one label map per sample for tune and every test split and
saves it as ``predictions_<split>/``: one PNG of class indices per sample and an
``index.parquet``. The metrics in ``metrics.json`` are computed from these label
maps, so they can be read back with
:meth:`soma.evaluation.LabelMapPredictions.load` and scored again without
rerunning inference (see :ref:`grid-predictions`). Overlays and per-pixel
probabilities are optional; see :doc:`outputs` for the artifact layout and
:doc:`evaluation` for evaluation settings.

Task head
---------

.. autoclass:: soma.models.heads.segmentation.SegmentationHead
   :members:
