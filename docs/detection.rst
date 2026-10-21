Detection
=========

Detection predicts cell or nucleus centroids and classes in a tile. A frozen
encoder supplies dense grids, a :doc:`decoder <decoders>` predicts per-class
heatmaps, and the detection head extracts points and scores them at matching
distance δ. It shares dense extraction and caching with :doc:`segmentation`.

.. figure:: /_static/figures/dense-prediction.svg
   :figclass: soma-figure
   :alt: A frozen foundation model produces a 2D feature grid; the same trained conv decoder feeds a detection branch (sigmoid heatmap, peak extraction, cell points) and a segmentation branch (per-pixel argmax, mask).

   Detection extracts heatmap peaks; segmentation selects a class per pixel.
   Both use the same decoder architecture, trained for their respective task.

The method
----------

The decoder output passes through a sigmoid to give one heatmap in ``[0, 1]``
per object class; background is the absence of a peak. The training target is a
**peak Gaussian** rendered at each annotated point (peak value 1, overlaps
merged by element-wise **max**, not a count-preserving density map), and the
loss is **foreground-weighted MSE**. At inference, the head recovers peaks per
channel by **local maxima + NMS** and keeps every peak above ``min_score``. A
**per-class score threshold**, chosen on the tune split, then selects the
points that are matched to ground truth with **class-aware F1@δ**.

Data contract
-------------

A ``detection`` head reads its supervision through
:class:`soma.data.AnnotationManifest` and :class:`soma.data.PointSource`. The
supervision is a per-sample **point file**, not a scalar ``label`` or a mask.

.. list-table::
   :header-rows: 1

   * - Column
     - Required
     - Meaning
   * - ``sample_id``
     - yes
     - Filename-safe id (cache key).
   * - ``image_path``
     - yes
     - Tile / ROI image.
   * - ``points_path``
     - yes
     - Per-sample point annotations.
   * - ``spacing_at_level_0``
     - no
     - Finite positive µm/px declaration for the source image's level-0 pixels. Required
       for flat PNG/JPEG extraction; WSI readers may resolve it from the slide.
   * - ``ignore_mask_path``
     - no
     - Region with no annotation. See :ref:`detection-ignore-masks`.
   * - ``source_wsi`` / ``tile_x`` / ``tile_y``
     - no
     - Parent slide id and tile origin, retained as metadata.
   * - ``label`` / ``patient_id``
     - no
     - Optional; supervision is the points.

The ``points_path`` file is CSV with ``x, y, class`` columns (a headerless
``x,y,class`` — OCELOT's format — or a 2-column single-class ``x,y`` is also accepted).
The ``class`` column holds the dataset's own annotated ids; :ref:`detection-classes`
says which ids form which class.

.. _detection-classes:

Classes
-------

.. code-block:: yaml

   learner:
     params:
       head:
         name: detection
         params:
           classes: {mnl: [0, 1]}
           drop: [2]

``classes`` names each class and the annotated id(s) that form it; several ids
merge into one class, which gets one heatmap channel. The class index is the
declaration order and the names are stored with the predictions. The ids need
not be 0-based: a dataset annotated with
``{1, 2}`` can be declared as ``classes: {background_cell: [1], tumor_cell: [2]}``.

``drop`` lists the annotated ids to discard. A dropped point leaves the
supervised set: it is absent from the target heatmap and from the ground truth
used for matching. Its location is therefore negative supervision, and a
prediction there counts as a false positive. To mark a region as not annotated
instead, use an :ref:`ignore mask <detection-ignore-masks>`.

An id belongs to one class or to ``drop``; listing it twice is a config error. A
point file holding an id declared in neither fails the run and names the sample.

The class scheme is not part of the feature cache key: regrouping classes reuses
the cached features. ``num_classes`` is derived from ``classes``. Point files
that already hold 0-based class indices may set ``num_classes`` alone.

.. _detection-ignore-masks:

Ignore masks
------------

Point-annotated datasets often label only part of each image, for example the
cells inside annotated ROI polygons. The optional ``ignore_mask_path`` column
marks the pixels that are not annotated, so that the model is neither trained
nor scored there.

**Format.** A flat, single-channel uint8 PNG in the image's own (level-0) pixel
frame. 255 means "not annotated" and 0 means "supervised", as with
segmentation's ``ignore_index``. Any other value fails the run with an error
that names the file and the sample. A sample without a mask is supervised
everywhere.

**Semantics.** Ignored pixels are "don't care":

- **Loss.** The loss is averaged over the supervised pixels only, so the
  prediction on an ignored pixel does not change it. Without a mask the loss is
  unchanged.
- **Ground truth.** Points on ignored pixels are removed from the target heatmap
  and from the ground truth used for matching. A point is removed when its
  level-0 pixel is ignored, or when its pixel in the ``target_size`` frame is
  ignored.
- **Predictions.** Peaks on ignored pixels are left out whenever predictions
  are scored: in the threshold sweep, in evaluation and in the detection
  benchmark. The saved predictions file keeps them.
- **Area.** The FROC per-mm² area of a sample (:mod:`soma.benchmark.monkey.froc`)
  covers its supervised pixels only.

The mask is mapped to the run's ``target_size`` frame like the points: each
target pixel takes the value of its nearest level-0 pixel.

**Tiling.** When an ROI row of the manifest given to
``python -m soma.benchmark.tiles`` has ``ignore_mask_path``, the mask
must have the size of the ROI image. The tiler then:

- pads the mask with 255 where it pads the ROI up to a full tile;
- crops the mask for each tile to ``ignore_masks/<tile_id>.png`` and sets the
  tile row's ``ignore_mask_path``;
- skips tiles with no supervised pixel;
- writes no point on ignored pixels to a tile, and counts these points in
  ``summary.json`` as ``points_in_ignored_region``;
- writes ``roi_valid_area_px``, the ROI's supervised pixel count (padding
  excluded), on every tile row. The stitched ROI area for FROC uses it.

Without the column, the tiler output does not change.

Coordinate convention — level-0 store, target compute
-----------------------------------------------------

Points are stored in the source image's level-0 pixels. The loader maps them
into the run's ``target_size`` frame for encoding and matching::

   x_target = x_level0 * (source_spacing_um / effective_spacing_um) - crop_left
   y_target = y_level0 * (source_spacing_um / effective_spacing_um) - crop_top

Both spacings come from each dense artifact's slide2vec sidecar;
``effective_spacing_um`` can differ slightly from
the requested ``features.dense.spacing_um`` when a WSI reader accepts a nearby native
level. Predicted points are written back to the source frame in the predictions
file.

Configuration
-------------

.. code-block:: yaml

   features:
     encoder: { name: uni }
     dense:
       target_size: 1024                  # supervision tile size
       spacing_um: 0.2                    # requested read scale; sidecar records effective scale
   learner:
     name: dense
     params:
       decoder:                           # the heatmap regressor
         name: lightweight_conv
       head:
         name: detection
         params:
           classes: {background_cell: [0], tumor_cell: [1]}   # OCELOT, as curated by soma
           match_distance: 3.0            # δ, in µm (OCELOT's 15 px at 0.2 µm/px)
           sigma: 1.0                     # target Gaussian σ in µm (default ≈ δ/3)
           matching: hungarian            # hungarian (default) | greedy (OCELOT-official)
           foreground_weight: 10.0        # MSE up-weight on near-peak pixels
   evaluation:
     metrics: [mean_f1, f1_per_class]

``match_distance`` (δ) is required. ``sigma`` defaults to δ/3 and
``nms_distance`` to δ. All three are specified in µm and converted to pixels
using each grid's persisted ``effective_spacing_um``. Samples may have different
source spacings, but a run requires one effective spacing and uniform grid
geometry.

The head's other parameters are:

- ``min_score`` (default ``0.01``): the lowest peak score the head emits. Keep
  it low; the operating threshold is chosen afterwards on the predictions.
- ``fixed_threshold`` (default ``0.5``): the operating threshold of a run without
  a tune split, and the default threshold of the detection overlays.
- ``foreground_weight`` (default ``10``): the MSE up-weight on near-peak pixels.
- ``matching``: ``hungarian`` (default) or ``greedy``, used for the threshold sweep
  and for scoring. Validation during training always matches greedily (see
  below).
- ``classes`` / ``drop``: the class scheme; see :ref:`detection-classes`.

Detection also runs with ``features.kind: live``, which re-encodes the tiles
at every step, and can then augment them with ``loaders.augment``; see
:ref:`live-mode`.

The default feature kind is ``patch_features``. To probe attention maps with
the same head, loss, and evaluator, use an attention-capable encoder and set:

.. code-block:: yaml

   features:
     dense:
       feature_kind: cls_attention
       attention_blocks: [-1]
       attention_include_registers: false

Attention maps retain the token-grid resolution; switching feature kind does
not add spatial samples. See :doc:`decoders` for the trained decoders.

To score the attention itself as the heatmap, with nothing trained, use the
``attention_map`` decoder. It averages the attention heads (``reduction: max``
takes their maximum), or only those listed in ``heads``, and min-max normalises
the map of each tile. Every class gets the same map. The run trains nothing but
still chooses its peak threshold on the tune split:

.. code-block:: yaml

   learner:
     name: dense
     params:
       decoder:
         name: attention_map
         params: {heads: null}   # [k] scores head k alone

``attention_map`` registers with the benchmark package (``soma.benchmark``),
which the ``soma`` CLI imports.

Metric — F1 at matching distance δ
----------------------------------

Predicted points are matched to ground truth **per class** (a prediction only matches a
same-class GT within δ) using optimal one-to-one **Hungarian** assignment (default) or
**greedy-by-confidence** (``matching: greedy``, OCELOT's official scorer — emit it for a
leaderboard-comparable number). Matched pairs are TP, unmatched predictions FP,
unmatched GT FN.

* **Score threshold** — the pipeline sweeps one threshold per class on the **tune**
  predictions to maximise F1 (:func:`soma.evaluation.sweep_threshold` with
  ``per_class=True``) and applies it unchanged to every test split (no test
  leakage). With greedy matching the sweep is exact on a grid of 0.001: one
  matching pass gives the F1 at every threshold. With Hungarian matching it tries
  a grid of candidates built from the scores. A run without a tune split
  (``cohort.allow_missing_tune``) scores at ``fixed_threshold`` instead: train
  stands in for tune there, and a sweep on the training samples would pick an
  in-sample cut.
* **Validation during training** — at each validation epoch, the head takes, for
  each class, the best tune F1 over every threshold at or above ``min_score``
  (the same one-pass sweep, so with greedy matching). ``mean_f1`` is the mean of
  these per-class bests: the tune ``mean_f1`` the sweep above would reach for that
  epoch, which ``checkpoint`` and ``early_stopping`` monitor. The per-class F1,
  precision and recall in the training history are read at those thresholds, and
  the history logs each class's threshold as ``threshold_class_<c>``.
* **Aggregation** — the headline ``mean_f1`` is **dataset-global** (counts pooled per
  class → one F1 per class → mean across classes, OCELOT-faithful). ``mean_f1_per_image``
  is available as a secondary (per-image macro). Per-class F1 / precision / recall are
  exposed via ``f1_per_class`` / ``precision`` / ``recall``.

Outputs
-------

Each fold writes ``metrics.json`` (tune + per test split) and
``predictions_<split>.parquet`` for tune and every test split. The predictions
are a :class:`~soma.evaluation.PointPredictions` with columns
``sample_id, x, y, class, score`` in **level-0** coordinates; ``class`` is the class
index. They hold every peak above ``min_score``, before any threshold, so a
different threshold or matching rule can be applied later without rerunning
inference (see :ref:`grid-predictions`). Point overlays and raw heatmaps are
optional; see :doc:`outputs`.

Task head
---------

.. autoclass:: soma.models.heads.detection.DetectionHead
   :members:

Benchmarks
----------

* :doc:`ocelot-detection-benchmark`: this path on the OCELOT 2023 cell-detection
  challenge, with recorded results for seven encoders.
* :doc:`midog-detection-benchmark`: mitotic-figure detection at 0.25 µm/px,
  scored with native stitched-ROI F1 across three seeds.
* :doc:`monkey-detection-benchmark`: inflammatory-cell detection at 0.242 µm/px,
  scored with merged inflammatory-cell FROC across five patient folds.
