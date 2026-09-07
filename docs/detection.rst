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

.. seealso::

   The :doc:`detection walkthrough <tutorials/walkthrough-detection>` runs
   detection end to end on a tiny synthetic dataset;
   :doc:`segmentation <tutorials/walkthrough-segmentation>` is the same dense flow
   with mask supervision, so you can see exactly what changes between the two.

The method
----------

For each tile:

1. Run the frozen ViT → dense patch-feature grid ``(d, grid_h, grid_w)`` (the same
   extraction / cache / store stack as the decoder segmentation path).
2. A **decoder** (``lightweight_conv`` by default) regresses a ``(C, grid)`` map; the
   head interpolates it to the padded ``encoded_size``, crops to ``target_size``, and
   applies a **sigmoid** → a per-class heatmap in ``[0, 1]`` (one channel per object
   class; background is the absence of a peak).
3. The training target is a **peak Gaussian** rendered at each annotated point (peak
   value 1, overlaps merged by element-wise **max** — *not* a count-preserving density
   map). Loss is **foreground-weighted MSE**.
4. At inference, peaks are recovered per channel by **local-maxima + NMS + a per-class
   score threshold**, then matched to ground truth with **class-aware F1@δ**.

Data contract
-------------

``dataset_type: detection`` uses :class:`soma.dataset.DetectionManifest`. The
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
   * - ``source_wsi`` / ``tile_x`` / ``tile_y``
     - no
     - Parent slide id and tile origin, retained as metadata.
   * - ``label`` / ``patient_id``
     - no
     - Optional; supervision is the points.

The ``points_path`` file is CSV with ``x, y, class`` columns (a headerless
``x,y,class`` — OCELOT's format — or a 2-column single-class ``x,y`` is also accepted).
Class ids must be **0-based** in ``[0, num_classes)``; map annotation labels (e.g.
OCELOT's ``{1, 2}``) to ``{0, 1}`` during ingestion.

Coordinate convention — level-0 store, target compute
-----------------------------------------------------

Points are stored in the source image's level-0 pixels. The loader maps them
into the run's ``target_size`` frame for encoding and matching::

   x_target = x_level0 * (source_spacing_um / effective_spacing_um) - crop_left
   y_target = y_level0 * (source_spacing_um / effective_spacing_um) - crop_top

Both values come from each dense artifact's slide2vec sidecar:
``source_spacing_um`` is the resolved physical scale of the stored point/source frame,
and ``effective_spacing_um`` is the scale actually sampled for the dense grid. The latter
can legitimately differ slightly from ``preprocessing.requested_spacing_um`` when a WSI
reader accepts a nearby native level. For flat tiles read at native resolution (equal
source/effective spacing, no crop) the transform is the identity. Predicted points are
written back to that source frame in the prediction CSV.

Configuration
-------------

.. code-block:: yaml

   data:
     dataset_type: detection
   preprocessing:
     requested_tile_size_px: 1024         # supervision tile size
     requested_spacing_um: 0.2            # requested read scale; sidecar records effective scale
   encoder: { name: uni }
   decoder:                               # the heatmap regressor
     name: lightweight_conv
   task:
     name: detection
     params:
       num_classes: 2                     # e.g. OCELOT: background-cell, tumor-cell
       match_distance: 3.0                # δ, in µm (OCELOT's 15 px at 0.2 µm/px)
       sigma: 1.0                          # target Gaussian σ in µm (default ≈ δ/3)
       matching: hungarian                # hungarian (default) | greedy (OCELOT-official)
       foreground_weight: 10.0            # MSE up-weight on near-peak pixels
   evaluation:
     metrics: [mean_f1, f1_per_class]

``match_distance`` (δ) is required. ``sigma`` defaults to δ/3 and
``nms_distance`` to δ. All three are specified in µm and converted to pixels
using each grid's persisted ``effective_spacing_um``. Samples may have different
source spacings, but a run requires one effective spacing and uniform grid
geometry. For example, 15 pixels at 0.2 µm/px corresponds to
``match_distance: 3.0``.

The default feature kind is ``patch_features``. To probe attention maps with
the same head, loss, and evaluator, use an attention-capable encoder and set:

.. code-block:: yaml

   preprocessing:
     feature_kind: cls_attention
     attention: { blocks: [-1], include_registers: false }

Attention maps retain the token-grid resolution; switching feature kind does
not add spatial samples. Detection requires a neural decoder. See
:doc:`decoders` for available architectures and :doc:`tutorials/detection`
for a runnable workflow.

Metric — F1 at matching distance δ
----------------------------------

Predicted points are matched to ground truth **per class** (a prediction only matches a
same-class GT within δ) using optimal one-to-one **Hungarian** assignment (default) or
**greedy-by-confidence** (``matching: greedy``, OCELOT's official scorer — emit it for a
leaderboard-comparable number). Matched pairs are TP, unmatched predictions FP,
unmatched GT FN.

* **Score threshold** — swept per class on the **tune** split to maximise F1, frozen
  into ``detection_thresholds.json``, and applied unchanged at test (no test leakage).
* **Aggregation** — the headline ``mean_f1`` is **dataset-global** (counts pooled per
  class → one F1 per class → mean across classes, OCELOT-faithful). ``mean_f1_per_image``
  is available as a secondary (per-image macro). Per-class F1 / precision / recall are
  exposed via ``f1_per_class`` / ``precision`` / ``recall``.

Outputs
-------

Each fold writes ``metrics.json`` (tune + per test split), ``detection_thresholds.json``
(the frozen per-class thresholds), and ``predictions_<split>.csv`` with columns
``sample_id, x, y, class, score`` in **level-0** coordinates.

Task head
---------

.. autoclass:: soma.tasks.detection.DetectionHead
   :members:

Scope
-----

Detection uses cached features and assumes uniform tile and grid sizes across
the cohort. Live re-encoding, geometric point-target augmentation, point-set
heads, and WSI-level stitching are not implemented.

Benchmarks
----------

* :doc:`ocelot-detection-benchmark` — this path reproduced on the OCELOT 2023
  cell-detection challenge, with the encoder × spacing ablation.

References
----------

* CellRegNet, *Point Annotation-Based Cell Detection in Histopathological Images via
  Density Map Regression* (2024).
* *Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology
  WSIs* (2025), `arXiv:2503.05678 <https://arxiv.org/abs/2503.05678>`_ — P2PNet on frozen
  features.
