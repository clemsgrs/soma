Detection (point detection)
===========================

A dense **point-detection** path for cell / nucleus detection: predict object
**centroids** (+ class) in a tile, not bounding boxes. detection-v1 reuses the
segmentation *front half* verbatim — a **frozen** foundation-model encoder produces a
dense ``(d, grid_h, grid_w)`` token grid (cached as ``feature_type="dense_grid"``) — and
only the output representation is detection-specific: a decoder regresses a per-class
**peak heatmap**, and a :class:`~soma.tasks.detection.DetectionHead` turns it back into
points and scores them with **F1 at a matching distance δ** (the OCELOT convention).

It sits alongside the segmentation paths: same manifest shape, splits, dense feature
cache, decoder registry, and streaming evaluator — the head, target encoding, loss,
postprocess, and metric are what differ.

The method
----------

For each tile:

1. Run the frozen ViT → dense patch-feature grid ``(d, grid_h, grid_w)`` (the same
   extraction / cache / store stack as the decoder segmentation path).
2. A **decoder** (``lightweight_conv`` by default) regresses a ``(C, grid)`` map; the
   head interpolates it to the supervision ``target_size``, crops via ``crop_box``, and
   applies a **sigmoid** → a per-class heatmap in ``[0, 1]`` (one channel per object
   class; background is the absence of a peak).
3. The training target is a **peak Gaussian** rendered at each annotated point (peak
   value 1, overlaps merged by element-wise **max** — *not* a count-preserving density
   map, so adjacent cells stay separable). Loss is **foreground-weighted MSE**.
4. At inference, peaks are recovered per channel by **local-maxima + NMS + a per-class
   score threshold**, then matched to ground truth with **class-aware F1@δ**.

Lineage: FCRN / CellRegNet (density-map regression). The P2PNet point-set head is a
planned follow-on; see the design note ``design/detection-design.md``.

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
     - Per-sample point file (replaces seg's ``mask_path``).
   * - ``level0_spacing``
     - no
     - µm/px of the frame the points are stored in (per-sample override; otherwise the
       task default).
   * - ``source_wsi`` / ``tile_x`` / ``tile_y``
     - no
     - Parent slide id + tile origin — retained now for deferred WSI stitching.
   * - ``label`` / ``patient_id``
     - no
     - Optional; supervision is the points.

The ``points_path`` file is CSV with ``x, y, class`` columns (a headerless
``x,y,class`` — OCELOT's format — or a 2-column single-class ``x,y`` is also accepted).
Class ids must be **0-based** in ``[0, num_classes)``; map annotation labels (e.g.
OCELOT's ``{1, 2}``) to ``{0, 1}`` during ingestion.

Coordinate convention — level-0 store, target compute
-----------------------------------------------------

Points are **persisted in level-0 (base full-resolution) pixels** — the pathology
convention (ASAP / QuPath / hs2p), invariant to the experiment. The loader maps them
into the run's ``target_size`` frame for encoding and matching::

   x_target = x_level0 * (level0_spacing / run_spacing) - crop_left
   y_target = y_level0 * (level0_spacing / run_spacing) - crop_top

where ``run_spacing`` is the µm/px the grids were extracted at. For flat tiles read at
their native resolution (``level0_spacing == run_spacing``, no crop) this is the
identity. Predicted points are written **back** to level-0 in the prediction CSV, so the
deferred WSI-stitching step needs no data migration.

Configuration
-------------

.. code-block:: yaml

   data:
     dataset_type: detection
   preprocessing:
     requested_tile_size_px: 1024         # supervision tile size
     requested_spacing_um: 0.2            # read + native encoder spacing (= run_spacing)
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
       level0_spacing: 0.2                # µm/px of the stored point frame (per-sample override via column)
   evaluation:
     metrics: [mean_f1, f1_per_class]

``match_distance`` (δ), ``sigma``, and ``nms_distance`` are **always given in µm** —
physically meaningful and spacing-invariant, so the same value means the same tolerance
regardless of which encoder / spacing the run uses, with no "px at which level?"
ambiguity. Each is resolved to target-frame pixels by dividing by ``run_spacing`` (the
µm/px the grids were extracted at), so a spacing is required (detection extraction always
records one). ``match_distance`` is required; ``sigma`` defaults to δ/3 and
``nms_distance`` to δ (so two detections cannot both satisfy one ground-truth point).
Benchmarks that define their tolerance in pixels are expressed in µm via the read
spacing: OCELOT's official **15 px** at ``requested_spacing_um: 0.2`` is
``match_distance: 3.0`` µm (``3.0 / 0.2 = 15`` px).

Feature substrate — patch features or attention grids
-----------------------------------------------------

The decoder is **input-agnostic**: it consumes whatever dense ``(d, grid)`` grid the
encoder emits, set by ``preprocessing.feature_kind`` (the same axis as
:doc:`attention-segmentation`). Two choices:

* ``patch_features`` *(default)* — the ViT patch-token grid (``d`` = the encoder's
  feature dim). The richest descriptor for sub-token localisation; the recommended
  baseline.
* ``cls_attention`` — per-head prefix-token self-attention as a ``(K, grid)`` grid (set
  ``attention: {blocks: [-1], include_registers: false}``). Switching to it is a **pure
  config flip** — nothing in the head, loss, peak extraction, or F1@δ evaluator changes
  (the decoder is simply built with ``input_dim = K``).

.. code-block:: yaml

   preprocessing:
     feature_kind: cls_attention
     attention: { blocks: [-1], include_registers: false }

Attention grids sit at the same token-grid resolution as patch features, so they do not
buy extra localisation resolution; they are best treated as an **ablation** against the
``patch_features`` baseline rather than an automatic win (a saliency scalar per head
carries less sub-token detail than a full patch descriptor). Run the ``patch_features``
baseline first so the attention number is interpretable relative to it. Multi-encoder
``composite:`` runs are supported on the decoder path and auto-concatenate at token-grid
resolution. The decoder-free pixel-classifier route used for attention *segmentation* does
**not** apply here: detection needs the decoder's spatial smoothing to shape clean,
separable peaks, which a per-pixel-independent model cannot provide.

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

Status & scope
--------------

detection-v1 is **cached-only** (no live re-encode / geometric point-target
augmentation) and assumes a uniform tile/grid size across the cohort. The P2PNet
point-set head, live augmentation, and WSI-level stitching are deferred increments. Full
rationale and the locked design decisions are in ``design/detection-design.md``.

References
----------

* CellRegNet, *Point Annotation-Based Cell Detection in Histopathological Images via
  Density Map Regression* (2024).
* *Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology
  WSIs* (2025), `arXiv:2503.05678 <https://arxiv.org/abs/2503.05678>`_ — P2PNet on frozen
  features.
* The decoder segmentation path and shared dense extraction (see :doc:`tasks`,
  :doc:`attention-segmentation`, and :doc:`preprocessing`).
