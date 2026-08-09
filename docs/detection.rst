Detection
=========

A dense **point-detection** path for cell / nucleus detection: predict object
**centroids** (+ class) in a tile, not bounding boxes. detection-v1 reuses the
:doc:`segmentation` *front half* verbatim — the shared **dense contract** (a **frozen**
foundation-model encoder produces a dense ``(d, grid_h, grid_w)`` token grid, cached as
``feature_type="dense_grid"``) — and only the output representation is detection-specific:
a decoder regresses a per-class **peak heatmap**, and a
:class:`~soma.tasks.detection.DetectionHead` turns it back into points and scores them with
**F1 at a matching distance δ** (the OCELOT convention).

It sits alongside the :doc:`segmentation` paths: same manifest shape, splits, dense feature
cache, decoder registry, and streaming evaluator — the head, target encoding, loss,
postprocess, and metric are what differ.

.. figure:: /_static/figures/dense-prediction.svg
   :figclass: soma-figure
   :alt: A frozen foundation model produces a 2D feature grid; the same trained conv decoder feeds a detection branch (sigmoid heatmap, peak extraction, cell points) and a segmentation branch (per-pixel argmax, mask).

   The dense path. A frozen encoder produces a 2D feature grid; the same trained
   lightweight conv decoder drives both dense tasks — detection reads peaks out of a
   per-class heatmap, segmentation argmaxes per pixel. This page covers the detection
   branch (left).

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
   * - ``spacing_at_level_0``
     - no
     - Finite positive µm/px declaration for the source image's level-0 pixels. Required
       for flat PNG/JPEG extraction; WSI readers may resolve it from the slide.
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

   x_target = x_level0 * (source_spacing_um / effective_spacing_um) - crop_left
   y_target = y_level0 * (source_spacing_um / effective_spacing_um) - crop_top

Both values come from each dense artifact's Slide2Vec sidecar:
``source_spacing_um`` is the resolved physical scale of the stored point/source frame,
and ``effective_spacing_um`` is the scale actually sampled for the dense grid. The latter
can legitimately differ slightly from ``preprocessing.requested_spacing_um`` when a WSI
reader accepts a nearby native level. For flat tiles read at native resolution (equal
source/effective spacing, no crop) the transform is the identity. Predicted points are
written **back** to level-0 in the prediction CSV, so deferred WSI stitching needs no data
migration.

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

``match_distance`` (δ), ``sigma``, and ``nms_distance`` are **always given in µm** —
physically meaningful and spacing-invariant, so the same value means the same tolerance
regardless of which encoder / spacing the run uses, with no "px at which level?"
ambiguity. Each is resolved to target-frame pixels by dividing by the persisted
``effective_spacing_um`` (the scale actually sampled), so resolved provenance is required
for every detection grid. One run may contain different source spacings, but all samples
must share one effective spacing because they share a decoder geometry. ``match_distance``
is required; ``sigma`` defaults to δ/3 and
``nms_distance`` to δ (so two detections cannot both satisfy one ground-truth point).
Benchmarks that define their tolerance in pixels are expressed in µm via the read
effective spacing: OCELOT's official **15 px** at ``effective_spacing_um = 0.2`` is
``match_distance: 3.0`` µm (``3.0 / 0.2 = 15`` px).

Feature substrate — patch features or attention grids
-----------------------------------------------------

The decoder is **input-agnostic**: it consumes whatever dense ``(d, grid)`` grid the
encoder emits, set by ``preprocessing.feature_kind`` (see :doc:`decoders`). Two choices:

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
baseline first so the attention number is interpretable relative to it.

Methods
-------

The dense grid admits two **feature substrates** (what the encoder emits) and two
**trainable components** on it; the neural decoder is detection's default and required
component. The :doc:`Detection tutorial <tutorials/detection>` lists the substrate and
component alternatives with the runnable walkthrough for each.

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

Status & scope
--------------

detection-v1 is **cached-only** (no live re-encode / geometric point-target
augmentation) and assumes a uniform tile/grid size across the cohort. The P2PNet
point-set head, live augmentation, and WSI-level stitching are deferred increments. Full
rationale and the locked design decisions are in ``design/detection-design.md``.

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
* The decoder :doc:`segmentation` path and shared dense extraction (see :doc:`decoders`,
  the decoder-free :doc:`decoders/pixel-classifier`, and :doc:`preprocessing`).
