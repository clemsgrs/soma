Detection
=========

Detection predicts class-labelled cell or nucleus centroids from image tiles.
It uses the frozen-grid pipeline in :doc:`decoders`, then represents objects as
peaks in one heatmap channel per class.

.. figure:: /_static/figures/dense-prediction.svg
   :figclass: soma-figure
   :alt: A frozen encoder produces a feature grid used by detection and segmentation decoders.

   Detection reads point peaks from class heatmaps; segmentation assigns a
   class to every pixel.

Data contract
-------------

``dataset_type: detection`` loads a
:class:`soma.dataset.DetectionManifest`. Each sample needs an image and a point
file:

.. list-table::
   :header-rows: 1

   * - Column
     - Required
     - Meaning
   * - ``sample_id``
     - yes
     - Filename-safe sample and cache identifier.
   * - ``image_path``
     - yes
     - Tile or ROI image.
   * - ``points_path``
     - yes
     - Point CSV for the sample.
   * - ``level0_spacing``
     - no
     - Microns per pixel in the stored coordinate frame; overrides the task default.
   * - ``source_wsi`` / ``tile_x`` / ``tile_y``
     - no
     - Parent-slide provenance and tile origin.

Point files accept ``x,y,class`` with or without a header, or ``x,y`` for a
single class. Class ids are zero-based in ``[0, num_classes)``.

Coordinates
-----------

Persist points in level-0 pixels. The loader transforms them to the run's
target frame for training and matching::

   x_target = x_level0 * (level0_spacing / run_spacing) - crop_left
   y_target = y_level0 * (level0_spacing / run_spacing) - crop_top

Predicted CSV coordinates are converted back to level 0. For native-resolution
tiles without cropping, the transform is the identity.

Configuration
-------------

.. code-block:: yaml

   data:
     dataset_type: detection
   preprocessing:
     requested_tile_size_px: 1024
     requested_spacing_um: 0.2
   encoder: {name: uni}
   decoder: {name: lightweight_conv}
   task:
     name: detection
     params:
       num_classes: 2
       match_distance: 3.0
       sigma: 1.0
       matching: hungarian
       foreground_weight: 10.0
       level0_spacing: 0.2
   evaluation:
     metrics: [mean_f1, f1_per_class]

``match_distance``, ``sigma``, and ``nms_distance`` are expressed in microns
and converted to target pixels using ``run_spacing``. ``match_distance`` is
required; ``sigma`` defaults to one third of it and ``nms_distance`` defaults
to the same value. For example, 15 pixels at ``0.2`` microns per pixel is a
``3.0`` micron match distance.

Training and inference
----------------------

The head renders a Gaussian peak at every annotated point, merging overlaps
with element-wise maximum. A decoder regresses these targets with
foreground-weighted MSE. At inference, local maxima are filtered by
non-maximum suppression and a per-class score threshold.

Thresholds are selected on the tune split, written to
``detection_thresholds.json``, and then fixed for test evaluation. Detection
currently uses cached dense features and requires uniform tile and grid sizes
across a cohort.

Metric
------

F1 matching is class-aware and one-to-one within ``match_distance``. Use
``matching: hungarian`` for optimal assignment or ``matching: greedy`` when a
benchmark requires confidence-ordered greedy assignment.

``mean_f1`` pools counts across the dataset, computes F1 for each class, then
averages classes. ``f1_per_class``, ``precision``, ``recall``, and
``mean_f1_per_image`` are also available. Default task metrics are indexed in
:doc:`tasks`; artifact controls are documented in :doc:`evaluation`.

Outputs
-------

Each evaluated split writes ``predictions_<split>.csv`` with
``sample_id,x,y,class,score`` in level-0 coordinates. Fold-level metrics and
the selected thresholds follow the layout in :doc:`outputs`.

Task head
---------

.. autoclass:: soma.tasks.detection.DetectionHead
   :members:

See :doc:`tutorials/detection` for runnable methods and
:doc:`ocelot-detection-benchmark` for the OCELOT reproduction.
