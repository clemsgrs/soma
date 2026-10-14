How soma works
==============

Define images, labels, and splits; soma preprocesses the images, extracts
frozen foundation-model features, trains a downstream model, and evaluates its
predictions.

Whether inputs are tiles, regions of interest, or whole slides, the same
modular workflow supports classification, regression, survival, segmentation,
and detection. The :doc:`modeling` guide describes which components fit each path.

.. figure:: /_static/figures/how-soma-works-workflow.svg
   :figclass: soma-figure
   :alt: Five modular blocks in sequence: Data, Preprocess, Encode with a frozen foundation model, Train with a trained downstream model, and Evaluate.

   A frozen foundation model produces reusable features. The downstream model
   is trained for the task.

One workflow, modular blocks
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 17 56 27

   * - Block
     - What you choose
     - Learn more
   * - Data
     - Images, labels, and train/tune/test or K-fold splits.
     - :doc:`dataset`
   * - Preprocess
     - Whole-slide tissue masking, spacing, tile size, and overlap.
     - :doc:`preprocessing`
   * - Encode
     - Frozen foundation models: soma applies model-specific transforms, then caches the features.
     - :doc:`encoders`
   * - Train
     - Feature aggregation or dense decoding, task-specific prediction, and optimization.
     - :doc:`modeling`
   * - Evaluate
     - Metrics and prediction visualizations.
     - :doc:`evaluation`

Explore and compare
-------------------

Use the :doc:`modular API <api>` to vary preprocessing, encoders, and downstream
models independently. Extracted features can be reused when only the downstream
model changes; :doc:`caching` explains when extraction must run again.

For controlled comparisons, hold the cohort, labels, splits, and remaining
protocol fixed while varying one component. Registered :doc:`benchmarks
<benchmarking>` provide fixed protocols for reproducing public results and
comparing encoders. Every run saves its resolved configuration, predictions,
and metrics in a :doc:`run bundle <outputs>`.

.. _use-cases:
.. _presets:

Use cases
---------

Each use case is a preset (or a task head swapped into one) over one source shape.
The end-to-end scenarios in the last column run it in soma's test suite, on synthetic
images with a weight-free encoder; `tests/e2e/README.md
<https://github.com/clemsgrs/soma/blob/main/tests/e2e/README.md>`_ lists what each
scenario checks.

.. list-table::
   :header-rows: 1
   :widths: 18 34 22 26

   * - Use case
     - Preset or head
     - Source
     - End-to-end scenario
   * - Slide classification
     - ``mil_classification(encoder, aggregator="abmil")``
     - Bags of tile features
     - ``slide_mil_binary_classification``, ``slide_mil_multiclass_classification``
   * - Survival
     - ``survival(encoder, aggregator)``; the ``cox_survival`` head
     - Bags of tile features
     - ``slide_mil_survival_nll``, ``slide_mil_survival_cox``
   * - Regression
     - The ``regression`` head in ``mil_classification``
     - Bags of tile features
     - ``slide_mil_regression``
   * - Tile-level classification
     - ``tile_classification(encoder)``
     - One embedding per image
     - ``tile_binary_classification``
   * - Segmentation
     - ``segmentation(encoder, decoder, num_classes=2)``
     - Dense token grids
     - ``dense_segmentation``, ``dense_preset_segmentation``, ``grid_live_segmentation``
   * - Detection
     - ``detection(encoder, decoder, num_classes=1, match_distance=3.0)``
     - Dense token grids
     - ``dense_detection``, ``dense_preset_detection``, ``grid_live_detection``

Next: :doc:`get started <getting-started>` or :doc:`choose a modeling path <modeling>`.
