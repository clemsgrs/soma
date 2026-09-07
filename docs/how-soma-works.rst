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
comparing encoders.

Each pipeline run saves the resolved configuration, predictions, and metrics in
a :doc:`run bundle <outputs>` so the experiment can be inspected and repeated.

Where to go next
----------------

* :doc:`Get started <getting-started>` — install soma and run an experiment.
* :doc:`Explore modeling paths <modeling>` — choose a downstream path.
* :doc:`Benchmark a component <benchmarking>` — compare one block or reproduce
  a published benchmark.
