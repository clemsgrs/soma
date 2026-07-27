How soma works
==============

soma was built to streamline computational pathology research with foundation models.
Define images, labels, and splits: soma takes care of preprocessing, feature
extraction, downstream training, and evaluation.

Whether inputs are tiles, regions of interest, or whole slides, the same
modular workflow supports classification, regression, survival, segmentation,
and detection. Change any block without rewriting the rest.

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
     - Whole-slide tiling, spacing, tile size, overlap, ...
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

Explore or benchmark
--------------------

Custom experimentation
~~~~~~~~~~~~~~~~~~~~~~

Explore preprocessing, encoders, and downstream models—alone or in
combination—to optimize a workflow for your data and evaluation objective. See
the :doc:`API <api>` for the composable interfaces.

Benchmarking
~~~~~~~~~~~~

Vary one block while holding the source cohort, labels, splits, and
the rest of the protocol fixed to measure its downstream effect. Encoder sweeps
are the most common use case, but the same applies to any building block like preprocessing or downstream models. See
:doc:`benchmarking` for controlled comparisons and public benchmark
reproduction.

Reproducible by design
----------------------

Each run records the resolved configuration. This provenance supports audit and reproduction. See :doc:`outputs`
for the saved result bundle.

Where to go next
----------------

* :doc:`Get started <getting-started>` — install soma and run an experiment.
* :doc:`Explore modeling paths <modeling>` — choose a downstream path.
* :doc:`Benchmark a component <benchmarking>` — compare one block or reproduce
  a published benchmark.
