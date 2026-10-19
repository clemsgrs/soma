How soma works
==============

soma takes a cohort of images with labels and splits, extracts features with a
frozen foundation model, trains a downstream model on those features, and scores
its predictions. Inputs can be tiles, regions of interest or whole slides, and the
same steps serve classification, regression, survival, segmentation and detection.

.. figure:: /_static/figures/how-soma-works-workflow.svg
   :figclass: soma-figure
   :alt: Five modular blocks in sequence: Data, Preprocess, Encode with a frozen foundation model, Train with a trained downstream model, and Evaluate.

   A frozen foundation model produces reusable features. The downstream model
   is trained for the task.

The steps
---------

Each step is a public function, a *verb*, that takes and returns plain objects you
can inspect:

.. list-table::
   :header-rows: 1
   :widths: 14 36 30 20

   * - Step
     - Verb
     - What you choose
     - Learn more
   * - Data
     - :class:`soma.Cohort` holds the targets and splits;
       :class:`soma.ImageManifest` holds the image paths
     - Images, targets, and train/tune/test or K-fold splits
     - :doc:`data`
   * - Preprocess
     - :func:`soma.extract` tiles the images
     - Tissue masking, spacing, tile size and overlap
     - :doc:`preprocessing`
   * - Encode
     - :func:`soma.extract` encodes the tiles and returns a feature *source*
     - A frozen foundation model; its features are cached for later runs
     - :doc:`encoders`
   * - Train
     - :func:`soma.fit` trains a *learner* and returns a :class:`soma.FitResult`;
       :func:`soma.save` writes the model as a bundle
     - An aggregator or a decoder, a task head, the optimizer and the budget
     - :doc:`components`, :doc:`training`
   * - Evaluate
     - :func:`soma.predict` returns :class:`soma.Predictions`;
       :func:`soma.evaluate` scores them and returns an :class:`soma.Evaluation`
     - Metrics and subgroups
     - :doc:`evaluation`

Features reach the downstream model in one of two shapes. A *set* is one feature
vector per sample, or a bag of tile features that an aggregator pools; the ``mil``
learner trains it, with a classification, regression or survival head. A *grid* is
a dense token grid per tile or region that a decoder maps back to pixels; the
``dense`` learner trains it, with a segmentation or detection head. :doc:`modeling`
helps you choose.

Three ways to run
-----------------

- **A preset** returns a complete experiment for a common task from a few
  arguments, such as ``soma.presets.mil_classification(encoder="uni2", aggregator="abmil")``, or
  ``soma run --preset`` on the command line.
- **A config**, an :class:`soma.ExperimentConfig` in Python or the same fields in
  YAML, states every step. :func:`soma.resolve` expands it into a
  :class:`soma.Plan` with every default filled in, and :class:`soma.Pipeline` runs
  it: extraction, every fold, and the run directory.
- **The verbs**, called one by one. ``Pipeline`` calls the same verbs in a fixed
  order, so a run composed by hand gives the same predictions and weights, and you
  can change any step and hand the result back to the pipeline.

The :doc:`getting-started notebook <getting-started>` takes one experiment through
all three. Every managed run writes a run directory: the config as given
(``config.yaml``), its plan (``plan.yaml``), the predictions, the metrics, and one
model bundle per fold (see :doc:`outputs`).

Compare experiments
-------------------

Features are cached under the image, preprocessing and encoder settings, so runs
that change only the downstream model reuse them; :doc:`caching` explains when
extraction must run again. For a controlled comparison, hold the cohort, labels,
splits and the rest of the protocol fixed while varying one component. Registered
:doc:`benchmarks <benchmarking>` are fixed protocols for reproducing public results
and comparing encoders.

.. _use-cases:
.. _presets:

Use cases
---------

Each use case is a preset over one source shape.
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
     - ``mil_classification(encoder, aggregator, spacing=None, tile_size=None)``
     - Bags of tile features
     - ``slide_mil_binary_classification``, ``slide_mil_multiclass_classification``
   * - Survival
     - ``survival(encoder, aggregator, loss, num_bins=None, spacing=None, tile_size=None)`` with ``loss="nll"`` (and ``num_bins``) or ``loss="cox"``
     - Bags of tile features
     - ``slide_mil_survival_nll``, ``slide_mil_survival_nll_train_quantiles``, ``slide_mil_survival_cox``
   * - Regression
     - ``mil_regression(encoder, aggregator, spacing=None, tile_size=None)``
     - Bags of tile features
     - ``slide_mil_regression``
   * - Tile-level classification
     - ``tile_classification(encoder)``
     - One embedding per image
     - ``tile_binary_classification``
   * - Segmentation
     - ``segmentation(encoder, decoder, num_classes)``
     - Dense token grids
     - ``dense_segmentation``, ``dense_preset_segmentation``, ``grid_live_segmentation``
   * - Detection
     - ``detection(encoder, decoder, num_classes, match_distance=3.0)``
     - Dense token grids
     - ``dense_detection``, ``dense_preset_detection``, ``grid_live_detection``

Next: :doc:`install soma <installation>`, then run the :doc:`getting-started
notebook <getting-started>`.
