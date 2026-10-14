Evaluation
==========

Scoring is two verbs with one noun between them. :func:`soma.predict` runs a model
over a loader and returns :class:`~soma.evaluation.Predictions`: one row per sample,
with the columns the task head declares. :func:`soma.evaluate` joins those
predictions with the cohort's targets by ``sample_id`` and returns an
:class:`~soma.evaluation.Evaluation`. Targets never travel inside predictions: they
come from :meth:`soma.data.Cohort.targets`, so predictions saved by a run, or made
outside soma, are scored the same way.

Score predictions
-----------------

.. code-block:: python

   import soma

   cohort = soma.Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
   predictions = soma.Predictions.from_parquet("run/predictions_test.parquet")
   # or, after soma.fit: predictions = soma.predict(result.learner, fold.tests["test"])
   evaluation = soma.evaluate(
       predictions,
       cohort.targets("test"),
       metrics=["auroc", "balanced_accuracy"],
       subgroups=["center"],
   )
   evaluation.to_json("evaluation.json")

``evaluate`` scores exactly the ids it is given. It refuses a prediction
without a target, a target without a prediction, and duplicate ids; the error
names the count. Fill the samples a model could not score with
:class:`~soma.evaluation.Baseline` (class prior, mean value or constant risk,
fitted on the training targets), or filter the targets. When the predictions
carry ``is_placeholder``, the metrics add ``coverage``, ``num_samples``,
``num_real_samples`` and ``num_placeholder_samples``. The pipeline fills
samples without features this way.

``soma evaluate`` does the same from the command line (see :doc:`cli`), without a
run directory.

``predict`` writes side artefacts through callbacks (see :ref:`custom-callbacks`):
``soma.evaluation.heatmaps.SaveAttention(dir)`` writes each bag's normalised tile
attention to ``<dir>/<sample_id>.npz``.

Choose metrics
--------------

``metrics`` is ``"default"`` (the family's defaults, listed in the :doc:`task zoo
<tasks>`), or a list of registry keys, dotted import paths and
``(predictions, targets) -> float`` callables. A callable receives the two frames
aligned by ``sample_id``, with labels already mapped to class indices.
:data:`soma.evaluation.metric_registry` lists each metric's families and direction;
a key outside the predictions' family is refused. In a config,
``evaluation.metrics`` takes the built-in names of the head's family (an empty
list means the defaults); score a metric of your own from Python, or with
``soma evaluate --metrics`` by its dotted path.

:func:`soma.evaluation.compare` runs a paired permutation test of several
prediction sets against the best one.

Subgroup metrics
----------------

``evaluation.subgroups`` names ``dataset.csv`` columns whose distinct values
receive their own metrics:

.. code-block:: yaml

   evaluation:
     subgroups: [center, grade]

For each column, every value with at least two samples gets the run's metrics
and its sample count. A significance test then compares each group with the
rest of the split by permuting group membership 1,000 times; it needs at least
10 samples in the group and two in the remainder. Patient-level runs read each
patient's metadata from its first slide. Results are written to
``subgroup_metrics_<split>.json`` for each test split.

Holding out test data
---------------------

Set ``evaluation.holdout_test: true`` to train and report tune results without
test inference or test artifacts. Checkpoint selection and tune evaluation
continue normally. This supports selecting a candidate on tune results before
evaluating it on the held-out test cohort.

Test identities are recorded in ``test_results.json``. Re-scoring an already
recorded test identity is skipped unless ``evaluation.overwrite_test: true``;
resuming unfinished folds is allowed. See :doc:`outputs` for experiment identity.

.. _grid-predictions:

Segmentation and detection predictions
--------------------------------------

For a segmentation or detection head, :func:`soma.predict` runs the head's
``postprocess`` and returns one of two prediction types instead of a table, and
:func:`soma.evaluate` scores it against a target source rather than a targets
frame:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - :class:`~soma.evaluation.LabelMapPredictions`
     - :class:`~soma.evaluation.PointPredictions`
   * - Holds
     - One ``(H, W)`` uint8 label map per sample
     - A frame ``sample_id, x, y, class, score`` in level-0 pixels, plus every
       sample id (a sample with no detection is still scored)
   * - Saved as
     - ``save(dir)``: one PNG per sample and ``index.parquet``;
       ``load(dir)`` reads it back
     - ``to_parquet(path)`` / ``from_parquet(path)``
   * - Scored against
     - :class:`~soma.data.LabelMapSource` or :class:`~soma.data.CachedLabelMapSource`
     - :class:`~soma.data.PointSource`
   * - Metrics
     - ``mean_dice``, ``mean_iou``, ``dataset_global_mean_dice``,
       ``foreground_mean_dice``, ``dice_per_class``
     - ``mean_f1``, ``f1_per_class``, ``precision``, ``recall``,
       ``mean_f1_per_image``

Label maps are scored from per-sample confusion counts. Points are matched to
the annotations per class within the matching distance δ, with Hungarian
assignment by default. A metric given as a ``{name, params}`` leaf can change
the matching: ``matching: greedy`` matches by descending score, and
``match_distance`` overrides δ (in level-0 pixels).

Point predictions keep every peak above the head's ``min_score``. Choosing an
operating point is a separate step: ``with_threshold`` takes one threshold or a
``{class: threshold}`` mapping, and :func:`~soma.evaluation.sweep_threshold`
picks the threshold that maximises a metric on the given targets (one per
class with ``per_class=True``). Sweep on tune and apply the result to test:

.. code-block:: python

   import soma
   from soma.evaluation import PointPredictions, sweep_threshold

   targets = soma.data.PointSource.from_manifest(soma.AnnotationManifest.from_csv("dataset.csv"))
   tune = PointPredictions.from_parquet("run/predictions_tune.parquet")
   test = PointPredictions.from_parquet("run/predictions_test.parquet")

   threshold, _ = sweep_threshold(tune, targets.subset(tune.sample_ids), per_class=True)
   evaluation = soma.evaluate(
       test.with_threshold(threshold),
       targets.subset(test.sample_ids),
       metrics=[{"name": "mean_f1", "params": {"matching": "greedy"}}],
   )

Grid ``predict`` callbacks write side artefacts:
:class:`~soma.evaluation.SaveOverlays` (predictions and supervision drawn over the
tile), :class:`~soma.evaluation.SaveProbabilities` (per-sample float16 softmax,
segmentation) and :class:`~soma.evaluation.SaveHeatmaps` (per-sample float16
heatmaps, detection), for example
``soma.predict(learner, loader, [SaveOverlays("overlays/test", images=images)])``.

Per-sample confusion matrices are recomputable from saved label maps with
:func:`soma.evaluation.confusion_records`.

``soma evaluate`` also accepts a label-map directory or a points parquet. It
scores label maps against the masks in the cohort CSV's ``label_mask_path``
column and points against the files in its ``points_path`` column; both must
already hold class indices.

Reference
---------

.. autoclass:: soma.config.EvaluationConfig

.. autofunction:: soma.evaluation.predict

.. autofunction:: soma.evaluation.evaluate

.. autoclass:: soma.evaluation.Predictions
   :members:

.. autoclass:: soma.evaluation.Evaluation
   :members:

.. autoclass:: soma.evaluation.Baseline
   :members:

.. autofunction:: soma.evaluation.compare

.. autoclass:: soma.evaluation.LabelMapPredictions
   :members: save, load

.. autoclass:: soma.evaluation.PointPredictions
   :members: with_threshold, to_parquet, from_parquet

.. autofunction:: soma.evaluation.sweep_threshold

.. autofunction:: soma.evaluation.confusion_records
