Evaluation
==========

Evaluation defines the metric contract for a run and the optional subgroup
breakdowns that appear in the saved outputs. Each task's default
metrics are listed in the :doc:`task zoo <tasks>`; detection additionally
exposes ``f1_per_class``, ``precision``, ``recall`` and ``mean_f1_per_image``
(see :doc:`detection`).

.. autoclass:: soma.config.EvalConfig
   :members:

.. autoclass:: soma.config.SubgroupConfig
   :members:

Subgroup metrics
----------------

``evaluation.subgroups.columns`` names ``dataset.csv`` columns whose distinct
values receive their own metrics:

.. code-block:: yaml

   evaluation:
     subgroups:
       columns: [center, grade]

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

Predictions and evaluations
---------------------------

Scoring is two verbs with one noun between them. :func:`soma.predict` runs a
model over a loader and returns :class:`~soma.evaluation.Predictions`: one row
per sample with the columns the task head declares. :func:`soma.evaluate` joins
those predictions with the cohort's targets by ``sample_id`` and returns an
:class:`~soma.evaluation.Evaluation`. Targets never travel inside predictions:
they come from :meth:`soma.data.Cohort.targets`.

.. code-block:: python

   import soma
   from soma.data import Cohort
   from soma.evaluation import Predictions

   cohort = Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
   predictions = Predictions.from_parquet("run/predictions_test.parquet")
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

``metrics`` is ``"default"`` (the family's defaults), or a list of registry
keys, dotted import paths and ``(predictions, targets) -> float`` callables.
A callable receives the two frames aligned by ``sample_id``, with labels
already mapped to class indices. :data:`soma.evaluation.metric_registry` lists
each metric's families and direction; a key outside the predictions' family is
refused. :func:`soma.evaluation.compare` runs a paired permutation test of
several prediction sets against the best one.

``soma evaluate`` does the same from the command line (see :doc:`cli`), so
predictions made outside soma can be scored without a run directory.

.. autofunction:: soma.evaluation.predict

.. autofunction:: soma.evaluation.evaluate

.. autoclass:: soma.evaluation.Predictions
   :members:

.. autoclass:: soma.evaluation.Evaluation
   :members:

.. autoclass:: soma.evaluation.Baseline
   :members:

.. autofunction:: soma.evaluation.compare
