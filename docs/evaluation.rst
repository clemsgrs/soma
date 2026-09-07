Evaluation
==========

Evaluation defines the metric contract for a run and the optional subgroup
breakdowns that appear in the saved outputs and reports.

The main configuration object is :class:`soma.config.EvalConfig`.

.. autoclass:: soma.config.EvalConfig
   :members:

.. autoclass:: soma.config.SubgroupConfig
   :members:

Metric families
---------------

The default metrics depend on the task family:

.. list-table::
   :header-rows: 1

   * - Task family
     - Default metrics
     - Notes
   * - ``binary_classification``
     - ``auroc``, ``balanced_accuracy``, ``auprc``, ``f1``
     - Standard binary classification reporting
   * - ``multiclass_classification``
     - ``auroc_macro``, ``balanced_accuracy``, ``f1_macro``
     - Multi-class classification
   * - ``ordinal_classification``
     - ``qwk``, ``balanced_accuracy``
     - Ordered labels
   * - ``regression``
     - ``mae``, ``r2``
     - Continuous targets
   * - ``survival``
     - ``c_index``
     - Time-to-event outcomes
   * - ``segmentation``
     - ``mean_dice``, ``mean_iou``
     - Dense per-pixel prediction
   * - ``detection``
     - ``mean_f1``
     - Class-aware **F1 at matching distance δ** (``f1_per_class`` /
       ``precision`` / ``recall`` / ``mean_f1_per_image`` also available); see
       :doc:`detection`

Use the smallest set of metrics that answers the scientific question, and
keep it fixed when comparing runs.

Subgroup metrics
----------------

Subgroup columns are read from ``dataset.csv`` and used to break down metrics
for each distinct value in the selected columns.

.. code-block:: yaml

   evaluation:
     metrics: [auroc, balanced_accuracy]
     subgroups:
       columns: [center, grade]

The run writes ``subgroup_metrics_<split>.json`` and includes the breakdowns in
the HTML report. See :doc:`reporting` for group-size requirements and statistical
tests.

Holding out test data
------------------------

Set ``evaluation.holdout_test: true`` to train and report tune results without
test inference or test artifacts. Checkpoint selection and tune evaluation
continue normally. This supports selecting a candidate on tune results before
evaluating it on the held-out test cohort.

Test identities are recorded in ``test_results.json``. Re-scoring an already
recorded test identity is skipped unless ``evaluation.overwrite_test: true``;
resuming unfinished folds is allowed. See :doc:`outputs` for experiment identity.

Evaluation results
------------------

The per-split evaluation output is represented by
:class:`soma.evaluation.report.EvaluationReport`.

.. autoclass:: soma.evaluation.report.EvaluationReport
   :members:

.. autoclass:: soma.evaluation.report.SamplePrediction
   :members:
