Evaluation
==========

Evaluation defines the metric contract for a run and the optional subgroup
breakdowns that appear in the saved outputs and reports. Each task's default
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
values receive their own metrics; see :doc:`reporting` for configuration,
group-size requirements and statistical tests.

Holding out test data
---------------------

Set ``evaluation.holdout_test: true`` to train and report tune results without
test inference or test artifacts. Checkpoint selection and tune evaluation
continue normally. This supports selecting a candidate on tune results before
evaluating it on the held-out test cohort.

Test identities are recorded in ``test_results.json``. Re-scoring an already
recorded test identity is skipped unless ``evaluation.overwrite_test: true``;
resuming unfinished folds is allowed. See :doc:`outputs` for experiment identity.

Evaluation results
------------------

.. autoclass:: soma.evaluation.report.EvaluationReport
   :members:

.. autoclass:: soma.evaluation.report.SamplePrediction
   :members:
