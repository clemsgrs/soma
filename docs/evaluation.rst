Evaluation
==========

Evaluation chooses reported metrics, protects held-out test sets, and adds
subgroup breakdowns to saved results. Task heads own the default metric set;
see :doc:`tasks`. Regression runs may add ``pearson``; spatial-expression
probes report per-gene coefficients and ``mean_pearson``.

.. autoclass:: soma.config.EvalConfig
   :members:

.. autoclass:: soma.config.SubgroupConfig
   :members:

Selection and test handling
---------------------------

Keep ``evaluation.metrics`` fixed when comparing candidates. Use
``holdout_test: true`` during model selection to skip all test inference and
artifacts, then evaluate only the selected configuration with the test set
enabled. Tune evaluation, threshold selection, and checkpoint selection still
run.

By default, soma refuses to replace results for a test identity that has
already been scored. Set ``overwrite_test: true`` only for an intentional
re-score. This operational flag does not change experiment identity.

.. code-block:: yaml

   evaluation:
     metrics: [auroc, balanced_accuracy]
     holdout_test: true

Subgroups
---------

Subgroup columns come from ``dataset.csv``. Each distinct value receives the
same metric calculation as the overall cohort:

.. code-block:: yaml

   evaluation:
     metrics: [auroc, balanced_accuracy]
     subgroups:
       columns: [center, grade]

Results are written to ``subgroup_metrics_<split>.json`` and included in the
HTML report. Comparison statistics are described in :doc:`reporting`.

Dense artifacts
---------------

Dense tasks save lightweight visual overlays by default. Probability tensors
for segmentation and heatmap arrays for detection are opt-in through
``save_segmentation_probabilities`` and ``save_detection_heatmaps``. Disable
the corresponding overlay flag for a metrics-only run.

Result objects
--------------

.. autoclass:: soma.evaluation.report.EvaluationReport
   :members:

.. autoclass:: soma.evaluation.report.SamplePrediction
   :members:
