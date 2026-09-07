Reporting
=========

Completed task-training pipelines write ``report.html``. Use the reporting
module to regenerate it, compare runs, or inspect subgroup metrics and
statistical tests. Task-free representation runs write metrics without a task
report; see :doc:`api`.

HTML reports
------------

A report is generated at the end of each ``Pipeline.run()`` call and written to
``<run_dir>/report.html``. It can also be regenerated from a completed run
directory without re-running training:

.. code-block:: python

   from soma.reporting import generate_report

   path = generate_report("output/my_run")
   print(f"Report written to {path}")

Contents of an HTML report:

- **Metrics summary** — fold-level and aggregated results for all splits
- **ROC and PR curves** — for classification tasks
- **Confusion matrix** — for classification tasks
- **Scatter and residual plots** — for regression tasks
- **Loss curves** — training and validation loss per epoch
- **Training timing** — elapsed time per epoch and total run time

To reuse predictions, metrics, and training history already in memory, pass a
:class:`soma.pipeline.PipelineResult`:

.. code-block:: python

   from soma import Pipeline
   from soma.reporting import generate_report_from_result

   result = Pipeline(config).run()
   path = generate_report_from_result(result, config)

Pass ``dataset=dataset`` when regenerating subgroup reports from memory; the
report needs the dataset metadata to populate those groups.

Run comparison
--------------

Given two or more completed run directories, ``compare_runs`` generates a
comparison report bundle that shows per-metric tables side by side, config
diffs (keys that differ between runs are highlighted), and statistical tests:

.. code-block:: python

   from soma.reporting import compare_runs

   path = compare_runs(
       ["output/run_abmil", "output/run_transmil"],
       labels=["ABMIL", "TransMIL"],
   )

When ``labels`` is omitted, labels are auto-derived from the config diff
(e.g., the aggregator name if that is the only varying field).

By default, the report is written beneath the shared ``output_root`` in
``comparisons/<comparison-id>/index.html``. Pass ``output_dir`` to override
the directory that receives the report bundle. An explicit ``output_dir`` is
required when the runs do not share an ``output_root``.

For supported metrics, comparisons use a sample-level paired permutation test
against the best run. Predictions are aligned by ``sample_id`` and randomly
swapped between runs for 1,000 iterations. This works for single-fold and
cross-validation runs with shared samples. P-values are corrected across
reported comparisons with the Benjamini–Hochberg procedure.

Subgroup analysis
-----------------

Set :attr:`soma.config.SubgroupConfig.columns` to a list of column names in
``dataset.csv`` to compute per-subgroup metrics alongside the overall results:

.. code-block:: yaml

   evaluation:
     subgroups:
       columns: [center, grade]

For each column, the pipeline computes metrics for distinct values with at
least two samples. Patient-level predictions require consistent subgroup
metadata across that patient's slides.

Subgroup significance tests compare each group with the rest of the cohort by
permuting group membership 1,000 times. Tests require at least 10 samples in the
group and two in the remainder; these limits do not suppress the descriptive
metrics for smaller groups.

Results are written to ``subgroup_metrics_<split>.json`` alongside the main
summary. They are also included in the HTML report as separate tables.
