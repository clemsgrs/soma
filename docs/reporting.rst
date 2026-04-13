Reporting
=========

Every completed run automatically produces an HTML report. The reporting
module also supports multi-run comparison and per-subgroup statistical analysis.
This is the detailed guide for report contents, subgroup breakdowns, and
comparison statistics.

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

The report can also be generated from an in-memory
:class:`soma.pipeline.PipelineResult` without any disk reads:

.. code-block:: python

   from soma.reporting import generate_report_from_result

   result = Pipeline(config).run()
   path = generate_report_from_result(result, config)

Run comparison
--------------

Given two or more completed run directories, ``compare_runs`` generates a
single HTML report that shows per-metric tables side by side, config diffs
(keys that differ between runs are highlighted), and statistical tests:

.. code-block:: python

   from soma.reporting import compare_runs

   path = compare_runs(
       ["output/run_abmil", "output/run_transmil"],
       labels=["ABMIL", "TransMIL"],
   )

When ``labels`` is omitted, labels are auto-derived from the config diff
(e.g., the aggregator name if that is the only varying field).

Subgroup analysis
-----------------

Set :attr:`soma.config.SubgroupConfig.columns` to a list of column names in
``dataset.csv`` to compute per-subgroup metrics alongside the overall results:

.. code-block:: yaml

   evaluation:
     subgroups:
       columns: [center, grade]

For each listed column, the pipeline computes metrics for every distinct value
(e.g., each hospital, each grade level). Groups smaller than
``SubgroupConfig.min_group_size`` (default: 10) are skipped.

Results are written to ``subgroup_metrics_<split>.json`` alongside the main
summary. They are also included in the HTML report as separate tables.

For comparison runs, a permutation test (paired sign-flip test over folds,
1000 iterations by default) is used to assess whether metric differences
between runs are statistically significant. P-values are corrected for
multiple comparisons using the Benjamini-Hochberg procedure.
