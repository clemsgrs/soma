Reporting
=========

Reporting turns completed result bundles into standalone HTML analyses and
cross-run comparisons. The underlying artifact layout is documented in
:doc:`outputs`.

Run reports
-----------

``Pipeline.run()`` writes ``<run_dir>/report.html`` automatically. Regenerate
it from saved artifacts without rerunning training:

.. code-block:: python

   from soma.reporting import generate_report

   report_path = generate_report("output/my_run")

Reports select task-appropriate views from metric summaries, prediction files,
and training history. These include classification curves and confusion
matrices, regression plots, loss curves, and timing when the required artifacts
are present.

An in-memory :class:`soma.pipeline.PipelineResult` can be rendered directly:

.. code-block:: python

   from soma.reporting import generate_report_from_result

   report_path = generate_report_from_result(result, config)

Run comparison
--------------

``compare_runs`` places metrics and configuration differences side by side:

.. code-block:: python

   from soma.reporting import compare_runs

   comparison_path = compare_runs(
       ["output/run_abmil", "output/run_transmil"],
       labels=["ABMIL", "TransMIL"],
   )

Without labels, names are inferred from the differing configuration fields.
The default destination is
``<output_root>/comparisons/<comparison-id>/index.html``; pass ``output_dir``
to override it.

Subgroup statistics
-------------------

Configure subgroup columns under ``evaluation.subgroups`` as described in
:doc:`evaluation`. Run comparisons align predictions by sample and use a
paired permutation test against the best run. Subgroup tests compare each
group with the rest of the cohort by permuting membership; groups with fewer
than 10 samples are omitted. Both views apply Benjamini-Hochberg correction.
