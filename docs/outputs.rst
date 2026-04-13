Run Outputs
===========

Each pipeline run writes a self-contained bundle beneath ``output_root``.
The bundle captures the resolved configuration, fold-level artifacts, and the
metrics needed to compare experiments reproducibly.

Run directory contents
----------------------

The main run directory contains:

- the resolved pipeline configuration
- fold checkpoints and fold-level summaries
- per-split predictions
- per-split subgroup metrics
- attention artifacts when heatmaps are enabled
- the final HTML report

Split-specific artifacts
------------------------

When a dataset defines multiple test splits, each split gets its own set of
artifacts. Common filenames include:

- ``predictions_<split>.csv``
- ``subgroup_metrics_<split>.json``
- ``fold_N/attention/<sample_id>.npz``
- ``fold_N/heatmaps/<sample_id>.png``

Metric keys in ``summary.json`` are prefixed by split name, for example
``test/auroc_mean`` and ``test_external/auroc_mean``.

Saved timing data
-----------------

Each fold writes ``training_history.json`` with the elapsed time and average
epoch time recorded during training. The HTML report includes the same timing
information in a dedicated training section, while the ETA remains a live-only
display field.

Heatmap artifacts
-----------------

When ``HeatmapConfig.enabled`` is true, the pipeline stores raw attention
scores under ``fold_N/attention/<sample_id>.npz`` and rendered overlays under
``fold_N/heatmaps/``. The rendered overlays can be regenerated with different
visual settings without rerunning inference.

Aggregators that support attention extraction: ``abmil``, ``clam_sb``,
``clam_mb``, ``dsmil``. Heatmaps are skipped for ``mean_pool``, ``max_pool``,
``transmil``, ``dtfdmil``, and ``hipt``.

Heatmap appearance is controlled by :class:`soma.config.HeatmapConfig`:
``cmap`` (colormap name, default ``jet``), ``alpha`` (overlay opacity),
``blur_sigma`` (Gaussian blur radius in pixels).

HTML report
-----------

Each run automatically generates an interactive HTML report containing metrics
summary tables, ROC/PR curves, confusion matrices (classification), scatter and
residual plots (regression), loss curves, and training timing. The report is
written to the run directory as ``report.html``.

Cache versus run outputs
------------------------

The shared cache stores reusable upstream artifacts such as tiling and feature
extraction. The run directory stores the outcome of one specific experiment and
should be treated as immutable once the run completes.
