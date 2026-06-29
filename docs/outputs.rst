Run outputs
===========

Each pipeline run writes a self-contained bundle beneath ``output_root``.
The bundle captures the resolved configuration, per-fold artifacts, and the
metrics needed to compare experiments reproducibly.

The shared cache, which stores reusable upstream artifacts such as tiling and
feature extraction, is documented separately in :doc:`caching`.

Run directory contents
----------------------

The main run directory contains:

- the resolved pipeline configuration
- model checkpoints and per-fold summaries
- per-split predictions
- per-split subgroup metrics
- attention artifacts when heatmaps are enabled
- the final HTML report

Experiment identity
-------------------

Managed outputs group runs by experiment identity before adding the per-run
timestamp/W&B suffix. The identity is derived from the data manifests and the
configuration choices that can change predictions, reported metrics, generated
artifacts, or the experiment indexes. Changing those choices creates a new
experiment namespace instead of mixing outputs under an existing one.

The identity covers the dataset and split checksums, preprocessing and sampling
choices, encoder or composite encoder choices, aggregation or dense decoder /
pixel-classifier choices, task and training settings, feature mode, live
augmentation, evaluation settings, and enabled heatmap artifact settings. The
training seed is intentionally a run-level value, so repeated seeds produce
separate runs under the same experiment when the rest of the configuration is
unchanged.

Evaluation settings are part of the identity because they change the reported
outputs. Heatmap settings are also part of the identity when heatmaps are
enabled because they change generated artifacts; inactive heatmap rendering
options do not fork the namespace.

Layout: single split vs cross-validation
-----------------------------------------

When ``splits.csv`` has no ``fold`` column (or a single fold value), all
artifacts are written directly inside the run directory:

- ``best_model.pt``, ``metrics.json``, ``training_history.json``
- ``predictions_<split>.csv``
- ``attention/<sample_id>.npz`` (if heatmaps enabled)
- ``heatmaps/<sample_id>.png``

When ``splits.csv`` defines multiple folds, each fold gets its own subdirectory:

- ``fold_0/``, ``fold_1/``, … containing the same per-fold files above

Split-specific artifacts
------------------------

When a dataset defines multiple test splits, each split gets its own set of
artifacts, e.g. ``predictions_test.csv`` and ``predictions_test_external.csv``.

Metric keys in ``summary.json`` are prefixed by split name:

- **Single fold**: ``test/auroc``, ``test_external/auroc``
- **Cross-validation**: ``test/auroc_mean``, ``test/auroc_std``

Saved timing data
-----------------

``training_history.json`` records the elapsed time and average epoch time for
each epoch. The HTML report includes the same timing information in a dedicated
training section, while the ETA remains a live-only display field.

Heatmap artifacts
-----------------

When ``HeatmapConfig.enabled`` is true, the pipeline stores raw attention
scores in ``attention/<sample_id>.npz`` and rendered overlays in ``heatmaps/``
(directly in the run directory for single-fold runs, inside each ``fold_N/``
subdir for cross-validation). The rendered overlays can be regenerated with
different visual settings without rerunning inference.

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

Run directory vs cache
----------------------

The run directory stores the outcome of one specific experiment and should be
treated as immutable once the run completes.
