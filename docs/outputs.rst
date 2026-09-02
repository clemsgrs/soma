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

Dataset checksums use :ref:`semantic manifest identity <semantic-manifest-identity>`:
machine-local artifact paths are excluded, while every semantic manifest value and
split assignment remains hashed. ``ExperimentSpec`` retains the resolved manifest
paths, while ``RunMetadata`` also stores whole-file SHA-256 provenance for the physical
``dataset.csv`` and ``splits.csv``; those provenance values may differ after
relocation. Cache identities are a separate physical-input contract and continue to
include the paths documented in :doc:`caching`.

This semantic checksum is a clean identity break: experiments and leaderboard triples
created by older soma versions are not aliased or migrated to the new values.

Evaluation settings are part of the identity because they change the reported
outputs. Heatmap settings are also part of the identity when heatmaps are
enabled because they change generated artifacts; inactive heatmap rendering
options do not fork the namespace.

Layout: single split vs cross-validation
-----------------------------------------

When ``splits.csv`` has no ``fold`` column (or a single fold value), all
artifacts are written directly inside the run directory:

- ``best_model.pt``, ``metrics.json``, ``training_history.json``
- ``feature_adapter.json`` (only when ``normalization`` or ``projection`` asks for a transform)
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

``*_std`` is the sample standard deviation (``ddof=1``) across folds. Threshold-free
metrics (AUROC, AUPRC, C-index, ...) are ``nan`` on a fold whose split holds a single
class or no comparable pairs; such folds are excluded from ``*_mean`` / ``*_std`` and
counted in ``test/<metric>_nan_folds``, which is only written when the count is non-zero.

Run index
---------

``<output_root>/indexes/runs.csv`` is append-only: every status change of a run
appends one line, and readers keep the last line per ``run_id``. Run
``soma compact-index <output_root>`` to rewrite it with one row per run.

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

Recoverable shared-storage mirrors
----------------------------------

Long-running jobs can keep active outputs on node-local storage while
publishing recovery bundles to shared storage by setting ``run.mirror_root``. The
mirror destination preserves the managed run path beneath that root. Leaving the
setting ``null`` is a no-op and does not change experiment identity.

Only completed folds are published. Each shared copy is staged beside its final
destination with the resolved ``config.yaml`` and a ``manifest.json`` containing every
file's SHA-256 digest and byte size. The staged copy is verified and exposed by one atomic
rename, so a partial copy never looks complete. Mirror errors never change a healthy local
training result; a later fold event or resumed run retries any completed local fold whose
atomic destination is still absent. Already-published destinations are not re-hashed on
this retry path.

If node-local run storage is lost, a pinned resume restores checksum-verified completed
folds from the corresponding mirror before checking which folds remain. A bare
``resume: true`` can do the same when exactly one recipe-compatible mirrored run exists
for the experiment. It fails loudly when several compatible runs exist so the user can
select one with ``run_id`` rather than letting recovery guess. Mid-fold checkpoints are
not mirrored; an incomplete fold resumes from local state when available or restarts
after node loss.
