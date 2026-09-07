Run outputs
===========

Each pipeline run writes a run bundle beneath ``output_root``.
The bundle captures the resolved configuration, per-fold artifacts, and the
metrics needed to compare experiments reproducibly.

The shared cache, which stores reusable upstream artifacts such as tiling and
feature extraction, is documented separately in :doc:`caching`.

Run directory contents
----------------------

For task-training pipelines, the main run directory contains:

- the resolved pipeline configuration
- model checkpoints and per-fold summaries
- per-split predictions
- per-split subgroup metrics
- attention artifacts when heatmaps are enabled
- the final HTML report

Experiment identity
-------------------

Managed outputs group runs by experiment identity, with a timestamp/W&B suffix
for each run. Identity version 2 hashes the ``train`` and ``tune`` manifest
slices, preprocessing, encoders, downstream model, task, training, evaluation,
feature mode, augmentation, normalization, projection, cache policy, and tags.
Representation-only runs instead hash their configured evaluation split and
representation settings.

Training seed and loader settings are run-level choices. Dense-output toggles
and ``evaluation.overwrite_test`` also do not change experiment identity.
Enabled attention-heatmap settings do, because they change generated artifacts;
inactive heatmap rendering options are excluded.

Adding or changing ``test*`` rows leaves the training experiment identity
unchanged. A separate test digest identifies the test assignments and their
sample rows; ``test_results.json`` records test identities and guards against
accidental re-scoring. See :doc:`evaluation` for test holdout and overwrite
controls.

Dataset checksums follow :ref:`semantic manifest identity
<semantic-manifest-identity>`: storage paths are excluded while semantic values
in the selected rows are hashed. ``ExperimentSpec`` retains resolved manifest
paths. ``RunMetadata`` also records the test digest and whole-file SHA-256 values
for the physical CSV files, so relocation or added test rows remain visible in
provenance. Cache identity separately includes physical inputs as described in
:doc:`caching`.

Older experiment identities are not aliased or migrated to version 2.

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

Completed task-training pipelines write ``report.html`` with task-specific
metrics, plots, training history, and timing. Task-free representation runs
write metrics without a task report. See :doc:`reporting` for report contents,
regeneration, and multi-run comparison.

Treat completed run artifacts as the recorded outcome of that run. Use a new
run for a new training attempt; shared features remain in the cache.

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
