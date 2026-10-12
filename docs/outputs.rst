Run outputs
===========

Each pipeline run writes a run bundle beneath ``output_root``.
The bundle captures the resolved configuration, per-fold artifacts, and the
metrics needed to compare experiments reproducibly.

The shared cache, which stores reusable upstream artifacts such as tiling and
feature extraction, is documented separately in :doc:`caching`.

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
in the selected rows are hashed. Run metadata still records the resolved
manifest paths and whole-file SHA-256 values, so relocation or added test rows
remain visible in provenance. Cache identity separately includes physical
inputs as described in :doc:`caching`.

Run directory layout
--------------------

The run directory holds the resolved ``config.yaml`` and ``summary.json``.
When ``splits.csv`` has no ``fold`` column (or a single fold value), the
per-fold artifacts are written directly inside it; with several folds, each
``fold_N/`` subdirectory holds its own set:

- ``bundle/``, the trained model (see :ref:`model-bundle`), ``metrics.json`` and
  ``training_history.json``
- ``feature_adapter.json`` when ``normalization`` or ``projection`` is active:
  the methods, ``eps`` floor and floored-channel count, dimensions, projection
  seed, PCA explained variance, and the number of samples and feature rows used
  to fit each stage
- ``predictions_<split>.parquet`` for each test split of a slide, patient or
  tile run, and ``subgroup_metrics_<split>.json`` when subgroup columns are
  configured
- ``attention/<split>/<sample_id>.npz`` and ``heatmaps/<sample_id>.png`` when
  heatmaps are enabled
- for segmentation and detection, the predictions of tune and every test split
  and the optional overlays, probabilities and heatmaps; see
  :ref:`grid-outputs`

.. _model-bundle:

Model bundle
------------

Each fold saves its selected model as a bundle with :func:`soma.save`. A bundle
is a directory of four files:

- ``weights.safetensors``: the model's full ``state_dict``. The fitted feature
  adaptor (``normalization`` / ``projection``) is stored here as buffers.
- ``model.json``: how to rebuild the model before the weights load: its shape,
  the builder and the builder's arguments (aggregator or decoder, task head,
  adaptor sections, and the encoder of a live run).
- ``preprocessing.json``: what scoring needs beyond the model: the task family,
  class names, target schema, metrics, and for grid runs the grid geometry, read
  spacing and tile size. A detection bundle also records the score threshold
  chosen on tune and the matching distance.
- ``provenance.json``: the soma, slide2vec and hs2p versions, the selected epoch
  and its monitor value, the fold, the run id and the identity of the feature
  source (the cache key and feature identity, not a path).

:func:`soma.load` rebuilds the model in eval mode and returns a
:class:`soma.ModelBundle` with ``model``, ``preprocessing`` and ``provenance``.
A live run's frozen encoder is built when the bundle loads, so missing encoder
weights fail at load time. A bundle written by another soma major version is
refused; another minor or patch version loads with a warning::

   import soma

   bundle = soma.load("runs/.../fold_0/bundle")
   predictions = soma.predict(bundle.model, loader)

``soma.save`` writes the bundle atomically: a reader sees the previous bundle
or the new one, never a partial one. A model built by hand in a notebook can be
saved too, but ``soma.load`` cannot rebuild it, because ``model.json`` records
no builder; load its ``weights.safetensors`` into the same model instead.

Resume is per fold. A fold is complete when its ``metrics.json`` and its
``bundle/`` exist. A relaunch into the same run directory (``run.resume`` or
``run.run_id``) skips complete folds and trains the others from epoch 0.

Split-specific artifacts
------------------------

When a dataset defines multiple test splits, each split gets its own set of
artifacts, e.g. ``predictions_test.parquet`` and
``predictions_test_external.parquet``.

Prediction files
----------------

``predictions_<split>.parquet`` holds one row per sample of the split: a
``sample_id`` column, the columns the task head declares (``prob_0`` …
``prob_<k-1>`` and ``predicted_label`` for classification, ``predicted_label``
and ``raw_score`` for ordinal classification, ``predicted_value`` for
regression (``value_0`` … ``value_<G-1>`` with ``num_outputs: G``),
``risk_score`` for survival), any column a custom learner adds, and
``is_placeholder``. A placeholder row is a baseline prediction for a sample
without features (see :doc:`evaluation`). Targets are not stored here; they
come from ``dataset.csv``. The parquet schema metadata records the task family,
the soma version, the role of each column and the class label of each
probability column. Read the file with
:meth:`soma.evaluation.Predictions.from_parquet` and score it again with
``soma evaluate`` (see :doc:`cli`).

Metric keys in ``summary.json`` are prefixed by split name:

- **Single fold**: ``test/auroc``, ``test_external/auroc``
- **Cross-validation**: ``test/auroc_mean``, ``test/auroc_std``

``*_std`` is the sample standard deviation (``ddof=1``) across folds. Threshold-free
metrics (AUROC, AUPRC, C-index, ...) are ``nan`` on a fold whose split holds a single
class or no comparable pairs; such folds are excluded from ``*_mean`` / ``*_std`` and
counted in ``test/<metric>_nan_folds``, which is only written when the count is non-zero.

.. _grid-outputs:

Segmentation and detection outputs
----------------------------------

Segmentation and detection runs write predictions for tune and for every test
split:

- **Segmentation:** ``predictions_<split>/``, one PNG label map of class indices
  per sample and an ``index.parquet``. Read it with
  :meth:`soma.evaluation.LabelMapPredictions.load`.
- **Detection:** ``predictions_<split>.parquet``, the level-0 point list
  ``sample_id, x, y, class, score`` with every peak above the head's
  ``min_score``, before any threshold. Read it with
  :meth:`soma.evaluation.PointPredictions.from_parquet`.

``soma evaluate`` scores either again (see :ref:`grid-predictions`). Optional
artifacts follow ``evaluation`` settings:

.. list-table::
   :header-rows: 1
   :widths: 36 36 28

   * - Setting
     - Writes
     - Default
   * - ``save_segmentation_overlays``
     - ``overlays/<split>/pred/`` and ``overlays/<split>/gt/``: predicted and
       annotated classes blended over each tile
     - ``true``
   * - ``save_segmentation_probabilities``
     - ``probabilities/<split>/<sample_id>.npz``: the float16
       ``(C, H, W)`` softmax
     - ``false``
   * - ``save_detection_overlays``
     - ``overlays/<split>/pred/`` and ``overlays/<split>/gt/``: predicted
       points (at ``validation_threshold``) and annotated points drawn over
       each tile
     - ``true``
   * - ``save_detection_heatmaps``
     - ``heatmaps/<split>/<sample_id>.npz``: the float16 ``(C, H, W)``
       heatmap
     - ``false``

Per-sample confusion matrices are not written; recompute them from the saved
label maps with :func:`soma.evaluation.confusion_records`.

Run index
---------

``<output_root>/indexes/runs.csv`` is append-only: every status change of a run
appends one line, and readers keep the last line per ``run_id``. Run
``soma compact-index <output_root>`` to rewrite it with one row per run.

Heatmap artifacts
-----------------

When ``HeatmapConfig.enabled`` is true, the pipeline scores each fold's bundle
once every fold is done and stores raw attention scores in
``attention/<split>/<sample_id>.npz`` and rendered overlays in ``heatmaps/``
(directly in the run directory for single-fold runs, inside each ``fold_N/``
subdir for cross-validation). The rendered overlays can be regenerated with
different visual settings without rerunning inference.

Aggregators that support attention extraction: ``abmil``, ``clam_sb``,
``clam_mb``, ``dsmil``. Heatmaps are skipped for ``mean_pool``, ``max_pool``,
``transmil``, ``dtfdmil``, and ``hipt``.

Heatmap appearance is controlled by :class:`soma.config.HeatmapConfig`:
``cmap`` (colormap name, default ``jet``), ``alpha`` (overlay opacity),
``blur_sigma`` (Gaussian blur radius in pixels).

Recoverable shared-storage mirrors
----------------------------------

Long-running jobs can keep active outputs on node-local storage while
publishing recovery bundles to shared storage by setting ``run.mirror_root``. The
mirror destination preserves the managed run path beneath that root. Leaving the
setting ``null`` is a no-op and does not change experiment identity.

Only completed folds are published, each with the resolved ``config.yaml`` and
a ``manifest.json`` of SHA-256 digests, through an atomic rename so a partial
copy never looks complete. Mirror errors never change the local training
result; a later fold event or resumed run retries any fold whose destination
is still absent.

If node-local storage is lost, a resume pinned to ``run_id`` restores
checksum-verified completed folds from the mirror before continuing. A bare
``resume: true`` does the same when exactly one compatible mirrored run exists
and fails when several do. Mid-fold checkpoints are not mirrored, so an
incomplete fold restarts after node loss.

Folds of one run can train side by side, one per GPU. Pin every launch to the
same ``run.run_id`` and give each its share with ``run.folds`` (for example
``run.folds: [2]``). A launch trains only its listed folds and leaves the other
pending folds alone; folds seed independently, so the result matches a single
launch. While folds remain pending the run is recorded as ``partial`` and no
``summary.json`` is written. The launch that finds every fold complete writes
the summary, so finish with one launch without ``run.folds`` if
no fold launch ended last. Two launches given the same fold would both train
it: the selection is not a lock.
