Run outputs
===========

Each pipeline run writes a run directory beneath ``run.output_dir``.
It holds the configuration as given and as resolved, per-fold
artifacts, and the metrics needed to compare experiments reproducibly.

The shared cache, which stores reusable upstream artifacts such as tiling and
feature extraction, is documented separately in :doc:`caching`.

Experiment identity
-------------------

Managed outputs group runs by experiment identity, with a timestamp/W&B suffix
for each run. A run started in the same second as an earlier run of the same
experiment gets ``-2``, ``-3``, … appended instead of reusing its directory.
Identity version 9 hashes the ``train`` and ``tune`` manifest
slices and the run's *plan* (see :ref:`plan-yaml`): the features section as
extraction resolves it, the cohort options, the loaders' batch size, sampler and
augmentation, the learner with every leaf's params, the trainer and the
evaluation. Representation-only runs instead hash their configured evaluation
split and representation settings.

The ``run`` section (output directory, seed, run id, resume, mirror root, fold
selection), the cohort file paths, the loader worker settings,
``trainer.device``, the grid-output save toggles and
``evaluation.overwrite_test`` do not change experiment identity. Enabled
attention-heatmap settings do, because they change generated artifacts.

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

The run directory holds ``config.yaml`` (the config as given), ``plan.yaml``
(the config as resolved, see :ref:`plan-yaml`) and ``summary.json``.
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
  ``evaluation.attention_heatmaps`` is set
- for segmentation and detection, the predictions of tune and every test split
  and the optional overlays, probabilities and heatmaps; see
  :ref:`grid-outputs`

.. _plan-yaml:

The plan
--------

:func:`soma.resolve` turns a config into a :class:`soma.Plan` before anything
runs: the config with every default of every component's params spelled out,
plus what the cohort decides (for example a classification head's label map or
a survival head's bin count). A value the config states wins over what the
cohort suggests; a contradiction (an explicit class count the cohort does not
match) is refused. Feature dimensions are not in the plan: they come from the
features at build time and are recorded in the bundle's ``model.json``.

``plan.yaml`` is the plan as YAML. It loads as a config, so ``soma run
plan.yaml`` replays the run with every value pinned. A component named by a
dotted import path keeps that path in the plan. Versions and code are not part
of the plan: the run records the code each component resolved to (see
:ref:`component-record`).

A resume into an existing run directory (``run.resume`` or ``run.run_id``)
compares the current plan with the saved ``plan.yaml`` and refuses on any
difference, except where and how the launch runs: ``run.output_dir``,
``run.run_id``, ``run.resume``, ``run.mirror_root`` and ``run.folds``. A changed
``run.seed`` is drift. GPU and preprocessing worker counts the config leaves unset
stay unset in the plan, so a run resumes on another machine. The cohort files'
content is then checked against the run's experiment identity, and the code of
the external components against the run's component record.

.. _component-record:

Component code
--------------

The experiment identity holds each component's name and params, not its code.
The run records the code instead: the ``components`` field of ``run.yaml`` and
the ``components`` list of each bundle's ``provenance.json`` hold one entry for
each component of the plan. These are the encoder, the sampler, the
augmentation, each trainer callback, the learner and its aggregator or decoder,
head, optimizer and schedulers, and each metric in ``evaluation.metrics``. An
entry holds:

- ``kind``, and ``name`` as the plan spells it
- ``object``: the ``module.qualname`` of the class or function the name resolved to
- ``distribution`` and ``version`` when the object's file belongs to an
  installed distribution (its ``RECORD`` lists the file), and ``editable: true``
  when that install is editable (the file is in its project directory)
- ``source_sha256``: the SHA-256 of the file that defines the object, or
  ``null`` when there is no file (for example a notebook cell)
- ``git``: the ``sha`` and ``dirty`` state of the git work tree that holds the
  file, when there is one
- ``fingerprint``: the ``version`` of a non-editable install, otherwise
  ``source_sha256``

A component is *external* when its object does not come from soma, slide2vec,
hs2p or torch. The run already records the versions of these four. Two checks
compare the fingerprints of the external components:

- **Resume.** A resume into a run whose ``run.yaml`` has a component record is
  refused when the fingerprint of an external component changed. The error
  names each component with its old and new fingerprint. No setting overrides
  this: start a fresh run.
- **Leaderboard.** ``soma leaderboard`` keeps the latest run of each seed of an
  experiment. If these runs disagree on the fingerprint of an external
  component, it stops with an error that names the experiment and the
  components. Rerun the stale seeds on the current code: the newer run of a
  seed replaces the older one.

Both checks also compare the ``object``: a name that now resolves to another
class or function is a change, even when the fingerprint is the same (two
classes of one unchanged file have the same ``source_sha256``). The error then
lists the old object as ``absent`` in the new run, and the new one as
``absent`` in the old run.

Runs without a component record are not compared. When an external component
has no fingerprint (no source file and no installed version), the run warns at
start that neither check can detect edits to it.

The checks cover runs, not the feature cache. A custom encoder is an external
component, so the record and both checks see an edit to it. The feature cache,
however, does not check an encoder's code or weights: after an edit, a new run
reuses the features of the old model while it records the new fingerprint.
After you change an encoder, delete its feature cache directory (see
:doc:`caching`).

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
- ``provenance.json``: the soma, slide2vec and hs2p versions, the component
  record (see :ref:`component-record`), the selected epoch and its monitor
  value, the fold, the run id and the identity of the feature source (the cache
  key and feature identity, not a path).

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
saved too; ``soma.load`` rebuilds it only when ``soma.learn.record_builder``
recorded the function that builds it, and otherwise refuses: load its
``weights.safetensors`` into the same model instead.

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
   * - ``save_overlays`` (segmentation)
     - ``overlays/<split>/pred/`` and ``overlays/<split>/gt/``: predicted and
       annotated classes blended over each tile
     - ``true``
   * - ``save_probabilities`` (segmentation)
     - ``probabilities/<split>/<sample_id>.npz``: the float16
       ``(C, H, W)`` softmax
     - ``false``
   * - ``save_overlays`` (detection)
     - ``overlays/<split>/pred/`` and ``overlays/<split>/gt/``: predicted
       points (at ``validation_threshold``) and annotated points drawn over
       each tile
     - ``true``
   * - ``save_heatmaps`` (detection)
     - ``heatmaps/<split>/<sample_id>.npz``: the float16 ``(C, H, W)``
       heatmap
     - ``false``

Per-sample confusion matrices are not written; recompute them from the saved
label maps with :func:`soma.evaluation.confusion_records`.

Run index
---------

``<run.output_dir>/indexes/runs.csv`` is append-only: every status change of a
run appends one line, and readers keep the last line per ``run_id``. Run
``soma compact-index <output_dir>`` to rewrite it with one row per run.

Heatmap artifacts
-----------------

When ``evaluation.attention_heatmaps`` is set, the pipeline scores each fold's bundle
once every fold is done and stores raw attention scores in
``attention/<split>/<sample_id>.npz`` and rendered overlays in ``heatmaps/``
(directly in the run directory for single-fold runs, inside each ``fold_N/``
subdir for cross-validation). The rendered overlays can be regenerated with
different visual settings without rerunning inference.

Aggregators that support attention extraction: ``abmil``, ``clam_sb``,
``clam_mb``, ``dsmil``. Heatmaps are skipped for ``mean_pool``, ``max_pool``,
``transmil``, ``dtfdmil``, and ``hipt``.

Heatmap appearance is set by the keys of ``evaluation.attention_heatmaps``
(``{}`` keeps the defaults): ``cmap`` (colormap name, default ``coolwarm``),
``alpha`` (overlay opacity, default 0.5), ``blur_sigma`` (Gaussian blur radius
in pixels, default 0).

Recoverable shared-storage mirrors
----------------------------------

Long-running jobs can keep active outputs on node-local storage while
publishing recovery bundles to shared storage by setting ``run.mirror_root``. The
mirror destination preserves the managed run path beneath that root. Leaving the
setting ``null`` is a no-op and does not change experiment identity.

Only completed folds are published, each with ``config.yaml``, ``plan.yaml`` and
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

.. _log-output:

Log output
----------

soma reports progress through Python's ``logging`` module, under the ``soma``
logger. The ``soma`` command prints soma's INFO messages to stderr; other
libraries print warnings and errors only.

In a Python script, soma checks the logging setup when you create a
``Pipeline`` or call ``soma.extract``:

- If you have not configured logging (no handler on the root logger or on the
  ``soma`` logger), soma adds one stderr handler to the ``soma`` logger and
  sets that logger to INFO. You see messages such as the fold summary without
  any setup. soma adds this handler only once. A level you set yourself is
  kept: on the ``soma`` logger, or on the root logger when it is not Python's
  default WARNING (``logging.getLogger().setLevel(logging.ERROR)`` keeps soma
  quiet).
- If you have configured logging, for example with ``logging.basicConfig``,
  soma changes nothing and your configuration decides what is shown.
- If you configure logging after you create a ``Pipeline``, soma's handler
  stops writing as soon as the root logger or the ``soma`` logger has a handler
  of yours, so each message prints once, through your handlers. The ``soma``
  logger stays at INFO. If you set ``propagate = False`` on the ``soma`` logger,
  its messages never reach your root handlers, so soma's handler keeps printing
  them unless you add a handler to the ``soma`` logger.

To change the output:

.. code-block:: python

   import logging

   soma_logger = logging.getLogger("soma")
   soma_logger.setLevel(logging.WARNING)   # hide soma's INFO messages
   soma_logger.setLevel(logging.DEBUG)     # or show soma's DEBUG messages too
   soma_logger.handlers.clear()            # remove soma's default handler
   soma_logger.propagate = False           # keep soma out of your root handlers
