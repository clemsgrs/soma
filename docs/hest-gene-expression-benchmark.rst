HEST
====

Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,
reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,
NeurIPS 2024).

HEST provides 9 registered datasets: CCRCC, COAD, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ, and SKCM.
All share the same closed-form Ridge+PCA probe protocol; see
:doc:`benchmarking` for the shared workflow.

**Pipeline:** spot tiles → frozen encoder → Ridge+PCA probe → mean Pearson

Prepare the data
----------------

Install soma with the optional HEST readers::

    pip install 'soma-pathology[hest]'

Use the Hugging Face CLI to download one task while excluding HEST's
precomputed ``fm_v1`` features; soma re-extracts them locally::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

Run the benchmark
-----------------

Choose a compatible tile-level :doc:`encoder <encoders>` and pass the downloaded
dataset directory as ``--raw-root``; curated manifests are written under
``<raw-root>/curated``. For example::

    soma reproduce hest/IDC --encoder virchow2 --raw-root /path/to/hest-bench/IDC

Or run HEST's 9 datasets in one go::

    soma reproduce hest --encoder virchow2 --raw-root /path/to/hest-bench

How the benchmark runs
----------------------

The benchmark trains no model through the pipeline; it composes soma's
verbs itself. It extracts each spot's tile features once into the feature
cache, then, for each fold:

1. fits :class:`~soma.benchmark.hest.probe.RidgePCAProbe`
   (``StandardScaler`` → ``PCA`` → ``Ridge``) on the train spots;
2. predicts the test spots as :class:`~soma.evaluation.Predictions` with one
   ``value_<k>`` column per gene;
3. scores them with :func:`soma.evaluate`: ``mean_pearson`` plus one
   ``pearson_geneNN_<symbol>`` metric per gene.

Each fold writes ``metrics.json`` and ``predictions_<split>.parquet``; the run
writes ``summary.json``.

Results
-------

Recorded mean Pearson scores alongside the packaged HEST references.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Task
     - Encoder
     - soma
     - HEST reference
   * - PAAD
     - ``uni2``
     - 0.501
     - 0.500
   * - PAAD
     - ``virchow2``
     - 0.472
     - 0.478
   * - PAAD
     - ``h-optimus-1``
     - 0.492
     - 0.496
   * - COAD
     - ``uni2``
     - 0.310
     - 0.301
   * - COAD
     - ``virchow2``
     - 0.262
     - 0.258
   * - COAD
     - ``h-optimus-1``
     - 0.319
     - 0.320
   * - LUNG
     - ``uni2``
     - 0.559
     - 0.559
   * - LUNG
     - ``virchow2``
     - 0.569
     - 0.569
   * - LUNG
     - ``h-optimus-1``
     - 0.577
     - 0.578

See the `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__ for the official reference leaderboard.

Protocol details
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - predictor
     - ``soma.benchmark.hest.probe.RidgePCAProbe``: a closed-form Ridge probe, no trained head, no gradient loop
   * - features
     - ``StandardScaler`` → ``PCA(n_components=256)`` fit on the fold's train spots (X only)
   * - estimator
     - ``Ridge(solver='lsqr', fit_intercept=False)``, penalty ``alpha = 0.0078125`` = 100 / (256·50)
   * - targets
     - 50-gene ``log1p(counts)`` vector per 112 µm spot (baked by the curator)
   * - metric
     - ``mean_pearson`` — per gene, pooled over test spots → mean over 50 genes → mean over folds; each gene's own Pearson is reported as ``pearson_geneNN_<symbol>``
   * - task family
     - ``regression``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/mean_pearson_mean`` (from ``summary.json``)
   * - canonical seeds
     - ``0`` (the probe is closed-form — one seed suffices)
