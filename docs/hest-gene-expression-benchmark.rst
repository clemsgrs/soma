HEST
====

Spatial gene-expression prediction from H&E with a frozen encoder, reproducing
`HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume *et al.*, NeurIPS 2024).
A closed-form probe on the frozen features predicts the expression of 50 genes at every
spot, so the benchmark measures the encoder alone.

Data and task
-------------

HEST-Benchmark pairs H&E images with spatial transcriptomics. Each spot is a 112 µm
tile with a measured expression vector. Nine tasks are registered, one per organ or
cancer type, as ``hest/<task>``: CCRCC, COAD, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ
and SKCM.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Task
     - Predict the ``log1p`` counts of each task's 50 most variable genes from the
       spot's tile.
   * - Splits
     - HEST's patient-level folds, read from the downloaded benchmark; soma never
       partitions the data itself.
   * - Metric
     - ``mean_pearson``: the Pearson correlation of each gene over the test spots,
       averaged over the 50 genes and then over the folds. Each gene's own
       correlation is reported as ``pearson_geneNN_<symbol>``.

Protocol
--------

The benchmark trains no network. It extracts every spot's tile features once into
the feature cache, then fits a Ridge probe per fold, exactly as HEST does.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Encoder
     - Any frozen tile encoder; the ``encoder`` axis of the benchmark.
   * - Features
     - ``StandardScaler`` then ``PCA`` to 256 components, both fitted on the fold's
       train spots only.
   * - Probe
     - ``Ridge(solver='lsqr', fit_intercept=False)`` with ``alpha = 100 / (256 · 50)``,
       the HEST default for 256 components and 50 genes.
   * - Seeds
     - ``0``. The probe is closed-form, so one seed suffices.

Each fold writes ``metrics.json`` and ``predictions_<split>.parquet``, and the run
writes ``summary.json``, like any soma run; the probe is
:class:`soma.benchmark.hest.probe.RidgePCAProbe`.

Results
-------

soma's ``mean_pearson`` on each task next to the value HEST publishes for the same
encoder. HEST extracts its features with TRIDENT and soma with slide2vec, so the two
are not expected to agree to the digit: the difference is shown as it is, without a
tolerance band, and reproduction is judged by whether the encoders keep their
published ranking.

.. include:: _generated/hest-results.rst

The published values are the `HEST-Benchmark leaderboard
<https://github.com/mahmoodlab/HEST#hest-benchmark>`__.

Reproduce
---------

**Prepare the data.** Install the optional HEST readers, then download one task from
the Hugging Face Hub, leaving out HEST's precomputed features (soma extracts its own)::

    pip install 'soma-pathology[hest]'
    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

**Run one task** or all nine; curated manifests are written under
``<raw-root>/curated``::

    soma reproduce hest/IDC --encoder virchow2 --raw-root /path/to/hest-bench/IDC
    soma reproduce hest --encoder virchow2 --raw-root /path/to/hest-bench

**Compare encoders** with ``--encoders``, and append a score to the results ledger with
``--record``. :doc:`benchmarking` describes the options shared by every benchmark.
