HEST
====

Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,
reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,
NeurIPS 2024).

HEST provides 9 registered datasets: CCRCC, COAD, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ, and SKCM.
All share the same closed-form :doc:`spatial-expression probe <regression>` protocol.

**Pipeline:** spot tiles → frozen encoder → Ridge+PCA probe → mean Pearson

Prepare the data
----------------

Install soma with the optional HEST readers::

    pip install 'soma-pathology[hest]'

Use the Hugging Face CLI to download one task while excluding HEST's
precomputed ``fm_v1`` features; soma re-extracts them locally::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

The ``hf`` CLI downloads the data. Omit ``--include`` to download
every registered task under the same local root.

Run the benchmark
-----------------

Pick any tile-level :doc:`encoder <encoders>` supported by soma and pass the
downloaded task directory as ``--raw-root``. ``soma reproduce`` runs the
built-in HEST curator automatically, writes the manifests under
``<raw-root>/curated``, preserves HEST's fold assignments, extracts features,
runs the Ridge probe, and reports the mean Pearson score. No separate curation
command is required. Some model weights require ``hf auth login``. For example::

    soma reproduce hest/IDC --encoder virchow2 --raw-root /path/to/hest-bench/IDC

Or run HEST's 9 datasets in one go::

    soma reproduce hest --encoder virchow2 --raw-root /path/to/hest-bench

Results
-------

We benchmarked three encoders: soma closely reproduces
HEST's published Pearson scores.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Task
     - Encoder
     - soma
     - HEST reference
   * - PAAD
     - ``uni2``
     - 0.5007
     - 0.5001
   * - PAAD
     - ``virchow2``
     - 0.4769
     - 0.4779
   * - PAAD
     - ``h-optimus-1``
     - 0.4916
     - 0.4964
   * - COAD
     - ``uni2``
     - 0.3105
     - 0.3015
   * - COAD
     - ``virchow2``
     - 0.2615
     - 0.2581
   * - COAD
     - ``h-optimus-1``
     - 0.3190
     - 0.3195
   * - LUNG
     - ``uni2``
     - 0.5593
     - 0.5587
   * - LUNG
     - ``virchow2``
     - 0.5520
     - 0.5685
   * - LUNG
     - ``h-optimus-1``
     - 0.5768
     - 0.5779

Across these 9 recorded task–encoder comparisons, the median relative difference is **0.21%**.

See the `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__ for the official reference leaderboard.

Protocol details
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - head
     - closed-form Ridge probe — no trained head, no gradient loop
   * - features
     - ``StandardScaler`` → ``PCA(n_components=256)`` fit on the fold's train spots (X only)
   * - estimator
     - ``Ridge(solver='lsqr', fit_intercept=False)``, penalty ``alpha = 0.0078125`` = 100 / (256·50)
   * - targets
     - 50-gene ``log1p(counts)`` vector per 112 µm spot (baked by the curator)
   * - metric
     - ``pearson`` — per gene, pooled over test spots → mean over 50 genes → mean over folds
   * - task family
     - ``regression``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/mean_pearson_mean`` (from ``summary.json``)
   * - canonical seeds
     - ``0`` (the probe is closed-form — one seed suffices)

See :doc:`benchmarking` for the shared benchmark workflow and :doc:`regression`
for the probe and metric.
