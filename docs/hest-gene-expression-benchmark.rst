HEST
====

What this benchmark measures
----------------------------

Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,
reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,
NeurIPS 2024). Each ``hest/<task>`` uses Soma's native slide2vec cache and the
same closed-form :doc:`spatial-expression probe <regression>`; only ``encoder``
varies. No ``hest`` or TRIDENT runtime is required.

**Pipeline:** spot tiles → frozen encoder → Ridge+PCA probe → mean Pearson.

Prepare the data
----------------

Install Soma with the optional HEST readers (the base package also provides
the ``hf`` download command)::

    pip install 'soma-pathology[hest]'

Some foundation-model weights require accepting their Hugging Face terms and
authenticating once with ``hf auth login``. The HEST data itself is downloaded
separately below.

Download one task while excluding HEST's precomputed ``fm_v1`` features; Soma
re-extracts features and prepares the downloaded tree locally::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

Omit ``--include`` to download every registered task under the same local root.
Start with one task and one encoder: feature extraction is the expensive step
and a GPU is strongly recommended.


Pass the downloaded task directory as ``--raw-root``. Soma writes its standard
manifests automatically and preserves HEST's supplied fold assignments.

Run the benchmark
-----------------

Reproduce IDC::

    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC

Or run every downloaded task::

    soma reproduce hest --raw-root /path/to/hest-bench

Select an encoder with ``--encoder`` (default ``uni2``).

What you can vary
-----------------

Choose one of the encoders supported by the published HEST campaign:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Encoder
     - HEST backbone
   * - ``uni2`` (default)
     - HEST-Benchmark UNI2-h; slide2vec default output (CLS, 1536-d)
   * - ``virchow2``
     - HEST-Benchmark Virchow2; slide2vec ``cls`` output (CLS-only, 1280-d)
   * - ``h-optimus-1``
     - HEST-Benchmark H-Optimus-1; slide2vec default output (CLS, 1536-d)

Run one registered tissue task or the complete downloaded family:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Benchmark
     - HEST task
   * - ``hest/CCRCC``
     - ``CCRCC``
   * - ``hest/COAD``
     - ``COAD``
   * - ``hest/IDC``
     - ``IDC``
   * - ``hest/LUNG``
     - ``LUNG``
   * - ``hest/LYMPH_IDC``
     - ``LYMPH_IDC``
   * - ``hest/PAAD``
     - ``PAAD``
   * - ``hest/PRAD``
     - ``PRAD``
   * - ``hest/READ``
     - ``READ``
   * - ``hest/SKCM``
     - ``SKCM``

All nine registered tissue tasks share the same protocol. HCC has no published score and is
not registered.

Results
-------

Soma closely reproduces HEST's published Pearson scores using native slide2vec
features. The table contains the task–encoder cells currently recorded for this
documentation; ``soma reproduce`` prints the matching reference for any registered
task. These external values are comparisons, not PASS/FAIL gates.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Task
     - Encoder
     - Soma
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

Across these 9 recorded task–encoder comparisons, the median relative difference is **0.21%**; the largest is **2.99%** (COAD with ``uni2``).

See the `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__ for the full published leaderboard.

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
