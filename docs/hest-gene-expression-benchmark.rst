HEST
====

Purpose
-------

Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,
reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,
NeurIPS 2024). Each ``hest/<task>`` uses Soma's native slide2vec cache and the
same closed-form :doc:`spatial-expression probe <regression>`; only ``encoder``
varies. No ``hest`` or TRIDENT runtime is required.

Protocol
--------

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

Supported campaign encoders:

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

Registered tasks:

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

All nine scored tasks share this protocol. HCC has no published score and is not
registered.

Prepare and run
---------------

Download one task while excluding HEST's precomputed ``fm_v1`` features; Soma
re-extracts features and curates the downloaded tree offline (ADR 0004)::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

Reproduce IDC::

    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC

Or run every downloaded task::

    soma reproduce hest --raw-root /path/to/hest-bench

Select an encoder with ``--encoder`` (default ``uni2``).

Results
-------

Published IDC references
~~~~~~~~~~~~~~~~~~~~~~~~

HEST's Ridge+PCA Pearson values are **external and non-gating**. The inline table
shows the supported campaign encoders; see the `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__ and `packaged reference CSV
<https://github.com/clemsgrs/soma/blob/main/soma/benchmarks/reference/hest.csv>`__
for the complete leaderboard and task × encoder evidence.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Encoder
     - Published ``pearson``
   * - ``h-optimus-1``
     - 0.6024
   * - ``virchow2``
     - 0.5971
   * - ``uni2``
     - 0.5898

Reproduction — is it sound?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soma extracts native slide2vec features rather than HEST's TRIDENT features. Published
HEST values are therefore **external, non-gating** references: no cross-stack delta
produces PASS/FAIL (ADR 0005). The joined results and reference ledgers show:

* **A — absolute agreement:** Pearson and signed slide2vec↔TRIDENT delta per cell.
* **B — rank agreement:** pooled encoder-pair concordance where HEST's separation is
  greater than 0.005, plus per-task Spearman ρ.
* **C — drift guard:** append-only commit and slide2vec provenance for Soma-to-Soma
  comparisons. This is the only comparison suitable for regression gating.

**A — per-cell agreement (published, not gated)**

.. list-table::
   :header-rows: 1

   * - Task
     - Encoder
     - soma
     - HEST
     - Δ
     - Δ %
     - Recorded
   * - PAAD
     - ``uni2``
     - 0.5007
     - 0.5001
     - +0.0006
     - +0.12%
     - 2026-07-10 @ ``e9fb89c``
   * - PAAD
     - ``virchow2``
     - 0.4769
     - 0.4779
     - -0.0010
     - -0.21%
     - 2026-07-10 @ ``e9fb89c``
   * - PAAD
     - ``h-optimus-1``
     - 0.4916
     - 0.4964
     - -0.0048
     - -0.97%
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``uni2``
     - 0.3105
     - 0.3015
     - +0.0090
     - +2.99%
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``virchow2``
     - 0.2615
     - 0.2581
     - +0.0034
     - +1.32%
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``h-optimus-1``
     - 0.3190
     - 0.3195
     - -0.0005
     - -0.16%
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``uni2``
     - 0.5593
     - 0.5587
     - +0.0006
     - +0.11%
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``virchow2``
     - 0.5520
     - 0.5685
     - -0.0165
     - -2.90%
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``h-optimus-1``
     - 0.5768
     - 0.5779
     - -0.0011
     - -0.19%
     - 2026-07-10 @ ``e9fb89c``

Across 9 cell(s) the parity gap is a median **0.21%** relative, worst **2.99%** (COAD/``uni2``). Stated, not gated: see ADR 0005.

**B — rank concordance (bonus)**

**Pooled pairwise rank concordance: 7/8 (88%)** on resolvable pairs (HEST separates them by more than 0.005); 1 within-noise pair(s) excluded.
Over *all* pairs (resolvable + within-noise): 8/9 (89%).

Resolvable pairs soma orders *differently* from HEST (reported, not gated):

* LUNG: HEST ``virchow2`` > ``uni2`` (Δref +0.0098) but soma reverses it (Δsoma -0.0073)

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Task
     - Spearman ρ (soma vs HEST)
   * - PAAD
     - +1.000
   * - COAD
     - +1.000
   * - LUNG
     - +0.500

**C — drift guard**

Recorded at soma commit(s) ``e9fb89c``, slide2vec 5.3.0. The ledger (``soma/benchmarks/results/hest.csv``) is append-only, so re-running a cell at a new commit adds a row — drift never overwrites history.

See :doc:`benchmarking` for reference semantics, :doc:`curation` for HEST input
contracts, and :doc:`regression` for the probe and metric.
