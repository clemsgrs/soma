HEST
====

*Maps to task:* :doc:`regression` — a **frozen** patch encoder scored on
**gene-expression-from-morphology**: predict a 50-gene expression vector from a
112 µm tile, reproducing the
`HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al., NeurIPS 2024).

.. note::

   This page is generated from the registered benchmark definition — the protocol
   summary and reference numbers from the ``Benchmark`` object's ``expected()`` rows
   (packaged ``soma/benchmarks/reference/hest.csv``), and the command from the benchmark name. Edit the registry
   (``soma/benchmarks/hest.py``) and the CSV, not this page; ``python docs/_generate_reference.py``
   re-emits it and ``tests/test_docs.py`` guards the two from drifting.

HEST is registered as **one sub-benchmark per task** (``hest/<task>``), each sharing
the same closed-form spatial-expression probe recipe and varying only the ``encoder``
axis. soma reproduces it **natively** — its own slide2vec encoder → its per-spot
feature cache → a closed-form Ridge+PCA probe — with **no dependency on the** ``hest``
**library or TRIDENT**. The vertical slice lands ``hest/IDC``; the eight remaining
tasks follow by data-provisioning plus a registration line (see *Adding a HEST task*
below).

Protocol
--------

Stated once, shared by every task; the ``encoder`` axis is the only variable:

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

Encoders
--------

The ``encoder`` axis maps a soma encoder onto a HEST leaderboard backbone. Any
slide2vec-registered encoder works (slide2vec validates the name); the variant is
pinned only where the leaderboard used a non-default one:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Encoder
     - HEST backbone
   * - ``uni2`` (default)
     - HEST-Benchmark UNI2-h; slide2vec default output
   * - ``virchow2``
     - HEST-Benchmark Virchow2; slide2vec ``cls`` output (CLS-only, 1280-d)

Tasks
-----

The registered sub-benchmark family (only ``hest/IDC`` now — the vertical slice):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Benchmark
     - HEST task
   * - ``hest/IDC``
     - ``IDC``

The eight remaining HEST-Benchmark tasks — ``PRAD``, ``PAAD``, ``SKCM``,
``COAD``, ``READ``, ``CCRCC``, ``LUNG``, ``LYMPH_IDC`` — are provisioned in fan-out
(*Adding a HEST task* below); the curator and probe already handle them.

Published leaderboard
---------------------

HEST's published **external, non-gating** Ridge+PCA Pearson on the IDC task, per
encoder (best first). There is **no gate row**: nothing is tolerance-checked.
``soma reproduce hest/IDC`` renders soma's Measured row *beside* these, making the
slide2vec↔TRIDENT extraction gap an explicit, non-gating delta. Source: `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Encoder
     - Published ``pearson``
   * - ``h-optimus-1``
     - 0.6024
   * - ``h-optimus-0``
     - 0.5976
   * - ``virchow2``
     - 0.5971
   * - ``uni2``
     - 0.5898
   * - ``uni``
     - 0.5890
   * - ``genbio-pathfm``
     - 0.5872
   * - ``h0-mini``
     - 0.5862
   * - ``virchow``
     - 0.5846
   * - ``midnight``
     - 0.5823
   * - ``hibou-l``
     - 0.5701
   * - ``gpfm``
     - 0.5660
   * - ``gigapath``
     - 0.5515
   * - ``lunit``
     - 0.5449
   * - ``conchv15``
     - 0.5440
   * - ``phikonv2``
     - 0.5408
   * - ``conch``
     - 0.5363
   * - ``phikon``
     - 0.5327
   * - ``musk``
     - 0.5248

Reproduced numbers
------------------

What soma has actually measured, recorded by ``soma reproduce --record`` into
``soma/benchmarks/results/hest.csv``. HEST's references are external, so the
Reference / Δ columns stay blank — compare against the published leaderboard
above:

No reproductions have been recorded yet. Run ``soma reproduce <name> --record`` to append a measured number + provenance to the results ledger.

Download one task
-----------------

The curator is hermetic and offline (ADR 0004): provision the raw task tree once,
out of band. Pull **only the needed task** and **exclude the** ``fm_v1/``
**precomputed foundation-model features** (soma re-extracts them natively via
slide2vec) — a few-GB task subtree, never the full multi-task / >1 TB HEST corpus::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

The scoped ``--include 'IDC/*'`` pulls just that task's ``patches/``, ``adata/``,
``splits/`` and ``var_50genes.json``; ``--exclude 'fm_v1/*'`` drops the precomputed
features. ``curate_hest`` then runs fully offline over the result.

Reproduce
---------

``soma reproduce`` curates the raw task tree, fits the closed-form probe over the
canonical seed, reads ``test/mean_pearson_mean`` from ``summary.json``, and
renders it beside the external reference::

    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC

Pick the encoder axis with ``--encoder`` (default ``uni2``; e.g. ``--encoder virchow2``).

Adding a HEST task
------------------

Fanning out to another task is **data + one registration line + reference rows** —
never new machinery. ``curate_hest`` and the closed-form probe are task-agnostic, so
adding a task **never touches the curator or the probe**:

**1. Download the task** (scoped; swap ``IDC`` for e.g. ``PRAD``)::

    hf download MahmoodLab/hest-bench --include 'PRAD/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

**2. Curate** it into a ``spatial_expression`` Manifest with the *same* curator::

    python -m soma.curation.hest --raw-root /path/to/hest-bench/PRAD \
        --output-dir /path/to/curated/PRAD --task PRAD

**3. Register** the sub-benchmark with a single line in ``soma/benchmarks/hest.py`` —
instantiate the *existing* class, no new curator/probe code::

    register_benchmark(HestBenchmark("PRAD"))

**4. Add external reference rows** for the task to ``soma/benchmarks/reference/hest.csv``
— one ``kind=external`` row per encoder (the published Pearson, a ``label``, a ``url``).

Then ``python docs/_generate_reference.py`` re-emits this page with the new task,
``soma list benchmarks`` shows ``hest/PRAD``, and ``soma reproduce hest/PRAD`` runs —
all from the same curator and the same probe.

.. seealso::

   * :doc:`regression` — the task family, the ``pearson`` metric, and the closed-form
     Ridge+PCA probe this benchmark drives.
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the HEST curator (``curate_hest``) and its split policy.
