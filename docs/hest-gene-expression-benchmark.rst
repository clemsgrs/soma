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
**library or TRIDENT**. All **9 HEST-Benchmark tasks** are registered (see *Tasks*);
reproduction soundness is proven by **rank agreement** across them, not by matching the
extraction stack (see *Reproduction — is it sound?*).

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
     - HEST-Benchmark UNI2-h; slide2vec default output (CLS, 1536-d)
   * - ``virchow2``
     - HEST-Benchmark Virchow2; slide2vec ``cls`` output (CLS-only, 1280-d)
   * - ``h-optimus-1``
     - HEST-Benchmark H-Optimus-1; slide2vec default output (CLS, 1536-d)

Tasks
-----

The registered sub-benchmark family — all 9 HEST-Benchmark tasks, spanning organs
(breast, prostate, pancreas, colon, rectum, kidney, lung, skin):

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

Each shares the *same* curator and closed-form probe; a task is data + a
registration line + reference rows (*Adding a HEST task* below). The hest-bench HF
dataset also ships an ``HCC`` (liver) tree, but HCC is **not** one of the 9 scored
tasks (no published leaderboard number), so it is deliberately not registered.

Published leaderboard (IDC)
---------------------------

HEST's published **external, non-gating** Ridge+PCA Pearson on the IDC task, per
encoder (best first). There is **no gate row**: nothing is tolerance-checked.
``soma reproduce hest/IDC`` renders soma's Measured row *beside* these, making the
slide2vec↔TRIDENT extraction gap an explicit, non-gating delta. The other 8 tasks'
references (our reproduction encoders × task) drive the reproduction proof below.
Source: `HEST-Benchmark leaderboard (mahmoodlab/HEST) <https://github.com/mahmoodlab/HEST#hest-benchmark>`__.

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

Reproduction — is it sound?
---------------------------

soma reproduces HEST **natively** — its own slide2vec features, not HEST's TRIDENT
extraction — so the proof of soundness is **not** that the numbers match to the
decimal (the extraction stacks differ), but that soma re-derives HEST's **ranking**
of encoders. Three views, computed from the results ledger joined to the published
reference:

* **A — absolute agreement** (per cell): soma's Pearson beside HEST's, and the signed
  delta. Shown, never gated — the delta is the accepted slide2vec↔TRIDENT parity gap.
* **B — rank agreement** (the headline): **pooled pairwise concordance** — over every
  (task, encoder-pair), the fraction soma orders the same way HEST does. A pair is
  *resolvable* when HEST separates it by more than 0.005 on the metric; the
  headline is concordance over resolvable pairs, so soma is not graded on within-noise
  coin-flips. Per-task Spearman ρ is shown alongside (coarse at few encoders).
* **C — drift guard**: the ledger is append-only and provenance-pinned (commit,
  slide2vec version), so a re-run at a new commit adds a row and drift is a visible diff.

**A — per-cell agreement**

.. list-table::
   :header-rows: 1

   * - Task
     - Encoder
     - soma
     - HEST
     - Δ
     - Recorded
   * - PAAD
     - ``uni2``
     - 0.5007
     - 0.5001
     - +0.0006
     - 2026-07-10 @ ``e9fb89c``
   * - PAAD
     - ``virchow2``
     - 0.4769
     - 0.4779
     - -0.0010
     - 2026-07-10 @ ``e9fb89c``
   * - PAAD
     - ``h-optimus-1``
     - 0.4916
     - 0.4964
     - -0.0048
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``uni2``
     - 0.3105
     - 0.3015
     - +0.0090
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``virchow2``
     - 0.2615
     - 0.2581
     - +0.0034
     - 2026-07-10 @ ``e9fb89c``
   * - COAD
     - ``h-optimus-1``
     - 0.3190
     - 0.3195
     - -0.0005
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``uni2``
     - 0.5593
     - 0.5587
     - +0.0006
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``virchow2``
     - 0.5520
     - 0.5685
     - -0.0165
     - 2026-07-10 @ ``e9fb89c``
   * - LUNG
     - ``h-optimus-1``
     - 0.5768
     - 0.5779
     - -0.0011
     - 2026-07-10 @ ``e9fb89c``

**B — rank concordance (headline)**

**Pooled pairwise rank concordance: 7/8 (88%)** on resolvable pairs (HEST separates them by more than 0.005); 1 within-noise pair(s) excluded.
Over *all* pairs (resolvable + within-noise): 8/9 (89%).

Resolvable pairs soma orders *differently* from HEST (honest failures):

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

    soma reproduce hest/CCRCC --raw-root /path/to/hest-bench/CCRCC
    soma reproduce hest/COAD --raw-root /path/to/hest-bench/COAD
    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC
    soma reproduce hest/LUNG --raw-root /path/to/hest-bench/LUNG
    soma reproduce hest/LYMPH_IDC --raw-root /path/to/hest-bench/LYMPH_IDC
    soma reproduce hest/PAAD --raw-root /path/to/hest-bench/PAAD
    soma reproduce hest/PRAD --raw-root /path/to/hest-bench/PRAD
    soma reproduce hest/READ --raw-root /path/to/hest-bench/READ
    soma reproduce hest/SKCM --raw-root /path/to/hest-bench/SKCM

Pick the encoder axis with ``--encoder`` (default ``uni2``; e.g. ``--encoder virchow2``).

Adding a HEST task
------------------

All 9 scored tasks are already registered. The one hest-bench task *not* registered
is ``HCC`` (liver): the HF hub ships its data tree, but HCC is **unscored** — no
published leaderboard number — so it carries no reference row. Adding it (or any future
task) is a **fan-out**: **data + one ``HEST_TASKS`` entry + reference rows** — never new
machinery. ``curate_hest`` and the closed-form probe are task-agnostic, so a new task
**never touches the curator or the probe**:

**1. Download the task** (scoped; e.g. ``HCC``)::

    hf download MahmoodLab/hest-bench --include 'HCC/*' --exclude 'fm_v1/*' \
        --repo-type dataset --local-dir /path/to/hest-bench

**2. Curate** it into a ``spatial_expression`` Manifest with the *same* curator::

    python -m soma.curation.hest --raw-root /path/to/hest-bench/HCC \
        --output-dir /path/to/curated/HCC --task HCC

**3. Register** it by adding the task id to ``HEST_TASKS`` in ``soma/benchmarks/hest.py``
— the module loop-registers ``HestBenchmark(task)`` for each, no new curator/probe code::

    HEST_TASKS = (..., "HCC")  # loop-registers hest/HCC

**4. Add external reference rows** for the task to ``soma/benchmarks/reference/hest.csv``
— one ``kind=external`` row per encoder (the published Pearson, a ``label``, a ``url``).
Without a published number a task can still run, but it has nothing to reproduce against.

Then ``python docs/_generate_reference.py`` re-emits this page with the new task,
``soma list benchmarks`` shows ``hest/HCC``, and ``soma reproduce hest/HCC`` runs —
all from the same curator and the same probe.

.. seealso::

   * :doc:`regression` — the task family, the ``pearson`` metric, and the closed-form
     Ridge+PCA probe this benchmark drives.
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the HEST curator (``curate_hest``) and its split policy.
