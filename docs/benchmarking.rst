Benchmarking
============

``soma`` ships **foundation-model benchmarks as first-class, registered code**. A
benchmark is not a folder of scripts you copy — it is a named entry in a registry
inside the wheel that wires an existing curator, a committed config (or a config
builder), a packaged reference table, and a scorer together behind one uniform
interface. Because the protocol is code, ``soma list benchmarks`` enumerates it,
``soma reproduce`` drives it, and the published numbers cannot silently drift from a
doc someone forgot to update — every :doc:`per-benchmark page <ocelot-detection-benchmark>`
renders its reference table straight from the same CSV the tolerance check reads.

This page is the conceptual guide: the shared **curate → configure → run →
leaderboard → reproduce** loop that every registered benchmark follows. The
:doc:`cli` page documents the exact command surface.

What a registered benchmark is
------------------------------

Each benchmark exposes a small, structural interface:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Piece
     - What it does
   * - ``name``
     - Registry key at per-dataset granularity (``ocelot``, ``eva/bach``).
   * - ``facet``
     - The canonical *fixed* recipe backbone vs the *varied* axes (what a
       leaderboard facets on).
   * - ``primary_metric``
     - The metric the tolerance band is defined on.
   * - ``canonical_seeds``
     - The seeds ``reproduce`` runs by default, so the check is like-for-like
       against the published number.
   * - ``curate(...)``
     - Turns a raw public dataset into soma ``dataset.csv`` / ``splits.csv``
       manifests (delegates to a :doc:`curator <curation>`).
   * - ``build_config(**axes)``
     - Emits the benchmark-faithful :class:`~soma.config.PipelineConfig` for the
       chosen axes.
   * - ``expected(**axes)``
     - The reference row(s) — from the packaged ``reference/<name>.csv`` — for
       those axes.
   * - ``score(run_dir)``
     - Reads the run's headline metric (the default reads ``summary.json``;
       OCELOT overrides it with its greedy matcher).

The five steps
--------------

**1. Curate.** Turn a raw public layout into soma manifests. The benchmark
delegates to a curator, so the output is the ordinary ``dataset.csv`` /
``splits.csv`` pair described in :doc:`dataset` — soma never invents splits, it
preserves the benchmark's own. ``soma reproduce`` runs this step for you from
``--raw-root``; you can also call the curator directly (see :doc:`curation`).

**2. Configure.** ``build_config`` assembles the protocol-faithful config for the
requested axes (encoder, spacing, …). This is the recipe that reproduces the
published number — the training hyper-parameters, the frozen-encoder settings, the
task head, and the metric are all fixed by the benchmark, not by you.

**3. Run.** A config runs like any other soma experiment — ``soma path/to/config.yaml``
(or ``python -m soma path/to/config.yaml``). Each run writes a self-contained,
:doc:`self-describing bundle <outputs>` under ``output_root``: the resolved
``config.yaml``, per-fold ``metrics.json``, and a run-level ``summary.json`` whose
keys are split-prefixed (``test/balanced_accuracy``). ``soma reproduce`` drives this
step for every canonical seed.

**4. Leaderboard.** ``soma leaderboard`` renders a faceted view over the completed
run dirs under an output root — no re-training, it reads what the runs already
wrote. A benchmark name supplies the canonical facet and reference band; ``--vary``
/ ``--fix`` / ``--like`` shape the facet on top of it. The leaderboard is a *view*,
so the same run dirs support many faceted tables.

**5. Reproduce.** ``soma reproduce <name>`` runs steps 1–4 end to end and
**tolerance-checks** the primary metric against the packaged reference band,
printing ``PASS`` / ``FAIL`` with the delta::

    soma reproduce ocelot --raw-root /path/to/ocelot
    soma reproduce eva/bach --raw-root /path/to/eva/bach

``--from-run-dir <dir>`` re-scores an existing run without re-training; ``--seeds 1``
is the quickest smoke; a family prefix (``soma reproduce eva``) fans out over every
member. Because the band lives in ``reference/<name>.csv`` and the check reads it
directly, a green ``reproduce`` is evidence the environment matches, not that a
number was typed correctly.

Registered benchmarks
---------------------

List what is registered in your install::

    soma list benchmarks

The bundled benchmarks each have a generated page with the protocol summary, the
reference table (verbatim from its CSV), and the exact ``soma reproduce`` command:

* :doc:`OCELOT <ocelot-detection-benchmark>` — the :doc:`detection` path on the
  OCELOT 2023 cell-detection challenge, with the encoder × spacing ablation.
* :doc:`EVA <eva-patch-classification-benchmark>` — the :doc:`classification` path
  on the kaiko-ai/eva patch suite (``eva/<dataset>``), varying the encoder.
* :doc:`HEST <hest-gene-expression-benchmark>` — the :doc:`regression` path on the
  HEST-Benchmark gene-expression tasks (``hest/<task>``), a closed-form Ridge+PCA
  probe, varying the encoder.

.. seealso::

   * :doc:`cli` — the exact ``reproduce`` / ``leaderboard`` / ``list benchmarks``
     command surface.
   * :doc:`curation` — the curators benchmarks delegate to.
   * :doc:`outputs` — the self-describing run bundle a leaderboard reads.
