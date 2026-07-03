OCELOT
======

*Maps to task:* :doc:`detection` — soma's :doc:`detection path <detection>`
reproduced on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_
cell-detection challenge.

.. note::

   This page is generated from the registered benchmark definition — the protocol
   summary from the ``Benchmark`` object, the reference table straight from the
   packaged ``soma/benchmarks/reference/ocelot.csv`` (same bytes), and the command from the benchmark name. Edit the
   registry (``soma/benchmarks/ocelot.py``) and the CSV, not this page; ``python docs/_generate_reference.py``
   re-emits it and ``tests/test_docs.py`` guards the two from drifting.

OCELOT 2023 provides paired cell + tissue patches from TCGA. This benchmark is
**cell-only**: a **frozen** foundation-model encoder produces a dense token grid,
a ``lightweight_conv`` decoder regresses a per-class peak heatmap, and the
:class:`~soma.tasks.detection.DetectionHead` scores it with OCELOT's class-aware
**mean F1 @ δ = 3 µm**, greedy-matched — the leaderboard-comparable operating
point (per-class score thresholds swept on ``tune``, frozen, applied once to
``test``). See :doc:`detection` for the canonical matcher and px↔µm definitions.

Protocol
--------

The recipe backbone is held fixed; the facet varies ``encoder`` × ``spacing``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Axis / setting
     - Value
   * - ``task``
     - ``detection``
   * - ``decoder``
     - ``lightweight_conv``
   * - ``matcher``
     - ``greedy_f1@delta=3um``
   * - varied axes
     - ``encoder``, ``spacing``
   * - primary metric
     - ``mean_f1``
   * - canonical seeds
     - ``0``
   * - anchor
     - ``virchow2`` @ 0.2 µm/px (seed 0)

Axes
----

``build_config`` resolves a committed config per ``(encoder, spacing)`` — the
2×2 magnification-alignment ablation plus the native anchor:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Encoder
     - Spacing (µm/px)
   * - ``uni2``
     - 0.25
   * - ``uni2``
     - 0.5
   * - ``virchow2``
     - 0.2
   * - ``virchow2``
     - 0.25
   * - ``virchow2``
     - 0.5

Reference numbers
-----------------

The tolerance band ``soma reproduce`` checks against — read verbatim from the
packaged reference CSV (config-agnostic banner; the ``source`` cell records
provenance and why the tolerance is what it is):

.. csv-table:: ``soma/benchmarks/reference/ocelot.csv``
   :file: ../soma/benchmarks/reference/ocelot.csv
   :header-rows: 1
   :widths: 8 8 8 12 10 10 46

Reference environment
---------------------

The recorded anchor environment the reference number was produced in:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component
     - Version
   * - ``soma``
     - ``1.5.1``
   * - ``slide2vec``
     - ``5.1.1``
   * - ``torch``
     - ``2.7.1+cu128``
   * - ``cuda``
     - ``12.8``
   * - ``gpu``
     - ``NVIDIA GeForce RTX 2080 Ti``

Reproduce
---------

One command curates the raw data, trains the anchor for the canonical seed,
greedy-scores it, and tolerance-checks ``mean_f1`` against the band above::

    soma reproduce ocelot --raw-root /path/to/ocelot

Fast paths: ``--from-run-dir <dir>`` re-scores an existing run with the greedy
matcher (no training); ``--seeds 1`` is the quickest smoke. Sweep the ablation
with ``--encoder`` / ``--spacing`` (e.g. ``soma reproduce ocelot --encoder uni2
--spacing 0.25 --raw-root ...``).

.. seealso::

   * :doc:`detection` — the detection modeling substrate (head, target encoding,
     loss, F1@δ evaluator).
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the OCELOT curator and split policy.
