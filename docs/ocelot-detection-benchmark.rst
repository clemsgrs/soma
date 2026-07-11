OCELOT
======

Purpose
-------

Evaluate Soma's :doc:`cell-detection path <detection>` on the
`OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ TCGA patches. A frozen
encoder produces a dense token grid; ``lightweight_conv`` predicts class peak
heatmaps. OCELOT's greedy matcher reports class-aware **mean F1 @ δ = 3 µm**.
Thresholds are selected on ``tune`` and applied once to ``test``.

Protocol
--------

The fixed recipe varies ``encoder`` × ``spacing``:

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

Available committed configurations:

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

Prepare and run
---------------

Curate, train the canonical seed, score, and check the gate::

    soma reproduce ocelot --raw-root /path/to/ocelot

Use ``--encoder`` and ``--spacing`` for an ablation, ``--seeds 1`` for a smoke
test, or ``--from-run-dir <dir>`` to rescore an existing run.

Results
-------

This **gate reference** is Soma's Virchow2 @ 0.2 µm/px, seed-0 frozen-probe
regression anchor. It is not an external leaderboard result:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Expected ± tolerance
   * - ``mean_f1``
     - 0.6995 ± 0.020

Guidance anchors (non-gating)
-----------------------------

These snapshotted `histoboard <https://wearewaiv.github.io/histoboard/>`__
values come from fully supervised, end-to-end systems, not Soma's frozen probe.
They provide context only; ``soma reproduce`` never gates on them:

* `OCELOT official baseline (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.70 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (low end / official challenge baseline). A different protocol from soma's frozen probe (end-to-end supervised, encoder not frozen, not tied to any encoder), so non-gating guidance. Snapshotted from histoboard 2026-07-03.
* `best reported (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.73 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (high end / best reported SOTA). A different protocol from soma's frozen probe, so non-gating guidance. Snapshotted from histoboard 2026-07-03.

Reference environment
~~~~~~~~~~~~~~~~~~~~~

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

See :doc:`benchmarking` for reference semantics, :doc:`curation` for the raw-data
layout, and :doc:`detection` for targets, loss, and matching.
