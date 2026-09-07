OCELOT
======

Evaluate frozen encoders on the `OCELOT 2023
<https://ocelot2023.grand-challenge.org/>`_ cell-detection challenge. The data
contains paired cell and tissue patches from TCGA; this benchmark uses only
the cell patches.

A frozen encoder produces a dense token grid and a ``lightweight_conv``
decoder predicts per-class peak heatmaps. The score is class-aware
**mean F1 @ δ = 3 µm**, using greedy matching. Per-class score thresholds
are selected on ``tune``, frozen, and applied once to ``test``. See
:doc:`detection` for matching and pixel-to-micrometer definitions.

Run the benchmark
-----------------

Prepare the raw data as described in :doc:`curation`, then run the default
Virchow2 encoder at 0.2 µm/px with canonical seed 0::

    soma reproduce ocelot --raw-root /path/to/ocelot

The command curates, trains, scores, and compares the result with its
packaged reference. Select another compatible encoder with ``--encoder``
or compare several with ``--encoders``. Use ``--from-run-dir <dir>`` to
rescore an existing run without training.

Protocol
--------

The recipe backbone is held fixed; ``soma reproduce`` varies only the ``encoder``
and fixes image spacing at the anchor.

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
     - ``encoder``
   * - primary metric
     - ``mean_f1``
   * - canonical seeds
     - ``0``
   * - anchor
     - ``virchow2`` @ 0.2 µm/px (seed 0)

Packaged spacing protocols
--------------------------

Reproduce fixes spacing at the anchor, but ``build_config`` still resolves a
committed protocol per ``(encoder, spacing)`` — the 2×2 magnification-alignment
ablation plus the native anchor. Use these for a custom spacing sweep compared on a
:doc:`leaderboard <benchmarking>`, like any other non-encoder axis:

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

Reference band
--------------

The packaged reference is soma's frozen-probe Virchow2 result at
0.2 µm/px, seed 0. ``soma reproduce`` uses its tolerance to highlight drift
for that encoder; the comparison is informational. External baselines
appear separately under *Guidance anchors* below.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Reference band (expected ± tolerance)
   * - ``mean_f1``
     - 0.6995 ± 0.020

Encoder results
---------------

Recorded test ``mean_f1`` scores use the protocol above at 0.2 µm/px.
``Seeds`` counts the runs aggregated in each entry; ``Δ`` is shown only
when a packaged reference matches the encoder.

.. list-table::
   :header-rows: 1

   * - Encoder
     - Metric
     - soma (mean ± std)
     - Seeds
     - Reference
     - Δ
     - Recorded (date @ commit)
   * - genbio-pathfm
     - ``mean_f1``
     - 0.733 ± 0.004
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``
   * - h-optimus-1
     - ``mean_f1``
     - 0.720 ± 0.002
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``
   * - h0-mini
     - ``mean_f1``
     - 0.719 ± 0.003
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``
   * - conchv15
     - ``mean_f1``
     - 0.711 ± 0.001
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``
   * - virchow2
     - ``mean_f1``
     - 0.703
     - 1
     - 0.700
     - +0.003
     - 2026-08-10 @ ``ba8e529``
   * - midnight
     - ``mean_f1``
     - 0.658 ± 0.002
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``
   * - dinov2-vitb14
     - ``mean_f1``
     - 0.656 ± 0.002
     - 3
     - —
     - —
     - 2026-07-15 @ ``54601e4``

These frozen-probe encoder results accompany an upcoming publication — Grisi
*et al.*, *Benchmarking foundation models for cell detection* (in preparation, 2026;
provisional citation).

Guidance anchors (non-gating)
-----------------------------

These packaged snapshots describe fully supervised, end-to-end methods.
They provide context for the frozen probe and never determine command
success:

* `OCELOT official baseline (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.70 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (low end / official challenge baseline). A different protocol from soma's frozen probe (end-to-end supervised, encoder not frozen, not tied to any encoder), so non-gating guidance. Snapshotted from histoboard 2026-07-03.
* `best reported (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.73 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (high end / best reported SOTA). A different protocol from soma's frozen probe, so non-gating guidance. Snapshotted from histoboard 2026-07-03.

Reference environment
---------------------

The anchor reference was measured in this environment:

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

.. seealso::

   * :doc:`detection` — the detection modeling substrate (head, target encoding,
     loss, F1@δ evaluator).
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the OCELOT curator and split policy.

.. note::

   Maintainers: edit ``docs/_generate_reference.py`` for prose, ``soma/benchmarks/ocelot.py``
   for the protocol, and ``soma/benchmarks/reference/ocelot.csv`` for references. Regenerate this page with
   ``python docs/_generate_reference.py``; ``tests/test_docs.py`` checks parity.
