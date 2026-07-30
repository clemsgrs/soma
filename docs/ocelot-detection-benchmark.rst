OCELOT
======

*Maps to task:* :doc:`detection` — soma's :doc:`detection path <detection>`
reproduced on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_
cell-detection challenge.

.. note::

   This page is generated from the registered benchmark definition — the protocol
   summary and reference numbers from the ``Benchmark`` object's ``expected()`` rows
   (packaged ``soma/benchmarks/reference/ocelot.csv``), and the command from the benchmark name. Edit the registry
   (``soma/benchmarks/ocelot.py``) and the CSV, not this page; ``python docs/_generate_reference.py``
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

The tolerance band ``soma reproduce`` uses to highlight potential drift — a
**config-agnostic** banner
(soma's own frozen-probe Virchow2 @ 0.2 µm/px seed-0 headline, used as a regression
anchor, not an external leaderboard number). The non-gating external anchors —
fully-supervised end-to-end baselines from a *different* protocol — are surfaced
with clickable links under *Guidance anchors* below:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Reference band (expected ± tolerance)
   * - ``mean_f1``
     - 0.6995 ± 0.020

Encoder results
---------------

Frozen-probe ``mean_f1`` on OCELOT test across foundation-model encoders — each a
**frozen** encoder feeding the same ``lightweight_conv`` decoder at 0.2 µm/px, with
per-class score thresholds swept on ``tune`` and applied once to ``test``, averaged
over 3 seeds. ``Δ`` is against the Virchow2 anchor band above.

.. list-table::
   :header-rows: 1

   * - Encoder
     - soma (mean ± std)
     - Seeds
     - Reference
     - Δ
     - Recorded (date @ commit)
   * - genbio-pathfm
     - 0.733 ± 0.004
     - 3
     - 0.700
     - +0.034
     - 2026-07-15 @ ``54601e4``
   * - h-optimus-1
     - 0.720 ± 0.002
     - 3
     - 0.700
     - +0.021
     - 2026-07-15 @ ``54601e4``
   * - h0-mini
     - 0.719 ± 0.003
     - 3
     - 0.700
     - +0.019
     - 2026-07-15 @ ``54601e4``
   * - conchv15
     - 0.711 ± 0.001
     - 3
     - 0.700
     - +0.011
     - 2026-07-15 @ ``54601e4``
   * - virchow2
     - 0.699 ± 0.004
     - 3
     - 0.700
     - -0.001
     - 2026-07-15 @ ``54601e4``
   * - midnight
     - 0.658 ± 0.002
     - 3
     - 0.700
     - -0.042
     - 2026-07-15 @ ``54601e4``
   * - dinov2-vitb14
     - 0.656 ± 0.002
     - 3
     - 0.700
     - -0.043
     - 2026-07-15 @ ``54601e4``
   * - virchow2
     - 0.700
     - 1
     - 0.700
     - +0.001
     - 2026-07-30 @ ``16d9ddb``

These frozen-probe encoder results accompany an upcoming publication — Grisi
*et al.*, *Benchmarking foundation models for cell detection* (in preparation, 2026;
provisional citation).

Guidance anchors (non-gating)
-----------------------------

External reference points shown for **context only** — the official challenge
baseline and best-reported numbers, snapshotted (not live-scraped) from
`histoboard <https://wearewaiv.github.io/histoboard/>`__. They measure a
*different* protocol than soma's frozen probe (fully-supervised, end-to-end, not
tied to any encoder), so ``soma reproduce`` **never gates** on them; they only show
how far the frozen-probe result stands from the best reported result:

* `OCELOT official baseline (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.70 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (low end / official challenge baseline). A different protocol from soma's frozen probe (end-to-end supervised, encoder not frozen, not tied to any encoder), so non-gating guidance. Snapshotted from histoboard 2026-07-03.
* `best reported (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.73 — Top fully-trained OCELOT cell-detection methods land ~0.70-0.73 mF1 (high end / best reported SOTA). A different protocol from soma's frozen probe, so non-gating guidance. Snapshotted from histoboard 2026-07-03.

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
greedy-scores it, and reports ``mean_f1`` beside the band above::

    soma reproduce ocelot --raw-root /path/to/ocelot

Fast paths: ``--from-run-dir <dir>`` re-scores an existing run with the greedy
matcher (no training); ``--seeds 1`` is the quickest smoke. Compare encoders with
``--encoder`` (e.g. ``soma reproduce ocelot --encoder uni2 --raw-root ...``); to
compare spacings, run per-spacing configs and a :doc:`leaderboard <benchmarking>`.

.. seealso::

   * :doc:`detection` — the detection modeling substrate (head, target encoding,
     loss, F1@δ evaluator).
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the OCELOT curator and split policy.
