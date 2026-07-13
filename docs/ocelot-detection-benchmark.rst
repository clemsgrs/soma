OCELOT
======

What this benchmark measures
----------------------------

Evaluate Soma's :doc:`cell-detection path <detection>` on the
`OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ TCGA patches. A frozen
encoder produces a dense token grid; ``lightweight_conv`` predicts class peak
heatmaps. OCELOT's greedy matcher reports class-aware **mean F1 @ δ = 3 µm**.
Thresholds are selected on ``tune`` and applied once to ``test``.

Prepare the data
----------------

Accept the OCELOT terms, download the public release, and unzip
``ocelot2023_v1.0.1``. Pass that directory as ``--raw-root``. Soma uses the
1024×1024 cell patches and point annotations, and preserves OCELOT's
train/validation/test split while preparing its standard manifests.

Run the benchmark
-----------------

Prepare the manifests, train the canonical seed, score, and check the gate::

    soma reproduce ocelot --raw-root /path/to/ocelot

Use ``--seeds 1`` for a smoke test or ``--from-run-dir <dir>`` to rescore an
existing run.

What you can vary
-----------------

Use ``--encoder`` and ``--spacing`` to select one of the registered cells:

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

For context, `fully supervised OCELOT systems on histoboard <https://wearewaiv.github.io/histoboard/>`__ report about 0.70–0.73 mean F1. They use a different,
end-to-end protocol, so this range is non-gating context rather than a Soma target.

Protocol details
----------------

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

See :doc:`benchmarking` for the shared benchmark workflow and :doc:`detection`
for targets, loss, and matching.
