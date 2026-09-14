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

Select another compatible encoder with ``--encoder``, compare several with
``--encoders``, or rescore an existing run with ``--from-run-dir <dir>``.

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

``build_config`` also resolves a committed protocol per ``(encoder, spacing)``:
the 2×2 magnification-alignment ablation plus the native anchor. Use these for
a custom spacing sweep compared on a :doc:`leaderboard <benchmarking>`:

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

Encoder results
---------------

Recorded test ``mean_f1`` scores use the protocol above at 0.2 µm/px. The
reference band is soma's frozen-probe Virchow2 result at seed 0; a value
outside its band is shown in red.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Reference band (expected ± tolerance)
   * - ``mean_f1``
     - 0.700 ± 0.020

.. list-table::
   :header-rows: 1

   * - Encoder
     - Metric
     - soma (mean ± std)
     - Seeds
     - Reference
   * - genbio-pathfm
     - ``mean_f1``
     - 0.733 ± 0.004
     - 3
     - —
   * - h-optimus-1
     - ``mean_f1``
     - 0.720 ± 0.002
     - 3
     - —
   * - h0-mini
     - ``mean_f1``
     - 0.719 ± 0.003
     - 3
     - —
   * - conchv15
     - ``mean_f1``
     - 0.711 ± 0.001
     - 3
     - —
   * - virchow2
     - ``mean_f1``
     - 0.703
     - 1
     - 0.700 ± 0.020
   * - midnight
     - ``mean_f1``
     - 0.658 ± 0.002
     - 3
     - —
   * - dinov2-vitb14
     - ``mean_f1``
     - 0.656 ± 0.002
     - 3
     - —

The anchor was measured with soma 1.5.1, slide2vec 5.1.1, and torch 2.7.1+cu128 on one NVIDIA GeForce RTX 2080 Ti.

These results accompany Grisi *et al.*, *Benchmarking foundation models for
cell detection* (in preparation).

Guidance anchors
----------------

These snapshots come from fully supervised, end-to-end methods with a
trainable encoder. They give context for the frozen probe and never gate
``soma reproduce``:

* `OCELOT official baseline (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.70
* `best reported (fully-supervised end-to-end) <https://wearewaiv.github.io/histoboard/>`__ — ``mean_f1`` ≈ 0.73

References
----------

* Ryu et al., *OCELOT: Overlapped Cell on Tissue Dataset for Histopathology*,
  CVPR 2023.
* *CellRegNet*, 2024 — a fully supervised point-detection baseline.
* `arXiv:2503.05678 <https://arxiv.org/abs/2503.05678>`__ — P2PNet-style
  point detection on frozen foundation-model features.

.. seealso::

   * :doc:`detection` — head, target encoding, loss, and F1@δ evaluator.
   * :doc:`benchmarking` — the shared benchmark workflow.
   * :doc:`curation` — the OCELOT curator and split policy.
