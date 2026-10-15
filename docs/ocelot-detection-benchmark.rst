OCELOT
======

Cell detection on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_
challenge data with a frozen encoder. soma's :doc:`detection` path trains a small
decoder on the encoder's dense features and scores the detected cells with the
challenge's own metric, so encoders can be compared on one fixed protocol.

Data and task
-------------

OCELOT (Ryu *et al.*, CVPR 2023) pairs cell patches with tissue patches cropped from
TCGA whole-slide images. The benchmark uses the cell patches only: 1024 × 1024
pixel H&E crops at 0.2 µm/px, each with a point per cell.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Task
     - Detect every cell as a point and classify it as a tumour cell or a
       background (non-tumour) cell.
   * - Splits
     - The official train, validation and test split, as one fold. Validation is
       soma's ``tune`` split, where the score thresholds are chosen.
   * - Metric
     - ``mean_f1``: F1 at a matching distance of 3 µm, averaged over the two
       classes. Matching is greedy, as in the official evaluation.

Protocol
--------

The recipe is fixed and only the encoder varies.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Input
     - The cell patches at their native 0.2 µm/px, tiled for the encoder.
   * - Encoder
     - Any frozen tile encoder with dense outputs; the ``encoder`` axis of the
       benchmark.
   * - Decoder
     - ``lightweight_conv`` on the frozen dense token grid, predicting one peak
       heatmap per class.
   * - Target and loss
     - A peak Gaussian at every annotated cell and a foreground-weighted MSE; see
       :doc:`detection`.
   * - Thresholds
     - One score threshold per class, chosen on ``tune`` and applied once to ``test``.
   * - Seeds
     - ``0``. The reference below is a seed-0 value, so the default run is seed 0.

The committed configs also hold a spacing ablation: ``uni2`` at 0.25 and 0.5 µm/px and
``virchow2`` at 0.2, 0.25 and 0.5 µm/px. ``soma reproduce`` keeps spacing at 0.2 µm/px;
the other configs serve a custom sweep ranked with ``soma leaderboard``.

Results
-------

soma's ``mean_f1`` on the OCELOT test split, three seeds per encoder, best first.

.. include:: _generated/ocelot-results.rst

Fully supervised, end-to-end methods with a trained encoder reach a mean F1 of
about 0.70 (the `official baseline
<https://wearewaiv.github.io/histoboard/>`__) to 0.73 (the `best reported
<https://wearewaiv.github.io/histoboard/>`__); they frame the frozen probe and never
gate ``soma reproduce``.

These results accompany Grisi *et al.*, *Benchmarking foundation models for cell
detection* (in preparation).

Reproduce
---------

**Prepare the data.** Download ``ocelot2023_v1.0.1.zip`` from `Zenodo record 8417503
<https://zenodo.org/records/8417503>`__ and unzip it. The benchmark's curator reads the
cell patches and their point files from that directory and writes soma's detection
manifests; :doc:`curation` describes what it writes.

**Run the benchmark** with its default encoder, Virchow2, or name another installed
encoder::

    soma reproduce ocelot --raw-root /path/to/ocelot2023_v1.0.1
    soma reproduce ocelot --encoder uni2 --raw-root /path/to/ocelot2023_v1.0.1

**Compare encoders** in one command, which also writes a leaderboard, or rescore a
finished run from its output root (by default ``soma_reproduce/ocelot/seed_<k>``)::

    soma reproduce ocelot --encoders virchow2 uni2 h-optimus-1 --raw-root /path/to/ocelot2023_v1.0.1
    soma reproduce ocelot --from-run-dir soma_reproduce/ocelot/seed_0

``--record`` appends the measured score to the results ledger that the table above
reads. :doc:`benchmarking` describes the options shared by every benchmark.

References
----------

* Ryu *et al.*, *OCELOT: Overlapped Cell on Tissue Dataset for Histopathology*,
  CVPR 2023.
* *CellRegNet*, 2024: a fully supervised point-detection baseline.
* `arXiv:2503.05678 <https://arxiv.org/abs/2503.05678>`__: P2PNet-style point
  detection on frozen foundation-model features.
