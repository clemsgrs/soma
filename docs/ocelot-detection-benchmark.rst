OCELOT
======

Cell detection on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_
challenge data with a frozen encoder. soma's :doc:`detection` path trains a small
decoder on the encoder's dense features and scores the detected cells with the
challenge's own metric, so encoders can be compared on one fixed protocol.

Data and task
-------------

OCELOT (Ryu *et al.*, CVPR 2023) pairs cell patches with tissue patches cropped from
TCGA whole-slide images of six organs: kidney, endometrium, bladder, prostate,
stomach, and head and neck. The benchmark uses the cell patches only: 1024 × 1024
pixel H&E crops at 0.2 µm/px, each with a point per cell. Each patch is one tile
of about 0.2 × 0.2 mm.

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

.. list-table:: Split sizes
   :header-rows: 1
   :stub-columns: 1

   * - Split
     - Patches
     - Slides
     - Patients
     - Area (mm²)
     - Tumour cells
     - Background cells
   * - train
     - 400
     - 173
     - 132
     - 15.8
     - 42,505
     - 23,332
   * - tune
     - 137
     - 65
     - 46
     - 5.4
     - 16,312
     - 8,400
   * - test
     - 126
     - 65
     - 57
     - 5.0
     - 12,874
     - 9,603
   * - total
     - 663
     - 303
     - 213
     - 26.2
     - 71,691
     - 41,335

The official split keeps each slide in one split but not each patient: 21 patients
have slides in more than one split, so the patient total is lower than the sum of
the splits. soma keeps the official split so scores stay comparable with other
OCELOT results.

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
     - A Gaussian peak (σ = 2 µm) at every annotated cell and an MSE loss with
       foreground weight 10; see :doc:`detection`.
   * - Training
     - AdamW with learning rate 3e-4 and weight decay 1e-4, cosine schedule over 50
       epochs, effective batch 8. Training stops after 10 epochs without a better tune
       ``mean_f1`` and keeps the best epoch. These values were chosen on ``tune``
       across seven encoders.
   * - Thresholds
     - One score threshold per class, chosen on ``tune`` and applied once to ``test``.
   * - Seeds
     - ``0``, ``1`` and ``2``. ``soma reproduce`` checks the mean of the three seeds
       against Virchow2's three-seed mean test score, **0.6985 ± 0.02**.

Results
-------

soma's ``mean_f1`` on the official OCELOT test set (126 patches in v1.0.1), three seeds
per encoder, best first.

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

**Compare encoders** in one command, which also writes a leaderboard::

    soma reproduce ocelot --encoders virchow2 uni2 h-optimus-1 --raw-root /path/to/ocelot2023_v1.0.1

**Rescore a finished run** without training. Point ``--from-run-dir`` at the output
root (by default ``soma_reproduce/ocelot``) to rescore all three seeds and check
their mean, or at one seed's directory as a spot check::

    soma reproduce ocelot --from-run-dir soma_reproduce/ocelot --record
    soma reproduce ocelot --from-run-dir soma_reproduce/ocelot/seed_0

``--record`` appends the measured score to the results ledger that the table above
reads. It records only a run over all three seeds, since the table holds three-seed
means. :doc:`benchmarking` describes the options shared by every benchmark.

References
----------

* Ryu *et al.*, *OCELOT: Overlapped Cell on Tissue Dataset for Histopathology*,
  CVPR 2023.
* *CellRegNet*, 2024: a fully supervised point-detection baseline.
* `arXiv:2503.05678 <https://arxiv.org/abs/2503.05678>`__: P2PNet-style point
  detection on frozen foundation-model features.
