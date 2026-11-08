MIDOG
=====

Mitotic-figure detection on the MIDOG 2022 public training data with a frozen
encoder. soma's :doc:`detection` path trains a small decoder on dense features
and scores its predictions after stitching them back to their source regions.

Data and task
-------------

MIDOG 2022 contains H&E regions from several tumour types and scanners, with
annotations for mitotic figures and hard-negative lookalikes.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Task
     - Detect mitotic figures as points, with one positive class. Hard-negative
       annotations do not form a second detection class.
   * - Splits
     - One deterministic, patient/domain-stratified local split of the public
       labelled data: 60% train, 20% tune and 20% test. All regions from a patient
       stay together.
   * - Metric
     - ``f1``: native stitched-ROI F1, using Hungarian one-to-one matching within
       7.5 µm and pooling the match counts across test regions.

The official challenge test set is hidden. These local held-out scores use a
different test set from the published MIDOG leaderboard.

Protocol
--------

The recipe is fixed and only the encoder varies.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Input
     - Native region pixels in one nominal 0.25 µm/px frame, approximating the
       scanner spacing differences. Regions are tiled into 1024 × 1024 pixel
       supervision tiles with 128 pixel overlap.
   * - Encoder
     - Any compatible frozen dense encoder; a 448 pixel sliding window with
       25% overlap supplies the token grids. The committed recipe caches fp16
       features.
   * - Decoder
     - ``lightweight_conv``, predicting one peak heatmap for mitotic figures.
   * - Target and loss
     - Peak Gaussians with σ = 5 µm (two thirds of the 7.5 µm matching distance)
       and foreground-weighted MSE with weight 10; see :doc:`detection`.
   * - Training
     - AdamW, learning rate 0.0003, weight decay 0.0001, cosine schedule, up to
       50 epochs. Batch size 1 with 8 accumulation steps; early stopping has
       patience 10. These settings, and σ as a fraction of the matching distance,
       were chosen on the OCELOT tune split and carried over unchanged.
   * - Selection
     - Checkpoints use tune tile F1. The final score threshold is selected on
       stitched tune ROIs and frozen before scoring stitched test ROIs.
   * - Seeds
     - ``0``, ``1`` and ``2`` on the same fixed split; results report the mean
       and standard deviation across seeds.

Results
-------

Native test F1, best first, as the mean and sample standard deviation across
the three seeds.

.. include:: _generated/midog-results.rst

All seven encoders were trained with the recipe above. The packaged Virchow2
target is the seed-0 test F1, **0.7626 ± 0.02**.

The `measurement provenance
<https://github.com/clemsgrs/soma/blob/main/soma/benchmark/midog/provenance.json>`__
records each seed's tune and test score, its selected epoch and the training
software for these rows. It also keeps the earlier recipe's results and their
history.

Reproduce
---------

**Prepare the data.** Place the public MIDOG images under ``images/`` and prepare
``MIDOG2022_training_enriched.json`` with the tumour and scanner metadata needed
for the canonical split. :doc:`curation` gives the metadata preparation commands
and the expected directory layout.

**Run the three-seed benchmark** with its default encoder, Virchow2, or choose
another compatible installed encoder::

    soma reproduce detection/midog --raw-root /path/to/midog --record
    soma reproduce detection/midog --encoder uni2 --raw-root /path/to/midog --record

**Reuse prepared manifests or rescore a completed run.** The latter reads saved
tune/test predictions and performs no training or feature extraction::

    soma reproduce detection/midog --curated-dir /path/to/midog/curated --record
    soma reproduce detection/midog --from-run-dir soma_reproduce/detection/midog/seed_0 --record

``--seeds 1`` selects seed 0 for the individual reference check. ``--encoders``
compares an encoder panel; :doc:`benchmarking` describes the shared options.
