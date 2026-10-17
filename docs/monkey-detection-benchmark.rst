MONKEY
======

Inflammatory-cell detection on the MONKEY public training data with a frozen
encoder. soma's :doc:`detection` path trains a two-channel decoder and scores
merged inflammatory-cell predictions in their source ROI polygons.

Data and task
-------------

MONKEY contains 81 PAS-stained kidney-transplant biopsy slides from four centres,
with lymphocyte and monocyte point annotations inside ROI polygons.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Task
     - Detect lymphocytes and monocytes as two classes during training; merge
       them into inflammatory cells for the headline score.
   * - Splits
     - Five deterministic patient folds, balanced across centres. In each
       rotation one fold is test, the next is tune and the other three are train;
       each patient is tested once.
   * - Metric
     - ``froc_inflammatory``: merged inflammatory-cell FROC with greedy matching
       within 5 µm, averaged over the five test folds.

The official challenge test set is hidden. These cross-validation scores use the
public training patients and a different test set from the published MONKEY
leaderboard.

Protocol
--------

The recipe is fixed and only the encoder varies.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Input
     - ROI crops at approximately 0.242 µm/px, tiled into 1024 × 1024 pixel
       supervision tiles with 128 pixel overlap. The valid area is each ROI
       polygon dilated by 5 µm; pixels outside it are ignored.
   * - Encoder
     - Any compatible frozen dense encoder; a 448 pixel sliding window with
       25% overlap supplies the token grids. The committed recipe caches fp16
       features.
   * - Decoder
     - ``lightweight_conv``, with one peak heatmap for lymphocytes and one for
       monocytes.
   * - Target and loss
     - Peak Gaussians with σ = 1.7 µm and foreground-weighted MSE with weight 10;
       see :doc:`detection`.
   * - Training
     - Adam, learning rate 0.0001, weight decay 0.00001, cosine schedule, up to
       50 epochs. Batch size 1 with 8 accumulation steps; early stopping has
       patience 10.
   * - Selection
     - Checkpoints use tune tile F1. Per-class score thresholds are selected on
       stitched tune ROIs and frozen before scoring stitched test ROIs.
   * - FROC
     - Pool the two classes and suppress cross-class duplicates within 5 µm.
       Average sensitivity at 10, 20, 50, 100, 200 and 300 false positives/mm²,
       using the valid ROI area. This merged score differs from the mean of
       the two per-class FROCs.
   * - Replicates
     - One seed, ``0``, across all five patient folds. Use ``--seeds 1`` explicitly:
       the generic detection CLI otherwise runs three seeds, each across five
       folds.

Results
-------

Mean merged inflammatory-cell FROC across the five test folds, best first. The
table reads the latest recorded result for each encoder. ``Seeds`` counts training
seeds; any displayed standard deviation is across seeds. A single five-fold run
therefore has one seed, and its fold spread is reported separately in the
provenance.

.. include:: _generated/monkey-results.rst

The table contains the historical seven-encoder panel, with the latest Virchow2
row replaced by a newer five-fold reproduction. Its mean **0.27549** is within
the packaged **0.2751 ± 0.02** target. Its population fold standard deviation is
**0.02515**, recorded separately from seed spread. The reference tolerance is
unchanged.

The `measurement provenance
<https://github.com/clemsgrs/soma/blob/main/soma/benchmark/monkey/provenance.json>`__
records each fold's score and population fold standard deviation. It identifies
historical panel rows harvested from native run summaries separately from rows
emitted by ``soma reproduce --record``, and preserves original training versions,
training and scoring commits, run paths and historical snapshot limitations.
Earlier training runs retain their original software identity when their results
are published here.

Reproduce
---------

**Prepare the data.** Download the public training slides and millimetre-frame
annotations into one directory::

    aws s3 sync s3://monkey-training /path/to/monkey --no-sign-request \
        --exclude "*" --include "images/pas-cpg/*" --include "annotations/json_mm/*"

:doc:`curation` describes the ROI crops, ignore masks and patient folds.

**Run one seed across all five folds** with the default encoder, Virchow2, or
choose another compatible installed encoder::

    soma reproduce detection/monkey --seeds 1 --raw-root /path/to/monkey --record
    soma reproduce detection/monkey --encoder uni2 --seeds 1 --raw-root /path/to/monkey --record

**Reuse prepared manifests or rescore a completed five-fold run.** Rescoring reads
saved tune/test predictions and performs no training or feature extraction::

    soma reproduce detection/monkey --seeds 1 --curated-dir /path/to/monkey/curated --record
    soma reproduce detection/monkey --from-run-dir soma_reproduce/detection/monkey/seed_0 --record

``--record`` stores the five-fold mean as one seed measurement. The native
``detection_score/metrics.json`` also retains every fold value and the population
fold standard deviation. :doc:`benchmarking` describes the shared options.
