EVA
===

Purpose
-------

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification
leaderboard with Soma's :doc:`classification` heads. Each ``eva/<dataset>`` entry
uses the same frozen-tile linear probe and varies only ``encoder``.

Protocol
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - head
     - linear probe (``aggregation: null`` — each patch is its own bag)
   * - optimizer
     - AdamW, lr ``0.0003``, weight_decay ``0.01``
   * - batch size
     - ``256``
   * - budget
     - eva's ``max_steps=12500`` mapped to soma epochs
   * - metric
     - ``balanced_accuracy``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/balanced_accuracy`` (from ``summary.json``)
   * - canonical seeds
     - ``0, 1, 2, 3, 4`` (averaged)

Encoder mappings:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Encoder
     - EVA backbone
   * - ``uni2`` (default)
     - eva ``mahmood_uni2_h``
   * - ``virchow2``
     - eva ``paige_virchow2``, slide2vec ``cls`` output

Dataset tasks and evaluation splits:

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Benchmark
     - Task head
     - Eval split
   * - ``eva/bach``
     - ``multiclass_classification``
     - EVA validation (``tune_is_test: true``)
   * - ``eva/breakhis``
     - ``multiclass_classification``
     - EVA validation (``tune_is_test: true``)
   * - ``eva/crc``
     - ``multiclass_classification``
     - EVA validation (``tune_is_test: true``)
   * - ``eva/gleason_arvaniti``
     - ``multiclass_classification``
     - EVA validation (``tune_is_test: true``)
   * - ``eva/mhist``
     - ``binary_classification``
     - EVA validation (``tune_is_test: true``)
   * - ``eva/patch_camelyon``
     - ``binary_classification``
     - EVA test (real val + test)

Train/validation-only datasets use validation as Soma ``test``; ``patch_camelyon``
retains its held-out test split.

Prepare and run
---------------

Reproduce one dataset::

    soma reproduce eva/bach --raw-root /path/to/eva/bach

Or run the family::

    soma reproduce eva --raw-root /path/to/eva

Select an encoder with ``--encoder`` (default ``uni2``).

Results
-------

Reproduced numbers
~~~~~~~~~~~~~~~~~~

Recorded cells from ``soma/benchmarks/results/eva.csv`` appear below with seed,
commit, and delta provenance. References are published EVA balanced accuracies
keyed by dataset × encoder from `kaiko-ai/eva pathology leaderboard <https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__. Unrecorded cells are omitted:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - soma (mean ± std)
     - Seeds
     - Reference
     - Δ
     - Recorded (date @ commit)
   * - bach
     - uni2
     - 0.914 ± 0.007
     - 5
     - 0.915
     - -0.001
     - 2026-06-19 @ ``7ef2d7c``
   * - bach
     - virchow2
     - 0.870 ± 0.010
     - 5
     - 0.883
     - -0.013
     - 2026-06-19 @ ``7ef2d7c``
   * - breakhis
     - uni2
     - 0.855 ± 0.006
     - 5
     - 0.859
     - -0.004
     - 2026-06-19 @ ``7ef2d7c``
   * - breakhis
     - virchow2
     - 0.812 ± 0.008
     - 5
     - 0.821
     - -0.009
     - 2026-06-19 @ ``7ef2d7c``
   * - crc
     - uni2
     - 0.966 ± 0.001
     - 5
     - 0.965
     - +0.001
     - 2026-06-19 @ ``7ef2d7c``
   * - crc
     - virchow2
     - 0.966 ± 0.001
     - 5
     - 0.967
     - -0.001
     - 2026-06-19 @ ``7ef2d7c``
   * - gleason_arvaniti
     - virchow2
     - 0.778 ± 0.010
     - 5
     - 0.783
     - -0.005
     - 2026-07-09 @ ``c8b320d``
   * - gleason_arvaniti
     - uni2
     - 0.779 ± 0.005
     - 5
     - 0.775
     - +0.004
     - 2026-07-09 @ ``9663253``

See :doc:`benchmarking` for gate semantics, :doc:`curation` for raw layouts, and
:doc:`classification` for task-head details.
