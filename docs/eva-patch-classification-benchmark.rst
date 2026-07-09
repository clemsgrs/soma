EVA
===

*Maps to task:* :doc:`classification` — frozen-tile linear-probe runs of the
binary / multiclass classification heads reproducing the
`kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification leaderboard.

.. note::

   This page is generated from the registered benchmark definition — the protocol
   summary and reference numbers from the ``Benchmark`` object's ``expected()`` rows
   (packaged ``soma/benchmarks/reference/eva.csv``), and the command from the benchmark name. Edit the registry
   (``soma/benchmarks/eva.py``) and the CSV, not this page; ``python docs/_generate_reference.py``
   re-emits it and ``tests/test_docs.py`` guards the two from drifting.

EVA is registered as **one sub-benchmark per dataset** (``eva/<dataset>``), each
sharing the same offline linear-probe recipe and varying only the ``encoder`` axis.
``soma reproduce eva`` fans out over the whole family; a single ``eva/<dataset>``
reproduces one dataset.

The frozen-tile-probe protocol
------------------------------

Stated once, shared by every dataset:

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

Encoders
--------

The ``encoder`` axis maps a soma encoder onto an EVA leaderboard backbone:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Encoder
     - EVA backbone
   * - ``uni2`` (default)
     - eva ``mahmood_uni2_h``
   * - ``virchow2``
     - eva ``paige_virchow2``, slide2vec ``cls`` output

Datasets
--------

Where EVA ships only train/validation, the EVA validation split becomes soma
``test`` and the run sets ``tune_is_test: true`` (train-on-all-train /
evaluate-on-validation); ``patch_camelyon`` has a real held-out test split:

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

Reproduced numbers
------------------

What soma has actually measured, recorded by ``soma reproduce --record`` into the
packaged results ledger (``soma/benchmarks/results/eva.csv``) alongside the commit
and slide2vec version that produced each number. The ``Reference`` column is the
published EVA balanced-accuracy band (keyed by ``dataset`` × ``encoder``, from `kaiko-ai/eva pathology leaderboard <https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__); only cells that have been run appear, each with its delta to that band:

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

Reproduce
---------

``soma reproduce`` curates the raw layout, trains the linear probe over the
canonical seeds, reads ``test/balanced_accuracy`` from ``summary.json``, and
tolerance-checks it against the band above. Reproduce one dataset::

    soma reproduce eva/bach --raw-root /path/to/eva/bach
    soma reproduce eva/breakhis --raw-root /path/to/eva/breakhis
    soma reproduce eva/crc --raw-root /path/to/eva/crc
    soma reproduce eva/gleason_arvaniti --raw-root /path/to/eva/gleason_arvaniti
    soma reproduce eva/mhist --raw-root /path/to/eva/mhist
    soma reproduce eva/patch_camelyon --raw-root /path/to/eva/patch_camelyon

…or fan out over the whole family in one go (each member owns a per-dataset
subdirectory)::

    soma reproduce eva --raw-root /path/to/eva

Pick the encoder axis with ``--encoder`` (default ``uni2``); ``--seeds 1`` runs a single-seed smoke.

.. seealso::

   * :doc:`classification` — the task heads the probe trains (binary, multiclass).
   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.
   * :doc:`curation` — the EVA curators and split policy.
