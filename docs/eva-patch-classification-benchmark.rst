EVA
===

*Maps to task:* :doc:`classification` — frozen-tile linear-probe runs of the
binary / multiclass classification heads reproducing the
`kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification leaderboard.

.. note::

   This page is generated from the registered benchmark definition — the protocol
   summary from the ``Benchmark`` object, the reference table straight from the
   packaged ``soma/benchmarks/reference/eva.csv`` (same bytes), and the command from the benchmark name. Edit the
   registry (``soma/benchmarks/eva.py``) and the CSV, not this page; ``python docs/_generate_reference.py``
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

Reference numbers
-----------------

The published EVA balanced-accuracy band, keyed by ``dataset`` × ``encoder`` —
read verbatim from the packaged reference CSV (``patch_camelyon`` carries both a
``test`` and a ``tune`` row):

.. csv-table:: ``soma/benchmarks/reference/eva.csv``
   :file: ../soma/benchmarks/reference/eva.csv
   :header-rows: 1
   :widths: 12 10 20 10 10 38

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
