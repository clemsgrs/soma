EVA
===

What this benchmark measures
----------------------------

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification
leaderboard with Soma's :doc:`classification` heads. Each ``eva/<dataset>`` entry
uses the same frozen-tile linear probe and varies only ``encoder``.

**Pipeline:** labelled patches → frozen encoder → linear head → balanced accuracy.

Prepare the data
----------------

Download a supported EVA dataset and point ``--raw-root`` at the directory with
the following contents. Soma converts it to the standard manifests automatically.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset
     - Raw-root contents
   * - ``bach``
     - ``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``
   * - ``breakhis``
     - the original BreaKHis tree; Soma selects 40× patches and EVA classes
   * - ``crc``
     - ``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``
   * - ``gleason_arvaniti``
     - ``train_validation_patches_750/`` or the original TMA and mask archives
   * - ``mhist``
     - ``images/*.png`` and ``annotations.csv``
   * - ``patch_camelyon``
     - ``{train,val,test}/<class>/`` images or the six official HDF5 files

For train/validation-only datasets, EVA validation becomes Soma ``test`` and
``tune_is_test`` preserves EVA's train-on-all-train protocol. Datasets with a real
test split retain validation as ``tune`` and test as ``test``.

Run the benchmark
-----------------

Reproduce one dataset::

    soma reproduce eva/bach --raw-root /path/to/eva/bach

Or run the family::

    soma reproduce eva --raw-root /path/to/eva

Select an encoder with ``--encoder`` (default ``uni2``).

What you can vary
-----------------

Compare the supported foundation-model encoders:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Encoder
     - EVA backbone
   * - ``uni2`` (default)
     - eva ``mahmood_uni2_h``
   * - ``virchow2``
     - eva ``paige_virchow2``, slide2vec ``cls`` output

Run one dataset or the complete family:

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

Results
-------

Reproduced numbers
~~~~~~~~~~~~~~~~~~

Recorded Soma scores appear beside the published EVA balanced accuracies from `kaiko-ai/eva pathology leaderboard <https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__. Unrecorded cells are omitted; detailed run provenance remains in the packaged
results CSV.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - Soma (mean ± std)
     - EVA reference
   * - bach
     - uni2
     - 0.914 ± 0.007
     - 0.915
   * - bach
     - virchow2
     - 0.870 ± 0.010
     - 0.883
   * - breakhis
     - uni2
     - 0.855 ± 0.006
     - 0.859
   * - breakhis
     - virchow2
     - 0.812 ± 0.008
     - 0.821
   * - crc
     - uni2
     - 0.966 ± 0.001
     - 0.965
   * - crc
     - virchow2
     - 0.966 ± 0.001
     - 0.967
   * - gleason_arvaniti
     - virchow2
     - 0.778 ± 0.010
     - 0.783
   * - gleason_arvaniti
     - uni2
     - 0.779 ± 0.005
     - 0.775

Protocol details
----------------

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

See :doc:`benchmarking` for the shared benchmark workflow and :doc:`classification`
for task-head details.
