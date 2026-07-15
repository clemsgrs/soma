EVA
===

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_
patch-classification leaderboard with frozen tile encoders and linear
:doc:`classification` heads.

EVA provides 6 registered datasets: bach, breakhis, crc, gleason_arvaniti, mhist, and patch_camelyon. All share the same linear-probe protocol.

**Pipeline:** labelled patches → frozen encoder → linear head → balanced accuracy

Prepare the data
----------------

soma does not download benchmark data. Download one dataset from its official
source and unpack it in the directory you will pass as ``--raw-root``:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset and source
     - Raw-root contents
   * - `BACH <https://zenodo.org/records/3632035>`__ (``bach``)
     - ``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``
   * - `BreaKHis <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>`__ (``breakhis``)
     - ``BreaKHis_v1/histology_slides/…/40X/*.png``; soma selects EVA classes
   * - `CRC <https://zenodo.org/records/1214456>`__ (``crc``)
     - ``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``
   * - `Gleason Arvaniti <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP>`__ (``gleason_arvaniti``)
     - the ``ZT{76_39,111_4,199_1,204_6}*.tar.gz`` TMA archives and ``Gleason_masks_train.tar.gz``
   * - `MHIST <https://bmirds.github.io/MHIST/#accessing-dataset>`__ (``mhist``)
     - ``images/*.png`` and ``annotations.csv``
   * - `PatchCamelyon <https://zenodo.org/records/2546921>`__ (``patch_camelyon``)
     - the six ``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5`` files

For example, prepare BACH from its public archive::

    mkdir -p /path/to/eva/bach
    curl -L 'https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1' -o /tmp/bach.zip
    unzip /tmp/bach.zip -d /path/to/eva/bach

Run the benchmark
-----------------

Pick any tile-level :doc:`encoder <encoders>` supported by soma and pass the
downloaded dataset directory as ``--raw-root``. ``soma reproduce`` runs the
built-in EVA curator automatically, writes the manifests under
``<raw-root>/curated``, extracts features, trains the linear probe, and reports
balanced accuracy. No separate curation command is required. For example::

    soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach

Or run EVA's 6 datasets in one go::

    soma reproduce eva --encoder virchow2 --raw-root /path/to/eva

Results
-------

We benchmarked two encoders: soma closely reproduces
EVA's published balanced accuracy scores.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - soma (mean ± std)
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

Across these 8 recorded dataset–encoder comparisons, the median relative difference is **0.48%**.

See the `kaiko-ai/eva pathology leaderboard <https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__ for the official reference leaderboard.

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
