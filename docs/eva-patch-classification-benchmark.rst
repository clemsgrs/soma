EVA
===

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_
patch-classification leaderboard with frozen tile encoders and linear
:doc:`classification` heads.

EVA provides 6 registered datasets: bach, breakhis, crc, gleason_arvaniti, mhist, and patch_camelyon. All share the same linear-probe protocol; see :doc:`benchmarking` for
the shared workflow.

**Pipeline:** labelled patches → frozen encoder → linear head → balanced accuracy

Prepare the data
----------------

Download one EVA dataset from its official
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

Run the benchmark
-----------------

Choose a compatible tile-level :doc:`encoder <encoders>` and pass the downloaded
dataset directory as ``--raw-root``; curated manifests are written under
``<raw-root>/curated``. For example::

    soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach

To run the whole family, prepare one subdirectory per dataset under
``/path/to/eva``::

    soma reproduce eva --encoder virchow2 --raw-root /path/to/eva

Results
-------

Recorded balanced accuracy scores alongside the packaged EVA references.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - soma (mean ± std)
     - EVA reference
   * - bach
     - uni2
     - 0.914 ± 0.008
     - 0.915
   * - bach
     - virchow2
     - 0.871 ± 0.011
     - 0.883
   * - breakhis
     - uni2
     - 0.855 ± 0.007
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
     - uni2
     - 0.779 ± 0.005
     - 0.775
   * - gleason_arvaniti
     - virchow2
     - 0.778 ± 0.010
     - 0.783

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
     - fixed step budget: ``max_steps=12500`` optimizer updates
   * - metric
     - ``balanced_accuracy``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/balanced_accuracy`` (from ``summary.json``)
   * - canonical seeds
     - ``0, 1, 2, 3, 4`` (averaged)
