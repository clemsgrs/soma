Curation
========

Soma includes small curators for converting known public benchmark layouts into
the standard ``dataset.csv`` and ``splits.csv`` manifests described in
:doc:`dataset`.

EVA patch-level classification
------------------------------

The first supported EVA slice covers patch-level classification datasets that
fit Soma's ``dataset_type: tile`` workflow:

- ``bach``
- ``mhist``
- ``crc``
- ``breakhis``
- ``gleason_arvaniti``
- ``patch_camelyon``

Use the curator from Python:

.. code-block:: python

   from soma.curation import curate_eva_patch_dataset

   manifest = curate_eva_patch_dataset(
       "mhist",
       raw_root="/path/to/mhist",
       output_dir="data/eva/mhist",
       # 0.0 reproduces the kaiko-ai/eva leaderboard (train-on-all-train, evaluate
       # on EVA validation via tune_is_test). Use a positive fraction (e.g. 0.2)
       # only if you want a separate carved-out tune split for model selection.
       tune_fraction=0.0,
   )

   print(manifest.dataset_csv)
   print(manifest.splits_csv)

The generated ``dataset.csv`` stores EVA numeric target indices in ``label`` and
keeps the readable class in ``class_name`` metadata. This preserves EVA's class
orientation for binary tasks.

.. note::

   The registered :doc:`EVA benchmark <eva-patch-classification-benchmark>`
   curates with ``tune_fraction=0.0`` for you (its ``CURATION_TUNE_FRACTION``), so
   ``soma reproduce eva/<dataset>`` follows the leaderboard protocol without any
   manual split choice.

Split policy
~~~~~~~~~~~~

For datasets where EVA provides only train/validation-style splits, Soma
reserves EVA validation as ``test`` and creates ``tune`` by deterministic
stratified sampling from EVA train.

To reproduce EVA's train-on-all-train/evaluate-on-validation protocol, curate
with ``tune_fraction=0.0`` and set ``training.tune_is_test: true`` in the run
config. The generated split file will contain only ``train`` and ``test`` rows.

For datasets where EVA provides train/validation/test splits, Soma preserves
EVA validation as ``tune`` and EVA test as ``test``. This lets one run report
both the EVA validation benchmark metric and the EVA test metric.

Raw layout expectations
~~~~~~~~~~~~~~~~~~~~~~~

``bach``
  ``ICIAR2018_BACH_Challenge/Photos/{Benign,InSitu,Invasive,Normal}/*.tif``.
  Pre-split extractions with
  ``ICIAR2018_BACH_Challenge/{train,test}/{Benign,InSitu,Invasive,Normal}/*.tif``
  are also accepted when they match the EVA train/validation counts.

``mhist``
  ``images/*.png`` and ``annotations.csv`` with ``Image Name``,
  ``Majority Vote Label``, and ``Partition`` columns.

``crc``
  ``NCT-CRC-HE-100K/<class>/*.tif`` and ``CRC-VAL-HE-7K/<class>/*.tif``.
  Extractions nested under ``original/<class>`` are also accepted.

``breakhis``
  BreaKHis images in the original nested layout. Only ``40X`` ``*.png`` images
  with EVA classes ``TA``, ``MC``, ``F``, and ``DC`` are used. The EVA
  validation patient-id list is used to assign validation samples to Soma
  ``test``.

``gleason_arvaniti``
  ``train_validation_patches_750/**/*.jpg``. Files from microarray ``ZT76`` are
  assigned to EVA validation and files from ``ZT111``, ``ZT199``, and ``ZT204``
  to EVA train. EVA reports GleasonArvaniti on the validation cohort and does not
  use ``test_patches_750`` — its test split "leads to unstable evaluation
  results" — so those patches are ignored even when present.

``patch_camelyon``
  Materialized image folders
  ``{train,val,test}/{normal|no_tumor,tumor}/*.{png,jpg,jpeg,tif,tiff}``.
  EVA's official HDF5 files must first be materialized as image files before
  they can be represented by Soma's current ``image_path`` manifest format.

Segmentation datasets from EVA, such as MoNuSAC, CoNSeP, and BCSS, are not
covered by this tile-classification curation path.

OCELOT 2023 cell detection
--------------------------

The OCELOT curator targets Soma's ``dataset_type: detection`` path. It converts
the unzipped `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ release
(Zenodo record 8417503, ``ocelot2023_v1.0.1.zip``) into Soma's detection
manifests. Like the EVA curators it does not download anything; accept the Zenodo
terms and unzip first (see ``examples/ocelot/README.md`` for the download step).

Curate from Python::

    from soma.curation.ocelot import curate_ocelot_detection

    curate_ocelot_detection("<raw>/ocelot2023_v1.0.1", "<out>/curated")

or from the command line::

    python -m soma.curation.ocelot \
        --raw-root <raw>/ocelot2023_v1.0.1 \
        --output-dir <out>/curated

OCELOT ships paired *cell* and *tissue* patches; detection-v1 uses the **cell**
patches only (1024×1024 JPEGs at ~0.2 µm/px). Each is paired with a headerless
``x,y,label`` point CSV whose 1-based cell label (``1`` = background cell, ``2`` =
tumor cell) is remapped to Soma's 0-based class ids (BC→0, TC→1). The curator
writes ``dataset.csv`` (``sample_id, image_path, points_path``), ``splits.csv``,
one ``points/<sample_id>.csv`` per sample, and ``summary.json``.

Split policy
~~~~~~~~~~~~

OCELOT's own train/val/test split is emitted verbatim as a single fold, with
train → ``train``, val → ``tune`` (threshold sweep / monitor), and test →
``test``. Soma never partitions the data itself.
