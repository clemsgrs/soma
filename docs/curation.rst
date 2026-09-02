Curation
========

soma includes small curators for converting known public benchmark layouts into
the standard ``dataset.csv`` and ``splits.csv`` manifests described in
:doc:`dataset`.

CRoMa cohorts
-------------

The :doc:`CRoMa <croma-robustness-benchmark>` benchmark uses three tile
cohorts from the PathoROB study. Acquire and decode the pinned sources
once::

    pip install 'soma-pathology[croma]'
    soma prepare-croma /data/croma

Or from Python::

    from soma.robustness import prepare_croma

    prepared = prepare_croma("/data/croma")

The command downloads the exact pinned dataset revisions, verifies every
checksum, and decodes the tiles. Allow at least 5 GB of free space. The source
datasets retain their upstream licenses: CC0-1.0 for Camelyon,
CC-BY-NC-SA-4.0 for TCGA, and CC-BY-SA-4.0 for Tolkach-ESCA.

Curate one cohort, or all three with family-wide sample-ID validation::

    from soma.curation import curate_croma_view, curate_croma_views

    camelyon = curate_croma_view(
        "/data/croma",
        "data/croma/camelyon",
        cohort="camelyon",
    )
    manifests = curate_croma_views("/data/croma", "data/croma")

The cohorts are fixed, balanced label-by-center grids: Camelyon has 20,400
rows, TCGA-4x4 has 5,760, and Tolkach-ESCA has 9,000. Every emitted row is
assigned to ``test``, fold ``0``. ``soma reproduce croma`` runs this curation
for you.

EVA patch-level classification
------------------------------

The first supported EVA slice covers patch-level classification datasets that
fit soma's ``dataset_type: tile`` workflow:

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
   )

   print(manifest.dataset_csv)
   print(manifest.splits_csv)

The generated ``dataset.csv`` stores EVA numeric target indices in ``label`` and
keeps the readable class in ``class_name`` metadata. This preserves EVA's class
orientation for binary tasks.

.. note::

   The registered :doc:`EVA benchmark <eva-patch-classification-benchmark>`
   curates the same way, so ``soma reproduce eva/<dataset>`` follows the
   leaderboard protocol without any manual split choice.

Split policy
~~~~~~~~~~~~

The curator follows EVA's official protocol and has no split knob. For datasets
where EVA provides only train/validation-style splits, every EVA train sample is
soma ``train`` and EVA validation is soma ``test``; the split file contains only
``train`` and ``test`` rows, and the run sets ``training.tune_is_test: true`` so
checkpoint selection happens on the reported split, as the leaderboard does.

For datasets where EVA provides train/validation/test splits, soma preserves
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
  validation patient-id list is used to assign validation samples to soma
  ``test``.

``gleason_arvaniti``
  ``train_validation_patches_750/**/*.jpg``. Files from microarray ``ZT76`` are
  assigned to EVA validation and files from ``ZT111``, ``ZT199``, and ``ZT204``
  to EVA train. EVA reports GleasonArvaniti on the validation cohort and does not
  use ``test_patches_750`` — its test split "leads to unstable evaluation
  results" — so those patches are ignored even when present.

  If ``train_validation_patches_750`` is absent, soma materializes it from the raw
  `Harvard Dataverse download <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP>`_:
  drop the train/validation TMA archives (``ZT{76,111,199,204}_*.tar.gz``) and
  ``Gleason_masks_train.tar.gz`` into ``<root>`` (already-extracted folders or a
  ``TMA_images/`` dir are also accepted). soma slices the 750×750 patches itself —
  a vendored, dependency-free port of ``create_patches.py`` from the
  `gleason_CNN <https://github.com/eiriniar/gleason_CNN>`_ repo — so you do not need
  to run that (Python 2-era) script. Materialization is atomic and idempotent: a
  completed ``train_validation_patches_750`` is reused as-is on later runs. Only the
  train/validation patches are produced; the ZT80 test cohort is skipped.

``patch_camelyon``
  Either materialized image folders
  ``{train,val,test}/{normal|no_tumor,tumor}/*.{png,jpg,jpeg,tif,tiff}``, or
  EVA's six official HDF5 files
  (``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5``) under the raw
  root. When only the HDF5 files are present the curator materializes them to
  the class-folder layout above on first use (writing PNGs under a writable raw
  root; idempotent, so an interrupted pass resumes on the next run).

Segmentation datasets from EVA, such as MoNuSAC, CoNSeP, and BCSS, are not
covered by this tile-classification curation path.

OCELOT 2023 cell detection
--------------------------

The OCELOT curator targets soma's ``dataset_type: detection`` path. It converts
the unzipped `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ release
(Zenodo record 8417503, ``ocelot2023_v1.0.1.zip``) into soma's detection
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
tumor cell) is remapped to soma's 0-based class ids (BC→0, TC→1). The curator
writes ``dataset.csv`` (``sample_id, image_path, points_path``), ``splits.csv``,
one ``points/<sample_id>.csv`` per sample, and ``summary.json``.

Split policy
~~~~~~~~~~~~

OCELOT's own train/val/test split is emitted verbatim as a single fold, with
train → ``train``, val → ``tune`` (threshold sweep / monitor), and test →
``test``. soma never partitions the data itself.
