Curation
========

soma includes curators for converting supported public benchmark layouts into
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
assigned to ``test``, fold ``0``.

EVA patch-level classification
------------------------------

The EVA patch curator supports these ``dataset_type: tile`` datasets:

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

``soma reproduce eva/<dataset>`` runs this curation for you (see the
:doc:`EVA benchmark <eva-patch-classification-benchmark>`).

The EVA segmentation datasets have their own curators, ``curate_consep`` and
``curate_monusac`` (``soma.curation.eva_segmentation``). They write 224 px
tiles and class-index masks that reproduce EVA's sample geometry, so the
curated directory holds images as well as the manifest.

The EVA slide-level datasets have ``curate_camelyon16_small`` and
``curate_panda_small`` (``soma.curation.eva_slide``). They choose each slide's
tiles with EVA's sampler and write one hs2p tiling artifact per slide; the
slide manifest names it in ``coordinates_path`` (see
:ref:`preprocessing-supplied-coordinates`). The curated directory holds these
artifacts and the manifest, not tile images.

Split policy
~~~~~~~~~~~~

The curator follows EVA's official protocol and has no split knob. EVA train
is soma ``train``. When EVA provides only a validation split, it becomes soma
``test`` and the run sets ``training.tune_is_test: true``. When EVA also
provides a test split, validation becomes ``tune`` and test becomes ``test``.

Raw layout expectations
~~~~~~~~~~~~~~~~~~~~~~~

HDF5 or archive inputs are materialized to image folders on first use; a
completed materialization is reused on later runs.

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

  If ``train_validation_patches_750`` is absent, soma slices the 750×750 patches
  itself from the raw
  `Harvard Dataverse download <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP>`_:
  place the ``ZT{76,111,199,204}_*.tar.gz`` archives and
  ``Gleason_masks_train.tar.gz`` under ``<root>`` (extracted folders or a
  ``TMA_images/`` dir are also accepted). The ZT80 test cohort is skipped.

``patch_camelyon``
  Either materialized image folders
  ``{train,val,test}/{normal|no_tumor,tumor}/*.{png,jpg,jpeg,tif,tiff}``, or
  EVA's six official HDF5 files
  (``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5``) under the raw
  root. HDF5 files are materialized as PNGs under a writable raw root.

Segmentation datasets from EVA, such as MoNuSAC, CoNSeP, and BCSS, are not
covered by this tile-classification curation path.

OCELOT 2023 cell detection
--------------------------

The OCELOT curator targets soma's ``dataset_type: detection`` path. It converts
the unzipped `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ release
(``ocelot2023_v1.0.1.zip``) into soma's detection manifests. Download and unzip
it first (see ``examples/ocelot/README.md``).

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
writes ``dataset.csv`` (``sample_id, image_path, points_path, spacing_at_level_0``),
``splits.csv``, one ``points/<sample_id>.csv`` per sample, and ``summary.json``.

Split policy
~~~~~~~~~~~~

OCELOT's own train/val/test split is emitted verbatim as a single fold, with
train → ``train``, val → ``tune`` (threshold sweep / monitor), and test →
``test``. soma never partitions the data itself.

MONKEY inflammatory-cell detection
----------------------------------

The MONKEY curator targets soma's ``dataset_type: detection`` path. It converts
the public training set of the
`MONKEY challenge <https://monkey.grand-challenge.org/>`_ (81 PAS-stained
kidney-biopsy slides from centres A–D, one slide per patient) into soma's
detection manifests. Download the slides and the annotations first. The bucket
is public, so no account is needed::

    aws s3 sync s3://monkey-training <raw_root> --no-sign-request \
      --exclude "*" --include "images/pas-cpg/*" --include "annotations/json_mm/*"

The curator expects this layout under ``<raw_root>``::

    images/pas-cpg/<case>_PAS_CPG.tif
    annotations/json_mm/<case>_lymphocytes.json
    annotations/json_mm/<case>_monocytes.json
    annotations/json_mm/<case>_inflammatory-cells.json

It fails with the download command when a folder or a file is missing.

Curate from Python::

    from soma.curation.monkey import curate_monkey_detection

    curate_monkey_detection("<raw_root>", "<out>/roi")

or from the command line::

    python -m soma.curation.monkey --raw-root <raw_root> --output-dir <out>/roi

Cells are annotated only inside ROI polygons, so the curator writes one sample
per ROI. It reads the polygon's level-0 bounding box from the slide, writes it as
a PNG, and writes the points in crop coordinates (lymphocytes → class 0,
monocytes → class 1). Each sample has an ignore mask (``ignore_mask_path``): the
valid region is the polygon dilated by 5 µm. Points further than 5 µm from their
polygon are dropped, and ``summary.json`` records the kept and dropped counts,
the valid area and the annotated ``area_rois``. ``dataset.csv`` also records the
source slide (``source_slide``) and the crop offset (``crop_x``, ``crop_y``).

The ROIs are larger than a training tile. The ``detection/monkey`` benchmark tiles
them with :func:`soma.curation.tile_detection.tile_detection_manifest` into
1024 px tiles with 128 px overlap, as for MIDOG, and the tiles carry the ignore
masks.

Split policy
~~~~~~~~~~~~

The official MONKEY test set is hidden, so the curator writes a 5-fold
cross-validation by patient, with each centre spread evenly over the folds. In
fold ``k``, fold ``k`` is ``test``, fold ``(k + 1) % 5`` is ``tune`` and the other
three folds are ``train``, so every slide is ``test`` exactly once. These results
are not comparable to the published leaderboard.
