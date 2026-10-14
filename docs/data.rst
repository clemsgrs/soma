Data
====

A run reads two CSV files: ``dataset.csv``, one row per sample with its image, its
targets and any metadata, and ``splits.csv``, which puts each sample in ``train``,
``tune`` or a ``test`` split, per fold. Three readers take what they need from the
same ``dataset.csv``:

- :class:`soma.Cohort` holds each sample's identity, targets and folds, and no file
  paths. It checks the splits when it is built, and refuses a patient whose samples
  fall in two splits of one fold.
- :class:`soma.ImageManifest` holds the image paths that extraction reads.
- :class:`soma.AnnotationManifest` holds the supervision files of segmentation and
  detection: label masks and point files.

Features enter a run through a *source*. :func:`soma.extract` reads the image
manifest (and the annotation manifest, to sample regions from whole slides), tiles
and encodes the images, and returns the source the run trains on, keyed by
``sample_id``. Features computed elsewhere enter as a source too; see
:ref:`bring-your-own-features`.

- :doc:`dataset`: the columns of both files, the split rules, and what makes two
  cohorts the same experiment
- :doc:`curation`: functions that write these files for supported public datasets
- :doc:`preprocessing`: tissue masking, tiling, spacing, and regions sampled from
  annotations

.. toctree::
   :maxdepth: 1
   :hidden:

   dataset
   curation
   preprocessing
