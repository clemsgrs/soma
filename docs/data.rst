Data
====

Every soma experiment starts with images, supervision, and a split definition.
The data layer keeps those inputs explicit while preprocessing converts them
into the samples consumed by foundation-model encoders.

A cohort holds each sample's identity, targets and folds, and no file paths.
The paths live in two manifests: :class:`soma.data.ImageManifest` (the images)
and :class:`soma.data.AnnotationManifest` (the supervision files). Features
enter through one verb, :func:`soma.extract`, which reads the manifests and
returns the source a run trains on.

.. toctree::
   :maxdepth: 1

   dataset
   curation
   preprocessing
