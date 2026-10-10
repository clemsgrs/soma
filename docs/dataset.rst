Dataset
=======

soma loads samples and split assignments from two CSV manifests:

- ``dataset.csv`` describes the samples, labels, and optional metadata.
- ``splits.csv`` assigns each sample to a fold and split.

Both files are validated on loading. Keep ``sample_id`` stable across them;
sample and patient IDs must be bare names without path separators.

Dataset format
--------------

``dataset.csv``
  | Required columns: ``sample_id``, ``image_path``, and the target columns the task head declares (``label`` for classification, ``value`` for regression, ``time`` and ``event`` (plus ``bin`` for discrete-time survival) for survival). ``task.targets`` maps a key to a differently named column, e.g. ``targets: {value: label}``.
  | Optional columns: ``mask_path`` (pre-computed tissue mask, valid for every ``dataset_type``), ``patient_id`` (required for ``dataset_type="patient"``; the sampling unit leakage checks protect), ``coordinates_path`` (user-supplied tile coordinates for whole slides; see :ref:`preprocessing-supplied-coordinates`), ``split`` / ``fold`` (a single CSV may carry its own splits).
  | Additional, unrecognized columns are carried along as per-sample metadata.

One CSV feeds three readers, each validating only its own columns:
:class:`soma.data.Cohort` (identity, targets, folds), :class:`soma.data.ImageManifest`
(what extraction reads) and :class:`soma.data.AnnotationManifest` (dense supervision
files). Records carry no file paths.

Dense-supervision manifests (``dataset_type="segmentation"`` / ``"detection"``)
replace the scalar ``label`` with a per-sample supervision file:

- **Segmentation** uses ``label_mask_path`` — a per-sample label mask. It is distinct
  from ``mask_path`` (the optional tissue mask): a segmentation row may carry both.
  A row is a pre-cropped tile or a whole slide; see :doc:`segmentation`.
- **Detection** uses ``points_path`` — a per-sample point file
  (read through :class:`soma.data.AnnotationManifest`), a CSV of object centroids with
  ``x, y, class`` columns (headerless ``x,y,class`` — OCELOT's format — or a
  2-column ``x,y`` for a single class). Points are stored in **level-0** pixels;
  an optional finite positive per-sample ``spacing_at_level_0`` column declares the
  source image's level-0 µm/px. It is required for flat PNG/JPEG dense extraction. See
  :doc:`detection` for the full column contract.

For ``dataset_type="spatial_expression"``, use ``target_index`` instead of
``label``. Each index selects a row in ``targets.npy`` (shape
``[n_rows, n_genes]``); ``genes.json`` lists the genes in column order. Both
sidecars must sit beside ``dataset.csv``; ``Cohort.from_csv(..., targets=["expression"])``
reads them into each record's ``expression`` target.

Splits format
-------------

``splits.csv``
  | Required columns: ``sample_id``, ``split``.
  | Optional column: ``fold`` (integer). Omit it for a single train/tune/test split; include distinct values (0, 1, 2, …) for cross-validation.
  | Valid split names: ``train``, ``tune``, or any name starting with ``test`` (e.g. ``test``, ``test_external``).
  | Every fold must contain at least one test split, unless ``training.tune_is_test`` uses its tune split for both roles.

Use explicit test split names for multiple held-out cohorts. See
:doc:`training` for checkpoint selection and missing-tune policies, and
:doc:`getting-started` for a complete run.

.. _semantic-manifest-identity:

Semantic manifest identity
--------------------------

Experiment and leaderboard dataset checksums describe semantic values in the
``dataset.csv`` rows selected by ``train`` and ``tune`` assignments. Test rows
have a separate identity, so changing a test cohort does not change the
training experiment. Representation-only runs use their configured evaluation
split instead. See :doc:`outputs` for the identity and provenance rules.

Dataset checksums exclude only the storage-location columns ``image_path``,
``mask_path``, ``label_mask_path``, ``points_path``, and ``coordinates_path``, so
relocating files does not change data identity. Split assignments have their own checksum. To make
artifact contents part of identity, add an explicit ``<path_column>_sha256``
column such as ``image_path_sha256``; soma never opens referenced files to derive
checksums itself.
