Dataset
=======

soma loads samples and split assignments from two CSV manifests:

- ``dataset.csv`` describes the samples, labels, and optional metadata.
- ``splits.csv`` assigns each sample to a fold and split.

Both files are validated on loading. Keep ``sample_id`` stable across them;
sample and patient IDs must be bare names without path separators.

soma reads ``sample_id`` and ``patient_id`` as literal strings in every CSV it
loads: ``001`` stays ``001``, not ``1``, and ``NA`` is an ID, not a missing value.
In these two columns only an empty cell is missing, so leave ``patient_id`` blank
for a sample without a patient. soma warns when a ``patient_id`` is spelled like a
missing value (``NA``, ``nan``, ``NULL``, ``None``, ...). Every other column is
parsed as usual: numbers stay numbers, and an empty cell or ``NA`` is missing.
A DataFrame you pass to the Python API (``Cohort.from_frames``,
``ImageManifest.from_frame``, ``AnnotationManifest.from_frame``) is used as you
built it, so parse its IDs as strings yourself.

Dataset format
--------------

``dataset.csv``
  | Required columns: ``sample_id``, ``image_path``, and the target columns the task head declares (``label`` for classification, ``value`` for regression, ``time`` and ``event`` (plus ``bin`` for discrete-time survival) for survival). The head's ``targets`` param maps a key to a differently named column, e.g. ``targets: {value: label}``.
  | Optional columns: ``mask_path`` (pre-computed tissue mask, valid for every run), ``patient_id`` (required with ``cohort.collapse: patient_id``; the sampling unit leakage checks protect), ``coordinates_path`` (user-supplied tile coordinates for whole slides; see :ref:`preprocessing-supplied-coordinates`), ``split`` / ``fold`` (a single CSV may carry its own splits).
  | Additional, unrecognized columns are carried along as per-sample metadata.

One CSV feeds three readers, each validating only its own columns:
:class:`soma.data.Cohort` (identity, targets, folds), :class:`soma.data.ImageManifest`
(what extraction reads) and :class:`soma.data.AnnotationManifest` (dense supervision
files). Records carry no file paths: ``soma.extract`` reads the image manifest (and,
to sample ROIs from whole slides, the annotation manifest) and returns the features
keyed by ``sample_id``.

Splits are sample-level. A sample is usually a slide, but it can be a patient or any
other unit; ``sample_id`` is the key. When extraction turns a sample into several
children, such as the ROIs annotation sampling draws from a slide, the children
inherit their sample's targets, patient and splits
(:meth:`soma.data.Cohort.derived`). The ROIs themselves stay inside extraction: the
grid source is keyed by ROI id, and its ``rois`` name each ROI's slide.

Dense-supervision manifests (segmentation and detection heads)
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

A gene-expression vector per sample, as in the
:doc:`HEST benchmark <hest-gene-expression-benchmark>`, uses a ``target_index``
column instead of ``label``. Each index selects a row in ``targets.npy`` (shape
``[n_rows, n_genes]``); ``genes.json`` lists the genes in column order. Both
sidecars must sit beside ``dataset.csv``; ``Cohort.from_csv(..., targets=["expression"])``
reads them into each record's ``expression`` target.

Splits format
-------------

``splits.csv``
  | Required columns: ``sample_id``, ``split``.
  | Optional column: ``fold`` (integer). Omit it for a single train/tune/test split; include distinct values (0, 1, 2, …) for cross-validation.
  | Valid split names: ``train``, ``tune``, or any name starting with ``test`` (e.g. ``test``, ``test_external``).
  | Every fold must contain at least one test split, unless ``cohort.tune_is_test`` uses its tune split for both roles.

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
