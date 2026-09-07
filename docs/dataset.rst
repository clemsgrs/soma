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
  | Required columns: ``sample_id``, ``image_path``, ``label``.
  | Optional columns: ``mask_path`` (pre-computed tissue mask, valid for every ``dataset_type``), ``patient_id`` (required for ``dataset_type="patient"``).
  | Additional, unrecognized columns are carried along as per-sample metadata.

Dense-supervision manifests (``dataset_type="segmentation"`` / ``"detection"``)
replace the scalar ``label`` with a per-sample supervision file:

- **Segmentation** uses ``label_mask_path`` — a per-sample label mask. It is distinct
  from ``mask_path`` (the optional tissue mask): a segmentation row may carry both.
  Segmentation manifests written before soma 1.11 used ``mask_path`` for the label
  mask; the loader rejects those explicitly — regenerate them with their curator.
- **Detection** uses ``points_path`` — a per-sample point file
  (:class:`soma.dataset.DetectionManifest`), a CSV of object centroids with
  ``x, y, class`` columns (headerless ``x,y,class`` — OCELOT's format — or a
  2-column ``x,y`` for a single class). Points are stored in **level-0** pixels;
  an optional finite positive per-sample ``spacing_at_level_0`` column declares the
  source image's level-0 µm/px. It is required for flat PNG/JPEG dense extraction. See
  :doc:`detection` for the full column contract.

For ``dataset_type="spatial_expression"``, use ``target_index`` instead of
``label``. Each index selects a row in ``targets.npy`` (shape
``[n_rows, n_genes]``); ``genes.json`` lists the genes in column order. Both
sidecars must sit beside ``dataset.csv`` and are validated by
:class:`soma.dataset.SpatialExpressionManifest`.

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

Dataset checksums exclude exactly the storage-location columns
``image_path``, ``mask_path``, ``label_mask_path``, and ``points_path``. Relocating
otherwise identical images, tissue masks, label masks, or point files therefore does
not change data identity. Every other column remains part of the checksum, including
scalar supervision, sample metadata, and sample IDs; fold, split, and membership
assignments remain part of the separate splits checksum.

When artifact contents must contribute to identity, a preparer should compute the
checksum and add an explicit ``<path_column>_sha256`` column such as
``image_path_sha256`` or ``label_mask_path_sha256``. These columns are ordinary semantic
metadata and are hashed. soma does not open referenced artifacts or derive their
checksums implicitly.
