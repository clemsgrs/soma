Dataset
=======

The first thing ``soma`` needs is a pair of CSV manifests:

- ``dataset.csv`` describes the samples, labels, and optional metadata.
- ``splits.csv`` assigns each sample to a fold and split.

These files are the contract between your data and the pipeline. They are
validated when the dataset and splits are loaded, and they define which samples
are used for training, tuning, and testing.

Dataset format
--------------

``dataset.csv``
  | Required columns: ``sample_id``, ``image_path``, ``label``.
  | Optional columns: ``mask_path`` (pre-computed tissue mask, valid for every ``dataset_type``), ``patient_id`` (required for ``dataset_type="patient"``).
  | Any additional columns are carried along as per-sample metadata.

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

.. _semantic-manifest-identity:

Semantic manifest identity
--------------------------

Experiment and leaderboard dataset checksums describe the semantic values in the
selected ``dataset.csv`` rows. They exclude exactly the storage-location columns
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

Splits format
-------------

``splits.csv``
  | Required columns: ``sample_id``, ``split``.
  | Optional column: ``fold`` (integer). Omit it for a single train/tune/test split; include it with distinct values (0, 1, 2, …) for cross-validation.
  | Valid split names: ``train``, ``tune``, or any name starting with ``test`` (e.g. ``test``, ``test_external``).
  | Every fold must contain at least one test split.

Practical notes
---------------

- Keep ``sample_id`` stable across both files.
- Use ``patient_id`` when you want patient-level evaluation or aggregation.
- Prefer explicit test split names when you have more than one held-out cohort.
- Keep at least one held-out test split in every fold so comparisons are
  reproducible.

For a quick example of how these manifests fit into a full run, see
:doc:`getting-started`.
