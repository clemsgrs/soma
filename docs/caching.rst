Caching
=======

The cache keeps repeated experiments inexpensive. Treat it as shared
infrastructure across sweeps, not as part of any single run.

The cache configuration is :class:`soma.config.CacheConfig`.

.. autoclass:: soma.config.CacheConfig
   :members:

What the cache stores
---------------------

.. list-table::
   :header-rows: 1

   * - Cache layer
     - Reused across runs
     - Why it matters
   * - Tiling
     - Yes, when preprocessing matches
     - Avoids re-reading whole-slide images
   * - Features
     - Yes, when encoder and geometry match
     - Avoids re-embedding the same data

Cache reuse rules
-----------------

The shared cache stores reusable upstream artifacts such as tiling and feature
extraction.

- Tiling payloads are reused per sample when preprocessing matches.
- Tiling reuse keys include sample identity
  ``(sample_id, image_path, mask_path)`` plus resolved preprocessing settings.
- Feature payloads are reused per sample (or per patient for patient-level
  embeddings) when the encoder and preprocessing match.
- Feature reuse keys include sample identity
  ``(sample_id, image_path, mask_path)`` plus encoder/preprocessing/execution
  settings.
- By default, sample identity uses paths only. Set
  ``cache.fingerprint_files: true`` to include SHA-256 hashes of slide and
  mask file contents in per-sample cache identity. This is safer when files
  may be replaced in place, but it requires reading each input file during
  cache resolution and can be expensive for large WSI cohorts.
- By default, feature cache validation checks metadata identity and payload
  existence. Dense grids also validate their per-sample sidecar metadata
  (feature dimension, channel axis, grid shape, target/encoded geometry, and
  dense input mode) before reuse, without loading the tensor payload. Set
  ``cache.validate_payloads: true`` to load cached tensors and verify rank,
  feature dimension, and dense grid shape before accepting a cache hit. This
  catches corrupt or wrong-shaped payloads, but it adds I/O proportional to the
  number of cached feature files.
- Cache metadata stores a normalized ``feature_type``:

  - ``tile``: 1-D embeddings for ``dataset_type="tile"``
  - ``bag``: 2-D WSI tile-bag embeddings
  - ``slide``: 1-D slide-level embeddings
  - ``patient``: 1-D patient-level embeddings
  - ``hierarchical``: 3-D hierarchical embeddings
  - ``dense_grid``: 3-D dense segmentation grids with shape stored in sidecars

- Two datasets can therefore reuse the same cached feature payload for shared
  samples while still recomputing non-overlapping samples.
- Cache hits do not replace the run directory, which still records one
  immutable experiment result.

See also
--------

Run-directory layout and experiment artifacts are documented in
:doc:`outputs`.
