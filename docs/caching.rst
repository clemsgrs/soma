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
- Cache metadata stores a normalized ``feature_type``:

  - ``tile``: 1-D embeddings for ``dataset_type="tile"``
  - ``bag``: 2-D WSI tile-bag embeddings
  - ``slide``: 1-D slide-level embeddings
  - ``patient``: 1-D patient-level embeddings
  - ``hierarchical``: 3-D hierarchical embeddings

- Two datasets can therefore reuse the same cached feature payload for shared
  samples while still recomputing non-overlapping samples.
- Cache hits do not replace the run directory, which still records one
  immutable experiment result.

See also
--------

Run-directory layout and experiment artifacts are documented in
:doc:`outputs`.
