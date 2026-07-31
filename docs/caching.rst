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
- Sample identity uses ``sample_id``, ``image_path``, and ``mask_path``. Replace
  files in place only after deleting the affected cache.
- By default, feature cache validation checks metadata identity and payload
  existence. Dense grids also validate their per-sample sidecar metadata
  (feature dimension, grid shape, target/encoded geometry, and — for
  ``cls_attention`` grids — the attention selection) before reuse, without
  loading the tensor payload. Set
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

Extraction geometry
-------------------

A feature cache also records the geometry it was extracted under
(``docs/adr/0008-cache-records-geometry-and-does-not-stamp-extraction-semantics.md``):
the tile size that was **requested**, the size actually **read** off each slide, and the
**effective encoder input** — the geometry of the tensor handed to the encoder.

Only the last is validated on reuse, because it is the one soma can derive from config
plus slide2vec's registry without loading a model, and the one whose change means the
cached features are registered to a different extent. A 224 px request reaches a
variable-input encoder at 224 px under its shipped transform and at 512 px under a
normalization-only one; reusing features across that shift would train on grids that do
not mean what the run thinks they mean. That is a **hard error**, not ordinary
incompleteness: soma raises ``CacheGeometryMismatch`` naming both sizes rather than
silently recomputing a feature set that may be hundreds of gigabytes. Delete the cache
directory to re-extract, or point the run at a different cache root.

What this deliberately does **not** catch is a change in *how* pixels are produced at
unchanged sizes — a different interpolation kernel, a resize moving stage, a corrected
photometric recipe. Those are not sizes, so no geometry record can see them; delete
caches by hand when upgrading slide2vec across such a change. Caches written before the
record exists have nothing to disagree with and stay reusable.

See also
--------

Run-directory layout and experiment artifacts are documented in
:doc:`outputs`.
