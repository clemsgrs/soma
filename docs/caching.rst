Caching
=======

The shared cache reuses tiling and frozen features across compatible runs.
Run-specific checkpoints, predictions, and reports live in :doc:`outputs`.

The cache configuration is :class:`soma.config.CacheConfig`.

.. autoclass:: soma.config.CacheConfig
   :members:

Cache reuse rules
-----------------

Tiling is reused per sample when preprocessing matches. Features also require
matching encoder and execution settings; patient embeddings are reused per
patient. Datasets with overlapping samples can share their payloads while
extracting only the missing samples.

Sample identity includes ``sample_id``, ``image_path``, ``mask_path``, and the
optional ``spacing_at_level_0`` declaration. ``mask_path`` is the tissue or
annotation-sampling mask; segmentation's ``label_mask_path`` is excluded from
ordinary feature identity but participates in the separate ROI-sampling cache.
Delete affected caches before replacing source files in place.

On a complete cache hit, soma does not load the foundation encoder. On a miss,
slide2vec extracts and persists features through its public ``Model`` interfaces.
For pooled slide encoders, soma passes the artifacts from ``Model.embed_tiles``
to ``Model.aggregate_tiles``. soma owns cache identity and completeness checks;
slide2vec owns extraction progress and writes to the selected directory.

Validation
----------

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
  - ``dense_grid``: 3-D dense segmentation/detection grids with shape stored in sidecars

Extraction geometry
-------------------

Feature caches record the requested tile size, the size read from the slide,
and the effective encoder input size. Only effective input size is validated
on reuse: soma can derive it from configuration and the encoder registry without
loading the model.

A mismatch raises ``CacheGeometryMismatch`` with both sizes. Delete the cache
directory to re-extract or choose a different cache root. This prevents reuse of
features whose spatial extent differs from the current encoder input.

Geometry checks cannot detect pixel-processing changes that preserve size,
such as a different interpolation kernel or photometric transform, and soma
does not invalidate caches by dependency version. Delete the affected cache
directories, or select a fresh ``cache.root_dir``, when upgrading slide2vec
across such a change.

Dense cache key
---------------

A dense cache key depends on:

- the encoder name, compute precision and storage dtype;
- the target size, patch size and pad mode;
- the dense input mode and, for sliding-window runs, the window size and overlap;
- the preprocessing settings, including the requested spacing;
- the annotation-sampling settings, for slide-level segmentation;
- the feature kind and the attention selection, for attention grids.

The key does not depend on the default ``input_size`` of the encoder registry.
A dense extraction feeds the encoder the configured target or window, so a
change of the registry default does not change the features. Pooled tile,
slide, patient and hierarchical keys still include the registry ``input_size``.

Migrate a legacy dense cache
----------------------------

Dense caches written by soma 1.16.0 or earlier include the registry
``input_size`` in their key. soma no longer finds them. The features are still
valid, so rename the caches instead of extracting again:

.. code-block:: bash

   # Dry run: print what would change.
   python scripts/migrate_dense_cache_keys.py /path/to/feature_cache

   # Rename the folders and update their metadata.
   python scripts/migrate_dense_cache_keys.py /path/to/feature_cache --apply

The script reads only the files recorded in each cache. It loads no encoder and
needs no GPU. For each legacy cache under ``dense/`` and ``dense_image/`` it:

- renames the folder to the new key;
- updates ``cache_key``, ``execution`` and the sample identities in
  ``cache_metadata.json``;
- writes ``MIGRATION.json`` with the old key and the date.

The script refuses a cache when the target folder already exists, and leaves
it unchanged. Remove the target folder if it is empty, then run the script
again. Do not migrate a cache while a job uses it.

GPU count is not part of the dense cache key. Multi-GPU sharding can change
grid bytes within slide2vec's tolerance contract, so a dense cache resumed at a
different GPU count is equivalent but not byte-identical.
