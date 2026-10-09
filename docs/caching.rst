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

On a complete cache hit, soma does not load the foundation encoder, except once
after a slide2vec upgrade (see `Feature identity`_). On a miss,
slide2vec extracts and persists features through its public ``Model`` interfaces.
For pooled slide encoders, soma passes the artifacts from ``Model.embed_tiles``
to ``Model.aggregate_tiles``. soma owns cache identity and completeness checks;
slide2vec owns extraction progress and writes to the selected directory.

A sample counts as cached only when an extraction committed its signature to
``cache_metadata.json``. Features found on disk without a committed signature,
for example after an interrupted run, are never signed from the cache manifest.
soma hands them back to slide2vec, which checks them or extracts them again.

Pre-cropped images
------------------

For ``dataset_type="tile"``, soma passes every image that needs work to one
``Model.embed_images`` call, whatever the dataset size:

1. soma resolves the cache. An image counts as cached when its signature is
   committed and both ``<sample_id>.pt`` and ``<sample_id>.meta.json`` exist.
   One directory listing decides this; soma reads no sidecar on a cache hit.
2. soma removes the committed signatures of the other images from
   ``cache_metadata.json``. After a failure, these images stay unsigned until
   an extraction succeeds, even if a later run requests another source path for
   them or only a subset of the dataset.
3. slide2vec reuses an image whose payload and sidecar record its source path
   and the current feature identity. It encodes every other image. soma passes
   ``on_image_mismatch="reencode"``, so an image recorded for another source
   path is replaced.
4. soma writes the feature dimension, the signatures and, for a new cache, the
   feature identity in one final update of ``cache_metadata.json``.

If the run stops, the next run resumes inside slide2vec: completed images are
not encoded again. ``cache_metadata.json`` is always replaced atomically, so a
failed write leaves the previous record readable.

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

Geometry checks cannot detect pixel-processing changes that preserve size. A
change of the encoder's image transform is detected by the feature identity
check below. A change in how slide2vec reads a tile from the slide, such as a
different interpolation kernel, is not detected: delete the affected cache
directories, or select a fresh ``cache.root_dir``, when upgrading slide2vec
across such a change.

Feature identity
----------------

A pooled cache key covers the encoder name, the output variant, the precision,
the registry ``input_size`` and the spacing. It does not cover the encoder's
image transform. A slide2vec release can change that transform (normalization
statistics, resize, crop) and leave the key unchanged.

slide2vec records the feature identity of every embedding it writes: the encoder,
the output variant, the precision, the tile geometry and the image transform.
When soma writes the first features of a tile, image, hierarchical, slide or
patient cache, it copies this identity into ``cache_metadata.json`` together
with the slide2vec version. For example:

.. code-block:: json

   {
     "feature_identity": {
       "slide2vec_version": "7.0.0",
       "identity": {
         "encoder_name": "uni2",
         "output_variant": "default",
         "precision": "fp16",
         "feature_dtype": "fp16",
         "transform": {
           "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
           "resize": null,
           "center_crop": null
         }
       }
     }
   }

A slide or patient cache records the identity of the tile cache it is aggregated
from, so its record includes the transform of the tile encoder. soma aggregates
only tiles whose cache records an identity.

When a pooled cache already holds features, soma checks the record before it
reuses the cache or adds samples to it:

- The recorded slide2vec version is the installed one. soma reuses the cache and
  does not load the encoder.
- The versions differ. soma loads the encoder once, on CPU, and compares the
  recorded identity with the one the installed slide2vec produces.

  - No field differs: soma reuses the cache and records the installed version,
    so later runs do not load the encoder.
  - A field differs, or the record lacks a field the installed slide2vec
    requires (reported as ``MISSING_FIELD``): the run stops with
    ``CacheFeatureIdentityMismatch``, which names the fields. Set
    ``cache.on_identity_mismatch: reextract`` to delete the cache and extract it
    again instead.

A pooled cache that holds features but records no feature identity, such as a
cache written by soma 1.17.0 or earlier, cannot be verified. soma logs a warning,
deletes the cache directory and extracts it again. soma never takes an identity
from existing sidecars to approve such a cache. A cache without features starts
a new record.

A run that stops before its first commit leaves a pending record (``identity``
is ``null``). The next run with the same slide2vec version keeps the cache and
hands its unsigned features to slide2vec. If the slide2vec version changed, soma
extracts the cache again. If slide2vec writes features without a feature
identity, soma raises ``MissingFeatureIdentity`` and does not commit them.

The settings ``cache.commit_every`` and ``cache.on_unrecorded_identity`` were
removed. soma rejects them as unknown keys, so delete them from your configs.

``reextract`` deletes the whole cache directory, including the features of
samples that the current dataset does not use. Do not use it while another job
reads the cache.

The check runs only when the slide2vec version changes. A transform that changes
without a slide2vec upgrade, for example after an upgrade of ``timm``, is not
detected. Dense caches are not checked.

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
