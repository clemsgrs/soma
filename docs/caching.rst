Caching
=======

The shared cache reuses tiling and frozen features across compatible runs.
Run-specific bundles, predictions, and metrics live in :doc:`outputs`.

The cache configuration is ``features.cache``, a :class:`soma.config.CacheOptions`.

.. autoclass:: soma.config.CacheOptions

Cache reuse rules
-----------------

Tiling is reused per sample when preprocessing matches. Features also require
matching encoder and execution settings; patient embeddings are reused per
patient. Datasets with overlapping samples can share their payloads while
extracting only the missing samples.

Sample identity includes ``sample_id``, ``image_path``, ``mask_path``, and the
optional ``spacing_at_level_0`` declaration, read from each row of the image
manifest. ``mask_path`` is the tissue or annotation-sampling mask; segmentation's
``label_mask_path`` is excluded from ordinary feature identity but participates in
the separate ROI-sampling cache. Delete affected caches before replacing source
files in place.

Annotation sampling caches each slide's ROI origins and mask crops under the
slide's identity, with its ``label_mask_path``. An ROI's grid is keyed by the ROI
id (``<slide>__x<x>_y<y>``) over its slide's image row, so a relaunch over the same
slides reuses both the ROIs and their grids.

On a complete cache hit, soma does not load the foundation encoder, except once
after a slide2vec upgrade (see `Feature identity`_). It still checks the encoder's
code, which needs no encoder. On a miss,
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

For a tile-level run (one image per row, a tile encoder without an aggregator),
soma passes every image that needs work to one
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
  ``features.cache.validate_payloads: true`` to load cached tensors and verify rank,
  feature dimension, and dense grid shape before accepting a cache hit. This
  catches corrupt or wrong-shaped payloads, but it adds I/O proportional to the
  number of cached feature files.
- Cache metadata stores a normalized ``feature_type``:

  - ``tile``: 1-D embeddings of pre-cropped images (tile-level runs)
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
directories, or select a fresh ``features.cache.root_dir``, when upgrading slide2vec
across such a change.

.. _feature-identity:

Feature identity
----------------

A feature cache key covers the encoder name, the output variant, the precision and
the geometry. It does not cover the encoder's code or its image transform. An edit
to the encoder, or a slide2vec release, can change the features and leave the key
unchanged.

slide2vec records the feature identity of every embedding and grid it writes: the
encoder and its code, the output variant, the precision, the geometry and the image
transform. When soma writes the first features of a cache, pooled or dense, it
copies this identity into ``cache_metadata.json`` together with the slide2vec
version. For example:

.. code-block:: json

   {
     "feature_identity": {
       "slide2vec_version": "7.1.4",
       "identity": {
         "encoder_name": "uni2",
         "encoder_code": {
           "object": "slide2vec.encoders.models.uni.UNI2",
           "fingerprint": "9b7499239d726642d6ab623084cfb5fe66935755ad062176a9193685cf14441b",
           "source": "MahmoodLab/UNI2-h"
         },
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

``encoder_code`` names the encoder's class (``object``) and its ``fingerprint``:
the SHA-256 of the file that defines the class, or the version of the installed
plugin package that holds it. A slide or patient cache records the identity of
the tile cache it is aggregated from, so its record also holds the transform and
the code of the tile encoder (``tile_encoder_code``). soma aggregates only tiles
whose cache records an identity. A dense record also holds the read settings of
one sample; they are not compared.

When a cache already holds features, soma checks the record before it reuses the
cache or adds samples to it. Two checks run.

- **The encoder's code, on every run.** soma compares the recorded
  ``encoder_code`` with the code the encoder registry holds now, and, for a slide
  or patient cache, the recorded ``tile_encoder_code`` with the code of the tile
  encoder. This check reads the registry and hashes one file: it does not load
  the encoder.
- **The whole identity, after a slide2vec upgrade.** When the recorded slide2vec
  version is not the installed one, soma loads the encoder once, on CPU, and
  compares the recorded identity with the one the installed slide2vec produces.
  When no field differs, soma records the installed version, so later runs skip
  this comparison.

When a check finds a difference, the run stops with
``CacheFeatureIdentityMismatch``. The error names the fields, or, for the code,
the encoder, its class and both fingerprints. Set
``features.cache.on_identity_mismatch: reextract`` to delete the cache and
extract it again instead.

A record that lacks a field the check needs, such as ``encoder_code`` in a cache
written before slide2vec 7.1.3, cannot be verified. soma treats it like a
difference: by default the run stops with an error that says the cache predates
the field, and ``reextract`` extracts the cache again.

An encoder whose class has no file, for example a class defined in a notebook
cell, records ``fingerprint: null``. soma reuses its cache and warns once per
encoder that it cannot detect changes to the encoder's code. Define the encoder
in a ``.py`` file.

A cache that holds features but records no feature identity cannot be verified:

- A pooled cache is a miss. soma logs a warning, deletes the cache directory and
  extracts it again.
- A dense cache, written before dense caches recorded an identity, follows
  ``on_identity_mismatch``. By default the run stops with an error that names the
  cache and says it predates the record. Set ``reextract`` to extract it again.
  soma never deletes the grids without this setting.

soma never takes an identity from existing sidecars to approve such a cache. A
cache without features starts a new record.

A run that stops before its first commit leaves a pending record (``identity``
is ``null``). The next run with the same slide2vec version keeps the cache and
hands its unsigned features to slide2vec. If the slide2vec version changed, soma
extracts the cache again. If slide2vec writes features without a feature
identity, soma raises ``MissingFeatureIdentity`` and does not commit them.

``reextract`` deletes the whole cache directory, including the features of
samples that the current dataset does not use. Do not use it while another job
reads the cache.

Some changes stay undetected:

- the encoder's weights, for example new weights at the same path;
- code that the encoder class inherits or imports from other files;
- a transform that changes without a slide2vec upgrade or an edit to the
  encoder's file, for example after an upgrade of ``timm``.

After such a change, delete the affected cache directories, or select a fresh
``features.cache.root_dir``.

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

GPU count is not part of the dense cache key. Multi-GPU sharding can change
grid bytes within slide2vec's tolerance contract, so a dense cache resumed at a
different GPU count is equivalent but not byte-identical.
