Caching and Output Layout
=========================

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
   * - Run outputs
     - No
     - Each experiment should still get a fresh result bundle

Output layout
-------------

The run directory contains the artifacts for a single experiment:

- saved config
- fold checkpoints
- predictions
- metrics
- summaries and HTML report

Treat the cache as reusable infrastructure and the run directory as the
immutable record of a specific experiment.
