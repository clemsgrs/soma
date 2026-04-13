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

- Tiling is reused when preprocessing matches.
- Features are reused when the encoder and geometry match.
- Cache hits do not replace the run directory, which still records one
  immutable experiment result.

See also
--------

Run-directory layout and experiment artifacts are documented in
:doc:`outputs`.
