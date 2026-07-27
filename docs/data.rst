Data
====

Every soma experiment starts with images, supervision, and a split definition.
The data layer keeps those inputs explicit while preprocessing converts them
into the samples consumed by foundation-model encoders.

* :doc:`dataset` defines samples, labels, and metadata.
* :doc:`curation` converts supported public datasets into soma manifests.
* :doc:`preprocessing` controls tiling, spacing, tissue masking, and geometry.

.. toctree::
   :maxdepth: 1
   :hidden:

   dataset
   curation
   preprocessing
