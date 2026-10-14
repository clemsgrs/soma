Components
==========

A soma model is a frozen foundation-model encoder followed by a trained downstream
model: an aggregator for a set or a decoder for a grid, then a task head. Each part
is a ``{name, params}`` leaf, chosen from a registry independently of the others.
``soma.components(kind)`` lists the registered entries of a kind with their params
and defaults, and :doc:`extending` shows how to add your own.

- :doc:`encoders`: the frozen tile-, slide- and patient-level encoders
- :doc:`modeling`: which downstream path fits your features and task
- :doc:`aggregators`: the MIL aggregators that pool a bag of tile features
- :doc:`decoders`: the decoders that map a dense feature grid back to pixels

The task heads are listed in :doc:`tasks`.

.. toctree::
   :maxdepth: 1
   :hidden:

   encoders
   modeling
   aggregators
   decoders
