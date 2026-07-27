Components
==========

soma separates frozen foundation-model encoding from the trainable downstream
model. Each block has a stable interface, so it can be selected independently
without rewriting the workflow around it.

* :doc:`encoders` turn images into frozen representations.
* :doc:`modeling` is the overview for choosing a downstream prediction path.
* :doc:`aggregators` pool bags of tile features for slide- and patient-level tasks.
* :doc:`decoders` recover dense spatial outputs from feature grids.

.. toctree::
   :maxdepth: 1
   :hidden:

   encoders
   modeling
   aggregators
   decoders
