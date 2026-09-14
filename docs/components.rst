Components
==========

soma separates frozen foundation-model encoding from the trainable downstream
model. Each block has a stable interface, so it can be selected independently
without rewriting the workflow around it.

.. toctree::
   :maxdepth: 1

   encoders
   modeling
   aggregators
   decoders
