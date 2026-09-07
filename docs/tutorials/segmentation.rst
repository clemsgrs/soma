Segmentation
============

Assign a class to each pixel using frozen encoder outputs. Choose a neural
decoder on patch features or a per-pixel classifier on attention features.
Either path can combine several encoders.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Neural decoder *(default)*
     - A lightweight conv decoder predicts a per-pixel class map; the
       :class:`~soma.tasks.segmentation.SegmentationHead` trains it with cross-entropy +
       soft-Dice.
     - :doc:`Segmentation <walkthrough-segmentation>`
   * - :doc:`Decoder-free pixel classifier </decoders/pixel-classifier>`
     - Fit XGBoost, random forest, logistic regression, or an MLP to dense
       features; the default input is per-head attention.
     - :doc:`Attention-based segmentation <walkthrough-attention-segmentation>`
   * - :doc:`Multi-encoder composite </encoders/composite>`
     - Concatenate the dense outputs of several foundation models into one
       per-position vector.
     - :doc:`Composite <walkthrough-composite>`

.. seealso::

   The dense contract, config, metric, and outputs are on the :doc:`Segmentation reference
   </segmentation>`.
