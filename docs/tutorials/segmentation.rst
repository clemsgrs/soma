Segmentation
============

Per-pixel semantic segmentation on a frozen-encoder token grid. The default path upsamples
the grid with a neural decoder; the rows below are the substrate and component choices a
segmentation run can swap in, with the walkthrough for each where one exists.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Neural decoder *(default)*
     - A lightweight conv decoder regresses a per-pixel class map; the
       :class:`~soma.tasks.segmentation.SegmentationHead` trains it with cross-entropy +
       soft-Dice.
     - :doc:`Dense prediction <walkthrough-dense>`
   * - :doc:`Decoder-free pixel classifier </decoders/pixel-classifier>`
     - Swap the neural decoder for a per-pixel classifier on the encoder's own per-head
       attention (XGBoost / RF / logistic / MLP) — no decoder, no checkpoints.
     - :doc:`Attention-based segmentation <walkthrough-attention-segmentation>`
   * - :doc:`Multi-encoder composite </encoders/composite>`
     - Concatenate the dense outputs of several foundation models into one richer
       per-position vector (the paper's +7.95% mean Dice headline).
     - :doc:`Composite <walkthrough-composite>`

.. seealso::

   The dense contract, config, metric, and outputs are on the :doc:`Segmentation reference
   </segmentation>`.
