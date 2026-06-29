Detection
=========

Point detection — cell / nucleus centroids — on a frozen-encoder token grid. The default
path regresses a per-class peak heatmap with a neural decoder; the rows below are the
substrate and component choices that path can swap in, with the walkthrough for each where
one exists.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Neural decoder *(default)*
     - A lightweight conv decoder regresses a per-class peak heatmap; the
       :class:`~soma.tasks.detection.DetectionHead` reads points back out and scores
       **F1@δ**.
     - :doc:`Dense prediction <walkthrough-dense>`
   * - :doc:`Multi-encoder composite </encoders/composite>`
     - Concatenate the dense outputs of several foundation models into one richer
       per-position vector before the decoder.
     - :doc:`Composite <walkthrough-composite>`
   * - :doc:`Decoder-free pixel classifier </decoders/pixel-classifier>`
     - A per-pixel classifier on the encoder's own attention. Not yet wired for detection —
       peaks need the decoder's spatial smoothing to stay separable.
     -

.. seealso::

   The full contract, config, metric, and outputs are on the :doc:`Detection reference
   </detection>`. The benchmark reproduction is :doc:`/ocelot-detection-benchmark`.
