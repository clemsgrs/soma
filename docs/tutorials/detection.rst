Detection
=========

Predict cell or nucleus centroids from a frozen encoder's token grid. A neural
decoder learns a per-class peak heatmap; choose a single encoder or combine
several encoders before the decoder.

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
     - :doc:`Detection <walkthrough-detection>`
   * - :doc:`Multi-encoder composite </encoders/composite>`
     - Concatenate the dense outputs of several foundation models into one
       per-position vector before the decoder.
     - :doc:`Composite <walkthrough-composite>`

The :doc:`pixel-classifier path </decoders/pixel-classifier>` currently supports
segmentation only.

.. seealso::

   The full contract, config, metric, and outputs are on the :doc:`Detection reference
   </detection>`. The benchmark reproduction is :doc:`/ocelot-detection-benchmark`.
