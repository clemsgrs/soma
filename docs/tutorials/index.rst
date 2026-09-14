Tutorials
=========

Each walkthrough runs one modeling path end to end on a small synthetic dataset.
Pick the row that matches your data and task.

Tile-level
----------

Classify pre-cropped tiles with a frozen encoder and a task head, as in EVA.
Each tile produces one feature vector, so this path needs neither tissue
masking and tiling nor a MIL aggregator.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Tile encoder + task head
     - A tile-level encoder emits one vector per tile; the classification head
       consumes it as-is (``aggregator=None``, no ``PreprocessingConfig``). The
       walkthrough runs binary and multiclass on the **same** extracted features.
     - :doc:`Tile-level <walkthrough-tile-level>`

Slide-level
-----------

Predict slide- or patient-level outcomes with a classification, regression, or
survival head. Extracted features can be reused across tasks; each task needs
its own labels and head.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Tile encoder + MIL aggregator
     - A tile-level encoder emits one vector per tile; an :doc:`aggregator </aggregators>`
       (ABMIL, CLAM, TransMIL, …) pools the bag into a slide vector before the head.
     - :doc:`Tile encoder + MIL <walkthrough-slide-mil>`
   * - Slide-level encoder
     - A slide-native encoder emits one vector per slide; the task head consumes
       it as-is, with no aggregator.
     - :doc:`Slide encoder <walkthrough-slide-encoder>`

Detection
---------

Predict cell or nucleus centroids from a frozen encoder's token grid. A neural
decoder learns a per-class peak heatmap; see the :doc:`detection reference
</detection>`.

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

Segmentation
------------

Assign a class to each pixel using frozen encoder outputs; see the
:doc:`segmentation reference </segmentation>`.

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

.. toctree::
   :maxdepth: 1
   :hidden:

   walkthrough-tile-level
   walkthrough-slide-mil
   walkthrough-slide-encoder
   walkthrough-detection
   walkthrough-segmentation
   walkthrough-attention-segmentation
   walkthrough-composite
