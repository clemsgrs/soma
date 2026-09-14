Modeling
========

Modeling begins with frozen foundation-model features and ends with predictions
for the research task; only the downstream path is trained.

Choose a modeling path
----------------------

The structure of the encoded features and the desired prediction determine the
downstream path. Each path uses the same task and training interfaces.

.. list-table::
   :header-rows: 1
   :widths: 27 39 34

   * - Feature structure
     - Modeling path
     - Typical outputs
   * - One feature vector per sample
     - Apply a task-specific predictor directly.
     - Classification or regression; survival for slide and patient embeddings.
   * - A bag of tile features per slide
     - Use :doc:`aggregators` to produce one slide-level representation.
     - Classification, regression, or survival predictions.
   * - A dense feature grid per tile or region
     - Use :doc:`decoders` to recover spatial detail before prediction.
     - Segmentation masks or detection predictions.

Tasks and training
------------------

Patient-level pipelines use a frozen patient encoder and a task head, with no
trainable aggregator. Segmentation also supports a decoder-free
:doc:`pixel classifier <decoders/pixel-classifier>`.

The :doc:`tasks` page defines the prediction target, loss, and compatible
metrics. :doc:`training` controls optimization and checkpoint selection, then
executes the folds defined by the data splits. These contracts stay consistent
when an aggregator, decoder, or task head is replaced.

Explore a path
--------------

The :doc:`tutorials <tutorials/index>` run each path end to end on a small
synthetic dataset.
