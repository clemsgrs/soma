Modeling
========

Modeling begins with frozen foundation-model features and ends with predictions
for the research task. soma keeps the foundation-model encoder frozen by design
and trains only the downstream path. Compatible experiments can therefore reuse
the same encoded features.

Choose a modeling path
----------------------

The structure of the encoded features and the desired prediction determine the
downstream path. In broad terms, features pass through an optional aggregator or
decoder, then a task head produces the prediction. Each path uses the same task
and training interfaces.

.. list-table::
   :header-rows: 1
   :widths: 27 39 34

   * - Feature structure
     - Modeling path
     - Typical outputs
   * - One feature vector per sample
     - Apply a task-specific predictor directly.
     - Tile or region classification and regression.
   * - A bag of tile features per slide or patient
     - Use :doc:`aggregators` to learn one slide- or patient-level representation.
     - Classification, regression, or survival predictions.
   * - A dense feature grid per tile or region
     - Use :doc:`decoders` to recover spatial detail before prediction.
     - Segmentation masks or detection predictions.

Tasks and training
------------------

The :doc:`tasks` page defines the prediction target, loss, and compatible
metrics. :doc:`training` controls optimization and checkpoint selection, then
executes the folds defined by the data splits. These contracts stay consistent
when an aggregator, decoder, or task head is replaced.

Explore a path
--------------

Use the workflow guides for a complete view of each modeling family:

* :doc:`slide-level workflow <tutorials/slide-level>` — aggregate tile features
  for slide-level prediction.
* :doc:`segmentation workflow <tutorials/segmentation>` — decode dense feature
  grids into masks.
* :doc:`detection workflow <tutorials/detection>` — decode dense feature grids
  into spatial detections.
