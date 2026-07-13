Reference
=========

Use this section when you already know what you are building and need an exact
configuration field, component contract, task behavior, or run artifact. For a
first experiment, start with :doc:`getting-started`; for the architecture and
available paths, start with :doc:`pipeline`.

Configure and run
-----------------

* :doc:`configuration` — canonical YAML and typed Python configuration.
* :doc:`api` — public Python entry points.
* :doc:`cli` — commands for experiments, discovery, benchmarks, and leaderboards.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Configure and run

   configuration
   api
   cli

Data and components
-------------------

* :doc:`dataset` and :doc:`preprocessing` define inputs and image preparation.
* :doc:`encoders`, :doc:`aggregators`, and :doc:`decoders` define the swappable
  model stages.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Data and components

   dataset
   preprocessing
   encoders
   encoders/composite
   aggregators
   decoders
   decoders/pixel-classifier

Prediction tasks
----------------

* :doc:`tasks` gives the task-head catalog.
* The individual pages define labels, losses, predictions, and metrics for each
  output type.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Prediction tasks

   tasks
   classification
   regression
   survival
   segmentation
   detection

Results and operations
----------------------

* :doc:`training` and :doc:`evaluation` cover fitting and metric computation.
* :doc:`outputs`, :doc:`reporting`, and :doc:`caching` cover reproducibility and
  repeated experimentation.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Results and operations

   training
   evaluation
   outputs
   reporting
   caching
