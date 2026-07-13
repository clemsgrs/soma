Slide-level
===========

Predict a slide- or patient-level label from a bag of tile features. You build the slide
representation one of two ways, then attach a task head. The walkthrough runs both paths
end to end and shows that classification, regression, and survival are just a different
head on the **same** extracted features.

.. list-table::
   :header-rows: 1
   :widths: 30 48 22

   * - Method
     - Summary
     - Walkthrough
   * - Tile encoder + MIL aggregator
     - A tile-level encoder emits one vector per tile; an :doc:`aggregator </aggregators>`
       (ABMIL, CLAM, TransMIL, …) pools the bag into a slide vector before the head.
     - :doc:`Slide-level <walkthrough-slide-level>`
   * - Slide-level encoder
     - A slide-native encoder emits one vector per slide directly — no aggregator; the task
       head consumes it as-is.
     - :doc:`Slide-level <walkthrough-slide-level>`

First slide-level run
---------------------

Create ``dataset.csv`` with one row per slide and its target label:

.. code-block:: text

   sample_id,image_path,label
   slide_001,/data/slides/slide_001.svs,0
   slide_002,/data/slides/slide_002.svs,1
   slide_003,/data/slides/slide_003.svs,0

Create ``splits.csv`` with explicit train, tune, and test ownership:

.. code-block:: text

   sample_id,split
   slide_001,train
   slide_002,tune
   slide_003,test

Start from the complete slide-classification configuration below. It uses UNI2
to encode tiles and ABMIL to learn a slide representation:

.. literalinclude:: ../../examples/slide_binary_classification.yaml
   :language: yaml

Copy it beside the manifests as ``slide_binary_classification.yaml``, adjust the
paths, then run:

.. code-block:: console

   soma slide_binary_classification.yaml

To compare components, keep the manifests, splits, preprocessing, task, metrics,
and training settings fixed. Change ``encoder.name`` between ``uni2`` and
``virchow2``; change ``aggregation.name`` between ``mean_pool`` and ``abmil``.
Feature caches are reused between aggregator runs for the same encoder.

See :doc:`/encoders` for model access and native spacing, :doc:`/preprocessing`
for tissue masking and tiling, and :doc:`/outputs` for the resulting metrics and
report.

.. seealso::

   Reference docs — task heads and metrics on :doc:`/tasks`, :doc:`/classification`,
   :doc:`/regression`, and :doc:`/survival`; the aggregator zoo on :doc:`/aggregators`.
