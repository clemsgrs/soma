Slide-level
===========

Predict a slide- or patient-level label from a bag of tile features. You build the slide
representation one of two ways, then attach a task head — and classification, regression,
and survival are just a different head on the **same** extracted features. Each walkthrough
below runs one path end to end.

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
     - A slide-native encoder emits one vector per slide directly — no aggregator; the task
       head consumes it as-is.
     - :doc:`Slide encoder <walkthrough-slide-encoder>`

.. seealso::

   Reference docs — task heads and metrics on :doc:`/tasks`, :doc:`/classification`,
   :doc:`/regression`, and :doc:`/survival`; the aggregator zoo on :doc:`/aggregators`.
