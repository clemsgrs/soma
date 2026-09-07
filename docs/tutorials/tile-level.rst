Tile-level
==========

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

.. seealso::

   How the tile path differs from whole-slide bags is summarized on
   :doc:`/components`. Reference docs — task heads and metrics on :doc:`/tasks`
   and :doc:`/classification`. The packaged benchmark on this path is
   :doc:`/eva-patch-classification-benchmark`.
