Tile-level
===========

Predict a class label for a single tile. Each sample is an already-cropped tile
image — the path behind patch-classification benchmarks like EVA. A tile encoder
maps each tile to one vector and a task head classifies it directly, so this is
the simplest path in soma: **no tissue masking or tiling** (the inputs are already
tiles) and **no MIL aggregator** (there is no bag to pool).

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
