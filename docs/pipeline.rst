Pipeline
========

``dataset_type`` selects the input contract and model path for a run:

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Type
     - Input
     - Model path
   * - ``tile``
     - Image tile + scalar label
     - Tile encoder → task head
   * - ``slide``
     - Whole slide + scalar label
     - Tile encoder → MIL aggregator → head, or slide encoder → head
   * - ``patient``
     - Patient-grouped slides + scalar label
     - Pretrained patient encoder → head *(experimental)*
   * - ``segmentation``
     - Image + mask
     - Frozen dense encoder → decoder or pixel classifier → segmentation head
   * - ``detection``
     - Image + point annotations
     - Frozen dense encoder → decoder → detection head
   * - ``spatial_expression``
     - Spot tile + gene-expression vector
     - Tile encoder → closed-form Ridge+PCA regression probe

The main configuration object is :class:`soma.config.PipelineConfig`:

.. autoclass:: soma.config.PipelineConfig
   :members:

Slide-level example
-------------------

The :doc:`quickstart <getting-started>` covers tile classification. A slide
run adds an MIL aggregator between the tile encoder and task head:

.. code-block:: python

   from soma import AggregatorConfig, EncoderConfig, EvalConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       encoder=EncoderConfig(name="dinov2-vitb14"),
       aggregator=AggregatorConfig(name="abmil"),
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["accuracy"]),
       training=TrainingConfig(epochs=20, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

Run outputs
-----------

The run directory layout is described in :doc:`outputs`.
