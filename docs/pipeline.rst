How Soma works
==============

Soma makes pathology foundation models useful as part of a complete experiment,
not just as feature extractors. Give it images and labels; it prepares the data,
extracts foundation-model features, trains the appropriate prediction path, and
writes predictions and performance metrics in a reproducible run directory.

One workflow, swappable parts
-----------------------------

Every experiment uses the same sequence. Each stage has a stable configuration
boundary, so you can change one choice without rewriting the rest of the pipeline.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Stage
     - What it does
     - Main configuration
   * - Prepare images
     - Read tiles or slides, find tissue, create tile coordinates, and align labels.
     - :doc:`dataset`, :doc:`preprocessing`
   * - Encode
     - Turn tiles, slides, or patients into frozen foundation-model features.
     - :doc:`encoders`
   * - Aggregate or decode
     - Combine tile features with MIL for slide-level tasks, or turn dense features into spatial predictions.
     - :doc:`aggregators`, :doc:`decoders`
   * - Predict
     - Apply a classification, regression, survival, segmentation, or detection task head.
     - :doc:`tasks`
   * - Evaluate
     - Compute split-aware metrics and produce predictions, summaries, and reports.
     - :doc:`evaluation`, :doc:`outputs`

:class:`~soma.config.PipelineConfig` composes these stages. The YAML and Python
interfaces describe the same experiment, making configurations straightforward
to generate, validate, and compare programmatically.

Choose the path that matches your labels
----------------------------------------

``dataset_type`` selects the input contract and the middle of the model path:

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Type
     - Input and labels
     - Model path
   * - ``tile``
     - Image tile + scalar label
     - Tile encoder → task head
   * - ``slide``
     - Whole slide + scalar label
     - Tile encoder → MIL aggregator → head, or slide encoder → head
   * - ``patient``
     - Patient-grouped slides + scalar label
     - Patient encoder → task head *(experimental)*
   * - ``segmentation``
     - Image + mask
     - Dense encoder → decoder or pixel classifier → segmentation head
   * - ``detection``
     - Image + point annotations
     - Dense encoder → decoder → detection head
   * - ``spatial_expression``
     - Spot tile + gene-expression vector
     - Tile encoder → closed-form Ridge+PCA regression probe

Compose a slide-level workflow
------------------------------

The :doc:`quickstart <getting-started>` covers tile classification. A slide run
adds an MIL aggregator between the tile encoder and prediction head:

.. code-block:: python

   from soma import AggregatorConfig, EncoderConfig, EvalConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       encoder=EncoderConfig(name="uni2"),
       aggregator=AggregatorConfig(name="abmil"),
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       training=TrainingConfig(epochs=20, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

Swap or sweep one component
---------------------------

Because the stages are independent config fields, an encoder or aggregator
comparison changes only that field. Keep the dataset, splits, task, training,
and evaluation protocol fixed; generate one config per value you want to compare:

.. code-block:: yaml

   encoder:
     name: virchow2
   aggregation:
     name: mean_pool

Replace ``virchow2`` with another registered foundation model, or ``mean_pool`` with
``abmil``, and run the same experiment again. The resolved config saved in every
:doc:`run bundle <outputs>` makes the comparison auditable. Registered benchmarks
use the same mechanism to define canonical encoder and spacing sweeps.

Where to go next
----------------

* Follow :doc:`workflow guides <tutorials/index>` for slide-level MIL,
  segmentation, and cell detection examples.
* Use :doc:`benchmarking` to reproduce included foundation-model comparisons.
* Consult :doc:`reference` when you need an exact config field, component, or
  output contract.
