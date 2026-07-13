soma
===================

Work with pathology foundation models from images and labels to performance
metrics. Compose a modular workflow, swap individual components, or reproduce a
published benchmark with a single command.

.. figure:: /_static/figures/pipeline-overview.svg
   :figclass: soma-figure soma-hero
   :alt: The soma pipeline — data, a frozen encoder, a trained decoder, and evaluation.

.. raw:: html

   <nav class="soma-route-list" aria-label="Documentation routes">
       <a class="soma-route" href="getting-started.html">
         <strong>Get started</strong>
         <span>Install Soma and run a first tiles-to-metrics experiment.</span>
       </a>
       <a class="soma-route" href="pipeline.html">
         <strong>Build a workflow</strong>
         <span>Compose and swap preprocessing, encoders, aggregators, task heads, and metrics.</span>
       </a>
       <a class="soma-route" href="benchmarking.html">
         <strong>Reproduce benchmark</strong>
         <span>Run EVA, OCELOT, or HEST against published references.</span>
       </a>
       <a class="soma-route" href="add-a-benchmark.html">
         <strong>Add a benchmark</strong>
         <span>Package your dataset preparation, protocol, metrics, and references.</span>
       </a>
   </nav>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Start

   getting-started
   How Soma works <pipeline>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Build workflows

   Workflow guides <tutorials/index>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Benchmark foundation models

   benchmarking
   EVA patch classification <eva-patch-classification-benchmark>
   OCELOT cell detection <ocelot-detection-benchmark>
   HEST gene expression <hest-gene-expression-benchmark>
   add-a-benchmark

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   reference
