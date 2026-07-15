soma
===================

``soma`` is a modular framework to streamline computational pathology research.

.. figure:: /_static/figures/pipeline-overview.svg
   :figclass: soma-figure soma-hero
   :alt: The soma pipeline — data, a frozen encoder, a trained decoder, and evaluation.

   Data and evaluation are fixed scaffolding; the encoder and decoder are the two
   swappable axes — change one in a single line and keep the rest.

It provides a unified API to go from a dataset of slides and labels to a full,
reproducible result report. Along the way, it makes it easy to sweep core
design choices such as preprocessing (spacing, field-of-view), encoding
(foundation models), and aggregation (MIL) so you can quickly find the
strongest configuration for your data.

Under the hood, it builds on two open-source projects of mine:

- `hs2p <https://github.com/clemsgrs/hs2p>`_ for fast whole-slide
  preprocessing
- `slide2vec <https://github.com/clemsgrs/slide2vec>`_ for fast whole-slide
  encoding

``soma`` supports tile, region-of-interest, slide, and patient workflows. You can use it either
as a full end-to-end :doc:`pipeline <getting-started>` or as a set of composable :doc:`building blocks <api>`
for custom experiment orchestration.

.. raw:: html

   <nav class="soma-route-list" aria-label="Documentation routes">
       <a class="soma-route" href="how-soma-works.html">
         <strong>How soma works</strong>
         <span>See how reusable pipeline blocks support custom workflows and reproducible benchmarks.</span>
       </a>
       <a class="soma-route" href="getting-started.html">
         <strong>Get started</strong>
         <span>Install soma and run one experiment through the modular API, pipeline, or CLI.</span>
       </a>
       <a class="soma-route" href="benchmarking.html">
         <strong>Benchmarking</strong>
         <span>Reproduce and compare fixed foundation-model evaluation protocols.</span>
       </a>
       <a class="soma-route" href="encoders.html#model-zoo">
         <strong>Foundation model zoo</strong>
         <span>Browse registered tile-, slide-, and patient-level encoders.</span>
       </a>
   </nav>

.. toctree::
   :maxdepth: 1
   :hidden:

   how-soma-works
   getting-started

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Data

   dataset
   curation
   preprocessing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Components

   encoders
   modeling
   aggregators
   decoders

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tasks

   tasks
   classification
   regression
   survival
   segmentation
   detection

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Training & Evaluation

   training
   evaluation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   tutorials/slide-level
   tutorials/detection
   tutorials/segmentation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Benchmarks

   benchmarking
   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api
   cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: System

   caching
   outputs
   reporting
