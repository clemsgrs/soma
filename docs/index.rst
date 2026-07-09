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

``soma`` supports tile, slide, and patient workflows. You can use it either
as a full end-to-end :doc:`pipeline <pipeline>` or as a set of composable :doc:`building blocks <api>`
for custom experiment orchestration.

Where to next? Pick the card that matches what you came to do.

.. raw:: html

   <section class="soma-section" style="margin-top: 1.5rem">
     <div class="soma-card-grid">
       <a class="soma-card" href="getting-started.html">
         <h3>Start here →</h3>
         <p>Install soma, run your first pipeline, and pick how you'll use it: pipeline API, building blocks, or CLI.</p>
       </a>
       <a class="soma-card" href="api.html">
         <h3>Compose building blocks →</h3>
         <p>Mix and match encoders, aggregators, and decoders for custom orchestration.</p>
       </a>
       <a class="soma-card" href="tasks.html">
         <h3>Pick a task →</h3>
         <p>Classification, regression, survival, segmentation, or detection — with the methods that fit each.</p>
       </a>
       <a class="soma-card" href="benchmarking.html">
         <h3>Reproduce a benchmark →</h3>
         <p>Curate, run, and tolerance-check a registered benchmark — the EVA patch suite or the OCELOT detection challenge.</p>
       </a>
     </div>
   </section>

.. toctree::
   :maxdepth: 1
   :hidden:

   getting-started
   pipeline
   api
   cli

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
   :caption: System

   caching
   outputs
   reporting
