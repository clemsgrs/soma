soma
===================

``soma`` is a modular framework to streamline computational pathology research.

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

.. raw:: html

   <section class="soma-section" style="margin-top: 1.5rem">
     <div class="soma-card-grid">
       <a class="soma-card" href="getting-started.html">
         <h3>Getting started</h3>
         <p>Install soma and run your first end-to-end pipeline in minutes.</p>
       </a>
       <a class="soma-card" href="pipeline.html">
         <h3>Pipeline</h3>
         <p>Configure a full end-to-end run from a single YAML file.</p>
       </a>
       <a class="soma-card" href="api.html">
         <h3>API Guide</h3>
         <p>Mix and match building blocks for custom experiment orchestration.</p>
       </a>
       <a class="soma-card" href="cli.html">
         <h3>CLI Guide</h3>
         <p>Run experiments from YAML configs and inspect registered presets.</p>
       </a>
       <a class="soma-card" href="tutorials/walkthrough-slide-level.html">
         <h3>Tutorials</h3>
         <p>Runnable, end-to-end notebooks for slide-level and dense tasks.</p>
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
   :caption: Tutorials

   tutorials/walkthrough-slide-level
   tutorials/walkthrough-dense

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
   :caption: Models

   encoders
   aggregators
   tasks
   attention-segmentation
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
   :caption: System

   caching
   outputs
   reporting
