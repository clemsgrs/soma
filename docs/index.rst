soma
====

``soma`` runs computational pathology experiments from images and labels to
predictions, metrics, and reports. Use a YAML pipeline for a complete run or the
Python API to compose preprocessing, frozen encoders, and downstream models.

.. figure:: /_static/figures/pipeline-overview.svg
   :figclass: soma-figure soma-hero
   :alt: The soma pipeline — data, a frozen encoder, a trained decoder, and evaluation.

   Reuse extracted features while comparing downstream models, or hold the
   training protocol fixed to compare encoders.

Workflows cover tiles, regions of interest, slides, and patients. Whole-slide
preprocessing uses `hs2p <https://github.com/clemsgrs/hs2p>`_; foundation-model
encoding uses `slide2vec <https://github.com/clemsgrs/slide2vec>`_.

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
   :maxdepth: 2
   :hidden:

   how-soma-works
   getting-started
   data
   components
   tasks
   training-evaluation
   tutorials/index
   benchmarking
   reference
   system
