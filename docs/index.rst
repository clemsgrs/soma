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

``soma`` supports tile, slide, and patient workflows. You can use  it either
as a full end-to-end :doc:`pipeline <pipeline>` or as a set of composable :doc:`building blocks <api>`
for custom experiment orchestration.

.. toctree::
   :maxdepth: 1

   getting-started
   api
   dataset
   pipeline
   preprocessing
   encoders
   aggregators
   tasks
   evaluation
   training
   caching
   outputs
   reporting
   reference
