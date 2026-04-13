soma
===================

`soma` is a modular framework to streamline computational pathology research.

It provides a unified API to go from a dataset of slides and labels to a full,
reproducible result report. Along the way, it makes it easy to sweep core
design choices such as preprocessing (spacing, field-of-view), encoding
(foundation models), and aggregation (MIL) so you can quickly find the
strongest configuration for your data.

You can use it either as a full end-to-end pipeline or as a set of composable
building blocks for custom experiment orchestration.

This is the canonical documentation home for the project.

.. toctree::
   :maxdepth: 1

   getting-started
   api
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
