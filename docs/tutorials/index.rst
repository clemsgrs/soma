Workflow guides
===============

Choose the guide that matches the labels you have and the prediction you need.
Each guide explains the pipeline choices first and links to a runnable notebook
when one is available.

* Start with the :doc:`tile quickstart <../getting-started>` when every image tile
  has one label.
* Use :doc:`slide-level` for classification, regression, or survival from whole
  slides with an MIL aggregator.
* Use :doc:`segmentation` for dense pixel labels.
* Use :doc:`detection` for point-based cell targets.

The :doc:`pipeline overview <../pipeline>` explains how these paths share the
same preprocessing, foundation-model encoding, prediction, and evaluation stages.

.. toctree::
   :maxdepth: 1
   :hidden:

   Slide-level prediction guide <slide-level>
   Slide-level runnable notebook <walkthrough-slide-level>
   Segmentation guide <segmentation>
   Cell-detection guide <detection>
   Dense prediction notebook <walkthrough-dense>
   Attention segmentation notebook <walkthrough-attention-segmentation>
   Composite encoder notebook <walkthrough-composite>
