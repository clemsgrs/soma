Attention-based segmentation
===============================

A pixel classifier predicts segmentation labels from a frozen encoder's dense
features, without a neural decoder. By default it uses per-head self-attention
maps. Available classifiers are XGBoost, random forest, logistic regression,
and a pointwise MLP.

This path shares the segmentation manifest, spacing-aware mask reader, metrics,
and prediction artifacts with :doc:`../segmentation`. Start with the
:doc:`attention walkthrough <../tutorials/walkthrough-attention-segmentation>`
for a complete example on a small synthetic dataset.

How it works
------------

The dense grid is interpolated to the padded encoded size and cropped to the
supervision target, giving one feature vector per pixel. A classifier is fitted
on class-stratified sampled training pixels and then classifies every pixel at
evaluation. Cached attention channels keep their individual heads, ordered
``[block][cls, reg…][head]``; the sidecar records that ordering. For features
from several encoders, use :doc:`../encoders/composite`.

Configuration
-------------

Choose exactly one of ``pixel_classifier`` and ``decoder``. The component and
feature kind are independent: ``feature_kind`` defaults to ``cls_attention``
with a pixel classifier, but can be set to ``patch_features`` explicitly.
Attention extraction requires an encoder that supports it.

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     feature_kind: cls_attention
     attention: { blocks: [-1], include_registers: false }
     dense_window_size: null              # whole padded tile; see window options below
   encoder: { name: uni }
   pixel_classifier:
     name: xgboost                        # xgboost | random_forest | logistic | mlp
     params: { n_estimators: 100, tree_method: hist }
   training:
     max_train_pixels: 2_000_000          # class-stratified training pixel budget
   task: { name: segmentation, params: { num_classes: 5 } }

XGBoost requires the optional ``pixel`` extra (``pip install 'soma-pathology[pixel]'``).
Set ``pixel_classifier.params.class_balanced_weights: true`` to weight fitting
by inverse class frequency. This path does not use the torch Trainer or its
fold checkpoints; the MLP runs its own minibatch loop with early stopping.

.. _native-window:

Choose the extraction window
----------------------------

Read spacing and encoder-window size control different aspects of the input:
spacing sets physical scale, while window size sets the context visible in one
forward pass. Select the spacing through ``requested_spacing_um``; soma does
not force every run to an encoder's recommended spacing.

.. list-table::
   :header-rows: 1

   * - ``dense_window_size``
     - Behavior
     - Context
   * - ``null`` (default)
     - Encode the whole padded tile in one forward pass.
     - All positions in the tile can attend to one another.
   * - Encoder's native input size
     - Slide native-size windows and stitch their token grids.
     - Attention is local to each window.
   * - Larger than the native input size
     - Use larger windows where supported by the encoder.
     - More context per window, with a different input geometry from pretraining.

Whole-tile and larger-window extraction require encoder support for the input
size. Native-size windows at a recommended spacing preserve those two
pretraining input settings, but stitched attention remains a mosaic of local
maps, not one global attention map. Set ``dense_window_overlap`` to overlap
windows and blend their grids.

soma does not resize a complete image to the encoder's native pixel size (the
method of Ramchandani et al.), because that also changes the physical scale.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models
  for Semantic Segmentation* (2026),
  `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_.
* Darcet et al., *Vision Transformers Need Registers* (2024),
  `arXiv:2309.16588 <https://arxiv.org/abs/2309.16588>`_.
