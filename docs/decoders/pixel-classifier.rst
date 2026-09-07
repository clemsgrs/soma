:orphan:

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

1. Extract the CLS-token attention rows from selected transformer blocks,
   retaining one spatial map per head. Optionally include register-token rows
   as additional channels.
2. Interpolate the maps to the padded encoded size and crop them to the
   supervision target, producing one feature vector per pixel.
3. Fit a classifier on class-stratified sampled training pixels. At evaluation,
   classify every pixel and compute the shared segmentation metrics.

Cached attention channels retain their individual heads, ordered
``[block][cls, reg…][head]``. The sidecar records that ordering. For features
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
by inverse class frequency. Each classifier owns its training and serialization
through :class:`~soma.pixel_classifiers.base.PixelClassifier`; this path does not
use the torch Trainer or its fold checkpoints. The MLP runs its own minibatch
training loop with early stopping.

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

This differs from resizing a complete image to the encoder's native pixel
size, as in the attention-probing method described by Ramchandani et al.
Resizing also changes effective physical scale: a 388 µm-wide field reduced to
224 pixels spans approximately 1.73 µm per pixel. soma exposes spacing and
window size separately rather than implementing that resize-to-native mode.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models
  for Semantic Segmentation* (2026),
  `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_.
* Darcet et al., *Vision Transformers Need Registers* (2024),
  `arXiv:2309.16588 <https://arxiv.org/abs/2309.16588>`_.
