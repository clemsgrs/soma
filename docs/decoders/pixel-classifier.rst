Attention-based segmentation
============================

This decoder-free method classifies each pixel from a frozen ViT's per-head
self-attention. It follows Ramchandani et al., *Benchmarking Computational
Pathology Foundation Models for Semantic Segmentation* (2026,
`arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_), while reading images at
the encoder's native physical scale instead of resizing them.

Method
------

For each tile, soma extracts CLS-token attention from selected transformer
blocks as a ``(K, grid_h, grid_w)`` dense grid. Register-token query rows may be
retained as additional channels. Channel order is
``[block][cls, register...][head]`` and is recorded with the cached grid.

The segmentation geometry upsamples attention to mask resolution. A registered
pixel classifier fits ``K -> class`` on a class-stratified pixel sample, then
predicts every pixel at evaluation. The manifest, mask reader, metrics, and
artifacts are the same as :doc:`../segmentation`; shared dense extraction and
cache behavior are documented in :doc:`../decoders`.

Native-window extraction
------------------------

Resizing a large pathology tile to an encoder's native pixel dimensions
changes microns per pixel and can put tissue structures out of distribution.
Native-window extraction instead reads at the encoder's native
spacing and stitches attention from native-size windows. This preserves scale
and detail, at the cost of limiting each CLS token to local context.

Choose the mode according to the required context:

.. list-table::
   :header-rows: 1

   * - Mode
     - ``dense_window_size``
     - Scale
     - Context
   * - Native window
     - Encoder native input size
     - Native
     - Local and in-distribution
   * - Larger window
     - Larger than native input
     - Native
     - Broader, with interpolated position embeddings
   * - Whole tile
     - ``null``
     - Native
     - Global, with one forward pass
   * - Resize to native pixels (paper)
     - Not implemented
     - Changed
     - Global, but scale-OOD

Use a larger or whole-tile window when global gland or tissue context matters.
The operational window contract lives in :doc:`../decoders`.

Configuration
-------------

``pixel_classifier`` and ``decoder`` are mutually exclusive. Setting a pixel
classifier defaults ``preprocessing.feature_kind`` to ``cls_attention``.

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
     feature_kind: cls_attention
     attention: {blocks: [-1], include_registers: false}
     dense_window_size: 224
   encoder: {name: uni}
   pixel_classifier:
     name: xgboost
     params:
       n_estimators: 100
       tree_method: hist
   training:
     max_train_pixels: 2_000_000
   task:
     name: segmentation
     params: {num_classes: 5}

Classifiers
-----------

Available choices are ``xgboost``, ``random_forest``, ``logistic``, and
``mlp``. Each implements
:class:`soma.pixel_classifiers.base.PixelClassifier`, including its own fit,
prediction, and serialization behavior. Set
``params.class_balanced_weights: true`` to weight training by inverse class
frequency. This path does not use the torch trainer or write ``.pt`` fold
checkpoints.

Composite encoders concatenate each member's attention at target resolution;
see :doc:`../encoders/composite`. The paper reports a 7.95% mean-Dice gain from
multi-encoder concatenation.

Run the complete method in the :doc:`attention-probing walkthrough
<../tutorials/walkthrough-attention-segmentation>`.

References
----------

* Darcet et al., *Vision Transformers Need Registers* (2024),
  `arXiv:2309.16588 <https://arxiv.org/abs/2309.16588>`_.
