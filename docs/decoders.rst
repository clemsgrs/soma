Decoders
========

Decoders are the trainable component for dense prediction. They transform a
cached grid from a frozen encoder into the map consumed by a
:doc:`segmentation` or :doc:`detection` head.

Shared dense substrate
----------------------

Dense tasks share this contract:

1. A frozen foundation-model encoder emits a ``(d, grid_h, grid_w)`` grid.
2. The grid is cached as ``feature_type="dense_grid"`` and reused across
   decoder runs; gradients do not flow through the encoder.
3. A decoder produces ``C`` channels at grid resolution.
4. The task head interpolates to ``target_size``, applies ``crop_box``, and
   owns activation, loss, postprocessing, metrics, and prediction artifacts.

``preprocessing.feature_kind`` selects the encoder output:

- ``patch_features`` uses the ViT patch-token grid and is the default for a
  neural decoder.
- ``cls_attention`` uses prefix-token self-attention, with channels selected
  by ``preprocessing.attention``.

Both substrates have token-grid resolution. Switching substrate changes the
decoder's inferred ``input_dim`` but not the task contract. Composite encoders
are concatenated at grid resolution with ``concat_resolution: grid``.

Encoder windowing
-----------------

``preprocessing.dense_window_size`` controls how the supervision tile reaches
the frozen encoder. ``null`` runs one padded forward pass; a positive value
slides patch-aligned windows and blends their grids using
``dense_window_overlap``. Read windows at the encoder's native spacing to keep
pixel scale consistent. The :doc:`pixel-classifier
<decoders/pixel-classifier>` guide provides the full mode table; extraction is
identical for neural decoders.

Decoder choices
---------------

.. list-table::
   :header-rows: 1

   * - Name
     - Use
   * - ``linear``
     - A single ``1x1`` convolution; the minimal dense linear probe.
   * - ``lightweight_conv``
     - The default convolutional decoder.
   * - ``heavy_conv``
     - Pyramid-pooling context fusion with learned upsampling.

``lightweight_conv`` and ``heavy_conv`` begin with a ``1x1`` projection from
encoder width ``d`` to decoder width ``D``. Decoder capacity therefore remains
comparable across encoders with different embedding dimensions.

.. code-block:: yaml

   preprocessing:
     feature_kind: patch_features
   decoder:
     name: lightweight_conv

For decoder-free segmentation, use a registered pixel classifier over
``cls_attention`` instead. See :doc:`decoders/pixel-classifier` and the
:doc:`attention-segmentation walkthrough
<tutorials/walkthrough-attention-segmentation>`.
