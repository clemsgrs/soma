Decoders
========

Decoders train on a frozen encoder's dense feature grids to produce spatial
predictions. They are used for :doc:`segmentation` and :doc:`detection`; only
the downstream model is trained. Segmentation also supports a decoder-free
:doc:`pixel classifier <decoders/pixel-classifier>`.

Choose a decoder
----------------

.. list-table::
   :header-rows: 1

   * - ``decoder.name``
     - Structure
   * - ``linear``
     - A single ``1x1`` convolution at token-grid resolution.
   * - ``lightweight_conv``
     - A ``1x1`` projection, bilinear upsampling and convolution blocks, then
       a class-output convolution.
   * - ``heavy_conv``
     - A ``1x1`` projection, pyramid-pooling context fusion, and learned
       upsampling through transposed convolutions.

The two convolutional decoders project the input channels to ``hidden_dim``.
Only this input projection depends on the encoder's embedding width; subsequent
layers have fixed width. A wider encoder therefore still increases the total
trainable parameter count.

.. code-block:: yaml

   decoder:
     name: lightweight_conv

The task head interpolates decoder outputs to the padded ``encoded_size``, then
uses ``crop_box`` to recover the supervision ``target_size``. Segmentation uses
per-pixel class logits; detection applies a sigmoid to produce one heatmap per
object class.

Choose the input features
-------------------------

``preprocessing.feature_kind`` selects the dense grid:

* ``patch_features`` (default with a decoder) supplies patch-token embeddings.
* ``cls_attention`` supplies per-head prefix-token attention maps. Configure
  the selected blocks and register tokens through ``preprocessing.attention``.

Both have shape ``(channels, grid_h, grid_w)``. The decoder's input dimension is
resolved from the grid's channel count; changing feature kind leaves the task
head and loss unchanged. Attention extraction requires encoder support.

:doc:`Composite encoders <encoders/composite>` combine several independently
cached grids. Decoder runs default to concatenation at a common token-grid
resolution.
