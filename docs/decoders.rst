Decoders
========

Decoders are the **dense trainable component**: the dense-grid analogue of
:doc:`aggregators`. Where an aggregator collapses a bag of tile features into a
single slide- or patient-level vector, a decoder consumes the dense
``(d, grid_h, grid_w)`` token grid a **frozen** foundation-model encoder emits and
produces a dense per-position output (a per-pixel segmentation map or a per-class
detection heatmap). No gradients flow through the backbone; only the decoder is
trained.

The dense paths share the same front half — a frozen encoder produces a dense
``(d, grid)`` grid (cached as ``feature_type="dense_grid"``) — and differ only in the
trainable component on that grid and its output representation. The decoder is the
default trainable component; the decoder-free :doc:`pixel-classifier
<decoders/pixel-classifier>` method is the alternative.

**Decoding methods**

- **lightweight_conv** — the default trainable neural decoder (documented below).
- **Decoder-free pixel classifier** — classifies the encoder's own attention per
  pixel, no neural decoder: :doc:`Attention-based segmentation
  <decoders/pixel-classifier>` ·
  :doc:`tutorial <tutorials/walkthrough-attention-segmentation>`.

lightweight_conv
----------------

``lightweight_conv`` is the default decoder. It regresses a ``(C, grid)`` map from the
dense token grid; the task head then interpolates it to the supervision ``target_size``,
crops via ``crop_box``, and applies the task-specific activation (a **sigmoid** per-class
heatmap for detection; per-pixel class logits for segmentation).

.. code-block:: yaml

   decoder:                               # the dense trainable component
     name: lightweight_conv

The decoder is **input-agnostic**: it consumes whatever dense ``(d, grid)`` grid the
encoder emits, set by ``preprocessing.feature_kind`` — ``patch_features`` (the ViT
patch-token grid, ``d`` = the encoder's feature dim) or ``cls_attention`` (per-head
prefix-token self-attention as a ``(K, grid)`` grid). Switching between them is a **pure
config flip** — the decoder is simply built with ``input_dim`` set to the emitted channel
count (``d`` or ``K``). Multi-encoder :doc:`encoders/composite` runs are supported and
auto-concatenate at token-grid resolution (``concat_resolution: grid``).
