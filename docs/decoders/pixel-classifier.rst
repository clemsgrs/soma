:orphan:

Attention-based segmentation
============================

A **decoder-free** method on the dense grid: an alternative trainable component to the
:doc:`../decoders` neural decoder. Instead of training a neural decoder on a frozen
encoder's patch features, it uses the encoder's own **per-head self-attention** as dense
per-pixel features and classifies each pixel with a lightweight, swappable classifier
(XGBoost, random forest, logistic regression, or a pointwise MLP). No gradients flow
through the backbone, and there is no neural decoder.

This re-implements Ramchandani et al., *Benchmarking Computational Pathology Foundation
Models for Semantic Segmentation* (2026, `arXiv:2602.18747
<https://arxiv.org/abs/2602.18747>`_), with one deliberate, well-motivated divergence in
how the frozen encoder is read (see :ref:`native-window`).

It sits alongside the neural-decoder segmentation path: both consume
``dataset_type="segmentation"`` (same manifest, splits, spacing-aware mask reader, dense
metrics, and prediction artifacts) — only the trainable component differs. Both land on
the same :doc:`../segmentation` dense contract.

.. seealso::

   The :doc:`attention-probing walkthrough <../tutorials/walkthrough-attention-segmentation>`
   runs this decoder-free path end to end on a tiny synthetic dataset — per-head
   CLS-attention grids into a per-pixel classifier. The
   :doc:`dense-prediction walkthrough <../tutorials/walkthrough-dense>` runs the
   neural-decoder segmentation path for contrast: same contract, only the trainable
   component differs.

The method
----------

For each tile:

1. Run the frozen ViT and take the **CLS-token self-attention of the chosen block(s)**,
   one ``(grid_h × grid_w)`` map **per head**. Optionally also keep the register-token
   query rows (Darcet et al.) as extra channels. This is a ``(K, grid_h, grid_w)`` grid —
   structurally just another dense feature grid (``K`` channels instead of the feature
   dim ``d``), so it reuses soma's whole dense extraction / cache / store stack.
2. Upsample the per-head maps to the mask's pixel resolution (the head's geometry, shared
   with the decoder path).
3. Fit a **per-pixel classifier** ``(K,) → class`` on class-stratified sampled pixels;
   predict **every** pixel at evaluation. Reuses the dense confusion-count metrics and the
   prediction-raster / overlay / CSV artifact writer verbatim.
4. Optionally **concatenate** attention from several foundation models for a richer
   per-pixel vector (the paper's headline, +7.95% mean Dice) — see
   :doc:`../encoders/composite`.

Per-head is always preserved — head specialization is the signal the classifier exploits;
reducing it would be lossy and irreversible in the cache. Channels are ordered
``[block][cls, reg…][head]`` and that order is recorded in each grid's sidecar.

.. _native-window:

How the encoder is read — the key divergence from the paper
-----------------------------------------------------------

A ViT is pretrained at a joint *(native pixel size, native mpp)*. To stay
**in-distribution** the input must match **both**. The paper **resizes** each patch to
the encoder's native *pixel* size, which matches only the pixel count:

   GlaS @ 20× (~0.5 µm/px), a 776×524 patch ≈ 388×262 µm field of view. Resized to 224 px,
   that 388 µm now spans 224 px ≈ **1.7 µm/px** — ~3.4× coarser than UNI's ~0.5 µm/px
   native scale. The backbone sees glands and nuclei at the wrong physical size
   (**scale-OOD**), plus detail loss from the downscale.

soma instead reuses its **window-as-knob** sliding extraction: read the tile at the
encoder's native **spacing** and slide a native-size **window**, stitching the token
grids. Every window is in-distribution on **both** pixel size and mpp. Three modes are
available, with explicit trade-offs:

.. list-table::
   :header-rows: 1
   :widths: 26 16 12 12 18 16

   * - Mode
     - ``dense_window_size``
     - Pixels
     - Scale (mpp)
     - Context
     - In-distribution
   * - **Native window @ native spacing** *(default)*
     - = native input
     - native
     - native
     - local (one window FOV)
     - ✓ both
   * - Larger-than-native window
     - > native
     - interp pos-embeds
     - native
     - more per window
     - partial
   * - Whole-patch (one forward)
     - ``null``
     - full tile
     - native
     - global (whole patch)
     - partial
   * - *Resize-to-native (paper)*
     - —
     - native
     - **wrong**
     - global
     - ✗ scale-OOD

**The context trade-off (not just scale).** Native-window sliding gives up *context*:
each window's CLS token attends only within its native field of view (~112 µm), and
stitching produces a **mosaic of local attentions**, not one global field. A whole-patch
forward lets the CLS token see the entire patch at once → *global* attention (a whole
gland can light up). For dense per-pixel segmentation, local-dense is the better default
(in-distribution, dense, reuses the machinery), but context can matter for large
structures — which is exactly why the larger-window and whole-patch modes are **kept**.
soma does **not** implement the paper's resize mode: it is the one option that is OOD on
scale *and* loses detail.

Configuration
-------------

Two orthogonal axes under ``dataset_type: segmentation``:

* **Axis 1 — what the encoder emits** (``preprocessing.feature_kind``):
  ``patch_features`` (the ViT patch grid, decoder path) or ``cls_attention`` (per-head
  attention, classifier path). Left unset, it is cross-defaulted from the component
  (a ``pixel_classifier`` ⇒ ``cls_attention``; a ``decoder`` ⇒ ``patch_features``).
* **Axis 2 — the trainable component** (``pixel_classifier`` XOR ``decoder``), mutually
  exclusive.

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     feature_kind: cls_attention          # auto when a pixel_classifier is set
     attention: { blocks: [-1], include_registers: false }
     requested_tile_size_px: 512          # supervision (mask) size
     requested_spacing_um: 0.5            # read + native encoder spacing
     dense_window_size: null              # null=whole; =native input for native-window mode
   encoder: { name: uni }                 # XOR `composite:` (multi-encoder)
   pixel_classifier:                      # XOR `decoder:`
     name: xgboost                        # xgboost | random_forest | logistic | mlp
     params: { n_estimators: 100, tree_method: hist }
   training:
     max_train_pixels: 2_000_000          # class-stratified pixel budget (train only)
   task: { name: segmentation, params: { num_classes: 5 } }

The classifiers are swappable and each owns its framework, training loop, and
serialization behind a common interface
(:class:`soma.pixel_classifiers.base.PixelClassifier`): XGBoost runs its boosting rounds,
the MLP runs internal mini-batch SGD epochs with early stopping — no torch ``Trainer`` and
no ``.pt`` fold checkpoints on this path. Add
``params.class_balanced_weights: true`` to weight the fit by inverse class frequency.

Multi-encoder concatenation
---------------------------

Concatenating attention from several foundation models is the paper's headline (+7.95%
mean Dice). For the pixel-classifier path, ``concat_resolution`` auto-resolves to
``target`` — each member upsamples its ``(K_i, grid_i)`` grid to the shared supervision
target via its own geometry, then channels stack into ``(ΣK_i, H, W)``. The full config
and concat-resolution semantics live on the :doc:`../encoders/composite` page.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models for Semantic
  Segmentation* (2026), `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_.
* Darcet et al., *Vision Transformers Need Registers* (2024),
  `arXiv:2309.16588 <https://arxiv.org/abs/2309.16588>`_.
* The neural-decoder :doc:`../segmentation` path and the window-as-knob sliding extraction
  it shares (see :doc:`../preprocessing`).
