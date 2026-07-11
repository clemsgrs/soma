Segmentation
============

Segmentation predicts a semantic class for every pixel in an image tile. It
uses the frozen-grid pipeline in :doc:`decoders`; the segmentation head owns
mask geometry, loss, metrics, and prediction artifacts.

Data contract
-------------

``dataset_type: segmentation`` loads a
:class:`soma.dataset.SegmentationManifest`. Each sample supplies an
``image_path`` and a registered ``mask_path`` instead of a scalar label. Masks
are read at the run spacing and aligned with the extracted token grid.

For whole-slide and annotation-mask inputs, configure
``preprocessing.masks`` and ``preprocessing.sampling`` as described in
:doc:`preprocessing`. The mapping defines class indices and ignored pixels.

Configuration
-------------

.. code-block:: yaml

   data:
     dataset_type: segmentation
   preprocessing:
     requested_tile_size_px: 512
     requested_spacing_um: 0.5
   encoder: {name: uni}
   decoder: {name: lightweight_conv}
   task:
     name: segmentation
     params: {num_classes: 5}
   evaluation:
     metrics: [mean_dice, mean_iou]

The decoder returns class channels at token-grid resolution. The head
interpolates them to ``target_size``, applies ``crop_box``, and trains with
cross-entropy plus soft Dice. ``tversky_alpha``, ``tversky_beta``, and
``tversky_gamma`` expose Tversky and focal-Tversky variants for imbalanced
structures.

Evaluation streams per-image confusion counts instead of retaining cohort-wide
logits. It reports ``mean_dice`` and ``mean_iou``; save controls for overlays
and probability tensors live in :doc:`evaluation`.

Methods
-------

``lightweight_conv`` is the default neural decoder. Composite encoders may be
concatenated at token-grid resolution. A decoder-free pixel classifier is also
available for attention grids; compare the methods in
:doc:`tutorials/segmentation` and :doc:`decoders/pixel-classifier`.

Task head
---------

.. autoclass:: soma.tasks.segmentation.SegmentationHead
   :members:
