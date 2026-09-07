:orphan:

Composite encoders
==================

A composite encoder concatenates several frozen encoders' dense features for
segmentation or detection. Each member is cached independently;
:class:`~soma.dense.composite.CompositeDenseFeatureStore` combines them at load
time. Use ``composite`` in place of the single ``encoder`` block.

.. code-block:: yaml

   composite:
     encoders:
       - { name: uni, feature_kind: cls_attention }
       - { name: phikon, feature_kind: cls_attention }
       - { name: lunit, feature_kind: patch_features, member_norm: l2 }

Each member can select its feature kind, attention settings, output variant,
and extraction window. If omitted, ``feature_kind`` resolves to
``cls_attention`` for a pixel classifier and ``patch_features`` for a decoder.
All members share the run's read spacing and supervision size; per-member
spacing is not supported. Set ``dense_window_size`` and
``dense_window_overlap`` on a member to override the shared preprocessing
settings, for example when an encoder requires a fixed input size.

Concatenation resolution
------------------------

``concat_resolution`` defaults according to the trainable component:

* ``target`` for a :doc:`pixel classifier <../decoders/pixel-classifier>`:
  each member is interpolated to its padded encoded size and cropped to the
  shared supervision target. Channels then stack into ``(Σd_i, H, W)``.
  This preserves each member's padding and crop geometry even when token-grid
  sizes differ.
* ``grid`` for a :doc:`decoder <../decoders>`: each member is resampled to a
  common token grid before channels are stacked. ``concat_grid_size`` sets
  that grid's ``(height, width)``; by default soma takes the largest height
  and width across members. This mode treats each grid as spanning the target
  field of view and ignores individual members' padding fractions.

Normalization follows resampling and precedes concatenation. Per-member
``member_norm`` accepts ``none``, ``l2``, or ``layernorm`` and operates across
channels at each position. Defaults are ``l2`` for patch features and ``none``
for attention maps. Use it to control differences in feature magnitude across
encoders.

Walkthrough
-----------

The :doc:`composite walkthrough <../tutorials/walkthrough-composite>` extracts
two ungated encoders on a small synthetic CPU dataset and trains a segmentation
decoder on their concatenated grids.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models
  for Semantic Segmentation* (2026),
  `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_. Reports a 7.95%
  average improvement over individual models for its three-model ensemble
  across four datasets.
