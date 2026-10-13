:orphan:

Composite encoders
==================

A composite encoder concatenates several frozen encoders' dense features for
segmentation or detection. Each member is cached independently;
:class:`~soma.dense.composite.CompositeDenseFeatureStore` combines them at load
time. Name the ``composite`` encoder in place of a single encoder. The
:doc:`composite walkthrough <../tutorials/walkthrough-composite>` trains a
segmentation decoder on two concatenated encoders.

.. code-block:: yaml

   features:
     encoder:
       name: composite
       params:
         members:
           - { name: uni, dense: { feature_kind: cls_attention } }
           - { name: phikon, dense: { feature_kind: cls_attention } }
           - { name: lunit, dense: { feature_kind: patch_features }, member_norm: l2 }

Each member can select its feature kind, attention settings and extraction
window (``dense``, fields of slide2vec's ``DenseOptions``), its encoder params
(``params``, such as ``output_variant``) and its ``execution`` precision and
batch size. If omitted, ``feature_kind`` resolves to ``patch_features``.
All members share the run's read spacing and supervision size
(``features.dense``); per-member spacing is not supported. Set
``window_size`` and ``overlap`` in a member's ``dense`` to override the shared
settings, for example when an encoder requires a fixed input size.

Concatenation resolution
------------------------

``concat_resolution`` selects where the members' grids are combined:

* ``grid`` (the default) resamples each member to a common token grid before
  channels are stacked, which is what a :doc:`decoder <../decoders>` consumes.
  ``concat_grid_size`` sets that grid's ``(height, width)``; by default soma
  takes the largest height and width across members. This mode treats each
  grid as spanning the target field of view and ignores individual members'
  padding fractions.
* ``target`` interpolates each member to its padded encoded size and crops it
  to the shared supervision target. Channels then stack into
  ``(Σd_i, H, W)``. This preserves each member's padding and crop geometry
  even when token-grid sizes differ.

Normalization follows resampling and precedes concatenation. Per-member
``member_norm`` accepts ``none``, ``l2``, or ``layernorm`` and operates across
channels at each position. Defaults are ``l2`` for patch features and ``none``
for attention maps. Use it to control differences in feature magnitude across
encoders.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models
  for Semantic Segmentation* (2026),
  `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_.
