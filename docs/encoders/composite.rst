Composite (multi-encoder)
=========================

A **composite** encoder concatenates the dense outputs of several foundation
models into one richer per-position vector. It is the single home for
multi-encoder concatenation across the dense paths (decoder segmentation,
detection, and the decoder-free :doc:`../decoders/pixel-classifier`).

Members live under a ``composite:`` block (XOR the single ``encoder:``). Each
member carries its own extraction spec and is cached independently; the composite
is a thin **load-time** concat view
(:class:`soma.dense.composite.CompositeDenseFeatureStore`).

.. code-block:: yaml

   composite:                               # XOR `encoder:`
     # concat_resolution auto: `target` (pixel_classifier) | `grid` (decoder/detection)
     encoders:
       - { name: uni,    feature_kind: cls_attention }
       - { name: phikon, feature_kind: cls_attention }
       - { name: cellvit, feature_kind: patch_features, member_norm: l2 }  # embedding, not attention

``member_norm`` (``{none, l2, layernorm}``) per-member normalizes before concat so a
large-magnitude encoder cannot dominate; it auto-defaults to ``l2`` for ``patch_features``
members and ``none`` for ``cls_attention``. v1 reads every member at the same spacing and
supervision size; per-member native spacing is deferred.

Concat resolution
-----------------

``concat_resolution`` controls where the per-member grids are stacked, and
auto-resolves from the trainable component:

* ``target`` *(auto for the :doc:`../decoders/pixel-classifier` path)* — each member
  upsamples its ``(K_i, grid_i)`` grid to the shared supervision target via its own
  geometry, then channels stack into ``(ΣK_i, H, W)``. Per-pixel resolution makes this
  **resolution-agnostic** — heterogeneous patch sizes / token grids simply land on the
  common target, no feature-space resampling.
* ``grid`` *(auto for the decoder / detection paths)* — members auto-concatenate at
  token-grid resolution before the decoder consumes the stacked ``(Σd_i, grid)`` grid.

Multi-resolution concat (planned)
---------------------------------

Reading each member at its own **native spacing** — so a composite can mix encoders
operating at different physical resolutions — is a planned increment. v1 reads every
member at the same spacing and supervision size; per-member native spacing is deferred.

References
----------

* Ramchandani et al., *Benchmarking Computational Pathology Foundation Models for Semantic
  Segmentation* (2026), `arXiv:2602.18747 <https://arxiv.org/abs/2602.18747>`_ — the
  multi-encoder concatenation headline (+7.95% mean Dice) on the decoder-free
  :doc:`../decoders/pixel-classifier` path.
