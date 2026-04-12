Aggregators
===========

Aggregators combine tile features into a bag-level representation for MIL.
Choose the simplest aggregator that matches the learning problem.

The shared base classes are :class:`soma.aggregators.base.Aggregator` and
:class:`soma.aggregators.base.AggregatorOutput`.

.. autoclass:: soma.aggregators.base.Aggregator
   :members:

.. autoclass:: soma.aggregators.base.AggregatorOutput
   :members:

Available presets
-----------------

.. list-table::
   :header-rows: 1

   * - Config name
     - Use when
     - Main knobs
   * - ``mean_pool``
     - Simple slide-level baseline
     - none beyond ``input_dim``
   * - ``max_pool``
     - Simple slide-level baseline
     - none beyond ``input_dim``
   * - ``abmil``
     - Standard attention MIL
     - ``hidden_dim``, ``activation``, ``gated``, ``dropout``
   * - ``clam_sb``
     - Strong general-purpose MIL baseline
     - ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``, ``instance_loss_mode``, ``low_attention_weight``, ``topk_target_weight``
   * - ``clam_mb``
     - Classification-only CLAM variant
     - ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``
   * - ``dsmil``
     - Dual-stream MIL
     - ``att_dim``, ``nonlinear_q``, ``nonlinear_v``, ``dropout``
   * - ``dtfdmil``
     - Two-stage feature distillation
     - ``hidden_dim``, ``n_groups``, ``distill_mode``, ``dropout``
   * - ``transmil``
     - Transformer MIL
     - ``att_dim``, ``n_layers``, ``n_heads``, ``n_landmarks``, ``pinv_iterations``, ``dropout``, ``use_mlp``
   * - ``hipt``
     - Hierarchical features
     - ``region_size``, ``patch_size``, ``embed_dim_region``, ``embed_dim_slide``, ``num_heads``, ``dropout``, ``pretrained_region_weights``

CLAM notes
----------

- ``clam_sb`` supports binary, multiclass, ordinal, and single-target regression.
- ``clam_mb`` is classification-only and emits one branch per class.
- The task head ultimately determines the valid loss/metric pairing.
