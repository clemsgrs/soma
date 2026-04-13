Aggregators
===========

Aggregators combine tile features into a bag-level representation for MIL.
Choose the simplest aggregator that matches the learning problem.

**Start with** ``abmil`` as the default: it balances interpretability and
performance on most tasks. Use ``mean_pool`` or ``max_pool`` for quick
debugging baselines. Use ``clam_sb`` when instance-level supervision (tile
labeling) is available. Reserve ``transmil`` and ``hipt`` for larger bags or
hierarchical feature structure.

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
     - Description
     - Main knobs
   * - ``mean_pool``
     - Unweighted mean of all tile features; baseline with no learnable parameters beyond the task head
     - none beyond ``input_dim``
   * - ``max_pool``
     - Element-wise max over tile features; baseline with no learnable parameters beyond the task head
     - none beyond ``input_dim``
   * - ``abmil``
     - Gated attention pooling (Ilse et al., 2018); standard first choice for slide-level classification and regression
     - ``hidden_dim``, ``activation``, ``gated``, ``dropout``
   * - ``clam_sb``
     - Single-branch CLAM (Lu et al., 2021); adds instance-level supervision with a small tile classifier trained jointly with the bag classifier
     - ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``, ``instance_loss_mode``, ``low_attention_weight``, ``topk_target_weight``
   * - ``clam_mb``
     - Multi-branch CLAM (Lu et al., 2021); one attention branch per class; classification only
     - ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``
   * - ``dsmil``
     - Dual-stream MIL (Li et al., 2021); critical-instance query-key attention between a bag-level stream and a max-pooling stream
     - ``att_dim``, ``nonlinear_q``, ``nonlinear_v``, ``dropout``
   * - ``dtfdmil``
     - Double-tier feature distillation MIL (Zhang et al., 2022); groups tiles into pseudo-bags and uses Grad-CAM distillation between tiers
     - ``hidden_dim``, ``n_groups``, ``distill_mode``, ``dropout``
   * - ``transmil``
     - Transformer MIL (Shao et al., 2021); uses Nystromformer self-attention with PPEG positional encoding; effective on large bags
     - ``att_dim``, ``n_layers``, ``n_heads``, ``n_landmarks``, ``pinv_iterations``, ``dropout``, ``use_mlp``
   * - ``hipt``
     - Hierarchical image pyramid transformer (Chen et al., 2022); processes tiles within regions, then regions within slides; requires hierarchical tiling
     - ``region_size``, ``patch_size``, ``embed_dim_region``, ``embed_dim_slide``, ``num_heads``, ``dropout``, ``pretrained_region_weights``

CLAM notes
----------

- ``clam_sb`` supports binary, multiclass, ordinal, and single-target regression.
- ``clam_mb`` is classification-only and emits one branch per class.
- The task head ultimately determines the valid loss/metric pairing.

HIPT notes
----------

HIPT processes tiles hierarchically: tiles are grouped into fixed-size regions,
which are then aggregated at the slide level. This requires hierarchical tiling
to be enabled in preprocessing.

Set ``region_tile_multiple`` in :class:`soma.config.PreprocessingConfig` to
control how many tiles fit inside one region. For example,
``region_tile_multiple=4`` with a 256 px tile produces 1024 px regions. The
framework derives ``hierarchical=True`` and the correct region geometry
automatically from the aggregator choice.
