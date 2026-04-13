Compact Parameter Reference
===========================

This page is generated from the public config dataclasses and component
registries. It provides a compact index of the main public surfaces.

Public API
----------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Description
   * - ``Pipeline``
     - Orchestrates the full pipeline: extract → train all folds → summarize
   * - ``train``
     - Train and evaluate all folds, then summarize
   * - ``Dataset``
     - Load and validate dataset.csv
   * - ``Splits``
     - Load and validate splits.csv
   * - ``FeatureExtractor``
     - Delegates tile/slide feature extraction to slide2vec
   * - ``FeatureStore``
     - Index and load precomputed tile embeddings from disk
   * - ``TileFeatureExtractor``
     - Encode individual tile images into 1D feature vectors

Configuration dataclasses
-------------------------

.. list-table::
   :header-rows: 1

   * - Config
     - Main fields
     - Purpose
   * - ``PreprocessingConfig``
     - ``backend``, ``requested_tile_size_px``, ``requested_spacing_um``, ``requested_region_size_px``, ``region_tile_multiple``, ``read_tile_size_px``, ``read_region_size_px``, ``tissue_method``, ``tissue_threshold``, ``overlap``, ``seg_downsample``, ``tolerance``, ``ref_tile_size_px``, ``a_t``, ``tissue_mask_tissue_value``, ``hierarchical``, ``npatch``, ``hierarchical_patch_size_px``
     - Whole-slide segmentation and tiling geometry
   * - ``EncoderConfig``
     - ``name``, ``precision``, ``batch_size``, ``num_workers``, ``adaptive_batching``, ``input_size``, ``spacing_um``, ``output_variant``, ``save_tile_features``
     - Foundation-model encoder selection and runtime behavior
   * - ``CacheConfig``
     - ``enabled``, ``root_dir``, ``reuse_policy``, ``save_tile_features_for_slide``
     - Shared cache policy
   * - ``AggregatorConfig``
     - ``name``, ``params``
     - MIL bag aggregation
   * - ``TaskConfig``
     - ``name``, ``params``
     - Task head selection
   * - ``EvalConfig``
     - ``metrics``, ``subgroups``
     - Metric and subgroup reporting
   * - ``TrainingConfig``
     - ``seed``, ``epochs``, ``learning_rate``, ``weight_decay``, ``optimizer``, ``scheduler``, ``patience``, ``batch_size``, ``gradient_accumulation``
     - Training hyperparameters
   * - ``HeatmapConfig``
     - ``enabled``, ``cmap``, ``alpha``, ``blur_sigma``
     - Attention heatmap rendering
   * - ``PipelineConfig``
     - ``dataset_csv``, ``splits_csv``, ``output_root``, ``dataset_type``, ``preprocessing``, ``cache``, ``encoder``, ``aggregator``, ``task``, ``evaluation``, ``training``, ``heatmaps``, ``tags``
     - Complete experiment specification

Aggregator registry
-------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Class
     - Constructor knobs
     - Notes
   * - ``mean_pool``
     - ``MeanPool``
     - ``input_dim``
     - Baseline pooling; ``input_dim`` is injected from the feature store
   * - ``max_pool``
     - ``MaxPool``
     - ``input_dim``
     - Baseline pooling; ``input_dim`` is injected from the feature store
   * - ``abmil``
     - ``ABMIL``
     - ``input_dim``, ``hidden_dim``, ``activation``, ``gated``, ``dropout``
     - Attention MIL; ``input_dim`` is injected from the feature store
   * - ``clam_sb``
     - ``CLAM_SB``
     - ``input_dim``, ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``, ``instance_loss_mode``, ``low_attention_weight``, ``topk_target_weight``
     - General-purpose CLAM; ``input_dim`` is injected from the feature store
   * - ``clam_mb``
     - ``CLAM_MB``
     - ``input_dim``, ``hidden_dim``, ``attn_dim``, ``gated``, ``dropout``, ``k_sample``, ``n_classes``, ``inst_loss``, ``use_negative_class_instance_loss``, ``bag_weight``
     - Multi-branch classification only; ``input_dim`` is injected from the feature store
   * - ``dsmil``
     - ``DSMIL``
     - ``input_dim``, ``att_dim``, ``nonlinear_q``, ``nonlinear_v``, ``dropout``
     - Dual-stream MIL; ``input_dim`` is injected from the feature store
   * - ``dtfdmil``
     - ``DTFDMIL``
     - ``input_dim``, ``hidden_dim``, ``n_groups``, ``distill_mode``, ``dropout``
     - Two-stage distillation MIL; ``input_dim`` is injected from the feature store
   * - ``hipt``
     - ``HIPT``
     - ``input_dim``, ``region_size``, ``patch_size``, ``embed_dim_region``, ``embed_dim_slide``, ``num_heads``, ``dropout``
     - Hierarchical aggregation; ``input_dim`` is injected from the feature store
   * - ``transmil``
     - ``TransMIL``
     - ``input_dim``, ``att_dim``, ``n_layers``, ``n_heads``, ``n_landmarks``, ``pinv_iterations``, ``dropout``, ``use_mlp``
     - Transformer MIL; ``input_dim`` is injected from the feature store

Task registry
-------------

.. list-table::
   :header-rows: 1

   * - Name
     - Class
     - Constructor knobs
     - Notes
   * - ``binary_classification``
     - ``BinaryClassificationHead``
     - ``input_dim``, ``num_classes``, ``metrics``
     - Requires exactly two classes
   * - ``multiclass_classification``
     - ``MulticlassClassificationHead``
     - ``input_dim``, ``num_classes``, ``metrics``
     - Two or more classes; use binary_classification when the problem is strictly binary
   * - ``branch_aware_classification``
     - ``BranchAwareClassificationHead``
     - ``input_dim``, ``num_classes``, ``metrics``
     - CLAM-MB compatible branch head
   * - ``ordinal_classification``
     - ``OrdinalClassificationHead``
     - ``input_dim``, ``num_classes``, ``metrics``
     - Ordered integer labels
   * - ``regression``
     - ``RegressionHead``
     - ``input_dim``, ``num_targets``, ``metrics``
     - Continuous targets

Use this page as a concise index. Use the guide pages for workflow and the
docstrings for the exact API contract.
