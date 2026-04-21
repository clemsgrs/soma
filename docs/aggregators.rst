Aggregators
===========

Aggregators combine tile features into a bag-level representation for MIL.
Start with the simplest preset that matches the task, then tune only the
knobs you need.

The shared base classes are :class:`soma.aggregators.base.Aggregator` and
:class:`soma.aggregators.base.AggregatorOutput`.

.. autoclass:: soma.aggregators.base.Aggregator
   :members:

.. autoclass:: soma.aggregators.base.AggregatorOutput
   :members:

Aggregator Zoo
--------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Description
     - Notes
   * - ``mean_pool``
     - Mean over all tile features.
     -
   * - ``max_pool``
     - Element-wise max over tile features.
     -
   * - ``abmil``
     - Gated attention pooling for slide-level aggregation.
     - Ilse et al., 2018
   * - ``clam_sb``
     - Single-branch CLAM with instance-level supervision.
     - Lu et al., 2021
   * - ``clam_mb``
     - Multi-branch CLAM with one attention branch per class.
     - Lu et al., 2021
   * - ``dsmil``
     - Dual-stream MIL with a critical-instance query.
     - Li et al., 2021
   * - ``dtfdmil``
     - Two-tier MIL with pseudo-bag distillation.
     - Zhang et al., 2022
   * - ``transmil``
     - Transformer-based MIL with Nystrom attention.
     - Shao et al., 2021
   * - ``hipt``
     - Hierarchical image pyramid transformer.
     - Chen et al., 2022

Aggregator details
------------------

The short notes below explain what each aggregator is for. The class docstrings
show the full constructor signatures and parameter descriptions.

ABMIL
~~~~~

``abmil`` applies gated attention pooling and returns tile-level attention
scores for interpretability and heatmap generation.

.. autoclass:: soma.aggregators.mil.abmil.ABMIL
   :members:

CLAM-SB
~~~~~~~

| ``clam_sb`` is the single-branch CLAM preset.
| It supports binary, multiclass, ordinal, and single-target regression tasks, and can mix bag-level
  and instance-level supervision.

.. autoclass:: soma.aggregators.mil.clam.CLAM_SB
   :members:

CLAM-MB
~~~~~~~

| ``clam_mb`` is the multi-branch CLAM preset.
| It is classification-only and creates one attention branch per class.

.. autoclass:: soma.aggregators.mil.clam.CLAM_MB
   :members:

DSMIL
~~~~~

``dsmil`` first scores instances to find a critical tile, then performs
query-key attention against that tile to build the bag representation.

.. autoclass:: soma.aggregators.mil.dsmil.DSMIL
   :members:

DTFDMIL
~~~~~~~

``dtfdmil`` partitions a bag into pseudo-bags, distills features from the
first tier, then aggregates the distilled set a second time.

.. autoclass:: soma.aggregators.mil.dtfdmil.DTFDMIL
   :members:

TransMIL
~~~~~~~~

``transmil`` uses Nystromformer-style self-attention with pyramid positional
encoding.

.. autoclass:: soma.aggregators.mil.transmil.TransMIL
   :members:

HIPT
~~~~

``hipt`` first aggregates tiles within regions, then aggregates regions into
a slide-level representation. This preset assumes hierarchical tiling in the
preprocessing pipeline.

.. autoclass:: soma.aggregators.mil.hipt.HIPT
   :members:

Notes
-----

- ``clam_sb`` is the only CLAM preset that supports regression and ordinal
  classification.
- ``clam_mb`` is classification-only and emits one branch per class.
- ``hipt`` requires hierarchical tiling; set
  ``region_tile_multiple`` in :class:`soma.config.PreprocessingConfig` to
  control how many tiles fit inside a region.
- The task head ultimately determines the valid loss and metric pairing.

Discovery helper
----------------

Use ``soma.list_aggregators()`` to inspect the registered aggregator names
from code when you are wiring configs or building a UI.
