Aggregators
===========

Aggregators combine a slide's tile features into a representation for
multiple-instance learning (MIL). In YAML, select a preset with
``aggregation.name``. In Python, pass ``AggregatorConfig(name=...)`` to
``PipelineConfig(aggregator=...)``. Patient-level pipelines use a frozen
patient encoder instead.

.. figure:: /_static/figures/slide-level.svg
   :figclass: soma-figure
   :alt: A frozen tile encoder turns a slide into a bag of features; a trained MIL aggregator or a frozen slide-level FM pools it into one slide-level vector for the task head.

   A trained MIL aggregator pools the bag of tile features into one slide-level
   vector.

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

ABMIL
~~~~~

``abmil`` returns tile-level attention scores for heatmap generation.

.. autoclass:: soma.aggregators.mil.abmil.ABMIL
   :members:

CLAM-SB
~~~~~~~

``clam_sb`` supports binary, multiclass, ordinal, and single-target regression
tasks, and can mix bag-level and instance-level supervision.

.. autoclass:: soma.aggregators.mil.clam.CLAM_SB
   :members:

CLAM-MB
~~~~~~~

``clam_mb`` requires ``multiclass_classification``.

.. autoclass:: soma.aggregators.mil.clam.CLAM_MB
   :members:

DSMIL
~~~~~

``dsmil`` requires ``binary_classification``.

.. autoclass:: soma.aggregators.mil.dsmil.DSMIL
   :members:

DTFDMIL
~~~~~~~

``instances_per_group`` (default 1) sets how many instances each pseudo-bag
contributes; the partition is random during training and contiguous in eval mode.

.. autoclass:: soma.aggregators.mil.dtfdmil.DTFDMIL
   :members:

TransMIL
~~~~~~~~

.. autoclass:: soma.aggregators.mil.transmil.TransMIL
   :members:

HIPT
~~~~

``hipt`` assumes hierarchical tiling; set ``preprocessing.region_tile_multiple``
to control how many tiles fit along each side of a region.

.. autoclass:: soma.aggregators.mil.hipt.HIPT
   :members:

Aggregator interface
--------------------

The shared base classes are :class:`soma.aggregators.base.Aggregator` and
:class:`soma.aggregators.base.AggregatorOutput`.

.. autoclass:: soma.aggregators.base.Aggregator
   :members:

.. autoclass:: soma.aggregators.base.AggregatorOutput
   :members:

``soma.list_aggregators()`` lists the registered names.
