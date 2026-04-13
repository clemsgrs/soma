Pipeline
=================

The pipeline is the orchestration layer for an experiment:

- read the dataset and split manifests
- preprocess and tile slides
- extract features with the selected encoder
- aggregate features when MIL is used
- train the task head
- evaluate and write a run bundle

The main configuration object is :class:`soma.config.PipelineConfig`.

.. autoclass:: soma.config.PipelineConfig
   :members:

Dataset types
-------------

``dataset_type`` determines which stages are active:

.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
     - Aggregator
   * - ``slide``
     - Whole-slide pipeline with optional MIL aggregation
     - | Required for MIL
       | ``None`` for direct slide heads
   * - ``tile``
     - Patch-level classification/regression
     - Must be ``None``
   * - ``patient``
     - Patient-level aggregation over slide features
     - Must be ``None``

Use the smallest change that answers the current question. In practice, that
usually means varying one stage at a time and keeping the rest of the pipeline
fixed.

Patient-level runs require a ``patient_id`` column in the dataset manifest and
produce one prediction per patient.

Run outputs
-----------

The run directory layout is described in :doc:`outputs`.
