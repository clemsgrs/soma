Encoders
========

Encoder choice sets the representation space. In `soma`, the encoder is
selected by ``EncoderConfig.name`` and configured through fields that describe
its native geometry and runtime behavior, including spacing, batch size, and
output variant.

The main configuration object is :class:`soma.config.EncoderConfig`.

.. autoclass:: soma.config.EncoderConfig
   :members:

Primary encoder fields
----------------------

.. list-table::
   :header-rows: 1

   * - Knob
     - Why it matters
   * - ``name``
     - Changes the representation family entirely
   * - ``spacing_um``
     - Indicates the preset's native spacing
   * - ``output_variant``
     - Chooses among supported feature variants for some presets
   * - ``batch_size``
     - Trades speed for memory
   * - ``input_size``
     - Indicates the preset's native input geometry when applicable

Supported presets
-----------------

Tile-level encoders
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Output dim
     - Spacing
     - Notes
   * - ``uni``
     - 1024
     - ``0.5``
     - UNI
   * - ``uni2``
     - 1536
     - ``0.5``
     - UNI2
   * - ``virchow``
     - 1280 / 2560
     - ``0.5``
     - Supports ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``virchow2``
     - 1280 / 2560
     - ``0.5``, ``1.0``, ``2.0``
     - Supports ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``conch``
     - 512
     - ``0.5``
     - CONCH
   * - ``conchv15``
     - 768
     - ``0.5``
     - CONCHv1.5
   * - ``gigapath``
     - 1536
     - ``0.5``
     - Alias: ``prov-gigapath``
   * - ``h-optimus-0``
     - 1536
     - ``0.5``
     - H-optimus-0
   * - ``h-optimus-1``
     - 1536
     - ``0.5``
     - H-optimus-1
   * - ``h0-mini``
     - 768 / 1536
     - ``0.5``
     - Supports ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``phikon``
     - 768
     - ``0.5``
     - Phikon
   * - ``phikonv2``
     - 1024
     - ``0.5``
     - Phikon-v2
   * - ``hibou-b``
     - 768
     - ``0.5``
     - Hibou-B
   * - ``hibou-l``
     - 1024
     - ``0.5``
     - Hibou-L
   * - ``midnight``
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Alias: ``kaiko-midnight``
   * - ``lunit``
     - 384
     - ``0.5``
     - Tile backbone for MOOZY

Slide-level encoders
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Tile encoder
     - Output dim
     - Notes
   * - ``gigapath-slide``
     - ``gigapath``
     - 768
     - Uses slide-level token aggregation
   * - ``prism``
     - ``virchow`` (``cls_patch_mean``)
     - 1280
     - PRISM slide encoder
   * - ``titan``
     - ``conchv15``
     - 768
     - TITAN slide encoder
   * - ``moozy-slide``
     - ``lunit``
     - 768
     - MOOZY slide encoder

Patient-level encoders
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Tile encoder
     - Output dim
     - Notes
   * - ``moozy``
     - ``lunit``
     - 768
     - Patient-level transformer over slide features

Compatibility is enforced by the code and by ``PipelineConfig`` validation.
Use this page to choose a valid starting point, then let the runtime validate
the final combination. If a preset accepts a non-default spacing or input size,
that is a compatibility choice and not necessarily a performance-improving
one.
