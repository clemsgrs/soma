Encoders
========

Encoder choice sets the representation space. In `soma`, the encoder is
selected by ``EncoderConfig.name`` and configured through fields that describe
its native geometry and runtime behavior, including spacing, batch size, and
output variant.

The main configuration object is :class:`soma.config.EncoderConfig`.

.. autoclass:: soma.config.EncoderConfig
   :members:

Model Zoo
---------

The tile encoder zoo below is grouped by output dimension for easier scanning,
with entries inside each bucket still following the existing date ordering.

Tile-level encoders
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Output dim
     - Spacing
     - Notes
   * - ``lunit``
     - 384
     - ``0.5``
     - Kang et al. (2023)
   * - ``prost40m``
     - 384
     - ``0.5``
     - Grisi et al. (2026)
   * - ``conch``
     - 512
     - ``0.5``
     - Lu et al. (2024)
   * - ``phikon``
     - 768
     - ``0.5``
     - Filiot et al. (2023)
   * - ``conchv15``
     - 768
     - ``0.5``
     - Lu et al. (2024)
   * - ``hibou-b``
     - 768
     - ``0.5``
     - Nechaev et al. (2024)
   * - ``h0-mini``
     - 768 / 1536
     - ``0.5``
     - Filiot et al. (2024)
   * - ``phikonv2``
     - 1024
     - ``0.5``
     - Filiot et al. (2024)
   * - ``hibou-l``
     - 1024
     - ``0.5``
     - Nechaev et al. (2024)
   * - ``uni``
     - 1024
     - ``0.5``
     - Chen et al. (2024)
   * - ``musk``
     - 1024 / 2048
     - ``0.25``, ``0.5``, ``1.0``
     - Xiang et al. (2024)
   * - ``virchow``
     - 1280 / 2560
     - ``0.5``
     - Vorontsov et al. (2024)
   * - ``virchow2``
     - 1280 / 2560
     - ``0.5``, ``1.0``, ``2.0``
     - Zimmermann et al. (2024)
   * - ``uni2``
     - 1536
     - ``0.5``
     - Chen et al. (2024)
   * - ``gigapath``
     - 1536
     - ``0.5``
     - Xu et al. (2024)
   * - ``h-optimus-0``
     - 1536
     - ``0.5``
     - Saillard et al. (2024)
   * - ``h-optimus-1``
     - 1536
     - ``0.5``
     - Saillard et al. (2024)
   * - ``midnight``
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - Karasikov et al. (2025)

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
     - Xu et al. (2024)
   * - ``titan``
     - ``conchv15``
     - 768
     - Ding et al. (2024)
   * - ``prism``
     - ``virchow`` (``cls_patch_mean``)
     - 1280
     - Shaikovski et al. (2024)
   * - ``moozy-slide``
     - ``lunit``
     - 768
     - Kotp et al. (2026)

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
     - Kotp et al. (2026)

Compatibility is enforced by the code and by ``PipelineConfig`` validation.
Use this page to choose a valid starting point, then let the runtime validate
the final combination. If a preset accepts a non-default spacing or input size,
that is a compatibility choice and not necessarily a performance-improving
one.

Discovery helpers
-----------------

Use ``soma.list_models()`` when you want the available encoder presets in code.
Pass ``level="tile"``, ``"slide"``, or ``"patient"`` to narrow the list.
