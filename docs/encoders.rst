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
     - HF model
     - Notes
   * - ``uni``
     - 1024
     - ``0.5``
     - ``MahmoodLab/UNI``
     - Chen et al. (2024)
   * - ``uni2``
     - 1536
     - ``0.5``
     - ``MahmoodLab/UNI2-h``
     - Chen et al. (2024)
   * - ``virchow``
     - 1280 / 2560
     - ``0.5``
     - ``paige-ai/Virchow``
     - Vorontsov et al. (2024); ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``virchow2``
     - 1280 / 2560
     - ``0.5``, ``1.0``, ``2.0``
     - ``paige-ai/Virchow2``
     - Zimmermann et al. (2024); ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``conch``
     - 512
     - ``0.5``
     - ``MahmoodLab/conch``
     - Lu et al. (2024)
   * - ``conchv15``
     - 768
     - ``0.5``
     - ``MahmoodLab/TITAN``
     - Lu et al. (2024); CONCHv1.5 tile backbone
   * - ``gigapath``
     - 1536
     - ``0.5``
     - ``prov-gigapath/prov-gigapath``
     - Xu et al. (2024); alias: ``prov-gigapath``
   * - ``h-optimus-0``
     - 1536
     - ``0.5``
     - ``bioptimus/H-optimus-0``
     - Saillard et al. (2024)
   * - ``h-optimus-1``
     - 1536
     - ``0.5``
     - ``bioptimus/H-optimus-1``
     - Saillard et al. (2024)
   * - ``h0-mini``
     - 768 / 1536
     - ``0.5``
     - ``bioptimus/H0-mini``
     - Saillard et al. (2024); ``output_variant="cls"`` or ``"cls_patch_mean"``
   * - ``phikon``
     - 768
     - ``0.5``
     - ``owkin/phikon``
     - Filiot et al. (2023)
   * - ``phikonv2``
     - 1024
     - ``0.5``
     - ``owkin/phikon-v2``
     - Filiot et al. (2024)
   * - ``hibou-b``
     - 768
     - ``0.5``
     - ``histai/hibou-b``
     - Nechaev et al. (2024)
   * - ``hibou-l``
     - 1024
     - ``0.5``
     - ``histai/hibou-L``
     - Nechaev et al. (2024)
   * - ``midnight``
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
     - ``kaiko-ai/midnight``
     - Campanella et al. (2025); alias: ``kaiko-midnight``
   * - ``lunit``
     - 384
     - ``0.5``
     - ``1aurent/vit_small_patch8_224.lunit_dino``
     - Kang et al. (2023); tile backbone for MOOZY
   * - ``prost40m``
     - 384
     - ``0.5``
     - ``waticlems/Prost40M``
     - fp32; prostate pathology specialist
   * - ``musk``
     - 1024 / 2048
     - ``0.25``, ``0.5``, ``1.0``
     - ``xiangjx/musk``
     - Xiang et al. (2024); requires ``pip install git+https://github.com/lilab-stanford/MUSK.git``; ``output_variant="cls"`` or ``"ms_aug"`` (default)

Slide-level encoders
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Tile encoder
     - Output dim
     - HF model
     - Notes
   * - ``gigapath-slide``
     - ``gigapath``
     - 768
     - ``prov-gigapath/prov-gigapath``
     - Xu et al. (2024); slide-level token aggregation
   * - ``prism``
     - ``virchow`` (``cls_patch_mean``)
     - 1280
     - ``paige-ai/Prism``
     - Shaikovski et al. (2024)
   * - ``titan``
     - ``conchv15``
     - 768
     - ``MahmoodLab/TITAN``
     - Ding et al. (2024)
   * - ``moozy-slide``
     - ``lunit``
     - 768
     - ``AtlasAnalyticsLab/MOOZY``
     - Slide-level variant; use when slides are the unit of analysis

Patient-level encoders
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Tile encoder
     - Output dim
     - HF model
     - Notes
   * - ``moozy``
     - ``lunit``
     - 768
     - ``AtlasAnalyticsLab/MOOZY``
     - Patient-level transformer over slide features

Compatibility is enforced by the code and by ``PipelineConfig`` validation.
Use this page to choose a valid starting point, then let the runtime validate
the final combination. If a preset accepts a non-default spacing or input size,
that is a compatibility choice and not necessarily a performance-improving
one.
