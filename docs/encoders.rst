Encoders
========

Encoder choice sets the representation space. In `soma`, the encoder is
selected by ``EncoderConfig.name`` and configured through runtime behavior
fields such as precision, batch size, and output variant. Geometry is handled
through preprocessing, not the encoder config itself.

**Encoding methods**

- **Single encoder** — pick one preset from the model zoo below (the common case).
- **Composite (multi-encoder)** — concatenate several presets into one richer
  per-position vector: :doc:`details <encoders/composite>` ·
  :doc:`tutorial <tutorials/walkthrough-composite>`.

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
     - Model
     - Output dim
     - Spacing
   * - ``lunit``
     - `Lunit ViT-S/8 <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`_
     - 384
     - ``0.5``
   * - ``prost40m``
     - `Prost40M <https://huggingface.co/waticlems/Prost40M>`_
     - 384
     - ``0.5``
   * - ``rudolfv2-s``
     - `RudolfV 2-S <https://huggingface.co/Aignostics/RudolfV-2-S>`_
     - 384 / 768
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``conch``
     - `CONCH <https://huggingface.co/MahmoodLab/conch>`_
     - 512
     - ``0.5``
   * - ``phikon``
     - `Phikon <https://huggingface.co/owkin/phikon>`_
     - 768
     - ``0.5``
   * - ``conchv15``
     - `CONCHv1.5 <https://huggingface.co/MahmoodLab/TITAN>`_
     - 768
     - ``0.5``
   * - ``hibou-b``
     - `Hibou-B <https://huggingface.co/histai/hibou-b>`_
     - 768
     - ``0.5``
   * - ``h0-mini``
     - `H0-mini <https://huggingface.co/bioptimus/H0-mini>`_
     - 768 / 1536
     - ``0.5``
   * - ``rudolfv2-b``
     - `RudolfV 2-B <https://huggingface.co/Aignostics/RudolfV-2-B>`_
     - 768 / 1536
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``phikonv2``
     - `Phikon-v2 <https://huggingface.co/owkin/phikon-v2>`_
     - 1024
     - ``0.5``
   * - ``hibou-l``
     - `Hibou-L <https://huggingface.co/histai/hibou-L>`_
     - 1024
     - ``0.5``
   * - ``uni``
     - `UNI <https://huggingface.co/MahmoodLab/UNI>`_
     - 1024
     - ``0.5``
   * - ``musk``
     - `MUSK <https://huggingface.co/xiangjx/musk>`_
     - 1024 / 2048
     - ``0.25``, ``0.5``, ``1.0``
   * - ``gpfm``
     - `GPFM <https://huggingface.co/majiabo/GPFM>`_
     - 1024
     - ``0.5``
   * - ``mstar``
     - `mSTAR <https://huggingface.co/Wangyh/mSTAR>`_
     - 1024
     - ``0.5``
   * - ``isight``
     - `iSight <https://huggingface.co/nirschl-lab/iSight>`_
     - 1024
     - ``0.5``
   * - ``phaet``
     - `Phaet <https://huggingface.co/wearewaiv/phaet>`_
     - 1024
     - ``0.5``
   * - ``virchow``
     - `Virchow <https://huggingface.co/paige-ai/Virchow>`_
     - 1280 / 2560
     - ``0.5``
   * - ``virchow2``
     - `Virchow2 <https://huggingface.co/paige-ai/Virchow2>`_
     - 1280 / 2560
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``uni2``
     - `UNI2 <https://huggingface.co/MahmoodLab/UNI2-h>`_
     - 1536
     - ``0.5``
   * - ``gigapath``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - 1536
     - ``0.5``
   * - ``h-optimus-0``
     - `H-Optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`_
     - 1536
     - ``0.5``
   * - ``h-optimus-1``
     - `H-Optimus-1 <https://huggingface.co/bioptimus/H-optimus-1>`_
     - 1536
     - ``0.5``
   * - ``rudolfv2``
     - `RudolfV 2 <https://huggingface.co/Aignostics/RudolfV-2>`_
     - 1536 / 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``mascaret``
     - `Mascaret <https://huggingface.co/wearewaiv/mascaret>`_
     - 1536
     - ``0.5``
   * - ``midnight``
     - `Midnight <https://huggingface.co/kaiko-ai/midnight>`_
     - 3072
     - ``0.25``, ``0.5``, ``1.0``, ``2.0``
   * - ``genbio-pathfm``
     - `GenBio-PathFM <https://huggingface.co/genbio-ai/genbio-pathfm>`_
     - 4608
     - ``0.5``

Natural-image control
~~~~~~~~~~~~~~~~~~~~~~~

A non-pathology baseline that shares the tile-encoder interface, for measuring
how much pathology pretraining actually contributes.

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Output dim
     - Spacing
   * - ``dinov2-vitb14``
     - `DINOv2 ViT-B/14 <https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m>`_
     - 768
     - ``0.5``

Slide-level encoders
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Output dim
   * - ``gigapath-slide``
     - `GigaPath <https://huggingface.co/prov-gigapath/prov-gigapath>`_
     - ``gigapath``
     - 768
   * - ``titan``
     - `TITAN <https://huggingface.co/MahmoodLab/TITAN>`_
     - ``conchv15``
     - 768
   * - ``prism``
     - `PRISM <https://huggingface.co/paige-ai/Prism>`_
     - ``virchow`` (``cls_patch_mean``)
     - 1280
   * - ``prism2``
     - `PRISM2 <https://huggingface.co/paige-ai/Prism2>`_
     - ``virchow2`` (``cls``)
     - 2560 (``base``) / 3072 (``diagnostic``)
   * - ``moozy-slide``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - 768

Patient-level encoders
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Preset
     - Model
     - Tile encoder
     - Output dim
   * - ``moozy``
     - `MOOZY <https://huggingface.co/AtlasAnalyticsLab/MOOZY>`_
     - ``lunit``
     - 768

Compatibility is enforced by the code and by ``PipelineConfig`` validation.
Use this page to choose a valid starting point, then let the runtime validate
the final combination. The spacing table is a reference for selecting
preprocessing geometry, not a separate encoder knob.

Resolving a fixed benchmark MIL recipe
--------------------------------------

A slide-level benchmark that accepts both tile- and slide-level encoders can
use :func:`soma.encoders.resolve_aggregator` when it builds each
``PipelineConfig``. The helper returns the supplied ``AggregatorConfig`` for a
tile encoder and ``None`` for a slide encoder, allowing the existing slide
representation path to pass the embedding directly to the task head. Patient
encoders are rejected because their representations are not slide-level.

Resolution reads only the encoder's registry metadata; it does not construct
the encoder or load weights, so config generation remains offline-safe. The
benchmark owns the scientific MIL recipe passed to ``tile_recipe`` because the
helper only decides whether that fixed recipe applies to the encoder level.

.. autofunction:: soma.encoders.resolve_aggregator

Discovery helpers
-----------------

Use ``soma.list_models()`` when you want the available encoder presets in code.
Pass ``level="tile"``, ``"slide"``, or ``"patient"`` to narrow the list.
