Croma 0.3 encoder panel
=======================

This compatibility audit translates the 26 tile-model names published with
`Croma 0.3.0 <https://github.com/clemsgrs/croma/releases/tag/0.3.0>`_ onto
soma's slide2vec encoder keys. soma validates every row against slide2vec's
runtime registry metadata without constructing a model or loading weights.

The check proves only that the soma encoder slug is a registered tile encoder and
that the explicit output variant resolves to the stated feature dimension. The
``default`` spelling is an explicit lock to that named variant; it is not permission
to consult a future registry default.

.. list-table::
   :header-rows: 1

   * - Published model
     - soma encoder
     - Output variant
     - Feature dimension
   * - Virchow2
     - ``virchow2``
     - ``cls_patch_mean``
     - 2560
   * - CONCH
     - ``conch``
     - ``default``
     - 512
   * - GenBio-PathFM
     - ``genbio-pathfm``
     - ``default``
     - 4608
   * - CONCHv1.5
     - ``conchv15``
     - ``default``
     - 768
   * - H0-mini
     - ``h0-mini``
     - ``cls_patch_mean``
     - 1536
   * - Virchow
     - ``virchow``
     - ``cls_patch_mean``
     - 2560
   * - Midnight-12k
     - ``midnight``
     - ``default``
     - 3072
   * - H-optimus-1
     - ``h-optimus-1``
     - ``default``
     - 1536
   * - DINOv2-B
     - ``dinov2-vitb14``
     - ``default``
     - 768
   * - H-optimus-0
     - ``h-optimus-0``
     - ``default``
     - 1536
   * - UNI2-h
     - ``uni2``
     - ``default``
     - 1536
   * - MUSK
     - ``musk``
     - ``ms_aug``
     - 2048
   * - mSTAR
     - ``mstar``
     - ``default``
     - 1024
   * - Prov-GigaPath
     - ``gigapath``
     - ``default``
     - 1536
   * - UNI
     - ``uni``
     - ``default``
     - 1024
   * - Hibou-B
     - ``hibou-b``
     - ``default``
     - 768
   * - GPFM
     - ``gpfm``
     - ``default``
     - 1024
   * - Phikon
     - ``phikon``
     - ``default``
     - 768
   * - Phikon-v2
     - ``phikonv2``
     - ``default``
     - 1024
   * - Prost40M
     - ``prost40m``
     - ``default``
     - 384
   * - Hibou-L
     - ``hibou-l``
     - ``default``
     - 1024
   * - Mascaret
     - ``mascaret``
     - ``default``
     - 1536
   * - Phaet
     - ``phaet``
     - ``default``
     - 1024
   * - RudolfV-2
     - ``rudolfv2``
     - ``cls_patch_mean``
     - 3072
   * - RudolfV-2-B
     - ``rudolfv2-b``
     - ``cls_patch_mean``
     - 1536
   * - RudolfV-2-S
     - ``rudolfv2-s``
     - ``cls_patch_mean``
     - 768

Limits of this audit
--------------------

Slug, output-variant, and dimension compatibility **does not prove exact numerical identity**
with Croma's published embeddings. The weights and checkpoint revision, preprocessing,
input geometry, normalization, precision, and implementation version remain unpinned and may all
change the resulting values. This mapping records no extraction identity and makes no numerical
reproduction claim.
