CRoMa
=====

The ``croma`` family measures how robust a frozen tile encoder is to
non-biological variation across medical centers. It runs the CRoMa protocol
from the `croma library <https://clemsgrs.github.io/croma/>`_
(`paper <https://arxiv.org/abs/2607.25497>`_), which supersedes PathoROB's
Robustness Index with a distance-aware, tail-sensitive robustness margin.

The three tile cohorts come from the `PathoROB study
<https://arxiv.org/abs/2507.17845>`_: ``croma/camelyon``, ``croma/tcga-4x4``, and ``croma/tolkach-esca``.
The protocol is fixed and task-free; the encoder is the only varied axis.

Run it
------

Install the preparation dependency and build the pinned raw tree once::

   pip install 'soma-pathology[croma]'
   soma prepare-croma /data/croma

Reproduce the whole family, or a single cohort::

   soma reproduce croma --raw-root /data/croma \
     --output-root runs/croma --cache-root /fast/croma-cache --record

   soma reproduce croma/camelyon --encoder uni2 --raw-root /data/croma \
     --output-root runs/croma/camelyon --cache-root /fast/croma-cache

Extraction is seed-independent, so repeat runs of the same cohort and encoder
reuse their cached tile features. One ``--cache-root`` may serve all three
cohorts.

Metrics
-------

Each run reports three metrics:

* ``test/croma_median`` — the median robustness margin; the primary metric,
  higher is better.
* ``test/croma_ltm10`` — the mean of the worst ten-percent tail; higher is
  better.
* ``test/croma_f0`` — the fraction of confounder-dominant samples; a
  diagnostic, lower is better.

There is no composite score. DINOv2-B (``dinov2-vitb14``) is the natural-image
control: it is measured and shown, but never ranked.

Encoder panel
-------------

The published panel covers 26 encoders: 25 pathology foundation models and the
DINOv2-B control. The full panel, per-cohort detail, and margin distributions
are on the `CRoMa results page <https://clemsgrs.github.io/croma/results/>`_.

In soma, every panel encoder resolves to its slide2vec registry default
output, except:

.. list-table::
   :header-rows: 1

   * - Published model
     - soma encoder
     - Output variant
   * - Virchow2
     - ``virchow2``
     - ``cls_patch_mean``
   * - H0-mini
     - ``h0-mini``
     - ``cls_patch_mean``
   * - Virchow
     - ``virchow``
     - ``cls_patch_mean``
   * - MUSK
     - ``musk``
     - ``ms_aug``
   * - RudolfV-2
     - ``rudolfv2``
     - ``cls_patch_mean``
   * - RudolfV-2-B
     - ``rudolfv2-b``
     - ``cls_patch_mean``
   * - RudolfV-2-S
     - ``rudolfv2-s``
     - ``cls_patch_mean``

soma validates this mapping against the slide2vec registry at configuration
time; it checks names, output variants, and feature dimensions, and makes no
claim of numerical identity with the published embeddings.

Results
-------

The recorded panel covers ``conchv15``, ``h0-mini``, ``uni``, and the ``dinov2-vitb14`` control across all three cohorts. Values are rounded to three decimals. A soma value would be shown in red if it deviated from the published value by more than 0.005; none does — across all 36 recorded values the largest deviation is 0.0007, and soma reproduces every published pair ordering.

**CRoMa median** (``test/croma_median``) — ranking metric:

.. list-table::
   :header-rows: 1
   :widths: 25 35 20 20

   * - Cohort
     - Encoder
     - soma
     - published
   * - camelyon
     - ``conchv15``
     - 0.187
     - 0.187
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - 0.050
     - 0.050
   * - camelyon
     - ``h0-mini``
     - 0.167
     - 0.167
   * - camelyon
     - ``uni``
     - -0.034
     - -0.034
   * - tcga-4x4
     - ``conchv15``
     - 0.153
     - 0.153
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - 0.006
     - 0.006
   * - tcga-4x4
     - ``h0-mini``
     - 0.121
     - 0.121
   * - tcga-4x4
     - ``uni``
     - 0.047
     - 0.047
   * - tolkach-esca
     - ``conchv15``
     - 0.392
     - 0.392
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - 0.176
     - 0.175
   * - tolkach-esca
     - ``h0-mini``
     - 0.380
     - 0.380
   * - tolkach-esca
     - ``uni``
     - 0.175
     - 0.175

**CRoMa LTM10** (``test/croma_ltm10``) — ranking metric:

.. list-table::
   :header-rows: 1
   :widths: 25 35 20 20

   * - Cohort
     - Encoder
     - soma
     - published
   * - camelyon
     - ``conchv15``
     - -0.145
     - -0.145
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - -0.184
     - -0.184
   * - camelyon
     - ``h0-mini``
     - -0.160
     - -0.160
   * - camelyon
     - ``uni``
     - -0.217
     - -0.217
   * - tcga-4x4
     - ``conchv15``
     - -0.130
     - -0.130
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - -0.122
     - -0.122
   * - tcga-4x4
     - ``h0-mini``
     - -0.188
     - -0.188
   * - tcga-4x4
     - ``uni``
     - -0.120
     - -0.120
   * - tolkach-esca
     - ``conchv15``
     - -0.029
     - -0.029
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - -0.069
     - -0.069
   * - tolkach-esca
     - ``h0-mini``
     - -0.074
     - -0.074
   * - tolkach-esca
     - ``uni``
     - -0.080
     - -0.080

**CRoMa F(0)** (``test/croma_f0``) — diagnostic, never ranked:

.. list-table::
   :header-rows: 1
   :widths: 25 35 20 20

   * - Cohort
     - Encoder
     - soma
     - published
   * - camelyon
     - ``conchv15``
     - 0.174
     - 0.174
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - 0.345
     - 0.345
   * - camelyon
     - ``h0-mini``
     - 0.180
     - 0.180
   * - camelyon
     - ``uni``
     - 0.652
     - 0.651
   * - tcga-4x4
     - ``conchv15``
     - 0.193
     - 0.193
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - 0.469
     - 0.468
   * - tcga-4x4
     - ``h0-mini``
     - 0.257
     - 0.257
   * - tcga-4x4
     - ``uni``
     - 0.320
     - 0.320
   * - tolkach-esca
     - ``conchv15``
     - 0.043
     - 0.043
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - 0.099
     - 0.099
   * - tolkach-esca
     - ``h0-mini``
     - 0.058
     - 0.058
   * - tolkach-esca
     - ``uni``
     - 0.086
     - 0.086

See :doc:`benchmarking` for the shared benchmark workflow.
