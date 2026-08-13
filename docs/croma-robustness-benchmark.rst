CRoMa
=====

The ``croma`` family measures how robust a frozen tile encoder is to
non-biological variation across medical centers. It runs the CRoMa protocol
from the `croma library <https://clemsgrs.github.io/croma/>`_, which
consolidates PathoROB's Robustness Index and introduces CRoMa, a
distance-aware, tail-sensitive robustness margin that supersedes it
(`paper <https://arxiv.org/abs/2607.25497>`_).

The three tile cohorts come from the `PathoROB study
<https://arxiv.org/abs/2507.17845>`_: ``croma/camelyon``, ``croma/tcga-4x4``,
and ``croma/tolkach-esca``. The protocol is fixed and task-free; the encoder is
the only varied axis.

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
   * - Virchow
     - ``virchow``
     - ``cls_patch_mean``
   * - Virchow2
     - ``virchow2``
     - ``cls_patch_mean``
   * - H0-mini
     - ``h0-mini``
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
