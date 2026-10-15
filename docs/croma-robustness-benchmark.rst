CRoMa
=====

How robust is a frozen tile encoder to the medical centre a slide comes from? CRoMa
measures it on the frozen embeddings alone, with no downstream model to train, using
the protocol of the `croma library
<https://clemsgrs.github.io/croma/>`_ (`paper <https://arxiv.org/abs/2607.25497>`_).
CRoMa supersedes PathoROB's Robustness Index with a distance-aware, tail-sensitive
robustness margin.

Data and task
-------------

The three tile cohorts come from the `PathoROB study
<https://arxiv.org/abs/2507.17845>`_ and are registered as ``croma/camelyon``,
``croma/tcga-4x4`` and ``croma/tolkach-esca``. Each holds tiles of known biological
class scanned at several medical centres, so that centre and biology can be told
apart.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Task
     - None is trained. Each tile is embedded once, and its biological class and
       medical centre are known; the metrics compare distances in the embedding
       space between tiles of the same biology from different centres and tiles of
       different biology from the same centre.
   * - Metrics
     - ``croma_median``, the median robustness margin and the primary metric, and
       ``croma_ltm10``, the mean of the worst ten percent: for both, higher is better.
       ``croma_f0``, the fraction of confounder-dominant tiles, is a diagnostic where
       lower is better. There is no composite score.

Protocol
--------

The protocol is fixed and task-free; the encoder is the only varied axis. Extraction
is seed-independent, so a cohort and encoder pair is embedded once and reused.

The published panel covers 26 encoders: 25 pathology foundation models and
DINOv2-B (``dinov2-vitb14``), the natural-image control, which is measured and shown
but never ranked. In soma every panel encoder resolves to its slide2vec registry
default output, except:

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

soma checks names, output variants and feature dimensions against the slide2vec
registry at configuration time. It makes no
claim of numerical identity with the published embeddings.

Results
-------

.. include:: _generated/croma-results.rst

The full panel, per-cohort detail and margin distributions are on the `CRoMa results
page <https://clemsgrs.github.io/croma/results/>`_.

Reproduce
---------

**Prepare the data.** Install the preparation dependency and build the pinned raw
tree once; the three cohorts share it::

    pip install 'soma-pathology[croma]'
    soma prepare-croma /data/croma

**Run the family** or a single cohort. One ``--cache-root`` serves all three cohorts,
and repeat runs of a cohort and encoder reuse their cached features::

    soma reproduce croma --raw-root /data/croma --output-root runs/croma --cache-root /fast/croma-cache
    soma reproduce croma/camelyon --encoder uni2 --raw-root /data/croma --output-root runs/croma/camelyon --cache-root /fast/croma-cache

**Compare encoders** with ``--encoders``, and append the scores to the results ledger
with ``--record``. :doc:`benchmarking` describes the options shared by every
benchmark.
