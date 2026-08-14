CRoMa
=====

The ``croma`` family measures how robust a frozen tile encoder is to
non-biological variation across medical centers. It runs the CRoMa protocol
from the `croma library <https://clemsgrs.github.io/croma/>`_, which
consolidates PathoROB's Robustness Index and introduces CRoMa, a
distance-aware, tail-sensitive robustness margin that supersedes it
(`paper <https://arxiv.org/abs/2607.25497>`_).

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

Pair resolvability
------------------

Rank agreement is judged over *resolvable* encoder pairs only — pairs the
published reference itself separates meaningfully. CRoMa's ranking metrics live
on a signed scale that crosses zero, where soma's historical scale-blind
absolute rule and a pure relative rule both misbehave, so this family uses a
hybrid rule (the inverse of ``math.isclose``)::

   resolvable  ⇔  |a − b| > max(0.005, 0.02 · max(|a|, |b|))

evaluated on the unrounded reference values exported from the Croma results,
with a strict boundary (a gap exactly at the threshold is a tie). The rule is
symmetric and independent of metric direction. At and near zero it behaves
explicitly: ``(0, 0)`` is a tie; ``0`` versus ``ε`` resolves only when
``ε > 0.005``; ``−ε`` versus ``+ε`` resolves only when
``2ε > 0.005`` —
opposite signs buy nothing beyond the honest gap magnitude, and an arbitrarily
small gap can never become resolvable merely because both values sit near
zero. At ordinary scale the relative term governs, so two encoders 0.005 apart
at a magnitude of 0.4 are a tie rather than a call.

The floor 0.005 is roughly 0.5–0.75 % of both ranking metrics' observed
ranges, and the thresholds were fixed from the published reference
distribution before any soma measurement was recorded (soma#321). One rule
serves both ranking metrics; ``test/croma_f0`` is a reported diagnostic and is
never ranked. Other benchmark families keep the absolute rule they were
published under.

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

**CRoMa median** (``test/croma_median``) — Ranking metric:

.. list-table::
   :header-rows: 1
   :widths: 22 28 18 18 14

   * - Cohort
     - Encoder
     - soma
     - published
     - Δ
   * - camelyon
     - ``conchv15``
     - +0.187013
     - +0.187014
     - -0.000001
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - +0.049686
     - +0.049735
     - -0.000049
   * - camelyon
     - ``h0-mini``
     - +0.166768
     - +0.166648
     - +0.000120
   * - camelyon
     - ``uni``
     - -0.033614
     - -0.033606
     - -0.000008
   * - tcga-4x4
     - ``conchv15``
     - +0.152964
     - +0.153000
     - -0.000036
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - +0.006115
     - +0.006134
     - -0.000019
   * - tcga-4x4
     - ``h0-mini``
     - +0.121186
     - +0.121174
     - +0.000012
   * - tcga-4x4
     - ``uni``
     - +0.047447
     - +0.047437
     - +0.000010
   * - tolkach-esca
     - ``conchv15``
     - +0.391840
     - +0.391815
     - +0.000025
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - +0.175520
     - +0.175370
     - +0.000150
   * - tolkach-esca
     - ``h0-mini``
     - +0.379612
     - +0.379595
     - +0.000017
   * - tolkach-esca
     - ``uni``
     - +0.174555
     - +0.174556
     - -0.000001

**CRoMa LTM10** (``test/croma_ltm10``) — Ranking metric:

.. list-table::
   :header-rows: 1
   :widths: 22 28 18 18 14

   * - Cohort
     - Encoder
     - soma
     - published
     - Δ
   * - camelyon
     - ``conchv15``
     - -0.144638
     - -0.144638
     - -0.000000
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - -0.184008
     - -0.184003
     - -0.000005
   * - camelyon
     - ``h0-mini``
     - -0.159867
     - -0.159871
     - +0.000004
   * - camelyon
     - ``uni``
     - -0.216738
     - -0.216723
     - -0.000015
   * - tcga-4x4
     - ``conchv15``
     - -0.130187
     - -0.130181
     - -0.000006
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - -0.121664
     - -0.121650
     - -0.000014
   * - tcga-4x4
     - ``h0-mini``
     - -0.187505
     - -0.187502
     - -0.000003
   * - tcga-4x4
     - ``uni``
     - -0.120455
     - -0.120443
     - -0.000012
   * - tolkach-esca
     - ``conchv15``
     - -0.028503
     - -0.028501
     - -0.000002
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - -0.068654
     - -0.068650
     - -0.000004
   * - tolkach-esca
     - ``h0-mini``
     - -0.073542
     - -0.073550
     - +0.000008
   * - tolkach-esca
     - ``uni``
     - -0.080229
     - -0.080223
     - -0.000006

**CRoMa F(0)** (``test/croma_f0``) — Reported diagnostic (never ranked):

.. list-table::
   :header-rows: 1
   :widths: 22 28 18 18 14

   * - Cohort
     - Encoder
     - soma
     - published
     - Δ
   * - camelyon
     - ``conchv15``
     - +0.173676
     - +0.173725
     - -0.000049
   * - camelyon
     - ``dinov2-vitb14`` (control)
     - +0.345098
     - +0.345490
     - -0.000392
   * - camelyon
     - ``h0-mini``
     - +0.180245
     - +0.180098
     - +0.000147
   * - camelyon
     - ``uni``
     - +0.651520
     - +0.651324
     - +0.000196
   * - tcga-4x4
     - ``conchv15``
     - +0.193056
     - +0.193056
     - -0.000000
   * - tcga-4x4
     - ``dinov2-vitb14`` (control)
     - +0.469097
     - +0.468403
     - +0.000694
   * - tcga-4x4
     - ``h0-mini``
     - +0.256597
     - +0.256597
     - +0.000000
   * - tcga-4x4
     - ``uni``
     - +0.319618
     - +0.319618
     - +0.000000
   * - tolkach-esca
     - ``conchv15``
     - +0.042778
     - +0.042778
     - -0.000000
   * - tolkach-esca
     - ``dinov2-vitb14`` (control)
     - +0.098778
     - +0.099000
     - -0.000222
   * - tolkach-esca
     - ``h0-mini``
     - +0.057556
     - +0.057556
     - -0.000000
   * - tolkach-esca
     - ``uni``
     - +0.085778
     - +0.085889
     - -0.000111

Rank agreement (pathology encoders, under the pair-resolvability rule above):

* **CRoMa median**: 9/9 resolvable pairs concordant (9 pairs total); camelyon ρ = +1.00, tcga-4x4 ρ = +1.00, tolkach-esca ρ = +1.00.
* **CRoMa LTM10**: 9/9 resolvable pairs concordant (9 pairs total); camelyon ρ = +1.00, tcga-4x4 ρ = +1.00, tolkach-esca ρ = +1.00.

No ranking flips: soma reproduces every published pair ordering.

Across all 36 recorded cells the largest absolute deviation from the published value is **0.000694**.

Every ledger row is provenance-pinned — soma commit ``7276473``; slide2vec 5.8.0; croma 0.3.0; recorded 2026-08-14.

Campaign notes
--------------

The recorded panel is the pinned reproduction campaign (soma#319): four
encoders — ``conchv15``, ``h0-mini``, ``uni``, and the ``dinov2-vitb14``
control — across all three cohorts, produced with::

   soma prepare-croma /var/tmp/croma-data
   soma reproduce croma --raw-root /var/tmp/croma-data \
     --output-root /var/tmp/croma-runs --cache-root /var/tmp/croma-cache \
     --encoders conchv15 dinov2-vitb14 h0-mini uni --record

Runs executed on shared-cluster GPU nodes (an RTX 3080 Ti node, completed on
an RTX 2080 Ti node after a pre-emption; extraction is deterministic, and the
interrupted cohort was re-extracted and re-scored from scratch on the second
node). Single canonical seed — extraction and CRoMa scoring are
seed-independent. Limitations: the remaining panel encoders are unmeasured
here (the published table covers them); rank agreement is over the three
pathology encoders only, so Spearman is coarse; F(0) is reported but never
ranked; nothing gates — every value renders next to its published reference.

See :doc:`benchmarking` for the shared benchmark workflow.
