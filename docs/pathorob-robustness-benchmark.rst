PathoROB robustness benchmarks
==============================

The ``pathorob`` family compares frozen tile encoders on three deterministic
PathoROB Robustness Index views. It contains ``pathorob/camelyon``,
``pathorob/tcga-4x4``, and ``pathorob/tolkach-esca``. Each member fixes the
same task-free representation protocol: Croma, ``medical_center`` as the
confounder, the ``test`` split, ``evaluation_design=all``, ``m=5``, and
``alpha=0.10``. The encoder is the only varied axis and the canonical seed is
``0``.

Prepare and reproduce
---------------------

Install the preparation dependency and build the pinned raw tree once::

   pip install 'soma-pathology[pathorob]'
   soma prepare-pathorob /data/pathorob

The family command uses that prepared root and fans out over all three cohorts::

   soma reproduce pathorob --raw-root /data/pathorob \
     --output-root runs/pathorob --cache-root /fast/pathorob-cache --record

A single member can be run or re-scored independently::

   soma reproduce pathorob/camelyon --encoder uni2 --raw-root /data/pathorob \
     --output-root runs/pathorob/camelyon --cache-root /fast/pathorob-cache
   soma reproduce pathorob/camelyon --encoder uni2 \
     --from-run-dir runs/pathorob/camelyon/seed_0 --record

``--from-run-dir`` reads the three existing summary metrics and performs no
curation, extraction, or training. Family-wide ``--from-run-dir`` is not
supported because one run belongs to one cohort.

Metrics and ranking
-------------------

All three Reported metrics are persisted and rendered, in this order:

* ``test/croma_median`` — higher is better; Ranking metric and the primary metric.
* ``test/croma_f0`` — lower is better; reported diagnostic only, with no rank
  and no participation in Pareto dominance.
* ``test/croma_ltm10`` — higher is better; Ranking metric exposing the worst
  ten-percent tail.

There is no composite robustness score. DINOv2-B (``dinov2-vitb14``) is the
natural-image control: all three measurements and reference deltas stay visible,
but it is excluded from per-metric ranks, pairwise rank agreement, and Pareto
analysis. The other 25 published pathology encoders are ranking-eligible.

References and provenance
-------------------------

Each measured value renders beside its Croma-published External reference and
the signed ``measured - reference`` delta. External rows provide context only:
their tolerance is blank and they never produce a pass/fail gate.

``reference/pathorob.csv`` contains the complete 26 encoder × 3 cohort × 3
metric panel. ``reference/pathorob.provenance.json`` separately pins the Croma
``0.3.0`` source release, the artifact producer (``croma_version=0.2.0``), the
2026-08-10 export, protocol, source-file checksums, and paper link. This source
artifact provenance is not the runtime environment. Rows added by
``--record`` store the Croma version actually installed for that soma run,
alongside the soma commit and slide2vec version, without rounding away
rank-relevant precision.

The published panel uses the audited Croma 0.3 encoder mapping. For a panel
encoder, configuration construction validates every explicit slide2vec output
variant and feature dimension. Other registered tile encoders remain runnable
with their registry default but have no packaged reference row.

Cache behavior
--------------

Extraction is seed-independent. Repeat runs and re-scoring of the same cohort ×
encoder reuse its cached tile features. One explicit ``--cache-root`` may serve
all three cohorts because their sample identifiers are globally unique. The
cohorts contain different tiles, however, so the first run of a second cohort
still encodes those uncached tiles; sharing a cache directory does not imply
cross-cohort compute reuse. Reuse across datasets occurs only when sample
identity and the complete cache key genuinely match.
