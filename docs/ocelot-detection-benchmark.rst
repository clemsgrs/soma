OCELOT
======

*Maps to task:* :doc:`detection` — this is soma's detection path reproduced on the
OCELOT 2023 cell-detection challenge.

A benchmark of soma's :doc:`detection path <detection>` on the
`OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ cell-detection challenge:
a single bundled protocol (config builder + recipe), a verified set of reference
numbers to land against, and a results table filled in by the run issues — **not**
invented here.

.. note::

   This page is a **scaffold**. The reference numbers below are verified from the
   literature; the soma result cells are intentionally left as ``TBD`` and are
   populated by the campaign run issues as the numbers are produced. Do not
   backfill them with estimates.

v1 is **cell-only**. Cell–tissue fusion — where the entire reported headroom over
the cell-only baseline lives — is a deliberate later increment, not smuggled into
the baseline.

Dataset & metric
----------------

OCELOT 2023 provides paired cell + tissue patches from TCGA. detection-v1 uses the
**cell** patches only: 1024×1024 JPEGs at **0.2 µm/px**, with headerless
``x, y, label`` point CSVs (``1`` = background cell → class 0; ``2`` = tumor cell →
class 1). Splits are **400 / 137 / 126** (v1.0.1 drops 4 under-annotated test
cases). See :doc:`curation` and ``examples/ocelot/README.md`` for the download and
curation recipe.

The metric is OCELOT's class-aware **mean F1@δ** over the two cell classes
(background cell, tumor cell), with detections pooled across the split — the
:doc:`detection` page is the canonical definition of the matcher and the px↔µm
conversion; for this benchmark δ and the matcher are fixed to OCELOT's official
settings, and soma's greedy matcher is verified faithful to it.

Operating point (leakage-free): per-class score thresholds are swept on the
**tune** split, frozen, and applied once to test — exactly a real submission.
The tune-frozen greedy test mF1 is the leaderboard-comparable headline. An
**oracle** ceiling (thresholds tuned on test) is reported alongside as a labelled
diagnostic upper bound, never as the result; a large oracle–headline gap signals a
fragile operating point.

Reference numbers
-----------------

Verified from `arXiv:2509.09153 <https://arxiv.org/abs/2509.09153>`_. These are
**targets, not gates** — the frozen cell-only probe is strictly weaker than a
trained cell-only model, so the cell-only baseline (63.54%) and the fusion ceiling
(~72%) are the two numbers to chase.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Entry
     - mF1
     - Notes
   * - Fine-tuned cell-only baseline
     - **63.54%** [95% CI 59.88–66.76]
     - trained deep cell-only model
   * - Fusion top-5 teams
     - 69.92% – **72.44%**
     - all use cell–tissue fusion
   * - Winner (Li et al.)
     - 72.44%
     - +7.99 over the baseline

Encoder × spacing ablation
--------------------------

The campaign runs a 2×2 encoder × spacing grid plus a native anchor, isolating the
**pretraining-alignment effect** (does dropping 0.5 → 0.25 µm/px *help* the
mixed-magnification Virchow2 but *hurt* the 20×-only UNI2). The headline probe is
**Virchow2**; UNI2 is an ablation point. Each cell is run for **3 seeds**
(mean ± std); the dense extraction is seed-independent and shared across seeds and
decoder ablations.

Watch per-class **recall** especially: token size is fixed at 14 px, so token-µm
scales with spacing (3.5 µm @ 0.25 vs 7 µm @ 0.5 against δ = 3 µm), and coarse
localization manifests as missed cells.

Results
-------

soma greedy test mF1 (headline), mean ± std over 3 seeds, with the Hungarian
secondary, the oracle ceiling, and per-class recall. **Populated by the run
issues — do not invent results.**

.. list-table::
   :header-rows: 1
   :widths: 16 12 22 16 14 20

   * - Encoder
     - Spacing (µm/px)
     - Greedy mF1 (headline) ± std
     - Hungarian mF1 (secondary)
     - Oracle ceiling
     - Per-class recall (BC / TC)
   * - Virchow2
     - 0.2 (native anchor)
     - TBD
     - TBD
     - TBD
     - TBD
   * - Virchow2
     - 0.25
     - TBD
     - TBD
     - TBD
     - TBD
   * - Virchow2
     - 0.5
     - TBD
     - TBD
     - TBD
     - TBD
   * - UNI2
     - 0.25
     - TBD
     - TBD
     - TBD
     - TBD
   * - UNI2
     - 0.5
     - TBD
     - TBD
     - TBD
     - TBD

Reproduce
---------

The full curate → train → greedy-score → check recipe lives at
``examples/ocelot/README.md``; the published anchor (frozen Virchow2 @ 0.2 µm/px)
is reproduced end to end by ``examples/ocelot/reproduce.py``. The encoder × spacing
grid is the set of ``examples/ocelot/ocelot_{virchow2,uni2}_*.yaml`` configs.

.. seealso::

   * :doc:`detection` — the detection modeling substrate (head, target encoding,
     loss, F1@δ evaluator).
   * ``examples/ocelot/README.md`` — download, curation, and run recipe.
   * ``design/ocelot-detection-benchmark-design.md`` — the full campaign design
     (gates, scoring fidelity, encoder choice, statistical rigor).
