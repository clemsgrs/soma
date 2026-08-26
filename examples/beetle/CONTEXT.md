# BEETLE Virchow2 validation

This project bundle owns the scientific protocol for the BEETLE rebuttal experiment.
It uses soma's generic Manifest, segmentation, sampling, evaluation, prediction, and
artifact-publication mechanisms without adding BEETLE policy to the Python package.
The official manuscript is still under review, so the authoritative evaluation contract
may change: this is a Project protocol, not yet a Benchmark. A published, sufficiently
fixed evaluation contract with reference evidence may make BEETLE a Benchmark and
eligible for promotion to a Built-in Benchmark, but promotion would be a separate
decision.

## Project language

**Full development cohort**: all 587 development slides from 527 patients in the
organizer-provided five folds.

**Native-spacing exceptions**: three public TCGA TIFFs tagged at 0.657476 µm/px. They
remain in their assigned folds and are read from native level-0 pixels; no finer tissue
pixels are synthesized.

**Primary OOF view**: confusion evidence pooled over all 527 held-out patients, each
appearing once across the five tune folds.

**Spacing-sensitivity view**: an evaluation-only 584-slide/524-patient view formed by
removing the three exception patients' confusion matrices without retraining.

**Sampling arms**: a uniform ROI arm and a class-conditioned arm with the same resolved
draw and optimizer-update budget. The conditioned arm declares its class-request ratios;
realized pixel and ROI exposure remain diagnostics, not balancing claims.

**Development selection**: choose the sampling arm from fold-macro class Dice on the
development folds only. The official external evaluation set is used once for the chosen
arm.

**Publication evidence**: the curated Manifest and checksums, completed hardware
preflight, locked encoder provenance, resolved arm configs, run/environment metadata,
ten decoder checkpoints, histories and sampler audits, recovery manifests, generic fold
confusion evidence, the project OOF/bootstrap report, and the development arm-selection
record. The gated encoder weights and large feature cache are referenced, not distributed.

**Patient bootstrap**: resample independent patients with replacement and recompute Dice
from summed confusion matrices. The publication protocol fixes the seed and number of
replicates; those values are project policy, not soma defaults.
