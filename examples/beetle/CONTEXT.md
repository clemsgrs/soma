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

**External submission**: average the selected arm's five fold-checkpoint pixel
probabilities, map every ROI prediction back to its exact supplied dimensions, translate
class indices to the organizer's labels 1 through 4, then validate the exact 170-name
grayscale-PNG set before writing one flat ZIP.

**External ROI sidecar**: the paper-lead-supplied, schema-v1 ROI-to-WSI map that declares
each flat PNG's filename, source WSI, patient, native spacing, width, and height. It is the
authoritative physical-scale and grouping provenance for External inference; image names
are never parsed to recover patients or source slides.

**External patient report**: an optional, label-gated report over 54 independent patients.
It groups all nested ROI confusion matrices through the External sidecar and reuses the
same fixed patient bootstrap as development reporting. Submission generation and
validation never require this report or the sequestered labels.

**Publication evidence**: the curated Manifest and checksums, completed hardware
preflight, locked encoder provenance, resolved arm configs, run/environment metadata,
ten decoder checkpoints, histories and sampler audits, recovery manifests, generic fold
confusion evidence, the project OOF/bootstrap report, and the development arm-selection
record; plus the validated External ROI sidecar, 170 submission PNGs, flat ZIP, and
submission audit. The External patient report joins the evidence only if the paper lead
supplies sequestered labels. The gated encoder weights and large feature cache are
referenced, not distributed.

**Patient bootstrap**: resample independent patients with replacement and recompute Dice
from summed confusion matrices. The publication protocol fixes the seed and number of
replicates; those values are project policy, not soma defaults.
