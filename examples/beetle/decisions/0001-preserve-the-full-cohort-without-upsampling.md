# Preserve the full cohort without upsampling

The protocol uses all 587 development slides from 527 patients so its held-out tune sets
match the BEETLE baseline and patient bootstrap. Three public TCGA TIFFs tagged at
0.657476 µm/px remain in their organizer-assigned folds and are sampled from native
level-0 pixels rather than interpolated to the nominal 0.5 µm/px analysis spacing.

Their identities, spacings, and native-read decisions are recorded with every run. The
primary out-of-fold result uses all 527 patients; a 584-slide/524-patient subset is a
derived evaluation-only view, not a separately trained experiment.
