# BEETLE Virchow2 validation

This directory is the tracked project protocol for the BEETLE rebuttal experiment. The
installable `soma` package supplies reusable mechanisms; this bundle owns the dataset
layout, cohort assertions, fold policy, experiment configs, reporting, and submission
rules.

BEETLE is not yet a Benchmark because its official manuscript remains under review. A
published, sufficiently fixed evaluation contract with reference evidence may make it a
Benchmark, but this repository will continue to treat it as a side project unless
maintainers separately promote it to a Built-in Benchmark.

Curate the strict development cohort:

```bash
python -m examples.beetle.curate --beetle-root data/beetle
```

Run a small non-publication curation smoke test:

```bash
python -m examples.beetle.curate --beetle-root data/beetle --slides 4
```

The base segmentation recipe is in `configs/segmentation.yaml`. `CONTEXT.md` defines
the project language and `decisions/` records project-level protocol decisions.
