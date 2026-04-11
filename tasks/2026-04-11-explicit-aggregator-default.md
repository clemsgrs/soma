# Explicit Aggregator Config

- [x] Remove the implicit `abmil` default from `AggregatorConfig`.
- [x] Make `PipelineConfig.aggregator` default to `None` for slide-level runs.
- [x] Update pipeline entrypoints that relied on the hidden default.
- [x] Add tests that enforce the explicit-aggregator contract.
- [x] Update project documentation to reflect the new config behavior.
