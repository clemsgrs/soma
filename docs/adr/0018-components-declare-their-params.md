# Components declare their params at registration

Every swappable component kind is a `soma.registry.Registry`: aggregators, augmentations, benchmarks, callbacks, decoders, heads, learners, optimizers, samplers and schedulers, plus a read-only view of slide2vec's encoders. A component is registered with `register(name, cls, *, params=<dataclass>, infer=None)`, and `soma.components(kind=None)` is the one discovery function.

## Why

Encoders, aggregators, decoders and benchmarks were listed by ad-hoc functions, and callbacks and samplers had no registry at all. A component's knobs and their defaults existed only in its constructor signature, so nothing could check a config before building it, list a component's params, or dump an editable config with every default filled. Task heads injected cohort-derived values through a classmethod, `auto_params(records)`, that every subclass inherited whether it applied or not.

## Decisions

- **Params are a dataclass passed at registration.** Its fields mirror the constructor keywords a user sets, with their defaults; validation lives in its `__post_init__`. Values the run supplies are not fields: `input_dim`, a head's `metrics`, a dense head's `geometry` and spacing. Every soma-owned entry declares one, and a conformance test checks that each field is a constructor keyword with the same default. A user registration may omit it: the entry is listed, its params pass unvalidated.
- **Signatures soma does not own are derived.** The torch optimizers and schedulers, and each benchmark's `build_config` axes, get a dataclass derived from the signature (`signature_params`), so they follow the upstream signature.
- **`infer(cohort, params) -> dict` replaces `auto_params`.** It returns the fields the cohort decides: the class count and label map, the survival bin count. Explicit params win when the two are merged. A hook belongs to a registration and is never inherited; the built-in hooks (`infer_label_map`, `infer_num_bins`) are public so a subclass registration can reuse them.
- **One convention, two rungs.** A dotted path (`mylab.models.FancyMIL`) is imported by `Registry.resolve` with a one-line warning, once per name; its params pass through unvalidated and it is never listed. A dotted path naming a registered class is that entry. `soma.register(kind, name, cls, params=..., infer=...)` adds validation, defaults and listing. A `Params` attribute on a class is not read.
- **Validation where params enter a build; values unchanged.** The pipeline still reads `PipelineConfig`. The learner, optimizer, scheduler and augmentation leaves, the aggregator and decoder params and the ROI batch sampler are validated as they are built, and the constructor receives the keys as given, so a run builds exactly what it built before. Head params are not validated yet: `task.params` still mixes config keys (`classes`, `ignore`, `drop`, `loss`) with constructor keywords until heads become `{name, params}` leaves. (Superseded by [ADR 0020](0020-config-mirrors-the-api-resolve-to-a-plan.md): heads are leaves validated like every other component, and `soma.resolve` expands every leaf's params.)
- **`soma.components(kind=None)`** returns the kind names, or one kind's entries as `{name, class, params, **metadata}`, where `params` maps each field to its default. `soma list [KIND]` and the docs generator print it. `list_models`, `list_aggregators`, `list_decoders`, `list_task_heads`, `list_benchmarks` and `Registry.list_with_metadata` are removed without aliases.

## Consequences

- The Cox head is registered as `cox_survival` (`survival` with `loss: cox` still selects it; [ADR 0020](0020-config-mirrors-the-api-resolve-to-a-plan.md) removes that selector), so the class name a bundle records for it resolves to a registered entry.
- The ROI batch strategies are the samplers `uniform` and `class_conditioned` (thin `SegmentationRoiBatchSampler` subclasses the pipeline builds by name); `event_balanced` registers the survival sampler.
- A config's task, aggregator and decoder names may be dotted import paths. A head named that way gets no `infer` hook, so its config sets `num_classes` and `label_map` itself.
- A registered subclass of a built-in head that relied on the inherited `auto_params` must pass the base's params dataclass and `infer` hook.
- Cache keys keep reading slide2vec's encoder metadata directly; soma's `encoders` view adds `class` and `params` for listing only.
- `soma list tasks` is now `soma list heads`; `soma list` without a kind prints the kinds.
