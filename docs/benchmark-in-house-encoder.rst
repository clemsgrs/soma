BYO Encoder
===========

An in-house encoder enters soma as an installed slide2vec preset. Keep the plugin in its
own package; soma needs only its registered name.

For packaging, encoder implementation, weight loading, credentials, installed-provider
discovery, and worker availability, follow the `slide2vec custom encoder plugin guide
<https://clemsgrs.github.io/slide2vec/models.html#custom-encoder-plugin-package>`_. Those
details belong to slide2vec and are not repeated here.

Install and inspect
-------------------

Install the plugin in the same Python environment as soma, then confirm that public
discovery exposes the preset::

   pip install ./my-slide2vec-encoders
   soma list encoders

The private name should appear beside the public model-zoo names. If it does not, repair
the plugin installation or provider failure before running a Benchmark.

Choose a compatible Benchmark
-----------------------------

List the registered protocols::

   soma list benchmarks

Choose one whose input mode and geometry fit the preset. A pooled tile encoder is a good
match for an EVA patch-classification member; a dense Benchmark additionally requires the
plugin's dense contract. The Benchmark remains fixed while the encoder varies.

Run private and public presets together
---------------------------------------

Name the private preset and every public comparator explicitly::

   soma reproduce eva/bach --encoders my-private-encoder uni2 --raw-root /data/eva/bach --output-root runs/eva-bach --seeds 1

Before curation or execution, soma preflights the complete panel through slide2vec's public
registry, capability, output, and geometry contracts. All preflight errors are reported
together and no Run starts until every cell is compatible. ``--seeds 1`` is a quick smoke;
omit it for the Benchmark's canonical seeds.

Each requested comparator is executed locally. A packaged Reference supplies context for
a matching public preset; it never substitutes for that Run. A private preset without a
packaged Reference is still a Measured row and reports ``REFERENCE SKIPPED``.

Failures and Benchmark families
-------------------------------

After preflight succeeds, runtime failures are isolated. Later encoders continue, completed
Runs remain valid, and soma writes a ``PARTIAL`` Leaderboard when at least one Run completed.
The command prints one failure summary and exits nonzero, so automation cannot mistake a
partial panel for success.

A family name applies the same ordered panel to every concrete member::

   soma reproduce eva --encoders my-private-encoder uni2 --raw-root /data/eva --output-root runs/eva --seeds 1

The output is one Leaderboard per concrete dataset, splits, and task. soma never combines
family members into a cross-dataset rank.

Interpret the Leaderboard
-------------------------

The automatically written Leaderboard holds the protocol fixed and varies only
``encoder``. Its Measured values come from completed local Runs; ``n`` and the spread show
how seeds were collapsed. Read any Reference and delta as packaged context for that encoder,
and compare the private Measured row directly with the locally Measured public rows. Run and
cache identity remain the ordinary preset-name-based soma identity, so reruns reuse the
normal cache rather than creating a plugin-specific format.
