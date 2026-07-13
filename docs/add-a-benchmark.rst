Add a benchmark
===============

Turn a pathology experiment into a named, repeatable command:

.. code-block:: console

   soma reproduce my-study --raw-root /path/to/data

A benchmark is a thin adapter around the same modular pipeline used for ordinary
experiments. It defines how to prepare the source data, build a fixed protocol,
score a completed run, and compare the result with a reference. Keep model and
training logic in reusable Soma components; keep only the benchmark-specific
choices here.

.. important::

   The current registry discovers benchmarks bundled with Soma; it does not yet
   expose a third-party plugin or entry-point mechanism. This guide is therefore
   for contributors working in a Soma source checkout or fork. Install that
   checkout in editable mode while developing (``pip install -e .``), and submit
   generally useful benchmarks upstream so installed releases can discover them.

What ``reproduce`` does
-----------------------

For every canonical seed, Soma performs the same sequence:

1. ``curate`` converts the downloaded dataset into Soma's ``dataset.csv`` and
   ``splits.csv`` contracts.
2. ``build_config`` combines those manifests with the benchmark protocol and
   selected axes, such as ``encoder`` or ``spacing``.
3. :class:`~soma.pipeline.Pipeline` runs the experiment.
4. ``score`` reads the primary metric from the run directory.
5. ``expected`` selects the matching published or regression reference.

Define the benchmark contract
-----------------------------

:class:`~soma.benchmarks.Benchmark` is a structural protocol: implement the
required attributes and methods; there is no base class to subclass. A typical
benchmark module has this shape:

.. code-block:: python

   from pathlib import Path
   from typing import Any

   from soma.benchmarks import (
       Facet,
       ReferenceRow,
       expected_rows,
       register_benchmark,
       score_from_summary,
   )
   from soma.config import PipelineConfig
   from soma.curation.manifest import CuratedManifest


   class MyBenchmark:
       name = "my-study"
       facet = Facet(
           fixed={"task": "binary_classification"},
           varied=("encoder",),
       )
       canonical_seeds = (0, 1, 2)
       primary_metric = "test/auroc_mean"
       reference_environment: dict[str, str] = {}

       def curate(
           self, raw_root: str | Path, out_dir: str | Path
       ) -> CuratedManifest:
           return curate_my_study(raw_root, out_dir)

       def build_config(
           self,
           *,
           encoder: str = "uni2",
           dataset_csv: str | Path,
           splits_csv: str | Path,
           output_root: str | Path,
           seed: int,
           overrides: dict[str, Any] | None = None,
           **kwargs: Any,
       ) -> PipelineConfig:
           return build_my_study_config(
               encoder=encoder,
               dataset_csv=dataset_csv,
               splits_csv=splits_csv,
               output_root=output_root,
               seed=seed,
               overrides=overrides,
           )

       def expected(self, **axes: Any) -> list[ReferenceRow]:
           return expected_rows("my-study", **axes)

       def score(self, run_dir: str | Path) -> dict[str, float]:
           return score_from_summary(run_dir)


   register_benchmark(MyBenchmark())

The CLI supplies ``dataset_csv``, ``splits_csv``, ``output_root``, ``seed``, and
cache ``overrides`` to ``build_config``. Your declared varied axes should also be
keyword arguments. Use ``score_from_summary`` when the pipeline already writes
your primary metric to ``summary.json``; implement a custom scorer only when the
benchmark has genuinely different matching or aggregation rules.

Add references
--------------

Create ``soma/benchmarks/reference/my-study.csv``. Columns before ``metric`` are
the axes used to select a row:

.. code-block:: text

   encoder,metric,expected,tolerance,source,kind,label,url
   uni2,test/auroc_mean,0.91,0.02,our reproducibility run,gate,,
   virchow2,test/auroc_mean,0.93,,published leaderboard,external,Official result,https://example.org

Use ``kind=gate`` for a Soma regression anchor with a tolerance band. Use
``kind=external`` for a published value from another implementation; it is shown
for comparison and never produces PASS or FAIL. Keep the row keys aligned with
the values returned by ``Facet`` and accepted by ``build_config``.

Register and verify
-------------------

Bundled benchmarks live in ``soma/benchmarks/``. Import the module from
``soma/benchmarks/__init__.py`` so registration happens when the package loads,
and include the reference CSV as package data.

Verify discovery before launching an expensive run:

.. code-block:: console

   soma list benchmarks
   soma reproduce my-study --raw-root /path/to/data --seeds 1

Then run the canonical seeds and inspect the comparison:

.. code-block:: console

   soma reproduce my-study --raw-root /path/to/data
   soma leaderboard my-study --root soma_reproduce

Test the smallest useful surface
--------------------------------

Add focused tests that prove:

* the benchmark is returned by ``get_benchmark`` and ``soma list benchmarks``;
* ``curate`` produces valid manifests from a minimal fixture;
* ``build_config`` resolves one explicit set of axes to the expected
  :class:`~soma.config.PipelineConfig`;
* ``expected`` selects the correct reference row; and
* ``score`` returns ``primary_metric`` from a minimal run directory.

For a family of related datasets, register one benchmark per member with names
such as ``my-study/cohort-a`` and store their references in a shared
``reference/my-study.csv`` keyed by the member axis. Running
``soma reproduce my-study`` then fans out over the registered family while
keeping each member's data and outputs separate.

.. note::

   The generic ``reproduce`` command currently exposes ``--encoder`` and
   ``--spacing`` as selectable axes. A benchmark may record other varied axes for
   leaderboard views, but making a new axis selectable from ``reproduce`` also
   requires adding its CLI option.

This guide is expanded in the benchmarking documentation work that follows.
