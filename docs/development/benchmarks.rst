:orphan:

Benchmark development
=====================

Adding a HEST task
------------------

HEST extensions reuse the task-agnostic ``curate_hest`` curator and closed-form probe. A
new task needs data, one ``HEST_TASKS`` entry, and external reference rows; it needs no new
curator or probe code.

1. Download only the task data, excluding precomputed features::

      hf download MahmoodLab/hest-bench --include 'HCC/*' --exclude 'fm_v1/*' \
          --repo-type dataset --local-dir /path/to/hest-bench

2. Curate it into a ``spatial_expression`` Manifest::

      python -m soma.curation.hest --raw-root /path/to/hest-bench/HCC \
          --output-dir /path/to/curated/HCC --task HCC

3. Add the task id to ``HEST_TASKS`` in ``soma/benchmarks/hest.py``. The module registers
   one ``HestBenchmark(task)`` per entry::

      HEST_TASKS = (..., "HCC")

4. Add one ``kind=external`` row per published encoder result to
   ``soma/benchmarks/reference/hest.csv``, including its Pearson value, label, and URL.
   A task without a published result can run, but has no external comparison.

After registration, verify that ``soma list benchmarks`` includes ``hest/HCC`` and run it
with ``soma reproduce hest/HCC``.
