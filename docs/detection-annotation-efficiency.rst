Detection annotation efficiency
===============================

Measure how many annotated training images each frozen encoder needs to approach
its full-data cell-detection score. The study trains the detection benchmark
recipe on nested subsets of the training set, on OCELOT and MIDOG 2022, and
reports a learning curve per encoder. Results will be published with the
accompanying paper.

Protocol
--------

**Annotation unit.** Subsets are counted in the unit a pathologist annotates:
one field-of-view patch for OCELOT (``sample_id``) and one region of interest for
MIDOG (``source_wsi``). All tiles cut from a MIDOG region stay together.

**Ladder.** Each dataset has a fixed ladder of subset sizes below its full
training set:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Dataset
     - Full training set
     - Subset sizes
   * - OCELOT
     - 400 patches
     - 256, 128, 64, 32, 16, 8
   * - MIDOG 2022
     - 212 regions
     - 128, 64, 32, 16, 8, 4

**Nested subsets.** For each seed (0, 1 and 2 by default), the training units are
sorted and shuffled once. A subset of size *N* is the first *N* units of that
order, so every subset contains all smaller subsets of the same seed. The draw
is random, with no stratification by object count or domain, so small subsets
can differ a lot in how many cells or mitoses they contain. The report records
the object count of every subset.

**Fixed tune and test.** Every subset is evaluated against the full ``tune`` and
``test`` splits. Only the training set changes. Because model selection always
uses the full ``tune`` split, results on small subsets are optimistic about
selection.

**Training recipe.** Each run uses the committed benchmark recipe for the
dataset: a frozen encoder with the ``lightweight_conv`` decoder, early stopping
and best-on-``tune`` checkpointing. Subsets smaller than one eighth of the full
training set would see too few optimizer steps, so they are stretched by a factor
``scale = ceil((full / 8) / N)``:

* the number of epochs is multiplied by ``scale``;
* ``tune`` is evaluated every ``scale`` epochs, so every run gets at most as many
  ``tune`` evaluations as a full-data run;
* patience is unchanged, since it counts ``tune`` evaluations.

**Scoring.** Scoring matches the :doc:`OCELOT <ocelot-detection-benchmark>` and
MIDOG benchmarks. Detection thresholds are selected on ``tune``, frozen, and
applied once to ``test``. OCELOT reports class-aware mean F1 at 3 µm with greedy
matching. MIDOG reports F1 at 7.5 µm with one-to-one (Hungarian) matching, after
tiles are stitched back to their region.

**Full-data reference.** The full-data score and its spread across seeds come
from the benchmark sweep's own runs. These runs are not repeated.

**Summary measures.** For each encoder and dataset:

* **N\* (own)**: the smallest subset whose mean score reaches 95% of the
  encoder's own full-data score;
* **N\* (global)**: the smallest subset that reaches 95% of the best full-data
  score of any encoder;
* **noise-band crossing**: the smallest subset whose mean score reaches the
  full-data mean minus one standard deviation;
* **rank agreement**: Spearman and Kendall correlation between the encoder
  ranking at each subset size and the full-data ranking;
* **cross-dataset agreement**: whether encoders ordered by N\* come in the same
  order on both datasets.

Run the study
-------------

The study reuses the outputs of the detection benchmark sweep: its cached dense
features and its scored full-data runs. You need a GPU and a Hugging Face token
for the gated encoders. The curated manifests must be in
``<data-root>/<dataset>/curated/`` (``dataset.csv`` and ``splits.csv``). See
:doc:`curation` for how to prepare OCELOT and MIDOG 2022.

1. From the repository root, run the benchmark sweep if you have not already. The
   ``extract`` phase caches the dense features, and the ``rank`` phase trains and
   scores the full-data runs::

       python examples/detection_benchmark/campaign.py extract \
           --data-root data --datasets ocelot midog
       python examples/detection_benchmark/campaign.py rank \
           --data-root data --datasets ocelot midog

2. Run the efficiency phase with its own output folder, pointing ``--full-root``
   at the sweep's output folder::

       python examples/detection_benchmark/campaign.py efficiency \
           --data-root data --datasets ocelot midog \
           --out-root output/detection_benchmark_efficiency \
           --full-root output/detection_benchmark

   The phase writes the subset splits, then trains and scores one run per
   dataset, encoder, subset size and seed, largest subsets first. It finishes by
   writing the report.

The phase can be stopped and restarted. A run that already has its scores is
skipped, so running the same command again resumes where it stopped. Add
``--dry-run`` to list the runs that are still to do, with their epochs, number of
training images and object counts.

These options narrow the work:

* ``--encoders`` picks encoders from the roster (all seven by default);
* ``--seeds`` picks seeds;
* ``--ladder`` replaces the subset sizes.

They also let you split the work across GPUs: start one process per GPU, each
with its own ``--encoders``. Each process writes a report covering only the
encoders, seeds and subset sizes it was given, and the last one to finish
overwrites the others. When all processes are done, rebuild the report over
everything without training anything::

    python examples/detection_benchmark/campaign.py efficiency-report \
        --data-root data --datasets ocelot midog \
        --out-root output/detection_benchmark_efficiency \
        --full-root output/detection_benchmark

``--full-source`` selects where the full-data reference comes from:

* ``sweep`` reads the scored runs under ``--full-root``;
* ``report`` reads the sweep's ``ranking_report.json``;
* ``train`` trains the full training set as one more subset, with the same code
  as the other subsets;
* ``auto`` (the default) uses ``sweep`` when scored runs exist and otherwise
  ``report``.

The option applies to every dataset in the command. To use a different source
per dataset, run one dataset per command.

Outputs
-------

Everything is written under ``--out-root``:

* ``splits/<dataset>/seed<S>_n<N>.csv`` and ``seed<S>_n<N>.dataset.csv``: the
  split and manifest of each subset;
* ``n<N>/<dataset>/<encoder>/replicate_<S>/``: one run per subset, encoder and
  seed, with its scores in ``metrics.json``;
* ``efficiency_report.json``: per dataset and encoder, the score of every seed at
  every subset size, the mean and standard deviation, the object counts, the
  epochs actually trained, the full-data reference, N\*, the noise-band crossing
  and the rank agreement, plus the cross-dataset comparison;
* ``efficiency_curves.png``: test score against subset size on a log scale, one
  line per encoder and one panel per dataset, with each encoder's full-data
  score shown as a band.

.. seealso::

   * :doc:`ocelot-detection-benchmark`: the OCELOT benchmark and its scoring.
   * :doc:`detection`: the detection head, target encoding and F1 evaluator.
   * :doc:`curation`: dataset curators and split policies.
