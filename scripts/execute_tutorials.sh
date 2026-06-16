#!/usr/bin/env bash
# Re-execute the tutorial notebooks in place, refreshing their committed outputs.
#
# The docs build never executes notebooks (nbsphinx_execute = "never"); this is
# how their outputs are kept in sync with the code. Run it after changing soma's
# API or the tutorial cells.
#
# Notebooks run on CPU with the ungated `phikon` encoder, so no GPU and no HF
# token are required (just a one-time weights download). They import soma from
# the *repo source*, not an installed wheel.
#
# Usage:
#   scripts/execute_tutorials.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=""        # force CPU — matches the "runs anywhere" promise
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# phikon is ungated, so no HF token is required; we inherit whatever is in the
# environment and do not force one either way.

for nb in docs/tutorials/walkthrough-slide-level.ipynb \
          docs/tutorials/walkthrough-dense.ipynb; do
    echo ">>> executing $nb"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb"
done

echo "done — committed outputs refreshed"
