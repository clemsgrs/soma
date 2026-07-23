#!/usr/bin/env bash
# Smoke-test the tutorial notebooks: execute each one and fail on any error.
#
# The committed notebooks ship WITHOUT outputs (the docs build never executes
# them — nbsphinx_execute = "never" — and rendering empty cells is intentional).
# This script therefore executes each notebook to a throwaway copy and discards
# it: it verifies the tutorial code still runs against the current soma API
# WITHOUT writing outputs back into the committed files. Run it after changing
# soma's API or a tutorial's cells.
#
# Notebooks run on CPU with ungated encoders (`phikon`, `hibou-b`, `moozy-slide`),
# so no GPU and no HF token are required (just a one-time weights download). They
# import soma from the *repo source*, not an installed wheel.
#
# Usage:
#   scripts/execute_tutorials.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=""        # force CPU — matches the "runs anywhere" promise
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Ungated encoders, so no HF token is required; we inherit whatever is in the
# environment and do not force one either way.

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for nb in docs/tutorials/walkthrough-tile-level.ipynb \
          docs/tutorials/walkthrough-slide-mil.ipynb \
          docs/tutorials/walkthrough-slide-encoder.ipynb \
          docs/tutorials/walkthrough-segmentation.ipynb \
          docs/tutorials/walkthrough-detection.ipynb \
          docs/tutorials/walkthrough-composite.ipynb \
          docs/tutorials/walkthrough-attention-segmentation.ipynb; do
    echo ">>> executing $nb"
    # --output-dir sends the executed copy to a temp dir; the committed file is
    # never modified, so outputs stay out of version control.
    jupyter nbconvert --to notebook --execute \
        --output-dir "$TMP" \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb"
done

echo "done — all tutorials executed without error (committed outputs unchanged)"
