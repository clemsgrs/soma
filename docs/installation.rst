Installation
============

soma requires Python 3.11 or later:

.. code-block:: bash

   pip install soma-pathology

Two optional extras add the dependencies of benchmark data preparation:
``[croma]`` for the CRoMa cohorts and ``[hest]`` for HEST curation, for example
``pip install 'soma-pathology[hest]'``.

The first run with a foundation model downloads its weights. A gated model such as
``uni2`` needs its terms accepted on Hugging Face and an access token in
``HF_TOKEN``. Feature extraction is faster on a GPU, but a GPU is not required.

Then run the :doc:`getting-started notebook <getting-started>`: one experiment on
synthetic slides with a weight-free encoder, which needs no GPU, no weights and no
network.
