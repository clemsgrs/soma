Installation
============

soma requires Python 3.11 or later:

.. code-block:: bash

   pip install soma-pathology

Optional extras add the dependencies of specific paths: ``[croma]`` for CRoMa
data preparation and ``[hest]`` for HEST curation, for example
``pip install 'soma-pathology[hest]'``.

The first run with a foundation model downloads its weights; gated models such as
``uni2`` need a Hugging Face access token (``HF_TOKEN``). Feature extraction is
faster on a GPU, but a GPU is not required.

Then run the :doc:`getting-started notebook <getting-started>`: one experiment on
synthetic slides with a weight-free encoder, which needs no GPU, no weights and no
network.
