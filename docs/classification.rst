Classification
==============

Choose binary classification for two classes, multiclass classification for an
unordered set of classes, or ordinal classification for ordered integer grades.

Binary
------

.. autoclass:: soma.tasks.classification.BinaryClassificationHead
   :members:

Multiclass
----------

.. autoclass:: soma.tasks.classification.MulticlassClassificationHead
   :members:

Ordinal
-------

.. autoclass:: soma.tasks.ordinal_classification.OrdinalClassificationHead
   :members:

Metric compatibility
--------------------

``multiclass_classification`` accepts ``qwk`` as an opt-in metric when the
class labels have an ordinal interpretation; the loss stays cross-entropy.

Benchmarks
----------

* :doc:`eva-patch-classification-benchmark` — frozen-tile-probe runs of these
  heads on the EVA patch-level datasets (``bach``, ``breakhis``, ``crc``,
  ``mhist``, ``patch_camelyon``).
