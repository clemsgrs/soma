Classification
==============

Classification heads map a feature vector to class predictions. The vector can
come directly from a frozen encoder or from a slide-level :doc:`aggregator
<aggregators>`. Choose binary classification for two classes, multiclass
classification for an unordered set of classes, or ordinal classification for
ordered integer grades.

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
class labels have an ordinal interpretation. The task still uses
cross-entropy loss; the metric only changes how results are summarized.

Benchmarks
----------

* :doc:`eva-patch-classification-benchmark` — frozen-tile-probe runs of these
  heads on the EVA patch-level datasets (``bach``, ``breakhis``, ``crc``,
  ``mhist``, ``patch_camelyon``).
