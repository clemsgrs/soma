Classification
==============

Classification task heads map the aggregated bag representation to class
predictions. ``soma`` ships binary, multiclass, and ordinal variants. The
aggregator is the method home for the bag → slide-level step that feeds these
heads — see :doc:`aggregators`.

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
