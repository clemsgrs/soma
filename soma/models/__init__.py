"""``soma.models``: what a learner trains.

The composed models (:class:`EmbeddingModel`, :class:`MILModel`, :class:`DenseModel`),
their components — aggregators, task heads (``heads``), decoders — with their
registries, the ``encoders`` registry and encoder validation, and the detection target
encoding (``detection_targets``).
"""

from soma.models.aggregators import (
    ABMIL,
    CLAM_MB,
    CLAM_SB,
    DSMIL,
    DTFDMIL,
    HIPT,
    Aggregator,
    AggregatorOutput,
    MaxPool,
    MeanPool,
    TransMIL,
    aggregator_registry,
)
from soma.models.decoders import (
    Decoder,
    HeavyConvDecoder,
    LightweightConvDecoder,
    LinearDecoder,
    decoder_registry,
)
from soma.models.heads import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    CoxSurvivalHead,
    DetectionHead,
    DiscreteSurvivalHead,
    MulticlassClassificationHead,
    OrdinalClassificationHead,
    RegressionHead,
    SegmentationHead,
    TaskHead,
    task_registry,
)
from soma.models.model import (
    DenseModel,
    DenseModelOutput,
    EmbeddingModel,
    EmbeddingModelOutput,
    MILModel,
    MILModelOutput,
)

__all__ = [
    # composed models
    "DenseModel",
    "DenseModelOutput",
    "EmbeddingModel",
    "EmbeddingModelOutput",
    "MILModel",
    "MILModelOutput",
    # aggregators
    "ABMIL",
    "Aggregator",
    "AggregatorOutput",
    "CLAM_MB",
    "CLAM_SB",
    "DSMIL",
    "DTFDMIL",
    "HIPT",
    "MaxPool",
    "MeanPool",
    "TransMIL",
    "aggregator_registry",
    # heads
    "BinaryClassificationHead",
    "BranchAwareClassificationHead",
    "CoxSurvivalHead",
    "DetectionHead",
    "MulticlassClassificationHead",
    "OrdinalClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "DiscreteSurvivalHead",
    "TaskHead",
    "task_registry",
    # decoders
    "Decoder",
    "HeavyConvDecoder",
    "LightweightConvDecoder",
    "LinearDecoder",
    "decoder_registry",
]
