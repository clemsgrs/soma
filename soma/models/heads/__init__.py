"""Task heads — map aggregated representations to predictions."""

from soma.models.heads.base import TaskHead
from soma.models.heads.registry import task_registry

# Import to trigger registration
from soma.models.heads.classification import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    ClassificationParams,
    MulticlassClassificationHead,
    infer_label_map,
    infer_ordinal_label_map,
)
from soma.models.heads.detection import DetectionHead, DetectionParams
from soma.models.heads.ordinal_classification import OrdinalClassificationHead
from soma.models.heads.regression import RegressionHead, RegressionParams
from soma.models.heads.segmentation import SegmentationHead, SegmentationParams, load_mask
from soma.models.heads.survival import (
    CoxSurvivalHead,
    CoxSurvivalParams,
    DiscreteSurvivalHead,
    DiscreteSurvivalParams,
    infer_num_bins,
)

__all__ = [
    "TaskHead",
    "task_registry",
    "ClassificationParams",
    "CoxSurvivalParams",
    "DetectionParams",
    "RegressionParams",
    "SegmentationParams",
    "DiscreteSurvivalParams",
    "infer_label_map",
    "infer_num_bins",
    "infer_ordinal_label_map",
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "BranchAwareClassificationHead",
    "OrdinalClassificationHead",
    "RegressionHead",
    "DiscreteSurvivalHead",
    "CoxSurvivalHead",
    "SegmentationHead",
    "DetectionHead",
    "load_mask",
]
