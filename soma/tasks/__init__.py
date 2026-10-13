"""Task heads — map aggregated representations to predictions."""

from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

# Import to trigger registration
from soma.tasks.classification import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    ClassificationParams,
    MulticlassClassificationHead,
    infer_label_map,
    infer_ordinal_label_map,
)
from soma.tasks.detection import DetectionHead, DetectionParams
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead, RegressionParams
from soma.tasks.segmentation import SegmentationHead, SegmentationParams, load_mask
from soma.tasks.survival import (
    CoxSurvivalHead,
    CoxSurvivalParams,
    SurvivalHead,
    SurvivalParams,
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
    "SurvivalParams",
    "infer_label_map",
    "infer_num_bins",
    "infer_ordinal_label_map",
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "BranchAwareClassificationHead",
    "OrdinalClassificationHead",
    "RegressionHead",
    "SurvivalHead",
    "CoxSurvivalHead",
    "SegmentationHead",
    "DetectionHead",
    "load_mask",
]
