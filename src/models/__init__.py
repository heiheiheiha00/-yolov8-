"""
Utilities for training and loading YOLOv8 pose models.
"""

from .training import train_pose_model, TrainingConfig
from .inference import PoseInferenceService, PoseComparisonResult

__all__ = [
    "train_pose_model",
    "TrainingConfig",
    "PoseInferenceService",
    "PoseComparisonResult",
]

