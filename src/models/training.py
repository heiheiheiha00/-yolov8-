from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


@dataclass(slots=True)
class TrainingConfig:
    data_yaml: str | Path
    model_architecture: str = "yolov8n-pose.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: Optional[str] = None
    project: str = "runs/train"
    name: str = "pose"


def train_pose_model(config: TrainingConfig) -> Path:
    """
    Launch training using the ultralytics YOLO API.

    Returns the path to the best trained weights.
    """
    model = YOLO(config.model_architecture)
    result = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=config.project,
        name=config.name,
    )
    if result is None:
        raise RuntimeError("Training did not return results.")
    best = Path(result.save_dir) / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected trained weights not found at {best}")
    return best

