from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class KeypointResult:
    frame_path: Path
    keypoints: np.ndarray  # shape: (num_people, num_keypoints, 3) (x, y, conf)
    scores: np.ndarray  # object confidence scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_path": str(self.frame_path),
            "keypoints": self.keypoints.tolist(),
            "scores": self.scores.tolist(),
        }


class PoseEstimator:
    """
    Wrapper around a YOLOv8 pose model for keypoint extraction.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n-pose.pt",
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.iou = iou

    def predict_on_frame(self, frame_path: str | Path) -> KeypointResult:
        result = self.model.predict(
            source=str(frame_path),
            save=False,
            stream=False,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )[0]

        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints else np.empty((0, 17, 3))
        scores = result.boxes.conf.cpu().numpy() if result.boxes else np.empty((0,))

        return KeypointResult(frame_path=Path(frame_path), keypoints=keypoints, scores=scores)

    def predict_on_frames(
        self,
        frames: Iterable[str | Path],
        *,
        save_json: Optional[str | Path] = None,
    ) -> list[KeypointResult]:
        outputs: list[KeypointResult] = []
        for frame in frames:
            outputs.append(self.predict_on_frame(frame))

        if save_json:
            data = [item.to_dict() for item in outputs]
            json_path = Path(save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return outputs

