from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.data_processing.features import FeatureSet


POSE_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_pose_on_frame(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    output = frame.copy()
    for x, y, conf in keypoints:
        if conf < 0.3:
            continue
        cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), thickness=-1)

    for start, end in POSE_EDGES:
        x1, y1, c1 = keypoints[start]
        x2, y2, c2 = keypoints[end]
        if c1 < 0.3 or c2 < 0.3:
            continue
        cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
    return output


def annotate_video_with_pose(
    sample: Iterable[FeatureSet],
    frame_paths: Iterable[Path],
    *,
    output_path: str | Path,
) -> Path:
    frame_paths = list(frame_paths)
    features = list(sample)
    if not frame_paths or not features:
        raise ValueError("Frame paths and features must not be empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    fps = 30
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_path, feature in zip(frame_paths, features):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        overlay = draw_pose_on_frame(frame, feature.keypoints)
        writer.write(overlay)

    writer.release()
    return output_path

